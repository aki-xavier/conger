"""OcclusionLayerModel: 两层遮挡 + 深度序的 EM 实例 (1D 探针)。

两层 L1, L2 (各为线性 a0+a1·x), 已知空间范围: L1 占据 [0, b], L2 占据
[a, 1], 重叠区 [a, b] (a<b)。观测: 非重叠区看到对应层, 重叠区看到「前
层」的值 (硬遮挡, 不是软叠加)。隐变量 D ∈ {0,1} 是深度序 (0 = L1 在
前, 1 = L2 在前); 参数 θ = (a0,a1,b0,b1)。

  E 步: 深度序后验 q = P(D=0|I) ∝ exp(−Σ_overlap (I−L1)²/2σ²)
  M 步: L1 用 [0,a] 全权 + 重叠区 q 权做加权线性拟合, L2 同理 (1−q)

非重叠区单独决定各层, 重叠区决定深度序 —— 这就是「遮挡 ↔ 深度序」
这一估计对的最小形态。与透明层 (软 alpha 叠加) 的区别: 这里是硬遮挡 +
单一深度序隐变量。
"""

from __future__ import annotations

import numpy as np


class OcclusionLayerModel:
    """两层遮挡 → 深度序软后验 E 步 + 加权线性拟合 M 步。"""

    def __init__(
        self, n: int = 96, a: float = 0.4, b: float = 0.6, sigma: float = 0.03
    ):
        if not (0.0 < a < b < 1.0):
            raise ValueError("需 0 < a < b < 1")
        self.n = n
        self.a, self.b = a, b
        self.sigma = sigma
        self.x = np.linspace(0.0, 1.0, n)
        self.xm = np.stack([np.ones(n), self.x], axis=1)  # (n,2)
        self.mask_1_only = self.x < a
        self.mask_2_only = self.x > b
        self.mask_overlap = (self.x >= a) & (self.x <= b)

    # ── 正向模型 ──────────────────────────────────────────────────

    def _layer(self, coeff: tuple[float, float]) -> np.ndarray:
        return coeff[0] + coeff[1] * self.x

    def render(
        self,
        params: tuple[float, float, float, float],
        front: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """(a0,a1,b0,b1) + 前层 front(0/1) → 观测。"""
        l1 = self._layer(params[:2])
        l2 = self._layer(params[2:])
        out = np.empty(self.n)
        out[self.mask_1_only] = l1[self.mask_1_only]
        out[self.mask_2_only] = l2[self.mask_2_only]
        out[self.mask_overlap] = (
            l1[self.mask_overlap] if front == 0 else l2[self.mask_overlap]
        )
        rng = rng or np.random.default_rng(0)
        return out + rng.normal(0.0, self.sigma, self.n)

    def sample(
        self,
        params: tuple[float, float, float, float],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """正向模型: 深度序按均匀先验采样 → 观测。"""
        rng = rng or np.random.default_rng(0)
        return self.render(params, 0 if rng.random() < 0.5 else 1, rng)

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self,
        params: tuple[float, float, float, float],
        observation: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """深度序后验 q = P(D=0|I) ∈ [0,1] (标量, 存成一维)。"""
        l1 = self._layer(params[:2])
        l2 = self._layer(params[2:])
        o = observation[self.mask_overlap]
        inv = 1.0 / max(temperature, 1e-8)
        r1 = float(np.sum((o - l1[self.mask_overlap]) ** 2)) * inv
        r2 = float(np.sum((o - l2[self.mask_overlap]) ** 2)) * inv
        logq = -r1 / (2.0 * self.sigma**2), -r2 / (2.0 * self.sigma**2)
        m = max(logq)
        e = np.exp(np.array(logq) - m)
        return e / e.sum()  # (2,) = [P(D=0), P(D=1)]

    # ── M 步 ──────────────────────────────────────────────────────

    def maximize(
        self,
        resp: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, float, float, float],
        damping: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """深度序后验下加权线性拟合重估两层。"""
        q = float(resp[0])
        w1 = np.where(self.mask_1_only, 1.0, np.where(self.mask_overlap, q, 0.0))
        w2 = np.where(self.mask_2_only, 1.0, np.where(self.mask_overlap, 1.0 - q, 0.0))
        c1 = self._fit(w1, observation)
        c2 = self._fit(w2, observation)
        if damping > 0.0:
            c1 = (1.0 - damping) * c1 + damping * np.array(params[:2])
            c2 = (1.0 - damping) * c2 + damping * np.array(params[2:])
        return (float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1]))

    def _fit(self, w: np.ndarray, observation: np.ndarray) -> np.ndarray:
        xwx = self.xm.T @ (w[:, None] * self.xm)
        xwy = self.xm.T @ (w * observation)
        return np.linalg.solve(xwx + 1e-8 * np.eye(2), xwy)

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(
        self, params: tuple[float, float, float, float], observation: np.ndarray
    ) -> float:
        """深度序混合对数似然: 非重叠区单层高斯, 重叠区 0.5 均匀混合。"""
        l1 = self._layer(params[:2])
        l2 = self._layer(params[2:])

        def logn(d: np.ndarray) -> np.ndarray:
            return -0.5 * (d / self.sigma) ** 2 - 0.5 * np.log(
                2 * np.pi * self.sigma**2
            )

        ll = float(np.sum(logn(observation[self.mask_1_only] - l1[self.mask_1_only])))
        ll += float(np.sum(logn(observation[self.mask_2_only] - l2[self.mask_2_only])))
        o = observation[self.mask_overlap]
        p = 0.5 * np.exp(-0.5 * ((o - l1[self.mask_overlap]) / self.sigma) ** 2) + (
            0.5 * np.exp(-0.5 * ((o - l2[self.mask_overlap]) / self.sigma) ** 2)
        )
        return ll + float(np.sum(np.log(p + 1e-12)))
