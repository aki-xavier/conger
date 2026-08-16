"""RetinexModel: 反照率↔光照的乘法分解 (Retinex 坐标上升, GenericEM 实例)。

生成模型: I(x) = A(x)·L(x), log 化后 log I = log A + log L。经典 Retinex
假设: 反照率 log A 分段常数 (K 段, 已知边界), 光照 log L 平滑 (线性)。

这是「坐标上升」而非带隐变量的 EM: 反照率 A 与光照 L 交替更新。映射到
GenericEM 框架时, 把光照 L=(l0,l1) 当参数 θ, 反照率 log A 当「中间量」
(E 步 responsibilities 返回), M 步固定 A 重估 L。规范自由度 (A→A·c,
L→L/c) 通过 E 步中心化 log A 打破 (DC 归光照)。

与透明层 (软 alpha 叠加) / 遮挡 (硬遮挡+深度序) 的区别: 这里是乘法
分解 (对数域加法), 且是坐标上升、无离散隐变量。
"""

from __future__ import annotations

import numpy as np


class RetinexModel:
    """反照率↔光照 → 分段常数反照率 + 平滑光照 (坐标上升)。"""

    def __init__(self, segments: np.ndarray, sigma: float = 0.03):
        self.segments = np.asarray(segments, dtype=int)
        if self.segments.ndim != 1:
            raise ValueError("segments 必须是一维")
        self.sigma = sigma
        self.n = self.segments.shape[0]
        self.k = int(self.segments.max()) + 1
        self.x = np.arange(self.n, dtype=float) / max(self.n - 1, 1)
        self.xm = np.stack([np.ones(self.n), self.x], axis=1)  # (n,2)

    # ── 正向模型 ──────────────────────────────────────────────────

    def _log_l(self, params: tuple[float, float]) -> np.ndarray:
        l0, l1 = params
        return l0 + l1 * self.x

    def render(
        self,
        log_a_seg: tuple[float, ...],
        params: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """(每段 log A, 光照系数) → 观测 I = exp(log A + log L + 噪声)。"""
        log_a = np.array([log_a_seg[s] for s in self.segments])
        rng = rng or np.random.default_rng(0)
        return np.exp(log_a + self._log_l(params) + rng.normal(0.0, self.sigma, self.n))

    def sample(
        self, params: tuple[float, float], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """正向模型 (全一反照率, 仅用于接口一致性; 测试用 render)。"""
        return self.render((0.0,) * self.k, params, rng)

    # ── E 步 (坐标上升: 固定光照, 估反照率) ───────────────────────

    def responsibilities(
        self, params: tuple[float, float], observation: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """固定 L, 分段常数估计 log A (中心化破规范), 返回 (n,)。"""
        log_i = np.log(observation + 1e-12)
        resid = log_i - self._log_l(params)
        log_a_seg = np.array(
            [resid[self.segments == k].mean() for k in range(self.k)]
        )
        sizes = np.array([np.sum(self.segments == k) for k in range(self.k)])
        log_a_seg = log_a_seg - float(np.sum(log_a_seg * sizes) / self.n)
        return np.array([log_a_seg[s] for s in self.segments])

    # ── M 步 (坐标上升: 固定反照率, 估光照) ───────────────────────

    def maximize(
        self,
        resp: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, float],
        damping: float = 0.0,
    ) -> tuple[float, float]:
        """固定 A, 线性拟合 log L = l0 + l1·x。"""
        log_i = np.log(observation + 1e-12)
        coef = np.linalg.lstsq(self.xm, log_i - resp, rcond=None)[0]
        new = (float(coef[0]), float(coef[1]))
        if damping > 0.0:
            new = tuple(
                (1.0 - damping) * a + damping * b for a, b in zip(new, params, strict=True)
            )
        return new

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, params: tuple[float, float], observation: np.ndarray) -> float:
        """重构残差负值 (log 域)。"""
        log_i = np.log(observation + 1e-12)
        log_a = self.responsibilities(params, observation)
        resid = log_i - log_a - self._log_l(params)
        return -float(np.sum(resid**2) / (2.0 * self.sigma**2))
