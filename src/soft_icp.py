"""SoftICPModel: EM-ICP 软对应 + 刚体变换 (GenericEM 实例)。

点集配准的标准 EM (对应 ↔ 几何 这一估计对的最小形态): 源点 S 与目标
点 T_obs, 隐变量 = 每目标点的软对应 (由哪个源点生成), 参数 = 刚体变换
(旋转角 θ + 平移 t)。

  E 步: q(j,i) ∝ exp(−‖t_j − (R·s_i + t)‖² / (2σ²·T))   (软对应)
  M 步: 期望源 ŝ_j = Σ_i q(j,i)·s_i → Kabsch 求 (R,t)     (加权 Procrustes)

与已退役的 SPN 质心压缩无关: 这里的软对应是配准问题的隐变量, M 步是
闭式刚体变换 (Kabsch/SVD), 不涉及把点平均到流形外。

这是 GenericEM 框架的第三个实例, 与前两个的区别是: 隐变量是「点对应」
(不是像素归属/外观), 参数是连续刚体变换, M 步是闭式 Procrustes。
"""

from __future__ import annotations

import numpy as np


class SoftICPModel:
    """EM-ICP: 软对应 E 步 + Kabsch M 步。"""

    def __init__(self, source: np.ndarray, sigma: float = 0.05):
        self.source = np.asarray(source, dtype=float)  # (N, 2)
        if self.source.ndim != 2 or self.source.shape[1] != 2:
            raise ValueError("source 必须是 (N, 2)")
        self.sigma = sigma
        self.n = self.source.shape[0]

    # ── 正向模型 ──────────────────────────────────────────────────

    @staticmethod
    def _rotation(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def _transform(self, params: tuple[float, float, float]) -> np.ndarray:
        theta, tx, ty = params
        r = self._rotation(theta)
        return self.source @ r.T + np.array([tx, ty])  # (N, 2)

    def sample(
        self, params: tuple[float, float, float], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """(θ, tx, ty) → 目标点云 = 变换源点 + 噪声 (生成时已知对应)。"""
        rng = rng or np.random.default_rng(0)
        return self._transform(params) + rng.normal(0.0, self.sigma, (self.n, 2))

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self, params: tuple[float, float, float], observation: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """每目标点的软对应 q(j,i) = P(target j ↔ source i) (N, N)。"""
        t = self._transform(params)
        d = ((observation[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
        inv = 1.0 / max(temperature, 1e-8)
        logq = -d / (2.0 * self.sigma**2) * inv
        logq = logq - logq.max(axis=1, keepdims=True)  # log-sum-exp 稳定
        q = np.exp(logq)
        return q / q.sum(axis=1, keepdims=True)

    # ── M 步 (Kabsch / 加权 Procrustes) ───────────────────────────

    def maximize(
        self,
        resp: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, float, float],
        damping: float = 0.0,
    ) -> tuple[float, float, float]:
        """期望源 ŝ_j = Σ_i q(j,i)·s_i → Kabsch 求刚体变换。"""
        expected_source = resp @ self.source  # (N, 2)
        r, t = self._kabsch(expected_source, observation)
        theta = float(np.arctan2(r[1, 0], r[0, 0]))
        new = (theta, float(t[0]), float(t[1]))
        if damping > 0.0:
            new = tuple(
                (1.0 - damping) * a + damping * b for a, b in zip(new, params, strict=True)
            )
        return new

    @staticmethod
    def _kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """两对应点集 → 最小二乘刚体变换 (R, t), 使 q ≈ R·p + t。"""
        p_mean = p.mean(axis=0)
        q_mean = q.mean(axis=0)
        pc = p - p_mean
        qc = q - q_mean
        h = pc.T @ qc  # (2,2) 互协方差
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:  # 反射修正
            vt[-1, :] *= -1.0
            r = vt.T @ u.T
        t = q_mean - r @ p_mean
        return r, t

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, params: tuple[float, float, float], observation: np.ndarray) -> float:
        """混合对应对数似然 Σ_j log Σ_i exp(−‖t_j−R·s_i−t‖²/2σ²)。"""
        t = self._transform(params)
        d = ((observation[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
        logp = -d / (2.0 * self.sigma**2)
        m = logp.max(axis=1, keepdims=True)
        return float(np.sum(m + np.log(np.exp(logp - m).sum(axis=1))))
