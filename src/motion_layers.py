"""MotionLayersModel: 运动分割↔光流的 EM (1D 光流场 → K 运动层)。

观测 u(x) = 每像素速度 (一维光流), 由 K 个运动层组成, 每层一个恒定速度
v_k。隐变量 = 每像素运动层归属; 参数 θ = (v_1, ..., v_K)。

  E 步 (responsibilities): 软归属 q_k(x) ∝ exp(−(u−v_k)²/2σ²·T), 再空间
     平滑 (鼓励分割连续, 这是「分割」与「聚类」的区别)
  M 步 (maximize): v_k = 按归属加权的平均速度

「运动分割 ↔ 光流」: 光流 (每像素速度) 是观测, 运动层速度是参数, 分割
归属是隐变量。与透明层 (强度 GMM) 结构相似但域不同 —— 这里是运动域,
且加了空间平滑先验使分割连续。
"""

from __future__ import annotations

import numpy as np


class MotionLayersModel:
    """光流场 → K 运动层 (软归属 + 空间平滑 + 加权速度)。"""

    def __init__(self, k: int = 2, n: int = 64, sigma: float = 0.05, smooth: int = 4):
        self.k = k
        self.n = n
        self.sigma = sigma
        self.smooth = smooth
        self.x = np.arange(n, dtype=float) / max(n - 1, 1)

    def sample(
        self, params: tuple[float, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """K 个速度 → 分段常数光流 (均匀分块) + 噪声。"""
        rng = rng or np.random.default_rng(0)
        seg = np.arange(self.n) * self.k // self.n  # 每像素所属层 (均匀分块)
        u = np.array([params[s] for s in seg], dtype=float)
        return u + rng.normal(0.0, self.sigma, self.n)

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self, params: tuple[float, ...], observation: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """软运动层归属 (n, k), 再空间平滑鼓励连续分割。"""
        v = np.asarray(params, dtype=float)
        inv = 1.0 / max(temperature, 1e-8)
        logq = -(observation[:, None] - v[None, :]) ** 2 / (2.0 * self.sigma**2) * inv
        logq = logq - logq.max(axis=1, keepdims=True)
        q = np.exp(logq)
        q = q / q.sum(axis=1, keepdims=True)
        for _ in range(self.smooth):
            q = (np.roll(q, 1, axis=0) + q + np.roll(q, -1, axis=0)) / 3.0
        return q

    # ── M 步 ──────────────────────────────────────────────────────

    def maximize(
        self,
        q: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, ...],
        damping: float = 0.0,
    ) -> tuple[float, ...]:
        """各层速度 = 按归属加权的平均光流。"""
        new = tuple(
            float(np.sum(q[:, j] * observation) / max(np.sum(q[:, j]), 1e-12))
            for j in range(self.k)
        )
        if damping > 0.0:
            new = tuple(
                (1.0 - damping) * a + damping * b for a, b in zip(new, params, strict=True)
            )
        return new

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, params: tuple[float, ...], observation: np.ndarray) -> float:
        """混合速度对数似然 Σ_x log Σ_k (1/K)·N(u|v_k,σ)。"""
        v = np.asarray(params, dtype=float)
        logp = -(observation[:, None] - v[None, :]) ** 2 / (2.0 * self.sigma**2)
        logp = logp - logp.max(axis=1, keepdims=True)
        return float(np.sum(np.log(np.exp(logp).mean(axis=1)) + logp.max(axis=1)))
