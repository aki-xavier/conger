"""DepthNormalModel: 深度↔法向的坐标上升 (GenericEM 实例)。

一维曲面: 深度 z(x), 法向的切向投影即斜率 s(x) = dz/dx。观测同时给出
噪声深度 ẑ (如双目) 与噪声斜率 ŝ (如明暗)。约束 s = dz/dx 把两者耦合:

  E 步 (responsibilities): 固定 z, 调和法向 s = (ŝ + dz/dx) / 2
  M 步 (maximize): 固定 s, 拟合深度 z = argmin ‖z−ẑ‖² + λ·‖Dz−s‖²

这是「深度 ↔ 法向」这一估计对的坐标上升: 深度给低频、法向给高频,
两者交替对齐到同一曲面。无离散隐变量, M 步是闭式 Tikhonov 线性解。

与前几例的区别: 这里两个「子估计」是同一物理量 (曲面) 的两种表示
(高度场 vs 梯度场), 靠微分算子 D 耦合, 不是隐变量/观测分解。
"""

from __future__ import annotations

import numpy as np


class DepthNormalModel:
    """深度↔法向 → 调和法向 E 步 + Tikhonov 深度拟合 M 步。"""

    def __init__(
        self,
        z_obs: np.ndarray,
        s_obs: np.ndarray,
        lam: float = 0.5,
        sigma: float = 0.02,
    ):
        self.z_obs = np.asarray(z_obs, dtype=float)  # (n,) 噪声深度
        self.s_obs = np.asarray(s_obs, dtype=float)  # (n-1,) 噪声斜率
        if self.z_obs.shape[0] != self.s_obs.shape[0] + 1:
            raise ValueError("s_obs 长度应比 z_obs 少 1")
        self.lam = lam
        self.sigma = sigma
        self.n = self.z_obs.shape[0]
        d = np.zeros((self.n - 1, self.n))
        for i in range(self.n - 1):
            d[i, i] = -1.0
            d[i, i + 1] = 1.0
        self.d = d

    @staticmethod
    def _grad(z: np.ndarray) -> np.ndarray:
        return np.diff(z)

    # ── E 步 (固定深度, 调和法向) ────────────────────────────────

    def responsibilities(
        self, z: tuple[float, ...], observation, temperature: float = 1.0
    ) -> np.ndarray:
        """s = (ŝ + dz/dx)/2: 观测斜率与当前深度梯度的调和。"""
        return (self.s_obs + self._grad(np.asarray(z))) / 2.0

    # ── M 步 (固定法向, Tikhonov 拟合深度) ───────────────────────

    def maximize(
        self,
        resp: np.ndarray,
        observation,
        z: tuple[float, ...],
        damping: float = 0.0,
    ) -> tuple[float, ...]:
        """z = argmin ‖z−ẑ‖² + λ·‖Dz−s‖² → (I + λDᵀD)z = ẑ + λDᵀs。"""
        a = np.eye(self.n) + self.lam * (self.d.T @ self.d)
        b = self.z_obs + self.lam * (self.d.T @ resp)
        z_new = np.linalg.solve(a + 1e-8 * np.eye(self.n), b)
        if damping > 0.0:
            z_new = (1.0 - damping) * z_new + damping * np.asarray(z)
        return tuple(float(v) for v in z_new)

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, z: tuple[float, ...], observation) -> float:
        """负联合残差 (深度数据项 + 斜率一致性项)。"""
        zz = np.asarray(z)
        return -float(
            np.sum((zz - self.z_obs) ** 2)
            + self.lam * np.sum((self._grad(zz) - self.s_obs) ** 2)
        ) / (2.0 * self.sigma**2)
