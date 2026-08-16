"""FigureGroundModel: 分割↔位姿的 EM (1D figure-ground)。

前景是一个区间 [c−r, c+r] (位姿 = 中心 c + 半宽 r), 强度 f; 背景强度 b。
隐变量 = 每像素前景归属; 参数 θ = (c, r, f, b)。

  E 步 (responsibilities): 软前景归属 q(x) (位姿先验 × 强度似然)
  M 步 (maximize): 强度用当前掩码均值重估; 位姿 (c,r) 坐标搜索最小化
     分段拟合残差

「分割 ↔ 位姿」: 分割 (哪像素是前景) 与位姿 (物体在哪、多大) 互相
约束 —— 位姿给分割先验, 分割给位姿/强度统计。M 步的坐标搜索让掩码能
随迭代移动/伸缩 (硬先验若直接用质心会卡死在初始掩码)。
"""

from __future__ import annotations

import numpy as np


class FigureGroundModel:
    """figure-ground → 软分割 E 步 + 位姿坐标搜索 M 步。"""

    def __init__(
        self,
        n: int = 100,
        sigma: float = 0.05,
        delta_c: float = 0.02,
        delta_r: float = 0.02,
    ):
        self.n = n
        self.sigma = sigma
        self.delta_c = delta_c
        self.delta_r = delta_r
        self.x = np.linspace(0.0, 1.0, n)

    def _fg_mask(self, c: float, r: float) -> np.ndarray:
        return np.abs(self.x - c) <= r

    def sample(
        self, params: tuple[float, float, float, float], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """(c, r, f, b) → 前景/背景分段的带噪观测。"""
        _, _, f, b = params
        rng = rng or np.random.default_rng(0)
        return np.where(self._fg_mask(params[0], params[1]), f, b) + rng.normal(
            0.0, self.sigma, self.n
        )

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self, params: tuple[float, float, float, float], observation: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """软前景归属 q(x) (位姿先验 × 强度似然)。"""
        c, r, f, b = params
        fg = self._fg_mask(c, r)
        inv = 1.0 / max(temperature, 1e-8)
        w_fg = np.where(fg, -0.5 * ((observation - f) / self.sigma) ** 2 * inv, -1e9)
        w_bg = np.where(~fg, -0.5 * ((observation - b) / self.sigma) ** 2 * inv, -1e9)
        m = np.maximum(w_fg, w_bg)
        e_fg = np.exp(w_fg - m)
        e_bg = np.exp(w_bg - m)
        return e_fg / (e_fg + e_bg)

    # ── M 步 (强度重估 + 位姿坐标搜索) ───────────────────────────

    def maximize(
        self,
        q: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, float, float, float],
        damping: float = 0.0,
    ) -> tuple[float, float, float, float]:
        c, r, _, _ = params
        mask = self._fg_mask(c, r)
        f = float(observation[mask].mean()) if mask.sum() else 0.0
        b = float(observation[~mask].mean()) if (~mask).sum() else 0.0

        def cost(cc: float, rr: float) -> float:
            m = self._fg_mask(cc, rr)
            if m.sum() == 0 or (~m).sum() == 0:
                return 1e9
            return float(np.sum((observation[m] - f) ** 2) + np.sum((observation[~m] - b) ** 2))

        best = (c, r, cost(c, r))
        for dc in (-self.delta_c, 0.0, self.delta_c):
            for dr in (-self.delta_r, 0.0, self.delta_r):
                cc, rr = c + dc, r + dr
                if rr <= 0.0:
                    continue
                e = cost(cc, rr)
                if e < best[2]:
                    best = (cc, rr, e)
        new = (best[0], best[1], f, b)
        if damping > 0.0:
            new = tuple(
                (1.0 - damping) * a + damping * o for a, o in zip(new, params, strict=True)
            )
        return new

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, params: tuple[float, float, float, float], observation: np.ndarray) -> float:
        """负分段拟合残差。"""
        c, r, f, b = params
        mask = self._fg_mask(c, r)
        resid = np.where(mask, observation - f, observation - b)
        return -float(np.sum(resid**2) / (2.0 * self.sigma**2))
