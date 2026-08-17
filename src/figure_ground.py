"""FigureGroundModel: 分割↔位姿的 EM (1D figure-ground)。

前景是一个区间 [c−r, c+r] (位姿 = 中心 c + 半宽 r), 强度 f; 背景强度 b。
隐变量 = 每像素前景归属; 参数 θ = (c, r, f, b)。

  E 步 (responsibilities): 软前景归属 q(x) = 软位姿先验 sigmoid × 强度似然
  M 步 (maximize): 强度用软归属 q 加权; 位姿 (c,r) 坐标搜索最小化
     负混合对数似然 (与软先验模型同一目标)

「分割 ↔ 位姿」: 分割 (哪像素是前景) 与位姿 (物体在哪、多大) 互相
约束 —— 位姿给分割先验, 分割给位姿/强度统计。M 步的坐标搜索让掩码能
随迭代移动/伸缩 (硬先验若直接用质心会卡死在初始掩码)。
"""

from __future__ import annotations

from typing import cast

import numpy as np


class FigureGroundModel:
    """figure-ground → 软分割 E 步 + 位姿坐标搜索 M 步。"""

    def __init__(
        self,
        n: int = 100,
        sigma: float = 0.05,
        delta_c: float = 0.02,
        delta_r: float = 0.02,
        boundary: float = 0.05,
    ):
        self.n = n
        self.sigma = sigma
        self.delta_c = delta_c
        self.delta_r = delta_r
        self.boundary = boundary
        self.x = np.linspace(0.0, 1.0, n)

    def _fg_mask(self, c: float, r: float) -> np.ndarray:
        return np.abs(self.x - c) <= r

    def _fg_prior(self, c: float, r: float) -> np.ndarray:
        """软空间先验 P(fg|x) = sigmoid((r−|x−c|)/w), 允许掩码随迭代移动。"""
        return 1.0 / (1.0 + np.exp((np.abs(self.x - c) - r) / self.boundary))

    def _mixture_ll(
        self, c: float, r: float, f: float, b: float, observation: np.ndarray
    ) -> float:
        """软先验混合对数似然 Σ_x log[P(fg|x)·N(I|f,σ) + P(bg|x)·N(I|b,σ)]。"""
        prior_fg = self._fg_prior(c, r)
        p = prior_fg * np.exp(-0.5 * ((observation - f) / self.sigma) ** 2) + (
            1.0 - prior_fg
        ) * np.exp(-0.5 * ((observation - b) / self.sigma) ** 2)
        return float(np.sum(np.log(p + 1e-12)))

    def sample(
        self,
        params: tuple[float, float, float, float],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """(c, r, f, b) → 前景/背景分段的带噪观测。"""
        _, _, f, b = params
        rng = rng or np.random.default_rng(0)
        return np.where(self._fg_mask(params[0], params[1]), f, b) + rng.normal(
            0.0, self.sigma, self.n
        )

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self,
        params: tuple[float, float, float, float],
        observation: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """软前景归属 q(x) = 软位姿先验 sigmoid × 强度似然。"""
        c, r, f, b = params
        prior_fg = self._fg_prior(c, r)
        inv = 1.0 / max(temperature, 1e-8)
        d_fg = ((observation - f) / self.sigma) ** 2
        d_bg = ((observation - b) / self.sigma) ** 2
        w_fg = np.log(prior_fg + 1e-12) - 0.5 * d_fg * inv
        w_bg = np.log(1.0 - prior_fg + 1e-12) - 0.5 * d_bg * inv
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
        # 强度用软归属 q 加权 (替代硬掩码均值, 与软先验 E 步自洽)
        sq = float(np.sum(q))
        f = float(np.sum(q * observation) / max(sq, 1e-12))
        b = float(np.sum((1.0 - q) * observation) / max(self.n - sq, 1e-12))

        # 位姿坐标搜索最小化负混合对数似然 (与软先验模型一致)
        def cost(cc: float, rr: float) -> float:
            if rr <= 0.0:
                return 1e9
            return -self._mixture_ll(cc, rr, f, b, observation)

        best = (c, r, cost(c, r))
        for dc in (-self.delta_c, 0.0, self.delta_c):
            for dr in (-self.delta_r, 0.0, self.delta_r):
                cc, rr = c + dc, r + dr
                if rr <= 0.0:
                    continue
                e = cost(cc, rr)
                if e < best[2]:
                    best = (cc, rr, e)
        new: tuple[float, float, float, float] = (best[0], best[1], f, b)
        if damping > 0.0:
            blended = tuple(
                (1.0 - damping) * a + damping * o
                for a, o in zip(new, params, strict=True)
            )
            new = cast(tuple[float, float, float, float], blended)
        return new

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(
        self, params: tuple[float, float, float, float], observation: np.ndarray
    ) -> float:
        """软先验混合对数似然 (与 E/M 步同一目标)。"""
        c, r, f, b = params
        return self._mixture_ll(c, r, f, b, observation)
