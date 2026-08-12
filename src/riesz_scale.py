"""RieszScale: 单尺度单演小波响应 (b0/b1/b2 正交三元组及派生量)。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass(slots=True)
class RieszScale:
    """单尺度单演小波响应: ψ 是各向同性带通, R₁ψ/R₂ψ 是它的
    Riesz 变换 (频域乘子 −j·ω/|ω|, 即 2D Hilbert 变换)。
    b0 偶对称、b1/b2 分别沿 x/y 奇对称, 三者构成正交三元组。"""

    b0: mx.array  # 带通响应 (偶)
    b1: mx.array  # Riesz-x 响应 (沿 x 奇)
    b2: mx.array  # Riesz-y 响应 (沿 y 奇)
    amp: mx.array = field(init=False)  # A = sqrt(b0²+b1²+b2²): 局部幅值
    phase: mx.array = field(init=False)  # φ = atan2(|R|, b0): 局部相位 ∈ [0, π]
    ori: mx.array = field(init=False)  # atan2(b2, b1): 结构法向 ∈ (−π, π]
    energy: mx.array = field(init=False)  # A²

    def __post_init__(self):
        """由 b0/b1/b2 派生 energy/amp/phase/ori。"""
        r2 = self.b1**2 + self.b2**2
        self.energy = self.b0**2 + r2
        self.amp = mx.sqrt(self.energy)
        self.phase = mx.arctan2(mx.sqrt(r2), self.b0)
        self.ori = mx.arctan2(self.b2, self.b1)

    def steer(self, theta: float) -> mx.array:
        """沿 θ 方向的一阶 Riesz 转向: cosθ·b1 + sinθ·b2。
        任意方向的奇对称滤波无需新卷积 —— 与 Gabor 多方向通道互为对偶:
        Gabor 用 N 个方向核逼近角度, Riesz 用 2 个基精确合成任意角度。"""
        return self.b1 * math.cos(theta) + self.b2 * math.sin(theta)
