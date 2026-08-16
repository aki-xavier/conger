"""TransparentLayerModel: 两层透明叠加的 EM 实例 (1D 探针)。

透明层的标准 EM 用「软归属混合」模型 (运动透明 / Weiss 1997):

    Z(x) ∈ {1,2} 隐变量, P(Z=1|x) = α(x)          (每像素混合权重)
    I(x) | Z(x)=k ~ N(c_k, σ²)                     (像素由第 k 层解释)

两层各取空间均匀强度 (c1, c2) —— 这是「两层透明均匀膜」的最小可辨识
形态: 单次观测下, 若层内容是自由逐像素均值, 则 L1=L2=平滑(I) 是退化
极大似然解 (每像素被两层同时解释); 均匀层把自由度压到 2, 使 EM 能
真正解耦两层。E 步算每像素软归属, M 步做加权平均重估两层强度。

注: 字面的 alpha 合成 (I = α·L1 + (1−α)·L2, 逐像素相加) 是线性模型、
无隐变量, EM 不是自然算法; 此处软归属混合是透明层 EM 的教科书形态,
空间变化层 (运动透明) 的完整版需要时间序列观测。

这是 GenericEM 框架的第一个非视觉验证域 (对应 ToySeriesFamily 之于
结构框架), 用来验证循环本身, 不追求视觉质量。
"""

from __future__ import annotations

import numpy as np


class TransparentLayerModel:
    """两透明层软归属混合 (均匀层) → 软归属 E 步 + 加权平均 M 步。"""

    def __init__(self, alpha: np.ndarray, sigma: float = 0.05):
        if alpha.ndim != 1:
            raise ValueError("alpha 必须是一维")
        if not (0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0):
            raise ValueError("alpha 必须在 [0,1]")
        self.alpha = alpha
        self.sigma = sigma
        self.n = alpha.shape[0]

    # ── 正向模型 ──────────────────────────────────────────────────

    def sample(
        self, params: tuple[float, float], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """(c1, c2) → 观测: 每像素按 α 软归属到一层 + 噪声。"""
        c1, c2 = params
        rng = rng or np.random.default_rng(0)
        z = rng.random(self.n) < self.alpha  # Z ~ Bernoulli(α)
        return np.where(z, c1, c2) + rng.normal(0.0, self.sigma, self.n)

    # ── E 步 ──────────────────────────────────────────────────────

    def responsibilities(
        self, params: tuple[float, float], observation: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """每像素属于层 1 的软后验 q(Z=1|x)。"""
        c1, c2 = params
        inv = 1.0 / max(temperature, 1e-8)
        logp1 = -0.5 * ((observation - c1) / self.sigma) ** 2 * inv
        logp2 = -0.5 * ((observation - c2) / self.sigma) ** 2 * inv
        w1 = np.log(self.alpha + 1e-12) + logp1
        w2 = np.log(1.0 - self.alpha + 1e-12) + logp2
        m = np.maximum(w1, w2)
        e1 = np.exp(w1 - m)
        e2 = np.exp(w2 - m)
        return e1 / (e1 + e2)

    # ── M 步 ──────────────────────────────────────────────────────

    def maximize(
        self,
        resp: np.ndarray,
        observation: np.ndarray,
        params: tuple[float, float],
        damping: float = 0.0,
    ) -> tuple[float, float]:
        """加权平均重估两层强度; damping 混入旧估计。"""
        c1_old, c2_old = params
        s = float(np.sum(resp))
        c1 = float(np.sum(resp * observation) / max(s, 1e-12))
        c2 = float(np.sum((1.0 - resp) * observation) / max(self.n - s, 1e-12))
        if damping > 0.0:
            c1 = (1.0 - damping) * c1 + damping * c1_old
            c2 = (1.0 - damping) * c2 + damping * c2_old
        return c1, c2

    # ── 收敛监控 ──────────────────────────────────────────────────

    def log_likelihood(self, params: tuple[float, float], observation: np.ndarray) -> float:
        """混合高斯观测对数似然 Σ_x log(α·N(I|c1,σ) + (1−α)·N(I|c2,σ))。"""
        c1, c2 = params
        p1 = self.alpha * np.exp(-0.5 * ((observation - c1) / self.sigma) ** 2)
        p2 = (1.0 - self.alpha) * np.exp(-0.5 * ((observation - c2) / self.sigma) ** 2)
        return float(np.sum(np.log(p1 + p2 + 1e-12)))
