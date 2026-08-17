"""SCMProxy: 外观机制代理 —— 黑盒 renderer 外观子图的乘法分解 (路线 ②)。

渲染因子图 (已知) 的外观子图:

    hue (反照率) → material ─┐
                             ├─→ 前景颜色 I_color
    lcol/ldir (光照) → light ┘

MeshStandardMaterial 的反照率×光照物理使前景平均 RGB 服从乘法机制:

    I_color(hue, lcol, ldir) ≈ albedo[hue] ⊙ lighting[lcol, ldir]

其中 lighting 项本身吸收了环境光与方向光着色 (对固定几何是常量)。
`AppearanceMechanism` 从全因子干预数据 (每个 (hue,lcol,ldir) 组合的
前景平均 RGB) 用交替最小二乘 (ALS) 估计两机制项, 把黑盒 renderer 的
外观子图拆成两个可独立 do-干预的模块。用途: 快速 do 查询 (反事实)
+ 秩一重构误差 = 模块性/不变性分数 (路线 ① 的验收判据)。只作对照/
校准件, 不替换 MixtureSPN 或 §3 分析-合成。
"""

from __future__ import annotations

import mlx.core as mx

from codebook import Codebook


class AppearanceMechanism:
    """外观机制的乘法分解代理: 拟合/预测/do 查询/不变性校验。"""

    def __init__(
        self,
        n_hue: int = Codebook.N_HUE,
        n_lcol: int | None = None,
        n_ldir: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-8,
    ):
        self.n_hue = n_hue
        self.n_lcol = len(Codebook.LIGHT_COLORS) if n_lcol is None else n_lcol
        self.n_ldir = len(Codebook.LIGHT_DIRS) if n_ldir is None else n_ldir
        self.max_iter = max_iter
        self.tol = tol
        self.albedo: mx.array | None = None  # (n_hue, 3)
        self.lighting: mx.array | None = None  # (n_lcol, n_ldir, 3)

    def fit(self, rgb: mx.array) -> AppearanceMechanism:
        """干预数据 rgb (n_hue,n_lcol,n_ldir,3) → 估计 albedo/lighting。

        乘法模型 `rgb[h,l,d] ≈ a[h] ⊙ g[l,d]` 的 ALS 最小二乘; 收敛后
        把 g 归一化为单位均值 (每通道), 尺度归 a, 使 a 可解释为「单位
        (平均) 光照下的反照率颜色」。
        """
        rgb = mx.array(rgb, dtype=mx.float32)
        a = mx.mean(rgb, axis=(1, 2))  # (n_hue, 3)
        g = mx.ones((self.n_lcol, self.n_ldir, 3), dtype=mx.float32)
        for _ in range(self.max_iter):
            den_g = mx.maximum(mx.sum(a[:, None, None, :] ** 2, axis=0), 1e-12)
            g_new = mx.sum(rgb * a[:, None, None, :], axis=0) / den_g
            den_a = mx.maximum(
                mx.sum(g_new[None, :, :, :] ** 2, axis=(1, 2)), 1e-12
            )
            a_new = mx.sum(rgb * g_new[None, :, :, :], axis=(1, 2)) / den_a
            change = float(
                mx.max(mx.abs(a_new - a)) + mx.max(mx.abs(g_new - g))
            )
            a, g = a_new, g_new
            if change < self.tol:
                break
        g_mean = mx.maximum(mx.mean(g, axis=(0, 1)), 1e-12)  # (3,)
        self.lighting = g / g_mean[None, None, :]
        self.albedo = a * g_mean
        return self

    def _assert_fitted(self) -> None:
        if self.albedo is None or self.lighting is None:
            raise RuntimeError("先调用 fit() 再查询")

    def predict(self, hue: int, lcol: int, ldir: int) -> mx.array:
        """机制组合 → 前景颜色 (3,) 的代理预测。"""
        self._assert_fitted()
        assert self.albedo is not None and self.lighting is not None
        return self.albedo[hue] * self.lighting[lcol, ldir]

    def do_lighting(
        self, hue: int, lcol: int, ldir: int, lcol_new: int, ldir_new: int
    ) -> mx.array:
        """反事实: do(lighting=lcol_new,ldir_new) 后同一反照率的颜色。"""
        return self.predict(hue, lcol_new, ldir_new)

    def reconstruct(self) -> mx.array:
        """代理的完整 (n_hue,n_lcol,n_ldir,3) 重构。"""
        self._assert_fitted()
        assert self.albedo is not None and self.lighting is not None
        return self.albedo[:, None, None, :] * self.lighting[None, :, :, :]

    def reconstruction_error(self, rgb: mx.array) -> float:
        """归一化重构误差 ||rgb − a⊗g|| / ||rgb|| (0=完美乘法分解)。"""
        rgb = mx.array(rgb, dtype=mx.float32)
        rec = self.reconstruct()
        num = float(mx.sqrt(mx.mean((rgb - rec) ** 2)))
        den = float(mx.sqrt(mx.mean(rgb**2)))
        return num / max(den, 1e-12)

    def albedo_invariance(self, rgb: mx.array) -> float:
        """模块性/不变性分数 = 1 − 重构误差。

        分数接近 1 ⟺ 反照率机制与光照机制可干净分离 (乘法分解精确),
        即反照率对光照不变 —— 路线 ① 的验收判据。环境光/着色等偏离
        纯乘法的项会压低分数, 暴露机制的非模块性。
        """
        return 1.0 - self.reconstruction_error(rgb)

    @staticmethod
    def foreground_mean_rgb(frame: mx.array, weights: mx.array) -> mx.array:
        """帧 (H,W,C) + 前景权重 (H,W) → 前景加权平均 RGB (3,)。"""
        w = weights if weights.ndim == 3 else weights[..., None]
        num = mx.sum(w * frame[..., :3].astype(mx.float32), axis=(0, 1))
        den = mx.maximum(mx.sum(w), 1e-8)
        return num / den
