"""CompositeGeometry: 附着组合物的部分感知双目几何锚点。

与 StereoLayers 不同, composite 的两个部件深度接近、语义上是一个对象,
不能靠视差 2-means 分层。这里利用 attached_on_top 结构先验: 在低分辨率
前景掩码上搜索水平接触线, 将上部 part 与下部 base 分别拟合成圆/方模板,
再在右图对应窗口内估计各部件视差。输出契约与 StereoLayers 相同:
[u0,v0,z0,area0, u1,v1,z1,area1] (0=base, 1=part)。
"""

from __future__ import annotations

import math

import mlx.core as mx

from codebook import Codebook
from joint_layer_optimizer import JointLayerOptimizer, LayerTemplate
from stereo import StereoDepth
from utils import Utils


class CompositeGeometry:
    """左右图 → base/part 两部分的几何统计。"""

    DOWN = 3
    MIN_PART_PIXELS = 5

    @staticmethod
    def _down(mask: mx.array) -> mx.array:
        q = CompositeGeometry.DOWN
        h, w = mask.shape
        m = mask[: h // q * q, : w // q * q]
        return mx.max(m.reshape(h // q, q, w // q, q), axis=(1, 3)) > 0

    @staticmethod
    def _centroid(weights: mx.array) -> tuple[float, float, float]:
        ys = mx.arange(weights.shape[0], dtype=mx.float32)[:, None]
        xs = mx.arange(weights.shape[1], dtype=mx.float32)[None, :]
        total = float(mx.sum(weights))
        if total <= 1e-8:
            c = (Codebook.W - 1) / 2.0
            return c, c, 0.0
        return (
            float(mx.sum(weights * xs) / total),
            float(mx.sum(weights * ys) / total),
            total,
        )

    @staticmethod
    def _area(t: LayerTemplate) -> float:
        q = CompositeGeometry.DOWN
        return (4.0 if t.shape == 2 else math.pi) * (t.r * q) ** 2

    @classmethod
    def _split_templates(
        cls, fg: mx.array
    ) -> tuple[LayerTemplate, LayerTemplate] | None:
        """前景掩码 → (base 模板, part 模板), 搜索 attached_on_top 接触线。"""
        fgd = cls._down(fg)
        h, w = fgd.shape
        idx = Utils.nonzero(fgd.reshape(-1))
        if idx.shape[0] < 4 * cls.MIN_PART_PIXELS:
            return None
        ys_idx = idx // w
        y0, y1 = int(mx.min(ys_idx)), int(mx.max(ys_idx))
        if y1 - y0 < 5:
            return None
        yy = mx.arange(h, dtype=mx.float32)[:, None]
        best: tuple[float, LayerTemplate, LayerTemplate] | None = None
        for split in range(y0 + 2, y1 - 1):
            top = fgd & (yy <= split)
            bottom = fgd & (yy > split)
            if (
                int(mx.sum(top.astype(mx.int32))) < cls.MIN_PART_PIXELS
                or int(mx.sum(bottom.astype(mx.int32))) < cls.MIN_PART_PIXELS
            ):
                continue
            part = JointLayerOptimizer._bbox_template(top, 0.0)
            base = JointLayerOptimizer._bbox_template(bottom, 0.0)
            if part.cy >= base.cy:
                continue
            # 附着关系: 中心纵向间距约 r0+r1−overlap; 横向偏移有限
            dy = base.cy - part.cy
            rr = base.r + part.r
            if not (0.45 * rr <= dy <= 1.35 * rr):
                continue
            if abs(base.cx - part.cx) > 1.2 * rr:
                continue
            a0 = cls._area(base)
            a1 = cls._area(part)
            ratio = math.sqrt(a1 / max(a0, 1e-8))
            # 采样尺度比 0.35–0.75; 轮廓面积比只加宽边界, 不硬卡死
            ratio_penalty = max(0.0, 0.25 - ratio, ratio - 0.95)
            union = base.mask(h, w) | part.mask(h, w)
            inter = mx.sum((union & fgd).astype(mx.float32))
            union_n = mx.sum((union | fgd).astype(mx.float32))
            mask_cost = 1.0 - float(inter / mx.maximum(union_n, 1.0))
            score = mask_cost + 2.0 * ratio_penalty
            if best is None or score < best[0]:
                best = (score, base, part)
        if best is None:
            return None
        return best[1], best[2]

    @staticmethod
    def _window_centroid(
        weights: mx.array, cx: float, cy: float, r: float
    ) -> tuple[float, float] | None:
        """以模板为中心的局部窗口质心 (右图对应点搜索用)。"""
        h, w = weights.shape
        pad = max(2.0, 0.25 * r)
        x0 = max(0, int(math.floor(cx - r - pad)))
        x1 = min(w, int(math.ceil(cx + r + pad + 1)))
        y0 = max(0, int(math.floor(cy - r - pad)))
        y1 = min(h, int(math.ceil(cy + r + pad + 1)))
        win = weights[y0:y1, x0:x1]
        total = float(mx.sum(win))
        if total <= 1e-8:
            return None
        ys = mx.arange(y0, y1, dtype=mx.float32)[:, None]
        xs = mx.arange(x0, x1, dtype=mx.float32)[None, :]
        return (
            float(mx.sum(win * xs) / total),
            float(mx.sum(win * ys) / total),
        )

    @classmethod
    def _part_depth(
        cls,
        wl: mx.array,
        wr: mx.array,
        t: LayerTemplate,
        d_global: float,
    ) -> float:
        """单部件模板 → 右图水平偏移窗口 → 部件视差深度。"""
        q = cls.DOWN
        cx, cy, r = t.cx * q, t.cy * q, t.r * q
        left = cls._window_centroid(wl, cx, cy, r)
        right = cls._window_centroid(wr, cx - d_global, cy, r)
        if left is None or right is None:
            return Codebook.CAM_Z - Codebook.FX * Codebook.STEREO_BASE / max(
                d_global, 1e-6
            )
        d = left[0] - right[0]
        d = min(max(d, 4.0), 14.0)  # 部件窗口噪声的物理限幅
        return Codebook.CAM_Z - Codebook.FX * Codebook.STEREO_BASE / d

    @classmethod
    def estimate(cls, fl: mx.array, fr: mx.array) -> tuple[float, ...]:
        """左右图 → [u,v,z,area]×2 (base, attached part)。"""
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        fg = wl > 0.01
        z_global, d_global, area_global = StereoDepth().estimate(fl, fr)
        templates = cls._split_templates(fg)
        q = cls.DOWN
        if templates is None:
            u, v, _ = cls._centroid(wl)
            return (u, v, z_global, 0.7 * area_global,
                    u, v, z_global, 0.3 * area_global)
        base, part = templates
        z0 = cls._part_depth(wl, wr, base, d_global)
        z1 = cls._part_depth(wl, wr, part, d_global)
        return (
            base.cx * q,
            base.cy * q,
            z0,
            cls._area(base),
            part.cx * q,
            part.cy * q,
            z1,
            cls._area(part),
        )
