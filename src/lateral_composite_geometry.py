"""LateralCompositeGeometry: mirror/repeat 横向组合的 part 几何锚点。

与 CompositeGeometry 的上下接触线不同, 横向组合在低分辨率前景上搜索
垂直分隔线; 0 号为左/base, 1 号为右/part, 两部件深度应接近。
"""

from __future__ import annotations

import math

import mlx.core as mx

from composite_geometry import CompositeGeometry
from joint_layer_optimizer import JointLayerOptimizer, LayerTemplate
from utils import Utils


class LateralCompositeGeometry(CompositeGeometry):
    """左右图 → 横向 base/part 的 [u,v,z,area]×2。"""

    @classmethod
    def split_score(
        cls, fg: mx.array
    ) -> tuple[float, LayerTemplate, LayerTemplate] | None:
        """前景掩码 → (横向组合得分, 左 base 模板, 右 part 模板)。"""
        fgd = cls._down(fg)
        h, w = fgd.shape
        idx = Utils.nonzero(fgd.reshape(-1))
        if idx.shape[0] < 4 * cls.MIN_PART_PIXELS:
            return None
        xs_idx = idx % w
        x0, x1 = int(mx.min(xs_idx)), int(mx.max(xs_idx))
        if x1 - x0 < 5:
            return None
        xx = mx.arange(w, dtype=mx.float32)[None, :]
        best: tuple[float, LayerTemplate, LayerTemplate] | None = None
        for split in range(x0 + 2, x1 - 1):
            left = fgd & (xx <= split)
            right = fgd & (xx > split)
            if (
                int(mx.sum(left.astype(mx.int32))) < cls.MIN_PART_PIXELS
                or int(mx.sum(right.astype(mx.int32))) < cls.MIN_PART_PIXELS
            ):
                continue
            base = JointLayerOptimizer._bbox_template(left, 0.0)
            part = JointLayerOptimizer._bbox_template(right, 0.0)
            if base.cx >= part.cx:
                continue
            dx = part.cx - base.cx
            rr = base.r + part.r
            if not (0.45 * rr <= dx <= 3.0 * rr):
                continue
            if abs(base.cy - part.cy) > 0.8 * rr:
                continue
            a0 = cls._area(base)
            a1 = cls._area(part)
            ratio = math.sqrt(a1 / max(a0, 1e-8))
            ratio_penalty = max(0.0, 0.25 - ratio, ratio - 1.1)
            union = base.mask(h, w) | part.mask(h, w)
            inter = mx.sum((union & fgd).astype(mx.float32))
            union_n = mx.sum((union | fgd).astype(mx.float32))
            mask_cost = 1.0 - float(inter / mx.maximum(union_n, 1.0))
            score = mask_cost + 2.0 * ratio_penalty
            if best is None or score < best[0]:
                best = (score, base, part)
        return best

    @classmethod
    def _split_templates(
        cls, fg: mx.array
    ) -> tuple[LayerTemplate, LayerTemplate] | None:
        out = cls.split_score(fg)
        return None if out is None else (out[1], out[2])
