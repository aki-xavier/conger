"""LateralCompositeGeometry: mirror/repeat 横向组合的 part 几何锚点。

与 CompositeGeometry 的上下接触线不同, 横向组合在低分辨率前景上搜索
垂直分隔线; 0 号为左/base, 1 号为右/part, 两部件深度应接近。

与父类 bbox 模板不同, 本类 override `estimate` 在模板足迹内做全分辨率
圆拟合 (面积 + 质心), 得到更准的 part 中心/半径, 并提供 kind 感知的
近端盖校正 `corrected_gap`, 供 mirror/repeat 判别消掉圆柱端盖投影的
表观半径偏置。
"""

from __future__ import annotations

import math

import mlx.core as mx

from codebook import Codebook
from composite_geometry import CompositeGeometry
from joint_layer_optimizer import JointLayerOptimizer, LayerTemplate
from stereo import StereoDepth
from utils import Utils


class LateralCompositeGeometry(CompositeGeometry):
    """左右图 → 横向 base/part 的 [u,v,z,area]×2。"""

    # kind → 近端面相对中心的深度偏移 (以 s 为单位): sphere 轮廓在中心
    # 平面 (δ=0); cylinder 可见端盖在 z+1.1s; box 前面在 z+s。
    NEAR_CAP_DELTA = (0.0, 1.1, 1.0)

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

    @classmethod
    def corrected_gap(
        cls, fl: mx.array, fr: mx.array, kind: int
    ) -> float | None:
        """kind 感知近端盖校正后的世界归一化间隔 g = |x1-x0|/(s0+s1)。

        在模板足迹内做全分辨率圆拟合 (消 max-pool 下采样膨胀), 再按 kind
        近端盖偏移反解真实世界半径 s 与中心 x。返回 None 表示无法可靠
        分割 (判别器应放弃)。与 `estimate` (重建锚点) 解耦 —— 判别证据
        直接取自原始帧, 不改动已训练模型的锚点契约。
        """
        wl = StereoDepth.foreground_weights(fl)
        fg = wl > 0.01
        split = cls.split_score(fg)
        if split is None:
            return None
        _, base, part = split
        q = cls.DOWN
        b = cls._disk_fit(fg, base, q)
        p = cls._disk_fit(fg, part, q)
        if b is None or p is None:
            return None
        u0, _, r0 = b
        u1, _, r1 = p
        z_global, _, _ = StereoDepth().estimate(fl, fr)
        zc = Codebook.CAM_Z - z_global
        if 0 <= kind < len(cls.NEAR_CAP_DELTA):
            delta = cls.NEAR_CAP_DELTA[kind]
        else:
            delta = 1.1  # 默认 cylinder
        s0 = r0 * zc / (Codebook.FX + delta * r0)
        s1 = r1 * zc / (Codebook.FX + delta * r1)
        c = (Codebook.W - 1) / 2.0
        x0 = (u0 - c) * (zc - delta * s0) / Codebook.FX
        x1 = (u1 - c) * (zc - delta * s1) / Codebook.FX
        return abs(x1 - x0) / max(s0 + s1, 1e-8)
