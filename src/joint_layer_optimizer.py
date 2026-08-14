"""JointLayerOptimizer: 遮挡双层的模板/视差/像素分配联合优化。

状态 = 前/后层各自的 (圆/方模板, 中心, 尺度, 视差中心)。候选可见区为
前层模板 T0 与 T1\\T0; 得分同时惩罚前景掩码不一致、视差不一致和
后层被过度遮挡, 避免先聚类再补全造成的错误级联。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from codebook import Codebook
from contour_completion import ContourCompleter
from utils import Utils


@dataclass(frozen=True)
class LayerTemplate:
    shape: int  # 0=圆, 2=方
    cx: float
    cy: float
    r: float
    d: float

    def mask(self, h: int, w: int) -> mx.array:
        return ContourCompleter._template(self.shape, self.cx, self.cy, self.r, h, w)


class JointLayerOptimizer:
    """低分辨率遮挡双层联合优化 (固定点交替 + 坐标搜索)。"""

    DOWN = 12
    CENTER_STEPS = (0.0,)  # 12×12 下中心来自初始掩码, 不再粗跳
    SCALE_STEPS = (0.85, 1.0, 1.15)
    D_STEPS = (-0.5, 0.0, 0.5)

    @staticmethod
    def _down_mask(mask: mx.array) -> mx.array:
        q = JointLayerOptimizer.DOWN
        h, w = mask.shape
        m = mask[: h // q * q, : w // q * q]
        return mx.max(m.reshape(h // q, q, w // q, q), axis=(1, 3)) > 0

    @staticmethod
    def _down_mean(x: mx.array) -> mx.array:
        q = JointLayerOptimizer.DOWN
        h, w = x.shape
        xc = x[: h // q * q, : w // q * q]
        return mx.mean(xc.reshape(h // q, q, w // q, q), axis=(1, 3))

    @classmethod
    def _bbox_template(
        cls, mask: mx.array, d: float, occluder: mx.array | None = None
    ) -> LayerTemplate:
        h, w = mask.shape
        idx = mx.arange(mask.size)[Utils.nonzero(mask.reshape(-1))]
        if idx.shape[0] == 0:
            return LayerTemplate(0, (w - 1) / 2, (h - 1) / 2, 4.0, d)
        xs, ys = idx % w, idx // w
        x0, x1 = float(mx.min(xs)), float(mx.max(xs))
        y0, y1 = float(mx.min(ys)), float(mx.max(ys))
        r = max(x1 - x0 + 1, y1 - y0 + 1) / 2.0
        # 圆/方轮廓先验: 与可见区 IoU 更高的作为初始形状
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        fits = []
        for shape in (0, 2):
            t = ContourCompleter._template(shape, cx, cy, r, h, w)
            if occluder is not None:
                t = t & ~occluder
            score = ContourCompleter._score(t, mx.zeros_like(mask), mask)
            fits.append((score, shape))
        return LayerTemplate(min(fits)[1], cx, cy, max(r, 1.5), d)

    @staticmethod
    def _score(
        front: LayerTemplate,
        back: LayerTemplate,
        fg: mx.array,
        disp: mx.array,
        valid: mx.array,
    ) -> float:
        """候选模板 → 遮挡可见区 + 前景/视差联合残差。"""
        if front.d <= back.d + 0.3:
            return float("inf")  # 前层必须是大视差层
        h, w = fg.shape
        tf = front.mask(h, w)
        tb = back.mask(h, w)
        vf = tf
        vb = tb & ~tf
        pred = vf | vb
        inter = mx.sum((pred & fg).astype(mx.float32))
        union = mx.sum((pred | fg).astype(mx.float32))
        mask_cost = 1.0 - float(inter / mx.maximum(union, 1.0))
        # 后层不能几乎全被遮住, 否则单层大模板会吞掉整个前景
        ratio = float(
            mx.sum(vb.astype(mx.float32))
            / mx.maximum(mx.sum(tb.astype(mx.float32)), 1.0)
        )
        occlusion_penalty = max(0.0, 0.25 - ratio) * 4.0
        d_cost = 0.0
        for tmpl, vis in ((front, vf), (back, vb)):
            m = valid & vis
            n = float(mx.sum(m.astype(mx.float32)))
            if n > 4:
                err = mx.where(m, (disp - tmpl.d) ** 2, 0.0)
                d_cost += float(mx.sum(err) / n) / 4.0
        return mask_cost + occlusion_penalty + d_cost

    @classmethod
    def _optimize_layer(
        cls,
        which: int,
        state: tuple[LayerTemplate, LayerTemplate],
        fg: mx.array,
        disp: mx.array,
        valid: mx.array,
    ) -> tuple[LayerTemplate, LayerTemplate]:
        best_score = cls._score(*state, fg, disp, valid)
        current = state[which]
        other = state[1 - which]
        for _ in range(1):
            improved = False
            candidates = []
            for shape in (current.shape,):
                for dx in cls.CENTER_STEPS:
                    for dy in cls.CENTER_STEPS:
                        for sm in cls.SCALE_STEPS:
                            for dd in cls.D_STEPS:
                                candidates.append(
                                    LayerTemplate(
                                        shape,
                                        current.cx + dx,
                                        current.cy + dy,
                                        max(current.r * sm, 1.5),
                                        current.d + dd,
                                    )
                                )
            for cand in candidates:
                state1 = (cand, other) if which == 0 else (other, cand)
                score = cls._score(*state1, fg, disp, valid)
                if score < best_score:
                    best_score, current, improved = score, cand, True
            if not improved:
                break
        return (current, other) if which == 0 else (other, current)

    @classmethod
    def optimize(
        cls,
        fg: mx.array,
        disp: mx.array,
        valid: mx.array,
        front0: mx.array,
        back0: mx.array,
        d_front: float,
        d_back: float,
    ) -> tuple[float, ...] | None:
        """初始前后层掩码 → (u0,v0,z0,area0,u1,v1,z1,area1)。"""
        fg_d = cls._down_mask(fg)
        disp_d = cls._down_mean(disp)
        valid_d = cls._down_mask(valid)
        front_mask = cls._down_mask(front0)
        back_mask = cls._down_mask(back0)
        front = cls._bbox_template(front_mask, d_front)
        back = cls._bbox_template(back_mask, d_back, front_mask)
        state = (front, back)
        for _ in range(1):
            state = cls._optimize_layer(0, state, fg_d, disp_d, valid_d)
            state = cls._optimize_layer(1, state, fg_d, disp_d, valid_d)
        q = cls.DOWN
        out = []
        for tmpl in state:
            area = (4.0 if tmpl.shape == 2 else math.pi) * (tmpl.r * q) ** 2
            z = Codebook.CAM_Z - Codebook.FX * Codebook.STEREO_BASE / max(
                tmpl.d, 1e-6
            )
            out.extend([tmpl.cx * q, tmpl.cy * q, z, area])
        return tuple(out)
