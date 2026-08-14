"""ContourCompleter: 后层可见轮廓 → 完整形状模板。

分析-合成补全: 候选完整模板 T 经前层遮挡后应为 T\\F; 与观测后层
可见掩码做对称差。当前场景族的投影近似各向同性, 圆模板覆盖
sphere/cylinder, 方模板覆盖 box; 低分辨率坐标搜索保持数据构建可用。
"""

from __future__ import annotations

import math

import mlx.core as mx

from utils import Utils


class ContourCompleter:
    """遮挡可见掩码 → (中心 u,v, 完整面积, 轮廓类型, 得分)。"""

    DOWN = 3
    CENTER_STEPS = (-6.0, -3.0, 0.0, 3.0, 6.0)
    SCALE_STEPS = (0.85, 1.0, 1.2, 1.45, 1.75)

    @classmethod
    def _down(cls, mask: mx.array) -> mx.array:
        q = cls.DOWN
        h, w = mask.shape
        m = mask[: h // q * q, : w // q * q]
        return mx.max(
            m.reshape(h // q, q, w // q, q), axis=(1, 3)
        ) > 0

    @staticmethod
    def _centroid(mask: mx.array) -> tuple[float, float]:
        idx = mx.arange(mask.size)
        sel = Utils.nonzero(mask.reshape(-1))
        n = sel.shape[0]
        if n == 0:
            return (mask.shape[1] - 1) / 2.0, (mask.shape[0] - 1) / 2.0
        ids = idx[sel]
        w = mask.shape[1]
        return float(mx.mean((ids % w).astype(mx.float32))), float(
            mx.mean((ids // w).astype(mx.float32))
        )

    @classmethod
    def _template(
        cls, shape: int, cx: float, cy: float, r: float, h: int, w: int
    ) -> mx.array:
        yy, xx = mx.meshgrid(
            mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )
        if shape == 2:  # box: 轴对齐正方形, r=半边长
            return (mx.abs(xx - cx) <= r) & (mx.abs(yy - cy) <= r)
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    @staticmethod
    def _score(template: mx.array, front: mx.array, back: mx.array) -> float:
        visible = template & ~front
        inter = mx.sum((visible & back).astype(mx.float32))
        union = mx.sum((visible | back).astype(mx.float32))
        return float(1.0 - inter / mx.maximum(union, 1.0))

    @classmethod
    def _fit_shape(
        cls, shape: int, front: mx.array, back: mx.array
    ) -> tuple[float, float, float, float]:
        """单形状坐标下降 → (cx,cy,r,score)。"""
        h, w = back.shape
        idx = Utils.nonzero(back.reshape(-1))
        xs, ys = idx % w, idx // w
        x0, x1 = float(mx.min(xs)), float(mx.max(xs))
        y0, y1 = float(mx.min(ys)), float(mx.max(ys))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # 可见外接框通常保留未遮挡侧边界, 比可见质心更接近完整轮廓
        r = max(x1 - x0 + 1, y1 - y0 + 1) / 2.0
        best = (cx, cy, r, float("inf"))
        for _ in range(3):
            for cx1 in [best[0] + d for d in cls.CENTER_STEPS]:
                for cy1 in [best[1] + d for d in cls.CENTER_STEPS]:
                    for mul in cls.SCALE_STEPS:
                        r1 = max(best[2] * mul, 1.5)
                        t = cls._template(shape, cx1, cy1, r1, h, w)
                        score = cls._score(t, front, back)
                        if score < best[3]:
                            best = (cx1, cy1, r1, score)
        return best

    @classmethod
    def complete(
        cls, front_mask: mx.array, back_mask: mx.array
    ) -> tuple[float, float, float, int, float]:
        """前后层可见掩码 → (u,v,完整面积,轮廓kind,score)。

        轮廓 kind: 2=方/box; 0=圆 (sphere/cylinder 在轮廓级不可分)。"""
        front = cls._down(front_mask)
        back = cls._down(back_mask)
        h, w = back.shape
        if int(mx.sum(back.astype(mx.int32))) < 8:
            q = cls.DOWN
            cx, cy = cls._centroid(back)
            return cx * q, cy * q, 0.0, 0, 1.0
        fits = [cls._fit_shape(shape, front, back) for shape in (0, 2)]
        best_i = min(range(2), key=lambda j: fits[j][3])
        shape = (0, 2)[best_i]
        cx, cy, r, score = fits[best_i]
        q = cls.DOWN
        area = (4.0 if shape == 2 else math.pi) * (r * q) ** 2
        return cx * q, cy * q, area, (2 if shape == 2 else 0), score
