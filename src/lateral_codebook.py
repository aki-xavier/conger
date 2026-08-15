"""LateralCompositeCodebook: mirror/repeat 横向同型组合模板。

参数序仍与双图元族兼容; 0 号为左/base, 1 号为右/part。part 与 base
同 kind/hue/depth, 只学习尺度比例和横向周期约束。该族用于物化
TemplateGrammar 的 mirror/repeat 子模板。
"""

from __future__ import annotations

import random

import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from template_lineage import TemplateLineage


class LateralCompositeCodebook(CompositeCodebook):
    """同 kind/hue 的横向 mirror/repeat 组合场景族。"""

    SAMPLE_V = 1
    STEREO_V = "lc1"
    RELATION = "mirror"
    GEOMETRY_FAMILY = "lateral"
    PART_PERIOD_RANGE = (0.15, 0.25)  # 原始 proposal lateral_ratio 范围
    SPACING_FACTOR = 5.0  # mirror: 0.2 约等于一个组合直径
    TEMPLATE_LINEAGE = TemplateLineage(
        family="lateral",
        parent_family="composite",
        operation="mirror",
        complexity=1.4,
        generation=3,
        delta={"relation": RELATION},
    )

    @classmethod
    def _sample_composite(
        cls,
        rng: random.Random,
        extrap: bool,
    ) -> tuple[float, ...]:
        """base 左 / part 右, 同深度横向组合 → 8 个几何量。"""
        for _ in range(64):
            if extrap:
                s0 = rng.choice(
                    [
                        rng.uniform(*Codebook.S_EXTRA[0]),
                        rng.uniform(*Codebook.S_EXTRA[1]),
                    ]
                )
                z0 = rng.choice(
                    [
                        rng.uniform(*Codebook.Z_EXTRA[0]),
                        rng.uniform(*Codebook.Z_EXTRA[1]),
                    ]
                )
            else:
                s0 = rng.uniform(*Codebook.S_RANGE)
                z0 = rng.uniform(*Codebook.Z_RANGE)
            s1 = s0 * rng.uniform(*cls.SCALE_RATIO)
            z1 = z0
            m0 = cls._margin(s0, z0)
            if 2 * m0 > Codebook.W - 4:
                continue
            u0 = rng.uniform(m0, Codebook.W - m0)
            v0 = rng.uniform(m0, Codebook.H - m0)
            x0, y0 = Codebook.unproject(u0, v0, z0)
            period = rng.uniform(*cls.PART_PERIOD_RANGE)
            x1 = x0 + period * cls.SPACING_FACTOR * (s0 + s1)
            zc1 = Codebook.CAM_Z - z1
            u1 = (Codebook.W - 1) / 2.0 + x1 * Codebook.FX / zc1
            v1 = (Codebook.H - 1) / 2.0 - y0 * Codebook.FY / zc1
            if cls._inside(u0, v0, s0, z0) and cls._inside(u1, v1, s1, z1):
                return u0, v0, s0, z0, u1, v1, s1, z1
        raise RuntimeError("LateralCompositeCodebook 取景拒绝重采失败")

    @classmethod
    def _block(cls, seed: int, extrap: bool = False) -> mx.array:
        """单复制块; mirror/repeat 约束 part 与 base 同 kind/hue。"""
        rng = random.Random(seed)
        rows = []
        for k0 in cls.BASE_KINDS:
            for h0 in cls.BASE_HUES:
                for lc in range(cls.N_LIGHT_COLORS):
                    for ld in range(cls.N_LIGHT_DIRS):
                        geom = cls._sample_composite(rng, extrap)
                        rows.append(
                            [
                                k0, geom[0], geom[1], geom[2], geom[3], h0,
                                k0, geom[4], geom[5], geom[6], geom[7], h0,
                                lc, ld,
                            ]
                        )
        return mx.array(rows, dtype=mx.float32)
