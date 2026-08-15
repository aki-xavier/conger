"""CompositeCodebook: 由两个已有图元组成的显式附着组合模板。

参数序与 LayeredCodebook 兼容:
[k0,u0,v0,s0,z0,h0, k1,u1,v1,s1,z1,h1, lcol,ldir]。

区别在于生成机制: 0 号图元是底座, 1 号图元不是独立采样, 而是按
"attached_on_top" 关系由底座导出 (尺度比例、横向偏移、接触重叠、轻微
深度抖动)。它用于验证“已有模板 → 稳定组合模板”的结构专家路径。
"""

from __future__ import annotations

import random

import mlx.core as mx

from codebook import Codebook
from layered_codebook import LayeredCodebook
from template_lineage import TemplateLineage


class CompositeCodebook(LayeredCodebook):
    """双图元附着组合场景族 (base + attached part) + 采样器。"""

    SAMPLE_V = 1  # 1 = attached_on_top 显式组合模板
    USES_LAYER_STATS = False
    USES_COMPOSITE_STATS = True  # base/part 模板拆分 + 部件视差
    STEREO_V = "cp2"
    RELATION = "attached_on_top"
    TEMPLATE_COMPLEXITY = 1.5  # 两图元但附着关系降低描述长度
    SCALE_RATIO = (0.35, 0.75)  # part/base 尺度比
    LATERAL_RATIO = 0.25  # 横向偏移占 s0+s1 的最大比例
    OVERLAP_RATIO = (0.03, 0.10)  # 接触处嵌入量占较小半径比例
    DEPTH_JITTER = (-0.06, 0.06)  # 附着件相对底座的轻微深度差
    TEMPLATE_LINEAGE = TemplateLineage(
        family="composite",
        parent_family="layered",
        operation="attach",
        complexity=TEMPLATE_COMPLEXITY,
        generation=2,
        delta={
            "relation": RELATION,
            "scale_ratio": SCALE_RATIO,
            "lateral_ratio": (-LATERAL_RATIO, LATERAL_RATIO),
            "depth_jitter": DEPTH_JITTER,
        },
    )

    @classmethod
    def _inside(cls, u: float, v: float, s: float, z: float) -> bool:
        """单个图元连同立体偏移是否完整落入画面。"""
        m = cls._margin(s, z)
        return m <= u <= Codebook.W - m and m <= v <= Codebook.H - m

    @classmethod
    def _sample_composite(
        cls,
        rng: random.Random,
        extrap: bool,
    ) -> tuple[float, ...]:
        """附着组合 → (u0,v0,s0,z0,u1,v1,s1,z1)。

        底座先采样; 附着件中心位于底座顶面附近, 允许小幅横向偏移和
        轻微嵌入, 使组合关系稳定但不过于模板化。取景失败则拒绝重采。
        """
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
            z1 = min(max(z0 + rng.uniform(*cls.DEPTH_JITTER), 2.2), 4.3)
            m0 = cls._margin(s0, z0)
            if 2 * m0 > Codebook.W - 4:
                continue
            u0 = rng.uniform(m0, Codebook.W - m0)
            v0 = rng.uniform(m0, Codebook.H - m0)
            x0, y0 = Codebook.unproject(u0, v0, z0)

            dx = rng.uniform(-cls.LATERAL_RATIO, cls.LATERAL_RATIO) * (s0 + s1)
            overlap = rng.uniform(*cls.OVERLAP_RATIO) * min(s0, s1)
            x1 = x0 + dx
            y1 = y0 + s0 + s1 - overlap  # 世界 Y 向上 → 图像 v1 < v0
            zc1 = Codebook.CAM_Z - z1
            u1 = (Codebook.W - 1) / 2.0 + x1 * Codebook.FX / zc1
            v1 = (Codebook.H - 1) / 2.0 - y1 * Codebook.FY / zc1
            if cls._inside(u0, v0, s0, z0) and cls._inside(u1, v1, s1, z1):
                return u0, v0, s0, z0, u1, v1, s1, z1
        raise RuntimeError("CompositeCodebook 取景拒绝重采失败")

    @classmethod
    def _block(cls, seed: int, extrap: bool = False) -> mx.array:
        """单复制块 → (2916,14), 离散因子全笛卡尔积。"""
        rng = random.Random(seed)
        rows = []
        for k0 in range(cls.N_KIND):
            for k1 in range(cls.N_KIND):
                for h0 in range(cls.N_HUE):
                    for h1 in range(cls.N_HUE):
                        for lc in range(cls.N_LIGHT_COLORS):
                            for ld in range(cls.N_LIGHT_DIRS):
                                geom = cls._sample_composite(rng, extrap)
                                rows.append(
                                    [
                                        k0, geom[0], geom[1], geom[2], geom[3], h0,
                                        k1, geom[4], geom[5], geom[6], geom[7], h1,
                                        lc, ld,
                                    ]
                                )
        return mx.array(rows, dtype=mx.float32)
