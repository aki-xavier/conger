"""LayeredCodebook: 双图元遮挡/前后层场景参数 ⇄ cga Scene。

参数序: [k0,u0,v0,s0,z0,h0, k1,u1,v1,s1,z1,h1, lcol,ldir]。
按深度规范排序 (z0 > z1, 0 是前层/更靠近相机), 消除两物体标签置换对称;
约 70% 样本强制投影重叠, 让遮挡成为数据分布的一等结构。
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import mlx.core as mx
from cga.engine import (  # pyright: ignore[reportMissingImports]
    AmbientLight,
    Color,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    Scene,
)

from codebook import Codebook
from template_lineage import TemplateLineage

if TYPE_CHECKING:
    from inverse_config import InverseConfig


class LayeredCodebook:
    """双物体遮挡场景族 (最小多层支持集) + 采样器。"""

    N_OBJECTS = 2
    H = W = Codebook.H
    N_KIND = Codebook.N_KIND
    N_HUE = Codebook.N_HUE
    LIGHT_COLORS = Codebook.LIGHT_COLORS
    LIGHT_DIRS = Codebook.LIGHT_DIRS
    N_LIGHT_COLORS = len(Codebook.LIGHT_COLORS)
    N_LIGHT_DIRS = len(Codebook.LIGHT_DIRS)
    N_COMBO = (
        N_KIND * N_KIND * N_HUE * N_HUE * N_LIGHT_COLORS * N_LIGHT_DIRS
    )  # 2916
    SAMPLE_V = 3  # 3 = 修正前层 z 方向 (z0>z1) + 可分辨层间距
    RENDER_V = Codebook.RENDER_V
    USES_LAYER_STATS = True
    USES_COMPOSITE_STATS = False
    STEREO_V = "sl8"  # 联合中心/深度 + soft-fusion 面积
    TEMPLATE_COMPLEXITY = 2.0  # 两个独立物体 + 层序
    TEMPLATE_LINEAGE = TemplateLineage(
        family="layered",
        parent_family="single",
        operation="layer",
        complexity=TEMPLATE_COMPLEXITY,
        generation=1,
        delta={"relation": "independent_front_back", "n_objects": 2},
    )
    BASE_KINDS = tuple(range(Codebook.N_KIND))
    PART_KINDS = tuple(range(Codebook.N_KIND))
    BASE_HUES = tuple(range(Codebook.N_HUE))
    PART_HUES = tuple(range(Codebook.N_HUE))
    TEMPLATE_VARIANT = ""
    GEOMETRY_FAMILY = "layered"
    # None = 父 layered 独立采样; 子模板可设置比例/横向/深度约束
    PART_SCALE_RANGE: tuple[float, float] | None = None
    PART_LATERAL_RANGE: tuple[float, float] | None = None
    DEPTH_GAP_RANGE = (0.7, 1.4)
    TARGET_IDX = (1, 2, 3, 4, 7, 8, 9, 10)
    CLASS_IDX = (0, 6, 5, 11, 12, 13)  # k0,k1,h0,h1,lcol,ldir
    CAT_SIZES = (3, 3, 6, 6, 3, 3)

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg

    @staticmethod
    def _margin(s: float, z: float) -> float:
        zc = Codebook.CAM_Z - z
        m = Codebook.EXTENT * s * Codebook.FX / zc + 2.0
        return m + Codebook.STEREO_BASE / 2 * Codebook.FX / zc

    @classmethod
    def _inside(cls, u: float, v: float, s: float, z: float) -> bool:
        """单个图元连同立体偏移是否完整落入画面。"""
        m = cls._margin(s, z)
        return m <= u <= Codebook.W - m and m <= v <= Codebook.H - m

    @classmethod
    def _sample_free(
        cls,
        rng: random.Random,
        extrap: bool,
        z_range: tuple[float, float] = (2.4, 3.4),
    ) -> tuple[float, ...]:
        """独立物体 → (u,v,s,z), 取景约束拒绝重采。"""
        s = z = m = 0.0
        for _ in range(8):
            if extrap:
                s = rng.choice(
                    [
                        rng.uniform(*Codebook.S_EXTRA[0]),
                        rng.uniform(*Codebook.S_EXTRA[1]),
                    ]
                )
            else:
                s = rng.uniform(*Codebook.S_RANGE)
            z = rng.uniform(*z_range)
            m = cls._margin(s, z)
            if 2 * m <= Codebook.W - 4:
                break
        u = rng.uniform(m, Codebook.W - m)
        v = rng.uniform(m, Codebook.H - m)
        return u, v, s, z

    @classmethod
    def _sample_pair(cls, rng: random.Random, extrap: bool) -> tuple[float, ...]:
        """前/后层连续参数; 父族独立采样, 子族可按 scale/lateral/depth 约束。"""
        for _ in range(8):
            u0, v0, s0, z0 = cls._sample_free(rng, extrap, (3.1, 4.2))
            scale_range = cls.PART_SCALE_RANGE
            if scale_range is None:
                break
            s1 = s0 * rng.uniform(*scale_range)
            z1 = max(z0 - rng.uniform(*cls.DEPTH_GAP_RANGE), 2.3)
            a0 = Codebook.EXTENT * s0 * Codebook.FX / (Codebook.CAM_Z - z0)
            a1 = Codebook.EXTENT * s1 * Codebook.FX / (Codebook.CAM_Z - z1)
            lateral = cls.PART_LATERAL_RANGE or (-0.75, 0.75)
            u1 = u0 + rng.uniform(*lateral) * (a0 + a1)
            v1 = v0
            if cls._inside(u0, v0, s0, z0) and cls._inside(u1, v1, s1, z1):
                return u0, v0, s0, z0, u1, v1, s1, z1
        else:
            raise RuntimeError("LayeredCodebook 子模板取景拒绝重采失败")

        z1 = max(z0 - rng.uniform(*cls.DEPTH_GAP_RANGE), 2.3)
        u1, v1, s1, _ = cls._sample_free(rng, extrap, (2.3, 3.5))
        z1 = min(z1, z0 - 0.05)
        if rng.random() < 0.7:
            a0 = Codebook.EXTENT * s0 * Codebook.FX / (Codebook.CAM_Z - z0)
            a1 = Codebook.EXTENT * s1 * Codebook.FX / (Codebook.CAM_Z - z1)
            reach = 0.75 * (a0 + a1)
            u1 = u0 + rng.uniform(-reach, reach)
            v1 = v0 + rng.uniform(-reach, reach)
            m1 = cls._margin(s1, z1)
            if not (m1 <= u1 <= Codebook.W - m1 and m1 <= v1 <= Codebook.H - m1):
                u1, v1, s1, _ = cls._sample_free(rng, extrap)
                z1 = max(z0 - rng.uniform(*cls.DEPTH_GAP_RANGE), 2.3)
        return u0, v0, s0, z0, u1, v1, s1, z1

    @classmethod
    def _block(cls, seed: int, extrap: bool = False) -> mx.array:
        """单复制块 → (2916,14), 离散因子全笛卡尔积。"""
        rng = random.Random(seed)
        rows = []
        for k0 in cls.BASE_KINDS:
            for k1 in cls.PART_KINDS:
                for h0 in cls.BASE_HUES:
                    for h1 in cls.PART_HUES:
                        for lc in range(cls.N_LIGHT_COLORS):
                            for ld in range(cls.N_LIGHT_DIRS):
                                u0, v0, s0, z0, u1, v1, s1, z1 = (
                                    cls._sample_pair(rng, extrap)
                                )
                                rows.append(
                                    [
                                        k0, u0, v0, s0, z0, h0,
                                        k1, u1, v1, s1, z1, h1,
                                        lc, ld,
                                    ]
                                )
        return mx.array(rows, dtype=mx.float32)

    @classmethod
    def sample(
        cls, replicates: int, seed: int, extrap: bool = False
    ) -> mx.array:
        """→ (2916×R,14)。逐复制块独立种子, R 增长纯追加。"""
        return mx.concatenate(
            [cls._block(seed * 1000 + r, extrap) for r in range(replicates)]
        )

    def to_scene(self, params: tuple[float, ...]) -> Scene:
        """双层参数 → cga Scene; renderer 的深度排序自然产生遮挡。"""
        assert len(params) == 14, f"双层参数应为 14 维, 得到 {len(params)}"
        scene = Scene(background=Color(self.cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        lcol, ldir = int(params[12]), int(params[13])
        scene.add(
            DirectionalLight(
                Color(Codebook.LIGHT_COLORS[lcol]),
                0.7,
                direction=Codebook.LIGHT_DIRS[ldir],
            )
        )
        for off in (0, 6):
            kind = int(params[off])
            u, v, s, z = (float(x) for x in params[off + 1 : off + 5])
            hue = int(params[off + 5])
            x, y = Codebook.unproject(u, v, z)
            material = MeshStandardMaterial(
                Color(Codebook.obj_color(hue)), roughness=0.55
            )
            scene.add(
                Mesh(Codebook.geometry(kind, s), material, position=(x, y, z))
            )
        return scene

    @staticmethod
    def targets(p: mx.array) -> mx.array:
        """双层参数 → 连续目标 [u0,v0,s0,z0,u1,v1,s1,z1]。"""
        return p[:, list(LayeredCodebook.TARGET_IDX)]

    @staticmethod
    def scene_classes(p: mx.array) -> mx.array:
        """双层参数 → 离散因子 [k0,k1,h0,h1,lcol,ldir]。"""
        return p[:, list(LayeredCodebook.CLASS_IDX)].astype(mx.int32)
