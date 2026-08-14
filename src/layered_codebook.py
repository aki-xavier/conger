"""LayeredCodebook: 双图元遮挡/前后层场景参数 ⇄ cga Scene。

参数序: [k0,u0,v0,s0,z0,h0, k1,u1,v1,s1,z1,h1, lcol,ldir]。
按深度规范排序 (z0 < z1, 0 是前层), 消除两物体标签置换对称;
约 70% 样本强制投影重叠, 让遮挡成为数据分布的一等结构。
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import mlx.core as mx
from cga.engine import (
    AmbientLight,
    Color,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    Scene,
)

from codebook import Codebook

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
    SAMPLE_V = 1  # 双层采样器版本 (入缓存指纹)
    RENDER_V = Codebook.RENDER_V
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
    def _sample_free(cls, rng: random.Random, extrap: bool) -> tuple[float, ...]:
        """独立物体 → (u,v,s,z), 取景约束拒绝重采。"""
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
            z = rng.uniform(2.4, 3.4)
            m = cls._margin(s, z)
            if 2 * m <= Codebook.W - 4:
                break
        u = rng.uniform(m, Codebook.W - m)
        v = rng.uniform(m, Codebook.H - m)
        return u, v, s, z

    @classmethod
    def _sample_pair(cls, rng: random.Random, extrap: bool) -> tuple[float, ...]:
        """前/后层连续参数; 70% 强制投影重叠 (遮挡训练支撑)。"""
        u0, v0, s0, z0 = cls._sample_free(rng, extrap)
        z1 = min(z0 + rng.uniform(0.25, 1.0), 4.3)
        u1, v1, s1, _ = cls._sample_free(rng, extrap)
        z1 = max(z1, z0 + 0.05)
        if rng.random() < 0.7:
            a0 = Codebook.EXTENT * s0 * Codebook.FX / (Codebook.CAM_Z - z0)
            a1 = Codebook.EXTENT * s1 * Codebook.FX / (Codebook.CAM_Z - z1)
            reach = 0.75 * (a0 + a1)
            u1 = u0 + rng.uniform(-reach, reach)
            v1 = v0 + rng.uniform(-reach, reach)
            m1 = cls._margin(s1, z1)
            if not (m1 <= u1 <= Codebook.W - m1 and m1 <= v1 <= Codebook.H - m1):
                u1, v1, s1, _ = cls._sample_free(rng, extrap)
                z1 = min(z0 + rng.uniform(0.25, 1.0), 4.3)
        return u0, v0, s0, z0, u1, v1, s1, z1

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
