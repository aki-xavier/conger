"""Codebook: 连续场景参数 ⇄ cga Scene (三维建模) + 领域常量。

场景: 暗背景 + 单个浅色图元 (sphere/cylinder/box), 像素位置 (u,v)、
尺寸 s、深度 z 全连续 —— 训练范围均匀采样, 外推探针采样范围外区间
(插值/外推分界即训练支撑集边界)。图元色绑定 kind (色度是 kind 的
合法判别线索; 颜色与种类解耦留待训练数据重设计)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import mlx.core as mx
from cga.engine import (
    AmbientLight,
    BoxGeometry,
    Color,
    CylinderGeometry,
    DirectionalLight,
    Mesh,
    MeshBasicMaterial,
    MeshStandardMaterial,
    PerspectiveCamera,
    Renderer,
    Scene,
    SphereGeometry,
)

if TYPE_CHECKING:
    from inverse_config import InverseConfig


class Codebook:
    """连续场景参数 (kind,u,v,s,z) ⇄ cga Scene + 领域常量。"""

    H = W = 144
    FX = FY = 90.0  # 引擎 fy = H/(2·tan(fov/2)) → 反解 fov
    FOV = 2.0 * math.degrees(math.atan((H / 2.0) / FY))
    CAM_Z = 5.5  # 相机位置 z (世界), 看向原点
    KINDS = ("sphere", "cylinder", "box")
    N_KIND = 3
    S_RANGE = (0.35, 0.6)  # 半径/半边长, 训练范围
    Z_RANGE = (2.5, 4.0)  # 图元中心世界 z, 训练范围
    # 外推探针区间 (训练支撑集之外; 位置不外推 —— 受图像边界物理限制)
    S_EXTRA = ((0.25, 0.35), (0.6, 0.75))
    Z_EXTRA = ((2.0, 2.5), (4.0, 4.5))
    EXTENT = 1.8  # 图元最大世界半径系数: box 半对角 √3≈1.732, 取余量
    # 光照: 默认右上光; 多光照训练用 5 方向池轮流渲染 → 光照不变;
    # TEST_LIGHT_DIR 为池外顶光, 验证真泛化
    LIGHT_DIRS: ClassVar[tuple] = (
        (0.3, -0.7, 0.4),
        (-0.6, -0.4, 0.7),
        (0.6, -0.4, 0.7),
        (-0.3, 0.7, 0.4),
        (0.0, 0.0, 1.0),
    )
    TEST_LIGHT_DIR = (0.0, -1.0, 0.0)  # 池外: 正上方顶光
    # 遮挡: 固定黄色竖柱 (图中央偏右, 像素 81,84 @ z=3.5)
    OCC_BOX = (0.5, 1.4, 0.5)
    OCC_UV = (81.0, 84.0)
    OCC_Z = 3.5
    OCC_COLOR = 0xF1C40F

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg

    @staticmethod
    def sample(n: int, key: mx.array, extrap: bool = False) -> mx.array:
        """→ (n,5) [kind,u,v,s,z]。位置边距按 s,z 逐样本计算,
        保证图元完整在画面内 (最大延伸 EXTENT·s 世界单位)。"""
        ks, kz, ku, kv, kk, ke = mx.random.split(key, 6)
        kind = mx.random.randint(0, Codebook.N_KIND, shape=(n,), key=kk)
        cb = Codebook
        if extrap:
            # 支撑集外两側区间等概率
            def uni_extra(rng, pair):
                side = mx.random.randint(0, 2, shape=(n,), key=rng)
                lo = mx.where(side == 0, pair[0][0], pair[1][0])
                hi = mx.where(side == 0, pair[0][1], pair[1][1])
                return lo + mx.random.uniform(shape=(n,), key=rng) * (hi - lo)

            s = uni_extra(ks, cb.S_EXTRA)
            z = uni_extra(kz, cb.Z_EXTRA)
        else:
            s = cb.S_RANGE[0] + mx.random.uniform(shape=(n,), key=ks) * (
                cb.S_RANGE[1] - cb.S_RANGE[0]
            )
            z = cb.Z_RANGE[0] + mx.random.uniform(shape=(n,), key=kz) * (
                cb.Z_RANGE[1] - cb.Z_RANGE[0]
            )
        margin = cb.EXTENT * s * cb.FX / (cb.CAM_Z - z) + 2.0  # 像素边距
        u = margin + mx.random.uniform(shape=(n,), key=ku) * (cb.W - 2 * margin)
        v = margin + mx.random.uniform(shape=(n,), key=kv) * (cb.H - 2 * margin)
        return mx.stack(
            [kind.astype(mx.float32), u, v, s, z], axis=1
        ).astype(mx.float32)

    @staticmethod
    def unproject(u: float, v: float, z0: float) -> tuple[float, float]:
        """像素 (u,v) + 深度 → 世界坐标 (投影点即像素点, 相机 Y 向下)。"""
        zc = Codebook.CAM_Z - z0
        x = (u - (Codebook.W - 1) / 2.0) * zc / Codebook.FX
        y = ((Codebook.H - 1) / 2.0 - v) * zc / Codebook.FY
        return x, y

    def to_scene(
        self, params: tuple[float, float, float, float, float], light=None
    ) -> Scene:
        """场景参数 (kind,u,v,s,z) → cga Scene。light: 覆盖光照方向
        (多光照/池外测试用), None = 默认 (test_light 配置则池外顶光)。"""
        cfg = self.cfg
        kind = int(params[0])
        u, v, s, z = (float(p) for p in params[1:])
        x, y = self.unproject(u, v, z)
        if kind == 0:
            geom = SphereGeometry(s)
        elif kind == 1:
            geom = CylinderGeometry(s, length=2.2 * s)  # 有限柱: 竖向可观测
        else:
            geom = BoxGeometry(2 * s, 2 * s, 2 * s)
        scene = Scene(background=Color(cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        ld = light or (
            self.TEST_LIGHT_DIR if cfg.test_light else self.LIGHT_DIRS[0]
        )
        scene.add(DirectionalLight(Color(0xFFFFFF), 0.7, direction=ld))
        if cfg.equal_luma:
            # 等亮度: 无明暗 (Basic 材质不接光照) → L 图均匀, 轮廓仅存于色度
            material = MeshBasicMaterial(Color(cfg.kind_colors[kind]))
        else:
            material = MeshStandardMaterial(
                Color(cfg.kind_colors[kind]), roughness=0.55
            )
        scene.add(Mesh(geom, material, position=(x, y, z)))
        if cfg.occlusion:
            scene.add(self.occluder())
        return scene

    def occluder(self) -> Mesh:
        """固定黄色竖柱遮挡物 (后添加 → 同深度时 z-buffer 赢)。"""
        xo, yo = self.unproject(*self.OCC_UV, self.OCC_Z)
        mat = (
            MeshBasicMaterial(Color(self.OCC_COLOR))
            if self.cfg.equal_luma
            else MeshStandardMaterial(Color(self.OCC_COLOR), roughness=0.55)
        )
        return Mesh(BoxGeometry(*self.OCC_BOX), mat, position=(xo, yo, self.OCC_Z))

    @staticmethod
    def make_renderer() -> tuple[Renderer, PerspectiveCamera]:
        renderer = Renderer(Codebook.H, Codebook.W, aa=1)
        cam = PerspectiveCamera(
            fov=Codebook.FOV,
            aspect=1.0,
            near=0.1,
            far=50.0,
            position=(0.0, 0.0, Codebook.CAM_Z),
            target=(0.0, 0.0, 0.0),
        )
        cam.look_at((0.0, 0.0, 0.0))
        return renderer, cam
