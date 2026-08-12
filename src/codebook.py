"""Codebook: 场景码 ⇄ cga Scene (三维建模) + 领域常量。"""

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
    from demo_config import DemoConfig


class Codebook:
    """场景码 ⇄ cga Scene (三维建模; 逆映射的落点) + 领域常量。

    场景: 暗背景 + 单个浅色图元 (sphere/cylinder/box), 中心投影 8×6
    网格, 尺寸两档, 深度四档 (近大远小, 单目深度线索; size 与 z 乘积
    混淆 = 熟悉尺寸歧义, 见 demo 输出)。彩色化: 三种 kind 不同色相
    (颜色成为合法判别线索, 色度通路才有信息)。
    """

    H = W = 144
    FX = FY = 90.0  # 引擎 fy = H/(2·tan(fov/2)) → 反解 fov
    FOV = 2.0 * math.degrees(math.atan((H / 2.0) / FY))
    CAM_Z = 5.5  # 相机位置 z (世界), 看向原点
    GRID = (8, 6)
    SIZES = (0.35, 0.6)  # 半径/半边长 两档
    KINDS = ("sphere", "cylinder", "box")
    N_KIND, N_GX, N_GY, N_SIZE = 3, 8, 6, 2
    Z0S = (2.5, 3.0, 3.5, 4.0)  # 图元中心世界 z, 4 档
    N_Z = len(Z0S)
    N_CODES = N_KIND * N_GX * N_GY * N_SIZE * N_Z  # 1152
    CARDS = (N_KIND, N_GX, N_GY, N_SIZE, N_Z)  # 各码列基数
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
    # 遮挡: 固定黄色竖柱 (图中央), 序数先验: 黄面积缺失 ⟹ 主图元在前
    OCC_BOX = (0.5, 1.4, 0.5)
    OCC_GRID = (4, 3)
    OCC_Z = 3.5
    OCC_COLOR = 0xF1C40F

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg

    @staticmethod
    def idx_to_code(i: int) -> tuple[int, int, int, int, int]:
        """码下标 → (kind, gx, gy, size, z); 枚举序字典序, z 在最低位。"""
        z = i % Codebook.N_Z
        i //= Codebook.N_Z
        size = i % Codebook.N_SIZE
        i //= Codebook.N_SIZE
        gy = i % Codebook.N_GY
        i //= Codebook.N_GY
        gx = i % Codebook.N_GX
        return (i // Codebook.N_GX, gx, gy, size, z)

    @staticmethod
    def code_to_idx(code: tuple[int, int, int, int, int]) -> int:
        """(kind, gx, gy, size, z) → 码下标 (idx_to_code 逆)。"""
        kind, gx, gy, size, z = code
        cb = Codebook
        return (
            (((kind * cb.N_GX + gx) * cb.N_GY + gy) * cb.N_SIZE + size) * cb.N_Z + z
        )

    @staticmethod
    def all_codes() -> mx.array:
        return mx.array(
            [list(Codebook.idx_to_code(i)) for i in range(Codebook.N_CODES)],
            dtype=mx.float32,
        )

    def project(self, gx: int, gy: int, z0: float) -> tuple[float, float]:
        """网格中心 → 世界坐标 (按深度反投影, 投影点始终网格中心)。"""
        u = (gx + 0.5) * self.W / self.N_GX
        v = (gy + 0.5) * self.H / self.N_GY
        zc = self.CAM_Z - z0
        x = (u - (self.W - 1) / 2.0) * zc / self.FX
        y = ((self.H - 1) / 2.0 - v) * zc / self.FY  # 相机 Y 向下 → 世界 Y 向上
        return x, y

    def to_scene(
        self, code: tuple[int, int, int, int, int], light: tuple | None = None
    ) -> Scene:
        """场景码 → cga Scene。light: 覆盖光照方向 (多光照/池外测试用),
        None = 默认右上光 (test_light 配置则池外顶光)。"""
        cfg = self.cfg
        kind, gx, gy, size, z = code
        x, y = self.project(gx, gy, self.Z0S[z])
        s = self.SIZES[size]
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
        scene.add(Mesh(geom, material, position=(x, y, self.Z0S[z])))
        if cfg.occlusion:
            scene.add(self.occluder())
        return scene

    def occluder(self) -> Mesh:
        """固定黄色竖柱遮挡物 (后添加 → 同深度时 z-buffer 赢)。"""
        xo, yo = self.project(self.OCC_GRID[0], self.OCC_GRID[1], self.OCC_Z)
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
