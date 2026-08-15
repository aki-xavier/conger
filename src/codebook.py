"""Codebook: 连续场景参数 ⇄ cga Scene (三维建模) + 领域常量。

场景: 暗背景 + 单个图元 (sphere/cylinder/box), 位置 (u,v)、尺寸 s、
深度 z 连续, 图元色 6 色相 (与 kind 解耦 —— kind 只剩形状线索,
颜色泄漏捷径拆除), 光色 3 / 光向 3 也是待重建的场景因子。离散因子
(kind × 图元色 × 光色 × 光向 = 162) 全笛卡尔积覆盖采样 (每组合
≥1 样本), 连续因子每样本独立随机。反照率×光照的联合歧义不再靠
丢弃光照标签回避, 而由全因子监督与条件后验显式建模。
"""

from __future__ import annotations

import colorsys
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
    MeshStandardMaterial,
    PerspectiveCamera,
    Renderer,
    Scene,
    SphereGeometry,
)

from template_lineage import TemplateLineage

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
    N_OBJECTS = 1
    N_COMBO = 162
    S_RANGE = (0.35, 0.6)  # 半径/半边长, 训练范围
    Z_RANGE = (2.5, 4.0)  # 图元中心世界 z, 训练范围
    # 外推探针区间 (训练支撑集之外; 位置不外推 —— 受图像边界物理限制)
    S_EXTRA = ((0.25, 0.35), (0.6, 0.75))
    Z_EXTRA = ((2.0, 2.5), (4.0, 4.5))
    EXTENT = 1.8  # 图元最大世界半径系数: box 半对角 √3≈1.732, 取余量
    STEREO_BASE = 0.2  # 双眼基线 (世界单位): 训练深度范围 d∈[6,12]px
    # 采样器版本 (入缓存指纹): 3 = 逐复制块独立种子 (增量追加友好)
    SAMPLE_V = 3
    # 渲染管线版本 (入缓存指纹): 2 = cga 引擎线性空间光照 + 输出端
    # sRGB 编码 (cga d71e0e4 重构, 着色数值变化, 几何不变)
    RENDER_V = 2
    # 几何统计契约: 单物体全局 [ẑ,area]; 由 DataBuilder/Reconstructor 共用
    USES_LAYER_STATS = False
    USES_COMPOSITE_STATS = False
    STEREO_V = "st4"
    TEMPLATE_VARIANT = ""  # 动态子模板非空, 进入缓存指纹
    GEOMETRY_FAMILY = "single"
    TEMPLATE_COMPLEXITY = 1.0  # 一个独立 primitive 的描述长度基准
    TEMPLATE_LINEAGE = TemplateLineage(
        family="single",
        parent_family=None,
        operation="primitive",
        complexity=TEMPLATE_COMPLEXITY,
        generation=0,
    )
    # 离散因子水平 (全笛卡尔积 = 3×6×3×3 = 162 组合)。水平数取最小
    # 可行集 (覆盖要求 = 组合存在即可), 省下的样本预算换连续复制数
    # R —— 稀疏平铺实测: R=1 时每组合 1 样本, 最近分量必色差失配,
    # 位置回归全崩 (R²≈0); 复制密度才是约束, 组合数不是
    N_HUE = 6  # 图元色: 60° 等距色相环
    LIGHT_COLORS: ClassVar[tuple] = (0xFFFFFF, 0xFF4040, 0x4040FF)
    WHITE = 0  # LIGHT_COLORS 中白色下标 (色恒常监督锚点)
    LIGHT_DIRS: ClassVar[tuple] = (
        (0.3, -0.7, 0.4),
        (-0.6, -0.4, 0.7),
        (0.6, -0.4, 0.7),
    )

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg

    @staticmethod
    def obj_color(hue_idx: int) -> int:
        """色相下标 → RGB hex: HSV(H, S=0.8, V=0.85)。"""
        h = hue_idx / Codebook.N_HUE
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.85)
        return (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)

    @staticmethod
    def sample(replicates: int, seed: int, extrap: bool = False) -> mx.array:
        """→ (162×R, 8)。逐复制块独立种子 (seed·1000+r): R 增长纯追加,
        已有块不重采 —— 增量训练的数据侧前提。"""
        return mx.concatenate(
            [
                Codebook._block(mx.random.key(seed * 1000 + r), extrap)
                for r in range(replicates)
            ]
        )

    @staticmethod
    def _block(key: mx.array, extrap: bool = False) -> mx.array:
        """单复制块 → (162, 8) [kind,u,v,s,z,hue,lcol,ldir]。离散因子全
        笛卡尔积 (组合覆盖保证), 连续因子每行独立随机; 位置边距按 s,z
        逐样本计算 (含立体偏移 FX·B/2z), 保证图元在两视图都完整在画面内。"""
        cb = Codebook
        combos = [
            (k, h, c, d)
            for k in range(cb.N_KIND)
            for h in range(cb.N_HUE)
            for c in range(len(cb.LIGHT_COLORS))
            for d in range(len(cb.LIGHT_DIRS))
        ]
        n = len(combos)  # 162

        def uni_extra(rng, pair):
            """支撑集外两側区间等概率采样 (n,) (外推探针用)。"""
            side = mx.random.randint(0, 2, shape=(n,), key=rng)
            lo = mx.where(side == 0, pair[0][0], pair[1][0])
            hi = mx.where(side == 0, pair[0][1], pair[1][1])
            return lo + mx.random.uniform(shape=(n,), key=rng) * (hi - lo)

        ks, kz, ku, kv = mx.random.split(key, 4)
        if extrap:
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
        margin = margin + cb.STEREO_BASE / 2 * cb.FX / (cb.CAM_Z - z)
        # 取景约束: margin ≤ W/2−2, 否则图元出画。角部组合 (大 s × 近 z)
        # 物理上放不下 → 拒绝重采 (相机取景的诚实约束, 非分布偏差)。
        # 实测: 不拒绝时角部样本出画 → 掩码为空 → 视差质心 1/d 爆炸
        for attempt in range(8):
            badm = 2 * margin > cb.W - 4.0
            nbad = int(mx.sum(badm.astype(mx.int32)))
            if nbad == 0:
                break
            ka, kb = mx.random.split(mx.random.key(1000 + attempt), 2)
            if extrap:
                s_new = uni_extra(ka, cb.S_EXTRA)
                z_new = uni_extra(kb, cb.Z_EXTRA)
            else:
                s_new = cb.S_RANGE[0] + mx.random.uniform(shape=(n,), key=ka) * (
                    cb.S_RANGE[1] - cb.S_RANGE[0]
                )
                z_new = cb.Z_RANGE[0] + mx.random.uniform(shape=(n,), key=kb) * (
                    cb.Z_RANGE[1] - cb.Z_RANGE[0]
                )
            s = mx.where(badm, s_new, s)
            z = mx.where(badm, z_new, z)
            margin = cb.EXTENT * s * cb.FX / (cb.CAM_Z - z) + 2.0
            margin = margin + cb.STEREO_BASE / 2 * cb.FX / (cb.CAM_Z - z)
        u = margin + mx.random.uniform(shape=(n,), key=ku) * (cb.W - 2 * margin)
        v = margin + mx.random.uniform(shape=(n,), key=kv) * (cb.H - 2 * margin)
        disc = mx.array(combos, dtype=mx.float32)  # (n,4) kind,hue,lcol,ldir
        return mx.concatenate(
            [disc[:, 0:1], u[:, None], v[:, None], s[:, None], z[:, None],
             disc[:, 1:]],
            axis=1,
        ).astype(mx.float32)

    @staticmethod
    def unproject(u: float, v: float, z0: float) -> tuple[float, float]:
        """像素 (u,v) + 深度 → 世界坐标 (投影点即像素点, 相机 Y 向下)。"""
        zc = Codebook.CAM_Z - z0
        x = (u - (Codebook.W - 1) / 2.0) * zc / Codebook.FX
        y = ((Codebook.H - 1) / 2.0 - v) * zc / Codebook.FY
        return x, y

    @staticmethod
    def geometry(kind: int, s: float):
        """kind × 尺度 → 图元几何 (单/多物体场景族共享)。"""
        if kind == 0:
            return SphereGeometry(s)
        if kind == 1:
            return CylinderGeometry(s, length=2.2 * s)  # 有限柱: 轴向可观测
        return BoxGeometry(2 * s, 2 * s, 2 * s)

    def to_scene(self, params: tuple[float, ...]) -> Scene:
        """场景参数 (kind,u,v,s,z,hue,lcol,ldir) → cga Scene。"""
        cfg = self.cfg
        kind = int(params[0])
        u, v, s, z = (float(p) for p in params[1:5])
        hue, lcol, ldir = (int(p) for p in params[5:8])
        x, y = self.unproject(u, v, z)
        geom = self.geometry(kind, s)
        scene = Scene(background=Color(cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(
            DirectionalLight(
                Color(self.LIGHT_COLORS[lcol]), 0.7,
                direction=self.LIGHT_DIRS[ldir],
            )
        )
        material = MeshStandardMaterial(
            Color(self.obj_color(hue)), roughness=0.55
        )
        scene.add(Mesh(geom, material, position=(x, y, z)))
        return scene

    @staticmethod
    def make_renderer(
        baseline: float = STEREO_BASE,
    ) -> tuple[Renderer, PerspectiveCamera, PerspectiveCamera]:
        """平行 rig: 左右相机 x 偏移 ±B/2, 光轴平行 (−z)。视差
        d = FX·B/zc 纯水平、与位置无关 (汇聚式有梯形畸变, 不用)。"""
        renderer = Renderer(Codebook.H, Codebook.W, aa=1)
        cams = []
        for sign in (-1.0, 1.0):
            px = sign * baseline / 2
            cam = PerspectiveCamera(
                fov=Codebook.FOV,
                aspect=1.0,
                near=0.1,
                far=50.0,
                position=(px, 0.0, Codebook.CAM_Z),
                target=(px, 0.0, 0.0),
            )
            cam.look_at((px, 0.0, 0.0))
            cams.append(cam)
        return renderer, cams[0], cams[1]
