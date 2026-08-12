"""逆渲染 demo: cga engine 渲染合成场景 → Riesz 特征 → 反推 3D 场景码。

双模型 (DemoConfig.model):
  nb  (默认) 全分辨率逐码对角高斯贝叶斯 (code_bayes.CodeBayes) —— 不池化,
      精确可增量, 码簿任务最优 (实测 0.965 vs spn 0.470, 秒级 vs 分钟级);
  spn 池化 (8×6) + SPNLearner 结构学习 —— 组合泛化/消融研究对照。

场景: 暗背景 + 单个浅色图元 (sphere / cylinder / box), 中心投影在 8×6
网格上、尺寸两档、深度四档 —— 场景码 (kind, gx, gy, size, z) 即 cga
三维建模的离散编码 (code → cga Scene 对象可逆)。

训练数据: 均匀随机采样场景码 → cga engine 渲染 144×144 → Riesz 特征
(深度通道改走亮度: engine 无深度输出) → 特征矩阵 → 模型。
推理: 枚举 1152 个场景码, 后验 argmax → 重建 cga 场景 (三维建模)。

评估: 码准确率 / 逐变量准确率 / 多数类与最近模板基线 / GT vs 重建渲染。

结构 (无游离状态: 配置集中 DemoConfig, 机制分属各类):
  Codebook        码 ⇄ cga 场景 (领域常量 + 投影)
  DemoConfig      运行配置 (feat/model/消融开关, 派生量全是 property)
  FeatureExtractor 帧 → 特征向量 (池化或全分辨率)
  DataBuilder     数据构建 (缓存) 与标准化
  Priors          码先验工厂 (edge/familiar/occlusion)
  Evaluator       评估与基线
  SequenceRunner  多帧运动先验 (贝叶斯滤波)
  DemoApp         主流程 (训练/推理/评估/可视化/自检)

运行: cd src && python demo_inverse.py [--model nb|spn] [--quick] [--no-cache]
自检: --quick 内置断言 (小数据集 + 阈值按全量运行标定)。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from code_bayes import CodeBayes
from riesz import RieszWavelet
from spn import SPN, SPNLearner
from utils import Utils


@dataclass(frozen=True)
class DemoConfig:
    """运行配置 (一切开关的唯一家); 派生量全是 property, 无游离全局。"""

    model: str = "nb"  # nb=全分辨率逐码贝叶斯; spn=池化+结构学习
    feat: str = "l"  # l=亮度 Riesz 3 通道; lhs=+色度; hs=仅色度; rgb=原始
    quick: bool = False
    use_cache: bool = True
    model_path: Path | None = None
    tree: bool = False
    prior_name: str = "flat"
    min_n: int | None = None  # spn 叶最小行数 (缺省 quick=8 / 全量=3)
    sigma_floor: float = 1e-6
    equal_luma: bool = False  # 等亮度消融: L 失效 / HS 补位
    occlusion: bool = False  # 遮挡场景: 固定黄柱 + 序数先验
    sequence: int = 0  # >0: 多帧运动先验 (每序列帧数)
    test_light: bool = False  # 光照鲁棒性评估 (需 --model-path)
    multi_light: bool = False  # 多光照训练 (5 方向池轮流)

    @property
    def full_res(self) -> bool:
        """逐码贝叶斯不池化 (SPN 结构学习需要低维)。"""
        return self.model == "nb"

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        return {
            "l": FeatureExtractor.FEAT_L,
            "lhs": FeatureExtractor.FEAT_LHS,
            "hs": FeatureExtractor.FEAT_HS,
            "rgb": FeatureExtractor.FEAT_RGB,
        }[self.feat]

    @property
    def n_feat(self) -> int:
        n = Codebook.N_GX * Codebook.N_GY
        return len(self.feat_spec) * (Codebook.H * Codebook.W if self.full_res else n)

    @property
    def code_cols(self) -> tuple[int, ...]:
        return tuple(range(self.n_feat, self.n_feat + 5))

    @property
    def card(self) -> dict[int, int]:
        return dict(
            zip(
                self.code_cols,
                (
                    Codebook.N_KIND,
                    Codebook.N_GX,
                    Codebook.N_GY,
                    Codebook.N_SIZE,
                    Codebook.N_Z,
                ),
            )
        )

    @property
    def kind_colors(self) -> tuple[int, int, int]:
        # 等亮度: 三色与背景同为亮度 0.10 (L 通路失效, 轮廓只剩色度可辨)
        if self.equal_luma:
            return (0x550000, 0x002B00, 0x0000E0)
        return (0xC0392B, 0x27AE60, 0x2980B9)

    @property
    def bg_color(self) -> int:
        return 0x1A1A1A if self.equal_luma else 0x141414


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


class FeatureExtractor:
    """渲染帧 → 特征向量 (池化 8×6 块均值或全分辨率, 由 cfg.full_res)。

    特征配置: (图像源, Riesz 通道) 列表, 双通路 L / L+HS (色度)。
    """

    FEAT_L: ClassVar[tuple] = (
        ("lum", "log_mag"), ("lum", "phase_coh"), ("lum", "ori_R"),
    )
    FEAT_HS: ClassVar[tuple] = (
        ("sat", "log_mag"), ("sat", "phase_coh"), ("sat", "ori_R"),
        ("hue", "log_mag"), ("hue", "phase_coh"), ("hue", "ori_R"),
    )
    FEAT_LHS: ClassVar[tuple] = FEAT_L + FEAT_HS
    # RGB 原始数据对照 (块均值, 光照敏感)
    FEAT_RGB: ClassVar[tuple] = (("rgb", "r"), ("rgb", "g"), ("rgb", "b"))

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg

    @staticmethod
    def frame_lum(frame: mx.array) -> mx.array:
        """(H,W,4) uint8 → (H,W) float32 亮度 [0,1] (Rec601)。"""
        rgb = frame[..., :3].astype(mx.float32) / 255.0
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    @staticmethod
    def frame_hs(frame: mx.array) -> tuple[mx.array, mx.array]:
        """(H,W,4) uint8 → (H, S) 色度图, 各 [0,1)。RGB→HSV, mlx where 链。

        H 是环形量 (0/1 相接): Riesz 对 H 图滤波在色相跳变处响应,
        wrap 只影响 0/1 边界像素带, 块池化后影响可忽略。
        """
        rgb = frame[..., :3].astype(mx.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mxv = mx.maximum(mx.maximum(r, g), b)
        mn = mx.minimum(mx.minimum(r, g), b)
        d = mxv - mn
        s = mx.where(mxv > 1e-6, d / mx.maximum(mxv, 1e-6), 0.0)
        max_r = r == mxv
        max_g = g == mxv
        h6 = mx.where(max_r, (g - b) / mx.maximum(d, 1e-9), 0.0)
        h6 = mx.where(max_g, (b - r) / mx.maximum(d, 1e-9) + 2.0, h6)
        h6 = mx.where((~max_r) & (~max_g), (r - g) / mx.maximum(d, 1e-9) + 4.0, h6)
        h = mx.where(d < 1e-6, 0.0, h6 / 6.0)  # 灰: 色相无定义 → 0
        return h, s

    @staticmethod
    def block_pool(fm: mx.array) -> mx.array:
        """(H,W) → (N_GY, N_GX) 块均值 (与场景网格对齐)。"""
        cb = Codebook
        return fm.reshape(cb.N_GY, cb.H // cb.N_GY, cb.N_GX, cb.W // cb.N_GX).mean(
            axis=(1, 3)
        )

    def labels(self) -> list[str]:
        """特征列语义名: 源:通道@(gx,gy), 与池化列序一致 (源-通道主序)。"""
        cb = Codebook
        return [
            f"{src}:{ch}@({gx},{gy})"
            for src, ch in self.cfg.feat_spec
            for gy in range(cb.N_GY)
            for gx in range(cb.N_GX)
        ]

    def of_frame(
        self, frame: mx.array, rw: RieszWavelet | None
    ) -> tuple[mx.array, RieszWavelet | None]:
        """渲染帧 → 特征向量 (n_feat,)。单 RieszWavelet 实例顺序 update
        (核只建一次); full_res 时不池化 (nb 模型)。"""
        cfg = self.cfg
        lum = self.frame_lum(frame)
        hue, sat = self.frame_hs(frame)
        if cfg.equal_luma:
            # 传感器噪声底: 等亮度残差对比 (~0.6 灰度级) 在真实相机被
            # 噪声淹没 → L 通路失效; S 轮廓 (0↔1 强对比) 不受影响 → HS 补位
            # (无 key = 全局 RNG, 每帧新噪声; 复现性由数据缓存保证)
            lum = lum + mx.random.normal(shape=lum.shape, scale=0.02)
        imgs = {"lum": lum, "sat": sat, "hue": hue}
        if rw is None and cfg.feat_spec[0][0] != "rgb":
            rw = RieszWavelet(imgs[cfg.feat_spec[0][0]])
        parts = []
        for src, ch in cfg.feat_spec:
            if src == "rgb":
                # 原始 RGB: 不经 Riesz (对照实验, 光照敏感)
                rgb = frame[..., :3].astype(mx.float32) / 255.0
                m = rgb[..., {"r": 0, "g": 1, "b": 2}[ch]]
            else:
                rw.update(imgs[src])
                m = getattr(rw.features(), ch)
            parts.append(
                m.reshape(-1) if cfg.full_res else self.block_pool(m).reshape(-1)
            )
        return mx.concatenate(parts), rw


class DataBuilder:
    """数据构建 (含缓存) 与标准化。"""

    def __init__(
        self, cfg: DemoConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def cache_tag(self, n_train: int, n_test: int) -> str:
        """配置指纹 → 缓存文件名 (任一相关配置变化 → 新缓存)。"""
        cfg = self.cfg
        cb = self.codebook
        feat_tag = "".join(f"{s[:2]}{c[:2]}" for s, c in cfg.feat_spec)
        col_tag = "".join(f"{c:x}" for c in cfg.kind_colors)
        eq_tag = "eqn" if cfg.equal_luma else "std"
        occ_tag = "occ" if cfg.occlusion else "noc"
        lt_tag = "ml" if cfg.multi_light else "sl"
        res_tag = "fr" if cfg.full_res else "pl"
        return (
            f"inv_{cb.H}x{cb.W}_g{cb.N_GX}x{cb.N_GY}_{feat_tag}_{col_tag}_"
            f"{eq_tag}_{occ_tag}_{lt_tag}_{res_tag}_{n_train}_{n_test}.safetensors"
        )

    def build(
        self, n_train: int, n_test: int, use_cache: bool
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """→ (Xtr, Ctr, Xte, Cte): 特征 (n, n_feat) + 码 (n, 5), 均 float32。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / self.cache_tag(n_train, n_test)
        if use_cache and path.exists():
            d = mx.load(str(path))
            return d["Xtr"], d["Ctr"], d["Xte"], d["Cte"]

        cb = self.codebook
        tr = mx.random.randint(
            0, cb.N_CODES, shape=(n_train,), key=mx.random.key(42)
        ).tolist()
        te = mx.random.randint(
            0, cb.N_CODES, shape=(n_test,), key=mx.random.key(99)
        ).tolist()
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None

        def feats_of(idxs: list[int]) -> mx.array:
            nonlocal rw
            out = []
            for n, i in enumerate(idxs):
                # 多光照训练: 每帧轮流取方向池 (确定性, 缓存可复现)
                light = (
                    cb.LIGHT_DIRS[n % len(cb.LIGHT_DIRS)]
                    if self.cfg.multi_light
                    else None
                )
                scene = cb.to_scene(cb.idx_to_code(i), light=light)
                vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
                # 逐帧立即求值: MLX 惰性求值会把数千帧的计算图累积到
                # 一次性 eval, 超 Metal 显存上限
                mx.eval(vec)
                out.append(vec)
            return mx.stack(out)

        x_tr = feats_of(tr)
        x_te = feats_of(te)
        c_tr = mx.array([list(cb.idx_to_code(i)) for i in tr], dtype=mx.float32)
        c_te = mx.array([list(cb.idx_to_code(i)) for i in te], dtype=mx.float32)
        mx.save_safetensors(
            str(path), {"Xtr": x_tr, "Ctr": c_tr, "Xte": x_te, "Cte": c_te}
        )
        print(f"数据缓存 → {path.name}")
        return x_tr, c_tr, x_te, c_te

    @staticmethod
    def standardize(
        x_tr: mx.array, x_te: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """逐特征 z-score (训练集统计) → (z_tr, z_te, mu, sd)。

        mu/sd 随模型保存: 加载模型推理必须用同一统计。"""
        mu = x_tr.mean(axis=0, keepdims=True)
        sd = mx.maximum(x_tr.std(axis=0, keepdims=True), 1e-6)
        return (x_tr - mu) / sd, (x_te - mu) / sd, mu, sd


class Priors:
    """码先验工厂 (外部知识注入, 对应 docs/prior.md 先验体系)。"""

    def __init__(self, cfg: DemoConfig, codebook: Codebook):
        self.cfg = cfg
        self.codebook = codebook

    def build(self, name: str) -> mx.array | None:
        """码先验 log P(c)。name 可逗号组合 (如 "edge,familiar"):
        各先验 log 相加 (= 概率相乘)。
          flat: 均匀先验 (None, 纯数据似然);
          edge: 一般视角 —— 图元中心不该贴图像边缘;
          familiar: 熟悉尺寸 —— 大尺寸更常见 (0.7/0.3);
          occlusion: 遮挡序数 (per-sample, 由 occlusion() 逐帧构造)。
        log 权重在 posterior 内 softmax 归一。"""
        names = [n.strip() for n in name.split(",")]
        if "occlusion" in names:
            # per-sample 先验, 与全局 (K,) 先验形状不兼容 → 不可组合
            if len(names) > 1:
                raise ValueError("occlusion 先验不可与其他先验组合 (per-sample)")
            return None
        if names == ["flat"]:
            return None
        cb = self.codebook
        w = mx.ones(cb.N_CODES)
        for n in names:
            if n == "flat":
                continue
            for i in range(cb.N_CODES):
                _, gx, gy, size, _ = cb.idx_to_code(i)
                if n == "edge":
                    if gx in (0, cb.N_GX - 1) or gy in (0, cb.N_GY - 1):
                        w[i] *= 0.3
                elif n == "familiar":
                    w[i] *= 0.7 if size == 1 else 0.3
                else:
                    raise ValueError(f"未知先验: {n}")
        return mx.log(w)

    def occlusion(self, frames: list[mx.array]) -> mx.array:
        """遮挡序数先验 (per-sample, N_CODES 每帧): 黄柱面积缺失
        (< 0.85·F0) ⟹ 主图元遮住黄柱 ⟹ 主不比遮挡物后 ⟹ 排除 z=4.0
        (其余档中性) —— 遮挡逻辑 (prior.md 物理先验): A 遮 B ⟹ A 在前。
        注意: z=3.5 同深时主图元先渲染 (z-buffer 严格 <) 也遮黄, 故不排除。"""
        f0 = self.occluder_f0()
        cb = self.codebook
        lp = mx.zeros((len(frames), cb.N_CODES))
        for i, fr in enumerate(frames):
            if self.yellow_area(fr) < 0.85 * f0:
                for j in range(cb.N_CODES):
                    if cb.idx_to_code(j)[4] == 3:  # z=4.0: 主在后, 不可能遮黄
                        lp[i, j] = math.log(0.1)
        return lp

    @staticmethod
    def yellow_area(frame: mx.array) -> float:
        """黄色遮挡物像素数 (色相阈值: 黄 H≈0.12, S>0.4)。"""
        h, s = FeatureExtractor.frame_hs(frame)
        mask = (s > 0.4) & (h > 0.07) & (h < 0.18)
        return float(mx.sum(mask))

    def occluder_f0(self) -> float:
        """黄柱无遮挡时的黄色像素数 (固定值, 离线预计算)。"""
        renderer, cam = Codebook.make_renderer()
        scene = Scene(background=Color(self.cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(
            DirectionalLight(Color(0xFFFFFF), 0.7, direction=Codebook.LIGHT_DIRS[0])
        )
        scene.add(self.codebook.occluder())
        return self.yellow_area(renderer.render(scene, cam))


class Evaluator:
    """评估与基线。"""

    @staticmethod
    def evaluate(pred_i: list[int], gt_i: list[int]) -> dict[str, float]:
        """码全对准确率 + 逐变量 (kind/gx/gy/size/z) 准确率。"""
        cb = Codebook
        pred_codes = [cb.idx_to_code(p) for p in pred_i]
        gt_codes = [cb.idx_to_code(g) for g in gt_i]
        n = len(gt_i)
        return {
            "code": sum(p == g for p, g in zip(pred_i, gt_i, strict=True)) / n,
            "kind": sum(
                p[0] == g[0] for p, g in zip(pred_codes, gt_codes, strict=True)
            )
            / n,
            "gx": sum(p[1] == g[1] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
            "gy": sum(p[2] == g[2] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
            "size": sum(
                p[3] == g[3] for p, g in zip(pred_codes, gt_codes, strict=True)
            )
            / n,
            "z": sum(p[4] == g[4] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
        }

    @staticmethod
    def baseline_majority(tr: list[int], te: list[int]) -> float:
        """多数类: 全测样本押训练集最常见的码。"""
        most = max(set(tr), key=tr.count)
        return sum(m == most for m in te) / len(te)

    @staticmethod
    def baseline_template(
        x_tr: mx.array, c_tr: mx.array, x_te: mx.array, te: list[int]
    ) -> float:
        """最近模板: 每码取训练特征均值, 测试特征 L2 最近邻 (未见码无法命中)。"""
        cb = Codebook
        code_i = [c_tr[:, j].astype(mx.int32) for j in range(5)]
        templates: list[mx.array] = []
        present: list[int] = []
        for i in range(cb.N_CODES):
            sel = mx.ones(x_tr.shape[0], dtype=mx.bool_)
            for j in range(5):
                sel = sel & (code_i[j] == cb.idx_to_code(i)[j])
            cnt = int(mx.sum(sel))
            if cnt == 0:
                continue
            idx = Utils.nonzero(sel)
            templates.append(mx.sum(x_tr[idx], axis=0) / cnt)
            present.append(i)
        tm = mx.stack(templates)  # (P, V)
        # 距离矩阵分块且逐块立即求值: 惰性图全量累积会超 Metal 显存上限
        dd_parts = []
        chunk = 20
        for i in range(0, x_te.shape[0], chunk):
            d = mx.sum((x_te[i : i + chunk, None, :] - tm[None, :, :]) ** 2, axis=2)
            mx.eval(d)
            dd_parts.append(d)
        dd = mx.concatenate(dd_parts)
        pred = [present[int(mx.argmin(d))] for d in dd]
        return sum(p == g for p, g in zip(pred, te, strict=True)) / len(te)


class SequenceRunner:
    """多帧运动先验 (prior.md 运动与时间先验): 贝叶斯前向滤波。"""

    def __init__(
        self, cfg: DemoConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def gen_sequence(self, seed: int, n_frames: int) -> list[tuple[int, ...]]:
        """运动序列: 起始码随机, gx/gy 每帧 ±1 格随机游走 (运动连续性),
        kind/size/z 固定 (物体属性不变, prior.md 时间一致性/不变性假设)。"""
        cb = self.codebook
        key = mx.random.key(seed)
        code = list(
            cb.idx_to_code(int(mx.random.randint(0, cb.N_CODES, shape=(1,), key=key)))
        )
        seq = [tuple(code)]
        for _ in range(1, n_frames):
            key, k1, k2 = mx.random.split(key, 3)
            dx = int(mx.random.randint(-1, 2, shape=(1,), key=k1))
            dy = int(mx.random.randint(-1, 2, shape=(1,), key=k2))
            code[1] = min(cb.N_GX - 1, max(0, code[1] + dx))
            code[2] = min(cb.N_GY - 1, max(0, code[2] + dy))
            seq.append(tuple(code))
        return seq

    def temporal_preds(self) -> list[list[int]]:
        """转移图: T(c'|c) 高 ⟺ 同 kind/size/z 且 |Δgx|+|Δgy|≤1 (运动连续性)。
        返回每个 c' 的前驱列表 (P(c_t|c_{t-1}) 非零的 c_{t-1})。"""
        cb = self.codebook
        preds: list[list[int]] = [[] for _ in range(cb.N_CODES)]
        for c in range(cb.N_CODES):
            k, gx, gy, s, z = cb.idx_to_code(c)
            for dgx in (-1, 0, 1):
                for dgy in (-1, 0, 1):
                    nx, ny = gx + dgx, gy + dgy
                    if 0 <= nx < cb.N_GX and 0 <= ny < cb.N_GY:
                        preds[cb.code_to_idx((k, nx, ny, s, z))].append(c)
        return preds

    def run(
        self,
        net: SPN | CodeBayes,
        mu: mx.array,
        sd: mx.array,
        n_seqs: int,
        n_frames: int,
        seq_seed: int,
    ) -> None:
        """序列推理: 逐帧后验, 对比单帧 MAP vs 贝叶斯前向滤波
        (马尔可夫时间先验: P(c_t|c_{t-1}) 同属性+邻域)。"""
        cb = self.codebook
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        codes = cb.all_codes()
        preds = self.temporal_preds()
        log_off = math.log(0.01)
        keys = ("code", "kind", "gx", "gy", "size", "z")
        acc_single: dict[str, float] = {k: 0.0 for k in keys}
        acc_filter: dict[str, float] = {k: 0.0 for k in keys}
        total = 0
        for s in range(n_seqs):
            seq = self.gen_sequence(seq_seed + s, n_frames)
            prev_post: mx.array | None = None
            for code in seq:
                scene = cb.to_scene(code)
                vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
                x = (vec - mu) / sd  # (1, V), 训练预处理统计
                like = net.posterior(x, codes)[0]  # (K,) log 似然
                pred1 = int(mx.argmax(like))
                if prev_post is not None:
                    # 贝叶斯滤波: P(c_t|I) ∝ P(I_t|c_t)·Σ_{c_{t-1}} T·P(c_{t-1})
                    agg = mx.full((cb.N_CODES,), log_off)
                    for c in range(cb.N_CODES):
                        agg[c] = mx.logsumexp(prev_post[preds[c]])
                    post_f = like + agg
                    post_f = post_f - mx.logsumexp(post_f)
                else:
                    post_f = like
                pred2 = int(mx.argmax(post_f))
                acc_single["code"] += pred1 == cb.code_to_idx(code)
                acc_filter["code"] += pred2 == cb.code_to_idx(code)
                c1, c2 = cb.idx_to_code(pred1), cb.idx_to_code(pred2)
                for name, ci in (
                    ("kind", 0), ("gx", 1), ("gy", 2), ("size", 3), ("z", 4)
                ):
                    acc_single[name] += c1[ci] == code[ci]
                    acc_filter[name] += c2[ci] == code[ci]
                prev_post = post_f
                total += 1
        fmt = "  ".join(f"{k} {acc_single[k]/total:.3f}" for k in keys)
        fmt2 = "  ".join(f"{k} {acc_filter[k]/total:.3f}" for k in keys)
        print(f"  单帧    : {fmt}")
        print(f"  时序滤波: {fmt2}")
        assert acc_filter["code"] > acc_single["code"], "时序先验应提升码准确率"
        print(
            f"  码准确率: 单帧 {acc_single['code']/total:.3f} → "
            f"滤波 {acc_filter['code']/total:.3f}"
        )
        print("demo_inverse: 序列自检 ✓")


class DemoApp:
    """主流程: 数据 → 训练/加载 → 推理 → 评估 → 可视化 → 自检。"""

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.codebook = Codebook(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.data = DataBuilder(cfg, self.codebook, self.extractor)
        self.priors = Priors(cfg, self.codebook)
        self.sequences = SequenceRunner(cfg, self.codebook, self.extractor)

    def run(self) -> None:
        cfg = self.cfg
        cb = self.codebook
        n_train = 600 if cfg.quick else 4000
        n_test = 80 if cfg.quick else 200
        min_n = cfg.min_n
        if min_n is None:
            min_n = 8 if cfg.quick else 3  # 叶最小行数: 小 = 叶码纯 (后验锐)
        print(
            f"[1/5] 数据: train {n_train} / test {n_test} "
            f"(cache={'on' if cfg.use_cache else 'off'}, "
            f"model={cfg.model}, min_n={min_n})"
        )
        x_tr, c_tr, x_te, c_te = self.data.build(n_train, n_test, cfg.use_cache)
        tr_codes = [
            cb.code_to_idx(tuple(int(v) for v in row)) for row in c_tr.tolist()
        ]

        # 模型: 存在 → 加载; 否则训练并保存。nb 用原始特征 (无预处理),
        # spn 用 z-score (mu/sd 随模型保存, 加载时复用)
        net: SPN | CodeBayes
        mu: mx.array | None
        sd: mx.array | None
        if cfg.model_path is not None and cfg.model_path.exists():
            print(f"[2/5] 加载模型 {cfg.model_path}")
            if cfg.model == "nb":
                net, extra = CodeBayes.load(cfg.model_path)
            else:
                net, extra = SPN.load(cfg.model_path)
            mu, sd = extra.get("mu"), extra.get("sd")
            if mu is not None:
                x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
        elif cfg.model == "nb":
            assert mx.all(mx.isfinite(x_tr)), "特征含 NaN/inf"
            print("[2/5] CodeBayes 逐码充分统计 (全分辨率, 精确可增量) ...")
            net = CodeBayes.fit(
                x_tr,
                mx.array(tr_codes, dtype=mx.int32),
                cards=(cb.N_KIND, cb.N_GX, cb.N_GY, cb.N_SIZE, cb.N_Z),
            )
            mu = sd = None
            if cfg.model_path is not None:
                net.save(cfg.model_path)
                print(f"      模型已保存 → {cfg.model_path}")
        else:
            x_tr, x_te, mu, sd = self.data.standardize(x_tr, x_te)
            assert mx.all(mx.isfinite(x_tr)), "特征含 NaN/inf"
            print("[2/5] SPNLearner 结构学习 ...")
            xj = mx.concatenate([x_tr, c_tr], axis=1)
            net = SPNLearner(
                disc_cols=set(cfg.code_cols),
                card=cfg.card,
                min_n=min_n,
                max_depth=14,
                sigma_floor=cfg.sigma_floor,
            ).learn(xj)
            print(f"      根节点: {type(net.root).__name__}")
            if cfg.model_path is not None:
                net.save(cfg.model_path, {"mu": mu, "sd": sd})
                print(f"      模型已保存 → {cfg.model_path}")
        if mu is None:  # nb 无预处理: 恒等占位 (序列/光照评估复用)
            mu = mx.zeros((1, cfg.n_feat))
            sd = mx.ones((1, cfg.n_feat))

        print("[3/5] 推理: 枚举场景码后验")
        # 分块: 全批输入矩阵 × 多棵 eval 图同时构建会超 Metal 显存上限;
        # 逐块 mx.eval (立即求值, 图小) 再拼接, 结果同全批
        codes = cb.all_codes()
        parts = []
        for i in range(0, n_test, 8):
            p = net.posterior(x_te[i : i + 8], codes)
            mx.eval(p)  # 立即求值, 释放该块 eval 图
            parts.append(p)
        post = mx.concatenate(parts)  # (n_test, N_CODES) log 后验
        assert mx.all(mx.isfinite(post)), "后验含 NaN/inf"
        pred_i = mx.argmax(post, axis=1).tolist()
        gt_i = [cb.code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()]

        print("[4/5] 评估 + 基线")
        acc = Evaluator.evaluate(pred_i, gt_i)
        base_maj = Evaluator.baseline_majority(tr_codes, gt_i)
        base_tpl = Evaluator.baseline_template(x_tr, c_tr, x_te, gt_i)
        base = {"majority": base_maj, "template": base_tpl}
        print(
            f"      码: {acc['code']:.3f}  kind: {acc['kind']:.3f}  "
            f"gx: {acc['gx']:.3f}  gy: {acc['gy']:.3f}  "
            f"size: {acc['size']:.3f}  z: {acc['z']:.3f}"
        )
        print(f"      基线: majority {base_maj:.3f} / template {base_tpl:.3f}")

        prior = self.priors.build(cfg.prior_name)
        if cfg.occlusion and cfg.prior_name == "occlusion":
            # 遮挡序数先验是 per-sample 的: 重渲染测试帧检测黄柱面积缺失
            renderer, cam = Codebook.make_renderer()
            frames = []
            for row in c_te.tolist():
                code = cb.idx_to_code(cb.code_to_idx(tuple(int(v) for v in row)))
                scene = cb.to_scene(code)
                frames.append(renderer.render(scene, cam))
            prior = self.priors.occlusion(frames)  # (n_test, N_CODES)
        if prior is not None:
            # occlusion 是 (M,K) 逐样本; 其余是 (K,) 广播
            post_p = net.posterior(x_te, cb.all_codes(), log_prior=prior)
            pred_p = mx.argmax(post_p, axis=1).tolist()
            acc_p = Evaluator.evaluate(pred_p, gt_i)
            print(
                f"      注入先验[{cfg.prior_name}]: 码 {acc_p['code']:.3f}  "
                f"kind {acc_p['kind']:.3f}  gx {acc_p['gx']:.3f}  "
                f"gy {acc_p['gy']:.3f}  size {acc_p['size']:.3f}  z {acc_p['z']:.3f}"
            )

        print("[5/5] 图 → artifacts/")
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        artifacts.mkdir(exist_ok=True)
        self.plot_panel(x_te, post, gt_i, pred_i, artifacts / "inverse_panel.png")
        self.plot_metrics(acc, base, artifacts / "inverse_metrics.png")

        if cfg.sequence > 0:
            print("\n[6/5] 多帧运动先验 (prior.md 运动与时间先验)")
            self.sequences.run(
                net, mu, sd, n_seqs=10, n_frames=cfg.sequence, seq_seed=0
            )
            return
        if cfg.test_light:
            print("\n[6/5] 光照鲁棒性评估 (训练右上光, 测试池外顶光)")
            self.run_test_light(net, mu, sd, n_test)
            return

        if cfg.tree:
            if cfg.model != "spn":
                print("--tree 仅 spn 模型 (nb 无结构可视化)")
            else:
                self.print_tree(net, artifacts)

        self.self_check(acc)

    # ── 可视化 ──────────────────────────────────────────────────────

    def plot_panel(
        self,
        x_te: mx.array,
        post: mx.array,
        gt_i: list[int],
        pred_i: list[int],
        out: Path,
    ) -> None:
        """3 个测试样本: GT/Pred 渲染 + 特征图 + P(gx,gy) 热图。"""
        cfg = self.cfg
        cb = self.codebook
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        n_show = min(3, len(gt_i))
        picks = (
            [0, len(gt_i) // 2, len(gt_i) - 1]
            if len(gt_i) >= 3
            else list(range(n_show))
        )
        fig, axes = plt.subplots(n_show, 5, figsize=(17, 3.4 * n_show))
        if n_show == 1:
            axes = axes[None, :]
        ch = cfg.n_feat // len(cfg.feat_spec)  # 每通道尺寸
        fshape = (
            (cb.N_GY, cb.N_GX) if ch == cb.N_GX * cb.N_GY else (cb.H, cb.W)
        )
        unit = "blocks" if ch == cb.N_GX * cb.N_GY else "map"
        cols = [
            "GT render",
            f"GT {cfg.feat_spec[0][1]} {unit}",
            "Pred render",
            f"Pred {cfg.feat_spec[0][1]} {unit}",
            "P(gx,gy|img)",
        ]
        for row, i in enumerate(picks):
            gt_scene = cb.to_scene(cb.idx_to_code(gt_i[i]))
            pd_scene = cb.to_scene(cb.idx_to_code(pred_i[i]))
            f_gt = renderer.render(gt_scene, cam)
            f_pd = renderer.render(pd_scene, cam)
            axes[row, 0].imshow(f_gt[..., :3].astype(mx.int32))
            axes[row, 2].imshow(f_pd[..., :3].astype(mx.int32))
            lg = x_te[i, :ch].reshape(fshape)
            axes[row, 1].imshow(lg, cmap="viridis")
            # Pred 特征图: 从重建渲染重算 (与 GT 同管线, 首通道)
            vec_pd, rw = self.extractor.of_frame(f_pd, rw)
            mx.eval(vec_pd)
            lg_p = vec_pd[:ch].reshape(fshape)
            axes[row, 3].imshow(lg_p, cmap="viridis")
            pg = post[i].reshape(cb.N_KIND, cb.N_GX, cb.N_GY, cb.N_SIZE, cb.N_Z)
            pgy = mx.exp(mx.logsumexp(pg, axis=(0, 3, 4)) - mx.logsumexp(pg))
            axes[row, 4].imshow(pgy.T, cmap="hot", origin="lower")
            for c in range(5):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            ok = "✓" if pred_i[i] == gt_i[i] else "✗"
            axes[row, 0].set_title(f"GT  code {cb.idx_to_code(gt_i[i])}")
            axes[row, 2].set_title(f"Pred code {cb.idx_to_code(pred_i[i])} {ok}")
        for c, name in enumerate(cols):
            if n_show == 1:
                axes[0, c].set_xlabel(name, fontsize=9)
            else:
                axes[0, c].set_title(name, fontsize=9)
        fig.suptitle(
            "inverse rendering: GT (cga 3D model) vs single-image reconstruction",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    @staticmethod
    def plot_metrics(acc: dict[str, float], base: dict[str, float], out: Path) -> None:
        names = ["code", "kind", "gx", "gy", "size", "z"]
        vals = [acc[n] for n in names]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        bars = ax.bar(range(len(names)), vals, color="#4C72B0")
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9,
            )
        for j, (name, v) in enumerate(base.items(), start=len(names)):
            ax.bar(j, v, color="#DD8452")
            ax.text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names) + len(base)))
        ax.set_xticklabels(names + list(base.keys()))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.axhline(1 / Codebook.N_CODES, color="gray", ls=":", lw=1)
        ax.text(
            len(names) + len(base) - 0.6, 1 / Codebook.N_CODES + 0.01,
            "chance", fontsize=8, color="gray",
        )
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    def print_tree(self, net: SPN | CodeBayes, artifacts: Path) -> None:
        """SPN 树结构文本可视化 (带语义列名) + 功能分工统计。"""
        assert isinstance(net, SPN)
        cb = self.codebook
        labels = dict(enumerate(self.extractor.labels()))
        labels.update(
            dict(zip(self.cfg.code_cols, ("kind", "gx", "gy", "size", "z")))
        )
        code_names = {
            self.cfg.code_cols[0]: dict(enumerate(cb.KINDS)),
            self.cfg.code_cols[1]: {i: f"gx={i}" for i in range(cb.N_GX)},
            self.cfg.code_cols[2]: {i: f"gy={i}" for i in range(cb.N_GY)},
            self.cfg.code_cols[3]: {i: f"s={cb.SIZES[i]}" for i in range(cb.N_SIZE)},
            self.cfg.code_cols[4]: {i: f"z={cb.Z0S[i]}" for i in range(cb.N_Z)},
        }
        txt = net.tree_str(labels, code_names)
        print(txt)
        (artifacts / "spn_tree.txt").write_text(txt)
        # 功能分工: 统计各分裂轴 (哪个码维度被哪些 Sum 节点负责)
        import re
        from collections import Counter

        axes = Counter(re.findall(r"分裂轴 (\w+):", txt))
        axes.pop("码分布相近", None)
        func_names = {
            "kind": "形状辨识 (sphere/cylinder/box)",
            "z": "深度估计 (近大远小, 单目线索)",
            "gx": "横向定位",
            "gy": "纵向定位",
            "size": "尺寸估计",
        }
        print("\n── 功能分工 (Sum 节点数 × 职责) ──")
        for ax, cnt in axes.most_common():
            print(f"  {ax:<5} ×{cnt:>3}  → {func_names.get(ax, ax)}")
        print("树结构 → artifacts/spn_tree.txt")

    def run_test_light(
        self, net: SPN | CodeBayes, mu: mx.array, sd: mx.array, n_test: int
    ) -> None:
        """光照变化评估: 池外顶光重渲染测试码 → 特征 → 后验。
        对比同一模型在正常光照下的准确率, 检验鲁棒性。"""
        cfg = dataclasses.replace(self.cfg, test_light=True)
        cb2 = Codebook(cfg)
        extractor2 = FeatureExtractor(cfg)
        cb = self.codebook
        te = mx.random.randint(
            0, cb.N_CODES, shape=(n_test,), key=mx.random.key(99)
        ).tolist()
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        feats = []
        for i in te:
            scene = cb2.to_scene(cb.idx_to_code(i))
            vec, rw = extractor2.of_frame(renderer.render(scene, cam), rw)
            mx.eval(vec)
            feats.append(vec)
        x_te = (mx.stack(feats) - mu) / sd
        codes = cb.all_codes()
        parts = []
        for i in range(0, n_test, 8):
            p = net.posterior(x_te[i : i + 8], codes)
            mx.eval(p)
            parts.append(p)
        post = mx.concatenate(parts)
        pred = mx.argmax(post, axis=1).tolist()
        gt = [cb.code_to_idx(cb.idx_to_code(i)) for i in te]
        acc = Evaluator.evaluate(pred, gt)
        print(
            f"  光照变化测试: 码 {acc['code']:.3f}  kind {acc['kind']:.3f}  "
            f"gx {acc['gx']:.3f}  gy {acc['gy']:.3f}  size {acc['size']:.3f}  "
            f"z {acc['z']:.3f}"
        )
        if self.cfg.multi_light:
            # 多光照增广应显著优于单光照的池外泛化 (单光照实测 0.080)
            assert acc["code"] > 0.15, f"多光照池外泛化不足 {acc['code']:.3f}"
        print("demo_inverse: 光照鲁棒性评估 ✓")

    # ── 自检断言 (阈值按 2026-08-11/12 实测标定, 留安全余量) ─────────

    def self_check(self, acc: dict[str, float]) -> None:
        cfg = self.cfg
        if cfg.model == "nb":
            if not cfg.equal_luma and not cfg.multi_light:
                # nb 标定 (2026-08-12): 全量 ≈0.96 (模板上限, fullres 实测);
                # quick N=600 实测 0.287 (码覆盖率上限 1−e^{−0.52}≈0.41 打头)
                if cfg.quick:
                    assert acc["code"] > 0.25, (
                        f"quick nb: 码准确率过低 {acc['code']:.3f}"
                    )
                else:
                    assert acc["code"] > 0.90, f"nb: 码准确率过低 {acc['code']:.3f}"
                    assert acc["kind"] > 0.93, f"nb: kind 过低 {acc['kind']:.3f}"
                print("demo_inverse: nb 自检 ✓")
            else:
                print("demo_inverse: nb 消融模式 (断言按 spn 标定, 跳过)")
            return
        if cfg.equal_luma:
            # 等亮度消融断言: 亮度通路失效 / 色度通路补位 (对照实验)
            if cfg.feat == "l":
                assert acc["code"] < 0.05, (
                    f"等亮度下 L 通路应失效 (轮廓不可见), 实测 {acc['code']:.3f}"
                )
            else:
                assert acc["code"] > 0.30, (
                    f"等亮度下 HS 应补位, 实测 {acc['code']:.3f}"
                )
            print("demo_inverse: 等亮度消融自检 ✓ (L 失效 / HS 补位)")
            return
        if cfg.multi_light:
            # 多光照模式实测: 正常 0.360 (5 光照分摊样本) / 池外 0.265
            assert acc["code"] > 0.30, f"多光照: 码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.70, f"多光照: kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.85, f"多光照: gx 过低 {acc['gx']:.3f}"
            assert acc["z"] > 0.50, f"多光照: z 过低 {acc['z']:.3f}"
        elif cfg.quick:
            # quick N=600/min_n=8 实测: code 0.025 kind 0.40 gx 0.55
            # gy 0.64 size 0.55 z 0.35
            assert acc["code"] > 0.02, f"quick: 码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.30, f"quick: kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.35, f"quick: gx 过低 {acc['gx']:.3f}"
            assert acc["gy"] > 0.50, f"quick: gy 过低 {acc['gy']:.3f}"
            assert acc["size"] > 0.45, f"quick: size 过低 {acc['size']:.3f}"
            assert acc["z"] > 0.30, f"quick: z 过低 {acc['z']:.3f}"
        else:
            # 全量 N=4000/min_n=3 实测 (码空间 1152): code 0.470 kind 0.835
            # gx 0.895 gy 0.855 size 0.885 z 0.735; template 0.965
            assert acc["code"] > 0.40, f"码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.78, f"kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.85, f"gx 过低 {acc['gx']:.3f}"
            assert acc["gy"] > 0.80, f"gy 过低 {acc['gy']:.3f}"
            assert acc["size"] > 0.83, f"size 过低 {acc['size']:.3f}"
            assert acc["z"] > 0.68, f"z 过低 {acc['z']:.3f}"
        print("demo_inverse: 自检 ✓")


    @staticmethod
    def parse_args() -> DemoConfig:
        """CLI → DemoConfig (一切开关的唯一家)。"""
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--model",
            default="nb",
            choices=("nb", "spn"),
            help="模型: nb=全分辨率逐码贝叶斯 (默认, 精确可增量, 码簿任务最优); "
            "spn=池化+结构学习 (组合泛化/消融研究对照)",
        )
        ap.add_argument("--quick", action="store_true", help="小数据集自检模式")
        ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
        ap.add_argument(
            "--model-path",
            default=None,
            help="模型存取路径 (safetensors); 存在则加载跳过学习, 否则训练后保存",
        )
        ap.add_argument(
            "--tree",
            action="store_true",
            help="打印 SPN 树结构 (带语义列名) 并存 artifacts/spn_tree.txt",
        )
        ap.add_argument(
            "--min-n",
            type=int,
            default=None,
            help="叶最小行数 (spn 结构复杂度先验); 缺省 quick=8 / 全量=3",
        )
        ap.add_argument(
            "--feat",
            default="l",
            choices=("l", "lhs", "hs", "rgb"),
            help="特征通路: l=亮度 Riesz 3 通道; lhs=亮度+色度 HS 双通路; "
            "hs=仅色度 (消融); rgb=原始 RGB 对照 (光照敏感)",
        )
        ap.add_argument(
            "--equal-luma",
            action="store_true",
            help="等亮度模式: 三色与背景同为亮度 0.10 且无明暗 → L 通路失效, "
            "展示 HS 补位 (断言: l 应失效, lhs 应补位)",
        )
        ap.add_argument(
            "--sigma-floor",
            type=float,
            default=1e-6,
            help="高斯叶 σ 下限 (spn, 平滑性先验 prior.md)",
        )
        ap.add_argument(
            "--occlusion",
            action="store_true",
            help="遮挡场景: 固定黄色竖柱 + 序数先验 (--prior occlusion 注入,"
            "黄柱被遮 ⟹ 主图元在前)",
        )
        ap.add_argument(
            "--sequence",
            type=int,
            default=0,
            help="多帧运动先验: 每序列帧数 (>0 启用), gx/gy 随机游走,"
            "时序平滑注入上一帧转移先验",
        )
        ap.add_argument(
            "--test-light",
            action="store_true",
            help="光照鲁棒性评估: 测试集换光照方向 (需 --model-path), 检验 "
            "Riesz gain_control 归一化 vs 原始 RGB",
        )
        ap.add_argument(
            "--multi-light",
            action="store_true",
            help="多光照训练: 5 方向池轮流渲染 (数据增广 → 光照不变); "
            "配合 --test-light 用池外顶光验证泛化",
        )
        ap.add_argument(
            "--prior",
            default="flat",
            help="推理时注入的码先验 (贝叶斯 P(S)), 逗号组合如 'edge,familiar': "
            "flat=均匀, edge=一般视角(不贴边), familiar=熟悉尺寸(size 偏态), "
            "occlusion=遮挡序数 (需 --occlusion)",
        )
        a = ap.parse_args()
        return DemoConfig(
            model=a.model,
            feat=a.feat,
            quick=a.quick,
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            tree=a.tree,
            prior_name=a.prior,
            min_n=a.min_n,
            sigma_floor=a.sigma_floor,
            equal_luma=a.equal_luma,
            occlusion=a.occlusion,
            sequence=a.sequence,
            test_light=a.test_light,
            multi_light=a.multi_light,
        )


if __name__ == "__main__":
    DemoApp(DemoApp.parse_args()).run()
