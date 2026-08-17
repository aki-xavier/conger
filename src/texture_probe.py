"""texture_probe.py — cga 纹理 → Riesz 结构轴/统计轴判别探针。

验证 architecture.md §2.3 的二分: `RieszWavelet.features()` 算全 11 张
跨尺度图, 其中 3 张是结构轴 (log_mag/phase_coh/ori_R, 当前 FEAT 已选),
8 张是统计轴 (slope/residual/bump/centroid/spread/skew/kurt/mean_ori,
纹理描述子, FEAT 未选)。

本脚本独立渲染带纹理 cga 场景, 在前景掩码内对每张图取 (mean, std)
区域描述子, 用 1-NN LOO 判别纹理身份, 比较结构轴 vs 统计轴、以及
lum vs chr_re+chr_im 两个源通道的可分性。

实验 (各独立; unlit 用 MeshBasicMaterial 直出贴图, 消除明暗污染):
  E1 色度纹理 (Rec601-sRGB 等亮度蓝↔红, 仅空间模式变): 应仅见于 chroma
  E2 灰度纹理 (白/灰棋盘·条纹·噪声):                   应仅见于 lum
  E3 roughness 0.2/0.55/0.9 (球面, 无贴图):           应见于 lum 统计轴
  E4 灰度棋盘尺度 (texel 2/4/8):                      应见于 lum 统计轴
  E5 谱斜率 (白/粉/蓝噪声, 全帧合成, 无渲染):         结构轴失明, 统计轴应分
  E6 谱斜率 + 对比度抖动 ×[1/3,3]:                     log_mag 崩, slope 不变
  E7 对比度纯判别 (同一粉噪声, c∈{0.5,1,2}):           log_mag 分, slope 不分

运行: python src/texture_probe.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from cga.engine import (
    AmbientLight,
    Color,
    DirectionalLight,
    Mesh,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Scene,
    Texture,
)

from codebook import Codebook
from feature_extractor import FeatureExtractor
from riesz import RieszWavelet
from stereo import StereoDepth

# 结构轴 (当前 FEAT) 与统计轴 (未选纹理描述子) 的图名
STRUCT_MAPS = ("log_mag", "phase_coh", "ori_R")
STAT_MAPS = (
    "slope", "residual", "bump", "centroid", "spread", "skew", "kurt", "mean_ori",
)


# ── 纹理生成 (sRGB 编码直接给 Texture.from_rgba) ─────────────────────

def _iso_pair_srgb(g: float, C: float, r_lo: float, r_hi: float):
    """等 Rec601 sRGB 亮度 (0.299R+0.587G+0.114B=C) 的蓝↔红色相对:
    固定 G=g, R 从 r_lo(蓝) 到 r_hi(红), B 由约束解出。frame_lum 用同一
    Rec601 权重 → unlit 下 lum 通道严格平坦。"""
    def col(r: float) -> tuple[float, float, float]:
        b = (C - 0.587 * g - 0.299 * r) / 0.114
        return (r, g, b)
    return col(r_lo), col(r_hi)


def _checker(size: int, c1, c2, tile: int) -> Texture:
    px = [
        [
            [*(c1 if ((i // tile) + (j // tile)) % 2 == 0 else c2), 1.0]
            for j in range(size)
        ]
        for i in range(size)
    ]
    return Texture.from_rgba(px)


def _stripes(size: int, c1, c2, period: int) -> Texture:
    px = [
        [
            [*(c1 if (j // period) % 2 == 0 else c2), 1.0]
            for j in range(size)
        ]
        for i in range(size)
    ]
    return Texture.from_rgba(px)


def _gray_noise(size: int, seed: int, lo: float, hi: float) -> Texture:
    arr = mx.random.normal((size, size), key=mx.random.key(seed))
    arr = (arr - mx.min(arr)) / (mx.max(arr) - mx.min(arr))
    arr = (arr * (hi - lo) + lo).tolist()
    return Texture.from_rgba([[[v, v, v, 1.0] for v in row] for row in arr])


def _colored_noise(shape: tuple[int, int], beta: float, seed: int) -> mx.array:
    """功率谱 ∝ |k|^beta 的合成噪声 (0=白, -1=粉 1/f, +1=蓝), 零 DC,
    对比度归一 (std=1)。频域乘 |k|^(beta/2) 后 IFFT 取实部。"""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal(shape))
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    k = np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1.0
    f = f * (k ** (beta / 2.0))
    f[0, 0] = 0.0
    out = np.real(np.fft.ifft2(f))
    out = (out - out.mean()) / (out.std() + 1e-9)
    return mx.array(out, dtype=mx.float32)


# ── 场景渲染 ─────────────────────────────────────────────────────────

class TextureProbe:
    """纹理 → Riesz 区域描述子 → 1-NN 判别报告。"""

    BG = 0x141414

    def __init__(self, kind: int = 2, s: float = 0.65, z: float = 2.7):
        self.renderer, self.cam_l, _ = Codebook.make_renderer()
        self.kind, self.s, self.z = kind, s, z

    def render(self, texture: Texture | None, roughness: float, lit: bool,
               hue_idx: int, lcol: int, ldir: int, u: float, v: float) -> mx.array:
        x, y = Codebook.unproject(u, v, self.z)
        geom = Codebook.geometry(self.kind, self.s)
        scene = Scene(background=Color(self.BG))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(DirectionalLight(
            Color(Codebook.LIGHT_COLORS[lcol]), 0.7,
            direction=Codebook.LIGHT_DIRS[ldir],
        ))
        # 基底白: 让 map 成为最终反照率; unlit 直出贴图 (无明暗)
        material = (
            MeshStandardMaterial(Color(0xFFFFFF), roughness=roughness, map=texture)
            if lit else MeshBasicMaterial(Color(0xFFFFFF), map=texture)
        )
        scene.add(Mesh(geom, material, position=(x, y, self.z)))
        return self.renderer.render(scene, self.cam_l)

    @staticmethod
    def frame_maps(
        frame: mx.array, rw: RieszWavelet | None
    ) -> tuple[dict, RieszWavelet]:
        """一帧 → {lum, chr_re, chr_im: FeatureMaps} (gc 与 FEAT 约定一致)。"""
        lum = FeatureExtractor.frame_lum(frame)
        chr_re, chr_im = FeatureExtractor.frame_chroma(frame)
        if rw is None:
            rw = RieszWavelet(lum)
        else:
            rw.update(lum)
        out = {"lum": rw.features(gain_control=True)}
        rw.update(chr_re)
        out["chr_re"] = rw.features(gain_control=False)
        rw.update(chr_im)
        out["chr_im"] = rw.features(gain_control=False)
        return out, rw

    @staticmethod
    def mask_of(frame: mx.array) -> mx.array:
        return StereoDepth.foreground_weights(frame) > 0.01

    @staticmethod
    def _region(fmap: mx.array, mask: mx.array) -> tuple[float, float]:
        w = mask.astype(mx.float32)
        tot = float(mx.sum(w))
        mean = float(mx.sum(fmap * w)) / tot
        d = fmap - mean
        std = float(mx.sqrt(mx.sum(d * d * w) / tot))
        return mean, std

    def descriptor(self, fmaps: dict, mask: mx.array,
                   sources: tuple[str, ...], maps: tuple[str, ...]) -> list[float]:
        parts: list[float] = []
        for src in sources:
            for name in maps:
                parts.extend(self._region(getattr(fmaps[src], name), mask))
        return parts

    @staticmethod
    def _full_descriptor(f, maps: tuple[str, ...]) -> list[float]:
        """无掩码全帧描述子: 每图 (mean, std) (E5 合成噪声用)。"""
        parts: list[float] = []
        for name in maps:
            a = getattr(f, name)
            parts.append(float(mx.mean(a)))
            parts.append(float(mx.std(a)))
        return parts

    @staticmethod
    def _loo_accuracy(descs: list[list[float]], labels: list[int]) -> float | None:
        xs = np.array(descs, dtype=np.float64)
        y = np.array(labels)
        # 丢弃近零方差列: 灰度纹理的 chroma、白材质的 chroma 全零, z-score
        # 会把浮点噪声放大成伪可分信号 (E3 chroma 0.44 的假象即源于此)。
        sd = xs.std(axis=0)
        keep = sd > 1e-6
        if not keep.any():
            return None  # 该描述子块无信号, 不可判别
        xs = (xs - xs.mean(axis=0)) / (sd + 1e-12)
        xs = xs[:, keep]
        correct = 0
        for i in range(len(y)):
            d = np.sum((xs - xs[i]) ** 2, axis=1)
            d[i] = np.inf
            correct += int(y[np.argmin(d)] == y[i])
        return correct / len(y)

    def collect(self, textures: dict[str, Texture | None],
                roughness: dict[str, float], lit: bool, jitter_seed: int = 0):
        """渲染 → (descs, labels, frames)。每纹理 9 样本 (3 ldir × 3 位置抖动)。"""
        rw: RieszWavelet | None = None
        descs, labels, frames = [], [], []
        rng = np.random.default_rng(jitter_seed)
        for label, name in enumerate(textures):
            for sample in range(9):
                ldir = sample % 3
                u = 72.0 + rng.uniform(-8, 8)
                v = 72.0 + rng.uniform(-8, 8)
                frame = self.render(
                    textures[name], roughness[name], lit, 0, 0, ldir, u, v
                )
                fmaps, rw = self.frame_maps(frame, rw)
                mask = self.mask_of(frame)
                descs.append(self.descriptor(
                    fmaps, mask, ("lum", "chr_re", "chr_im"),
                    STRUCT_MAPS + STAT_MAPS,
                ))
                labels.append(label)
                if sample == 0:
                    frames.append((name, frame))
        return descs, labels, frames

    # ── 实验 ─────────────────────────────────────────────────────────

    def run_e1_chromatic(self) -> None:
        blue, red = _iso_pair_srgb(0.35, 0.35, 0.11, 0.483)
        textures = {
            "checker": _checker(16, blue, red, 4),
            "stripes_fine": _stripes(16, blue, red, 2),
            "stripes_coarse": _stripes(16, blue, red, 6),
        }
        self.report("E1 色度纹理 unlit (3 类)", *self.collect(
            textures, {k: 0.55 for k in textures}, lit=False))

    def run_e2_luminance(self) -> None:
        w, g = (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)
        textures = {
            "checker": _checker(16, w, g, 4),
            "stripes_fine": _stripes(16, w, g, 2),
            "stripes_coarse": _stripes(16, w, g, 6),
            "noise": _gray_noise(16, 0, 0.3, 0.7),
        }
        self.report("E2 灰度纹理 unlit (4 类)", *self.collect(
            textures, {k: 0.55 for k in textures}, lit=False))

    def run_e3_roughness(self) -> None:
        probe = TextureProbe(kind=0, s=0.6, z=2.8)  # 球面: specular 瓣空间可见
        textures = {"r=0.2": None, "r=0.55": None, "r=0.9": None}
        probe.report("E3 roughness 球面 (3 类)", *probe.collect(
            textures, {"r=0.2": 0.2, "r=0.55": 0.55, "r=0.9": 0.9}, lit=True))

    def run_e4_scale(self) -> None:
        w, g = (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)
        textures = {
            "checker_t2": _checker(16, w, g, 2),
            "checker_t4": _checker(16, w, g, 4),
            "checker_t8": _checker(16, w, g, 8),
        }
        self.report("E4 灰度棋盘尺度 unlit (3 类)", *self.collect(
            textures, {k: 0.55 for k in textures}, lit=False))

    def _spectral_disc(self, title: str, specs, contrast: float | None,
                       jitter_seed: int = 7) -> None:
        """白/粉/蓝噪声 1-NN 判别 (可选对数对称对比度抖动)。gc=False:
        Wiener 收缩的噪声 floor 按每图最细尺度 MAD 估, 随 β 变, 会把 β
        泄漏进 log_mag/phase_coh 使结构轴假可分。"""
        print(f"\n== {title} ==")
        descs, labels = [], []
        rng = np.random.default_rng(jitter_seed)
        for label, (name, beta) in enumerate(specs):
            for sample in range(8):
                img = _colored_noise((256, 256), beta, seed=1000 * label + sample)
                if contrast is not None:
                    img = img * float(rng.uniform(1.0 / contrast, contrast))
                f = RieszWavelet(img).features(gain_control=False)
                if sample == 0:
                    print(f"  {name:10s} slope 图均值 = {float(f.slope.mean()):+.3f}")
                descs.append(self._full_descriptor(f, STRUCT_MAPS + STAT_MAPS))
                labels.append(label)
        n_struct = len(STRUCT_MAPS) * 2
        n_stat = len(STAT_MAPS) * 2
        cols = {
            "结构轴 (log_mag/phase_coh/ori_R)": slice(0, n_struct),
            "统计轴 (8 张)": slice(n_struct, n_struct + n_stat),
        }
        for name, sl in cols.items():
            acc = self._loo_accuracy([d[sl] for d in descs], labels)
            shown = "无信号" if acc is None else f"{acc:.2f}"
            print(f"  {name:28s} ({sl.stop - sl.start:2d}d): 1-NN LOO = {shown}")
        print("  逐图 (mean,std 2d) 1-NN:")
        for k, mname in enumerate(STRUCT_MAPS + STAT_MAPS):
            acc = self._loo_accuracy([d[2 * k : 2 * k + 2] for d in descs], labels)
            shown = "无信号" if acc is None else f"{acc:.2f}"
            print(f"    {mname:10s}: 1-NN LOO = {shown}")

    def run_e5_spectral(self) -> None:
        self._spectral_disc(
            "E5 谱斜率 (白/粉/蓝噪声, 全帧合成, 无渲染)",
            (("white β=0", 0.0), ("pink β=-1", -1.0), ("blue β=+1", 1.0)),
            contrast=None,
        )

    def run_e6_contrast(self) -> None:
        """对比度抖动: 每样本随机全局对比度 c∈[1/3,3] 作类内扰动。
        实测 log_mag 只 1.00→0.96 未崩到 chance —— β 的带通能量差
        (粉噪声低频被 DC 剥除, 差 ~10–100×) 远大于 ×3 对比度。"""
        self._spectral_disc(
            "E6 谱斜率 + 对比度抖动 ×[1/3,3] (全帧合成)",
            (("white β=0", 0.0), ("pink β=-1", -1.0), ("blue β=+1", 1.0)),
            contrast=3.0,
        )

    def run_e7_contrast_only(self) -> None:
        """对比度纯判别: 同一粉噪声谱 (β 固定), 只变全局对比度 c∈{0.5,1,2}。
        log_mag (能量 0 阶矩) 随 c² 平移应 1.00 分离; slope 等谱形状/比值量
        对 c 不变应 chance —— 反证统计轴对对比度(光照)的鲁棒性。"""
        print("\n== E7 对比度判别 (同一粉噪声, 3 档对比度, 无渲染) ==")
        descs, labels = [], []
        for label, c in enumerate((0.5, 1.0, 2.0)):
            for sample in range(8):
                img = _colored_noise((256, 256), -1.0, seed=2000 + sample) * c
                f = RieszWavelet(img).features(gain_control=False)
                descs.append(self._full_descriptor(f, STRUCT_MAPS + STAT_MAPS))
                labels.append(label)
        print("  逐图 (mean,std 2d) 1-NN:")
        for k, mname in enumerate(STRUCT_MAPS + STAT_MAPS):
            acc = self._loo_accuracy([d[2 * k : 2 * k + 2] for d in descs], labels)
            shown = "无信号" if acc is None else f"{acc:.2f}"
            print(f"    {mname:10s}: 1-NN LOO = {shown}")

    # ── 报告 ─────────────────────────────────────────────────────────

    def report(self, title: str, descs, labels, frames) -> None:
        D = 2  # mean + std
        n_struct = len(STRUCT_MAPS) * D
        n_stat = len(STAT_MAPS) * D
        n_src = n_struct + n_stat
        cols = {
            "lum 结构轴": slice(0, n_struct),
            "lum 统计轴": slice(n_struct, n_src),
            "chroma 结构轴": slice(n_src, n_src + 2 * n_struct),
            "chroma 统计轴": slice(n_src + 2 * n_struct, n_src + 2 * n_src),
        }
        print(f"\n== {title} ==")
        for name, frame in frames:
            lum = FeatureExtractor.frame_lum(frame)
            re, im = FeatureExtractor.frame_chroma(frame)
            m = self.mask_of(frame).astype(mx.float32)
            tot = float(mx.sum(m))
            def rng(a):
                mean = float(mx.sum(a * m)) / tot
                return float(mx.sqrt(mx.sum((a - mean) ** 2 * m) / tot))
            print(f"  {name:14s} 前景 lum σ={rng(lum):.3f}  "
                  f"chroma σ={rng(mx.sqrt(re * re + im * im)):.3f}")
        for name, sl in cols.items():
            acc = self._loo_accuracy([d[sl] for d in descs], labels)
            shown = "无信号" if acc is None else f"{acc:.2f}"
            print(f"  {name:14s} ({sl.stop - sl.start:2d}d): 1-NN LOO = {shown}")


if __name__ == "__main__":
    probe = TextureProbe()
    probe.run_e1_chromatic()
    probe.run_e2_luminance()
    probe.run_e3_roughness()
    probe.run_e4_scale()
    probe.run_e5_spectral()
    probe.run_e6_contrast()
    probe.run_e7_contrast_only()
