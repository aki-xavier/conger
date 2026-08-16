"""FeatureExtractor: 渲染帧 → 全分辨率特征向量 + 特征配置。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import mlx.core as mx

from riesz import RieszWavelet

if TYPE_CHECKING:
    from inverse_config import InverseConfig


class FeatureExtractor:
    """渲染帧 → 全分辨率特征向量 (9 × H × W)。

    池化已随离散码网格一起退役: 块均值会擦除块内位置信息, 而那是
    连续回归要的信号。

    特征配置 (唯一, L+复数色相双通路): (图像源, Riesz 通道) 列表。
    色度走复数色相 S·e^{i2πH} 的实/虚两个源图 —— H 是环形量, 直接滤波
    H 图会在 0/1 切口产生假边缘 (环绕瑕疵), 复数表示无切口。

    色度源关 gain_control: 色相信息 = chr_re/chr_im 的边缘幅度比,
    对比度归一化 (Retinex 式局部除能) 会把它抹平 —— kind 就不可辨。
    代价是色度通道不光照不变; 亮度源保留归一化 (抗光照)。色相辨识
    与对比度不变性不可兼得, 这是固有的信息出口选择。

    另加两个原始 (未滤波) 色度通道: Riesz 特征是能量量, 符号盲
    (chr_im 差一个负号的两个色相, 能量特征全同 —— 等亮度绿/蓝
    实测不可分); 拮抗色信号的符号 = 色相身份, 必须有个带符号的
    信息出口 (生理上拮抗通道本就是有符号的)。
    """

    FEAT: ClassVar[tuple] = (
        ("lum", "log_mag"), ("lum", "phase_coh"), ("lum", "ori_R"),
        ("chr_re", "log_mag"), ("chr_re", "phase_coh"), ("chr_re", "ori_R"),
        ("chr_im", "log_mag"), ("chr_im", "phase_coh"), ("chr_im", "ori_R"),
        ("chr_re", "raw"), ("chr_im", "raw"),
    )  # 3 源 × 3 Riesz 通道 + 2 原始色度 = 11

    # shape 轴 8 张谱形图 (roughness 专用头的区域描述子, 见 shape_descriptor)
    SHAPE_MAPS: ClassVar[tuple] = (
        "slope", "residual", "bump", "centroid", "spread", "skew", "kurt", "mean_ori",
    )

    def __init__(self, cfg: InverseConfig):
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
    def frame_chroma(frame: mx.array) -> tuple[mx.array, mx.array]:
        """(H,W,4) uint8 → 复数色相 S·e^{i2πH} 的 (实部, 虚部) 图。

        色相环形量在复平面连续 (0.98 与 0.02 相邻), 滤波无环绕假边缘。"""
        h, s = FeatureExtractor.frame_hs(frame)
        ang = h * (2.0 * math.pi)
        return s * mx.cos(ang), s * mx.sin(ang)

    def of_frame(
        self, frame: mx.array, rw: RieszWavelet | None
    ) -> tuple[mx.array, RieszWavelet | None]:
        """渲染帧 → 全分辨率特征向量 (n_feat,)。单 RieszWavelet 实例
        顺序 update (核只建一次)。"""
        lum = self.frame_lum(frame)
        chr_re, chr_im = self.frame_chroma(frame)
        imgs = {"lum": lum, "chr_re": chr_re, "chr_im": chr_im}
        if rw is None:
            rw = RieszWavelet(lum)
        parts = []
        for src, ch in self.cfg.feat_spec:
            if ch == "raw":  # 原始源图 (带符号色度, 不过 Riesz)
                parts.append(imgs[src].reshape(-1))
                continue
            rw.update(imgs[src])
            gc = src == "lum"  # 色度关 gain_control (保色相幅度), 见类 docstring
            m = getattr(rw.features(gain_control=gc), ch)
            parts.append(m.reshape(-1))
        return mx.concatenate(parts), rw

    @staticmethod
    def shape_descriptor(frame: mx.array) -> list[float]:
        """前景掩码内 8 张谱形图的 (mean, std) → 16 维 (roughness 谱形头)。

        gc=False: Wiener 收缩的噪声 floor 按最细尺度 MAD 估, 随 roughness
        变 (粗糙高光瓣 → 高 MAD → 高 floor), 会把 roughness 泄漏进结构图;
        谱形判别须关 gc 或统一 floor (§2.3 实测)。
        """
        from stereo import StereoDepth  # 惰性导入防 stereo→feature_extractor 环

        lum = FeatureExtractor.frame_lum(frame)
        f = RieszWavelet(lum).features(gain_control=False)
        m = StereoDepth.foreground_weights(frame) > 0.01
        w = m.astype(mx.float32)
        tot = float(mx.sum(w))
        out: list[float] = []
        for name in FeatureExtractor.SHAPE_MAPS:
            a = getattr(f, name)
            mean = float(mx.sum(a * w)) / tot
            d = a - mean
            out += [mean, float(mx.sqrt(mx.sum(d * d * w) / tot))]
        return out
