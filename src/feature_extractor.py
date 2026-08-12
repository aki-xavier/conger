"""FeatureExtractor: 渲染帧 → 特征向量 (池化或全分辨率) + 特征配置。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import mlx.core as mx

from codebook import Codebook
from riesz import RieszWavelet

if TYPE_CHECKING:
    from inverse_config import InverseConfig


class FeatureExtractor:
    """渲染帧 → 特征向量 (池化 8×6 块均值或全分辨率, 由 cfg.full_res)。

    特征配置 (唯一, L+复数色相双通路): (图像源, Riesz 通道) 列表。
    色度走复数色相 S·e^{i2πH} 的实/虚两个源图 —— H 是环形量, 直接滤波
    H 图会在 0/1 切口产生假边缘 (环绕瑕疵), 复数表示无切口。
    """

    FEAT: ClassVar[tuple] = (
        ("lum", "log_mag"), ("lum", "phase_coh"), ("lum", "ori_R"),
        ("chr_re", "log_mag"), ("chr_re", "phase_coh"), ("chr_re", "ori_R"),
        ("chr_im", "log_mag"), ("chr_im", "phase_coh"), ("chr_im", "ori_R"),
    )  # 3 源 × 3 通道 = 9

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
        chr_re, chr_im = self.frame_chroma(frame)
        if cfg.equal_luma:
            # 传感器噪声底: 等亮度残差对比 (~0.6 灰度级) 在真实相机被
            # 噪声淹没 → L 通路失效; 色度轮廓不受影响 → HS 补位
            # (无 key = 全局 RNG, 每帧新噪声; 复现性由数据缓存保证)
            lum = lum + mx.random.normal(shape=lum.shape, scale=0.02)
        imgs = {"lum": lum, "chr_re": chr_re, "chr_im": chr_im}
        if rw is None:
            rw = RieszWavelet(lum)
        parts = []
        for src, ch in cfg.feat_spec:
            rw.update(imgs[src])
            m = getattr(rw.features(), ch)
            parts.append(
                m.reshape(-1) if cfg.full_res else self.block_pool(m).reshape(-1)
            )
        return mx.concatenate(parts), rw

    def of_frame_pair(
        self, frame: mx.array, rw: RieszWavelet | None
    ) -> tuple[mx.array, mx.array, RieszWavelet | None]:
        """→ (全分辨率向量, 池化向量, rw): 一次 Riesz 双分辨率输出,
        实验双臂 (全分辨率模型 + 池化 SPN) 共用 —— 机制单家。"""
        lum = self.frame_lum(frame)
        chr_re, chr_im = self.frame_chroma(frame)
        imgs = {"lum": lum, "chr_re": chr_re, "chr_im": chr_im}
        if rw is None:
            rw = RieszWavelet(lum)
        full, pooled = [], []
        for src, ch in self.cfg.feat_spec:
            rw.update(imgs[src])
            m = getattr(rw.features(), ch)
            full.append(m.reshape(-1))
            pooled.append(self.block_pool(m).reshape(-1))
        return mx.concatenate(full), mx.concatenate(pooled), rw
