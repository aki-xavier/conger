"""StereoDepth: 双眼视差 → 深度 (平行相机 rig 的几何管线)。

管线: 立体帧对 → 色度幅度分割前景 (背景 unlit, 任意光色下恒 S=0)
→ 软质心水平差 d = cx_L − cx_R (平行 rig 纯水平视差, 亚像素) →
z = CAM_Z − FX·B/d。

为什么质心而非逐像素匹配: 均匀色块内部是孔径问题 (法向流不可定),
但单凸物体 + 无遮挡场景里, 质心差是唯一无歧义的全局视差量
(懒惰但正确)。已知系统偏差: 可见面质心深度 ≠ 图元中心深度
(球前半球更近 → d 偏大 → ẑ 系统性偏近; box 前面恒在 zc−s) ——
偏差是 kind 的确定函数, 留给下游模型标定, 不在此修正。
"""

from __future__ import annotations

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor


class StereoDepth:
    """立体帧对 → (ẑ, 视差 d, 左掩码面积)。"""

    def __init__(self, baseline: float = Codebook.STEREO_BASE):
        self.b = baseline

    @staticmethod
    def foreground_weights(frame: mx.array) -> mx.array:
        """(H,W,4) → 前景权重 m = S² + (lum−bg)²。

        色度能量 (背景灰 S=0) + 亮度对比 (交叉光色下图元近黑 → 色度
        失效时亮度对比兜底); 背景亮度取帧四角中位 (角点保证是背景)。
        渲染残差精炼复用同一前景定义。"""
        re, im = FeatureExtractor.frame_chroma(frame)
        lum = FeatureExtractor.frame_lum(frame)
        corners = mx.concatenate(
            [lum[:8, :8].reshape(-1), lum[:8, -8:].reshape(-1),
             lum[-8:, :8].reshape(-1), lum[-8:, -8:].reshape(-1)]
        )
        bg = mx.median(corners)
        dl = lum - bg
        return re * re + im * im + dl * dl

    @staticmethod
    def _centroid(frame: mx.array) -> tuple[float, float]:
        """(H,W,4) → (加权 x 质心, 掩码像素数)。掩码版本 st2
        (统计入缓存, 改掩码须 bump)。"""
        m = StereoDepth.foreground_weights(frame)
        xs = mx.arange(m.shape[1], dtype=mx.float32)[None, :]
        tot = float(mx.sum(m))
        cx = float(mx.sum(m * xs)) / max(tot, 1e-8)
        # 像素计数 (表观尺寸代理): 阈值 0.01 —— 等亮度 lum 噪声底
        # (0.02σ)²≈4e-4 的 25 倍, 物体信号 S²~0.6 / dl²~1e-2 之上
        area = float(mx.sum((m > 0.01).astype(mx.float32)))
        return cx, area

    def estimate(
        self, frame_l: mx.array, frame_r: mx.array
    ) -> tuple[float, float, float]:
        """→ (ẑ 世界单位, d 视差 px, area 左掩码面积 px²)。
        ẑ 截断到物理范围: 近不可见样本的质心噪声会产生野值,
        野值经白化放大后会主导近邻选择 (实测 u/v R² 0.90→0.73) ——
        截断让误差停留在边界, 不污染度量。"""
        cx_l, area = self._centroid(frame_l)
        cx_r, _ = self._centroid(frame_r)
        d = cx_l - cx_r  # 平行 rig: d = FX·B/zc > 0
        z = Codebook.CAM_Z - Codebook.FX * self.b / max(d, 1e-6)
        z = min(max(z, 0.5), Codebook.CAM_Z + 1.0)  # 物理截断
        return z, d, area


