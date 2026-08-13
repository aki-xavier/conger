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
    def _centroid(frame: mx.array) -> tuple[float, float]:
        """(H,W,4) → (加权 x 质心, 掩码像素数)。权重 m = S² + (lum−bg)²:
        色度能量 (背景灰 S=0) + 亮度对比 (交叉光色下图元近黑 → 色度
        失效时亮度对比兜底, 实测外推爆炸源); 背景亮度取帧四角中位
        (角点保证是背景)。掩码版本 st2 (统计入缓存, 改掩码须 bump)。"""
        re, im = FeatureExtractor.frame_chroma(frame)
        lum = FeatureExtractor.frame_lum(frame)
        corners = mx.concatenate(
            [lum[:8, :8].reshape(-1), lum[:8, -8:].reshape(-1),
             lum[-8:, :8].reshape(-1), lum[-8:, -8:].reshape(-1)]
        )
        bg = mx.median(corners)
        dl = lum - bg
        m = re * re + im * im + dl * dl
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


def _selftest() -> None:
    """黑盒自检: 合成位移帧对, 契约全部来自几何第一性原理。"""
    h = w = 144
    # 左帧: 红色方块 (H,W,4) uint8; 右帧 = 左移 k px (d = k)
    k = 8
    fl = mx.zeros((h, w, 4), dtype=mx.uint8)
    fl[50:90, 60:100, 0] = 200  # R
    fl[50:90, 60:100, 3] = 255
    fr = mx.zeros((h, w, 4), dtype=mx.uint8)
    fr[50:90, 60 - k : 100 - k, 0] = 200
    fr[50:90, 60 - k : 100 - k, 3] = 255
    z, d, area = StereoDepth(baseline=0.2).estimate(fl, fr)
    # 位移不变性: 视差必须等于位移 (亚像素容差 0.1px, 质心是精确量)
    assert abs(d - k) < 0.1, f"视差 {d} ≠ 位移 {k}"
    # 深度公式: ẑ = CAM_Z − FX·B/d = 5.5 − 90·0.2/8 = 3.25
    assert abs(z - 3.25) < 0.05, f"深度 {z}"
    # 面积: 40×40 = 1600 (软权重 S²=1 每像素)
    assert abs(area - 1600) < 1.0, f"面积 {area}"
    # 背景鲁棒: 纯灰背景帧的质心权重应全在物体上 —— 本构造已隐含
    # (背景 S=0); 交换左右帧 → 视差变号 (对称性)
    z2, d2, _ = StereoDepth(baseline=0.2).estimate(fr, fl)
    assert abs(d2 + k) < 0.1, f"交换视差 {d2}"
    print(f"stereo 自检 ✓ (d={d:.2f}px, ẑ={z:.3f}, area={area:.0f})")


if __name__ == "__main__":
    _selftest()
