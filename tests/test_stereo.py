"""StereoDepth 黑盒测试: 合成位移帧对, 契约全部来自几何第一性原理。

从 src/stereo.py 内嵌自检迁移。
"""

import mlx.core as mx
import pytest

from stereo import StereoDepth

K = 8  # 右帧 = 左帧左移 k px (平行 rig: d = k)


@pytest.fixture(scope="module")
def frames() -> tuple[mx.array, mx.array]:
    """左帧: 红色方块 (H,W,4) uint8; 右帧 = 左移 K px。"""
    h = w = 144
    fl = mx.zeros((h, w, 4), dtype=mx.uint8)
    fl[50:90, 60:100, 0] = 200  # R
    fl[50:90, 60:100, 3] = 255
    fr = mx.zeros((h, w, 4), dtype=mx.uint8)
    fr[50:90, 60 - K : 100 - K, 0] = 200
    fr[50:90, 60 - K : 100 - K, 3] = 255
    return fl, fr


def test_disparity_pipeline(frames: tuple[mx.array, mx.array]) -> None:
    fl, fr = frames
    z, d, area = StereoDepth(baseline=0.2).estimate(fl, fr)
    # 位移不变性: 视差必须等于位移 (亚像素容差 0.1px, 质心是精确量)
    assert abs(d - K) < 0.1, f"视差 {d} ≠ 位移 {K}"
    # 深度公式: ẑ = CAM_Z − FX·B/d = 5.5 − 90·0.2/8 = 3.25
    assert abs(z - 3.25) < 0.05, f"深度 {z}"
    # 面积: 40×40 = 1600 (软权重 S²=1 每像素)
    assert abs(area - 1600) < 1.0, f"面积 {area}"
    # 背景鲁棒: 纯灰背景帧的质心权重应全在物体上 —— 本构造已隐含
    # (背景 S=0); 交换左右帧 → 视差变号 (对称性)
    _, d2, _ = StereoDepth(baseline=0.2).estimate(fr, fl)
    assert abs(d2 + K) < 0.1, f"交换视差 {d2}"
