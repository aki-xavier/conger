"""StereoLayers 黑盒测试: 遮挡双层视差 → 前后层几何。"""

import mlx.core as mx
import pytest

from codebook import Codebook
from stereo_layers import StereoLayers


@pytest.fixture(scope="module")
def layered_frames() -> tuple[mx.array, mx.array]:
    """两个随机纹理层: 前层 d=12, 后层 d=6, 左图前层遮挡后层。"""
    h = w = Codebook.H
    fl = mx.full((h, w, 4), 20, dtype=mx.uint8)
    fr = mx.full((h, w, 4), 20, dtype=mx.uint8)
    fl[..., 3] = 255
    fr[..., 3] = 255
    rng_l, rng_r = mx.random.split(mx.random.key(9))
    back = (mx.random.uniform(shape=(40, 40, 3), key=rng_l) * 255).astype(mx.uint8)
    front = (mx.random.uniform(shape=(40, 40, 3), key=rng_r) * 255).astype(mx.uint8)
    # 左图: 后层先写, 前层覆盖重叠区
    fl[45:85, 55:95, :3] = back
    fl[60:100, 70:110, :3] = front
    # 右图: 平行 rig 下同一深度层整体左移 d px
    fr[45:85, 49:89, :3] = back
    fr[60:100, 58:98, :3] = front
    return fl, fr


def test_layered_disparity(layered_frames: tuple[mx.array, mx.array]) -> None:
    """前后层深度/中心/面积应与构造值一致 (可见层统计)。"""
    fl, fr = layered_frames
    out = StereoLayers.estimate(fl, fr)
    u0, v0, z0, a0, u1, v1, z1, a1 = out
    # d=12 → z=4.0; d=6 → z=2.5
    assert abs(z0 - 4.0) < 0.25, f"前层 z {z0}"
    assert abs(z1 - 2.5) < 0.35, f"后层 z {z1}"
    assert abs(u0 - 89.5) < 4.0 and abs(v0 - 79.5) < 4.0
    # 后层被前层遮住右下 25×25, 可见部分质心约 (69.7,59.7)
    assert abs(u1 - 69.7) < 4.0 and abs(v1 - 59.7) < 4.0
    assert a0 > a1 > 0.0, f"遮挡后前层可见面积应更大: {a0}, {a1}"
