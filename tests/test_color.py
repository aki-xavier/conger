"""光学先验黑盒测试 (白平衡 / 对数色度)。从 src/color.py 内嵌自检迁移。"""

import mlx.core as mx
import pytest

from color import Color


@pytest.fixture(scope="module")
def base() -> mx.array:
    """三块表面 (红/绿/蓝灰) 条带 (32, 96, 3)。"""
    b = mx.zeros((32, 96, 3))
    b = b.at[:, :32].add(mx.array([0.7, 0.2, 0.2]))
    b = b.at[:, 32:64].add(mx.array([0.2, 0.6, 0.3]))
    return b.at[:, 64:].add(mx.array([0.5, 0.5, 0.55]))


def test_gray_world_wb(base: mx.array) -> None:
    """暖光源 (R×1.3, B×0.75) → 校正后通道均值近相等; 4D 批量广播一致。"""
    cast = base * mx.array([1.3, 1.0, 0.75])
    wb = Color.gray_world_wb(cast)
    means = mx.mean(wb, axis=(0, 1))
    spread = float(mx.max(means) - mx.min(means))
    assert spread < 0.05, f"校正后通道均值应近等: {means.tolist()}"
    # 4D 批量输入广播 (回归: gain 维度曾错位)
    wb4 = Color.gray_world_wb(mx.stack([cast, cast]))
    assert wb4.shape == (2, 32, 96, 3)
    assert float(mx.max(mx.abs(wb4[0] - wb))) < 1e-6, "批量应与单图一致"


def test_log_chromaticity(base: mx.array) -> None:
    """同表面两强度 → 色度相同 (阴影不变性); 不同表面色度可分。"""
    c_bright = Color.log_chromaticity(base)
    c_dark = Color.log_chromaticity(base * 0.3)
    diff = float(mx.max(mx.abs(c_bright - c_dark)))
    assert diff < 1e-3, f"强度缩放不应改色度: {diff}"
    gap = float(mx.abs(c_bright[16, 16] - c_bright[16, 48]).sum())
    assert gap > 0.5, f"红/绿表面色度应可分: {gap}"
