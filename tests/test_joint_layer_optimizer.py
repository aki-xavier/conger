"""JointLayerOptimizer 黑盒测试: 模板 × 遮挡 × 视差联合一致性。"""

import mlx.core as mx

from joint_layer_optimizer import JointLayerOptimizer


def test_joint_layer_optimization() -> None:
    """已知合成层: 联合得分应恢复前圆后方、中心、面积和深度。"""
    h = w = 144
    yy, xx = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    front = (xx - 87) ** 2 + (yy - 90) ** 2 <= 24**2
    back_full = (mx.abs(xx - 60) <= 27) & (mx.abs(yy - 63) <= 27)
    back = back_full & ~front
    fg = front | back
    disp = mx.zeros((h, w), dtype=mx.float32)
    disp = mx.where(front, 10.0, disp)
    disp = mx.where(back, 6.5, disp)
    out = JointLayerOptimizer.optimize(
        fg, disp, fg, front, back, d_front=10.0, d_back=6.5
    )
    assert out is not None
    u0, v0, z0, a0, u1, v1, z1, a1 = out
    assert abs(u0 - 87) < 8.0 and abs(v0 - 90) < 8.0
    assert abs(u1 - 60) < 8.0 and abs(v1 - 63) < 8.0
    assert abs(z0 - 3.7) < 0.2 and abs(z1 - 2.73) < 0.3
    assert abs(a0 - 3.1416 * 24**2) / (3.1416 * 24**2) < 0.2
    assert abs(a1 - 54**2) / 54**2 < 0.2
