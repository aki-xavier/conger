"""ContourCompleter 黑盒测试: 模板−遮挡物 = 可见后层。"""

import mlx.core as mx

from contour_completion import ContourCompleter


def test_square_contour_completion() -> None:
    """方形后层右下被前层遮挡 → 恢复完整中心/面积/轮廓类型。"""
    h = w = 144
    yy, xx = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    back = (xx >= 50) & (xx < 90) & (yy >= 50) & (yy < 90)
    front = (xx - 88) ** 2 + (yy - 82) ** 2 <= 22**2
    visible = back & ~front
    u, v, area, kind, score = ContourCompleter.complete(front, visible)
    assert abs(u - 69.5) < 5.0 and abs(v - 69.5) < 5.0
    assert abs(area - 1600.0) / 1600.0 < 0.25
    assert kind == 2
    assert score < 0.35


def test_circle_contour_completion() -> None:
    """圆形后层被遮挡 → 圆模板恢复中心/面积。"""
    h = w = 144
    yy, xx = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    back = (xx - 60) ** 2 + (yy - 60) ** 2 <= 20**2
    front = (xx - 77) ** 2 + (yy - 66) ** 2 <= 18**2
    visible = back & ~front
    u, v, area, kind, score = ContourCompleter.complete(front, visible)
    assert abs(u - 60.0) < 5.0 and abs(v - 60.0) < 5.0
    assert abs(area - 1256.6) / 1256.6 < 0.25
    assert kind == 0
    assert score < 0.35
