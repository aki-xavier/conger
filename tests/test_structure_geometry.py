"""StructureGeometry 测试: 三类结构族的观测级几何证据。"""

import math

import mlx.core as mx
import pytest

from codebook import Codebook
from composite_codebook import CompositeCodebook
from inverse_config import InverseConfig
from lateral_codebook import LateralCompositeCodebook
from layered_codebook import LayeredCodebook
from structure_geometry import StructureGeometry


@pytest.fixture(scope="module")
def frames() -> dict[str, tuple[mx.array, mx.array]]:
    renderer, cam_l, cam_r = Codebook.make_renderer()
    families = {
        "single": Codebook(InverseConfig(scene_family="single")),
        "layered": LayeredCodebook(InverseConfig(scene_family="layered")),
        "composite": CompositeCodebook(InverseConfig(scene_family="composite")),
        "lateral": LateralCompositeCodebook(InverseConfig(scene_family="composite")),
    }
    out = {}
    for i, (name, cb) in enumerate(families.items()):
        prm = tuple(float(x) for x in cb.sample(1, 777 + i)[0].tolist())
        scene = cb.to_scene(prm)
        out[name] = (
            renderer.render(scene, cam_l), renderer.render(scene, cam_r)
        )
    return out


def test_structure_geometry_costs(frames: dict[str, tuple[mx.array, mx.array]]) -> None:
    """每个真实结构的观测几何代价都应小于另外两类。"""
    for true, (fl, fr) in frames.items():
        costs = StructureGeometry.costs(fl, fr)
        assert costs[true] == min(costs.values())


def _lateral_stats(gap_ratio: float) -> tuple[float, ...]:
    """构造横向组合几何统计 [u,v,z,area]×2, 像素空间归一化间隔 = gap_ratio。"""
    r0, r1 = 36.0, 18.0
    u0 = 58.5
    u1 = u0 + gap_ratio * (r0 + r1)
    a0 = math.pi * r0**2
    a1 = math.pi * r1**2
    return (u0, 72.0, 3.0, a0, u1, 72.0, 3.0, a1)


def test_lateral_gap_cost_discriminates_mirror_vs_repeat() -> None:
    """mirror/repeat 判别: 正确操作的横向间隔代价应低于错误操作。"""
    delta_m = {"period_ratio": (0.18, 0.22)}
    delta_r = {"period_ratio": (0.18, 0.22)}
    mirror_stats = _lateral_stats(0.20 * 5.0)  # g = 1.0
    repeat_stats = _lateral_stats(0.20 * 7.5)  # g = 1.5
    m_on_m = StructureGeometry.lateral_gap_cost("mirror", delta_m, mirror_stats)
    r_on_m = StructureGeometry.lateral_gap_cost("repeat", delta_r, mirror_stats)
    r_on_r = StructureGeometry.lateral_gap_cost("repeat", delta_r, repeat_stats)
    m_on_r = StructureGeometry.lateral_gap_cost("mirror", delta_m, repeat_stats)
    # 正确操作代价更低, 且正确操作接近零
    assert m_on_m < r_on_m
    assert r_on_r < m_on_r
    assert m_on_m < 0.01
    assert r_on_r < 0.01
