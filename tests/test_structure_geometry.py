"""StructureGeometry 测试: 三类结构族的观测级几何证据。"""

from typing import cast

import mlx.core as mx
import pytest

from codebook import Codebook
from composite_codebook import CompositeCodebook
from inverse_config import InverseConfig
from lateral_codebook import LateralCompositeCodebook
from lateral_composite_geometry import LateralCompositeGeometry
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
        prm = tuple(
            float(x) for x in cast(list, cb.sample(1, 777 + i)[0].tolist())
        )
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


def test_lateral_gap_cost_discriminates_mirror_vs_repeat(monkeypatch) -> None:
    """mirror/repeat 判别: 正确操作的横向间隔代价应低于错误操作。

    corrected_gap 用 monkeypatch 注入受控 g (省渲染), 只验证判别带逻辑。
    """
    state = {"g": 0.0}
    monkeypatch.setattr(
        LateralCompositeGeometry,
        "corrected_gap",
        staticmethod(lambda fl, fr, kind: state["g"]),
    )
    delta = {"period_ratio": (0.18, 0.22), "part_kinds": [1]}
    dummy = mx.zeros((4, 4, 4))
    state["g"] = 1.0  # mirror 归一化间隔
    m_on_m = StructureGeometry.lateral_gap_cost("mirror", delta, dummy, dummy)
    r_on_m = StructureGeometry.lateral_gap_cost("repeat", delta, dummy, dummy)
    state["g"] = 1.5  # repeat 归一化间隔
    r_on_r = StructureGeometry.lateral_gap_cost("repeat", delta, dummy, dummy)
    m_on_r = StructureGeometry.lateral_gap_cost("mirror", delta, dummy, dummy)
    # 正确操作代价更低, 且正确操作接近零
    assert m_on_m < r_on_m
    assert r_on_r < m_on_r
    assert m_on_m < 0.01
    assert r_on_r < 0.01
