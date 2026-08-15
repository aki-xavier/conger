"""StructureGeometry 测试: 三类结构族的观测级几何证据。"""

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
