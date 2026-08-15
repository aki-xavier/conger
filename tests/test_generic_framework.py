"""通用结构框架的非视觉验证: 时间序列机制专家。"""

import mlx.core as mx

from generic_expert_registry import GenericExpertRegistry
from generic_structure_gate import GenericStructureGate
from structure_birth import StructureBirthController
from toy_series_expert import ToySeriesExpert
from toy_series_family import ToySeriesFamily


def _registry() -> GenericExpertRegistry:
    experts = {
        "linear": ToySeriesExpert.train("linear", n=192, seed=1),
        "sine": ToySeriesExpert.train("sine", n=192, seed=2),
    }
    return GenericExpertRegistry(
        experts,
        gate=GenericStructureGate(birth_residual=0.30),
        birth_controller=StructureBirthController(min_cases=2),
    )


def test_nonvisual_structure_gating() -> None:
    """线性/振荡观测应分别选择对应结构专家。"""
    registry = _registry()
    x = ToySeriesFamily.X
    linear_y = 1.2 * x - 0.3
    sine_y = 1.1 * mx.sin(3.3 * x + 0.4)
    out_l = registry.decide(linear_y)
    out_s = registry.decide(sine_y)
    assert out_l.estimate.structure_id == "linear"
    assert out_s.estimate.structure_id == "sine"
    assert out_l.posterior["linear"] > 0.8
    assert out_s.posterior["sine"] > 0.8
    assert not out_l.needs_new_structure
    assert not out_s.needs_new_structure


def test_nonvisual_structure_birth() -> None:
    """二次机制在两个现有专家下都不兼容 → 聚合后产生出生请求。"""
    registry = _registry()
    x = ToySeriesFamily.X
    unknown = 1.5 * x * x - 0.2
    first = registry.decide(unknown)
    assert first.needs_new_structure
    assert registry.last_birth_request is None
    registry.decide(unknown)
    req = registry.last_birth_request
    assert req is not None
    assert len(req.cases) == 2
    assert req.residual_mean > 0.30
