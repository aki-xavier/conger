"""GenericStructureGate 两级层级后验门控黑盒测试。"""

from generic_structure_gate import GenericStructureGate
from structured_hypothesis import StructuredHypothesis


def _est(
    name: str,
    residual: float,
    family: str | None = None,
    complexity: float | None = None,
    geometry_cost: float | None = None,
) -> StructuredHypothesis:
    return StructuredHypothesis(
        structure_id=name,
        params=(0.0,),
        residual=residual,
        complexity=complexity,
        geometry_cost=geometry_cost,
        geometry_family=family,
    )


def test_hierarchical_posterior_decomposes_and_normalizes() -> None:
    """联合后验 = p(family)×p(expert|family), 且两级各自归一。"""
    estimates = {
        "single": _est("single", 0.4, family="single"),
        "composite": _est("composite", 0.5, family="composite"),
        "composite_attach": _est("composite_attach", 0.3, family="composite"),
    }
    out = GenericStructureGate().decide_hierarchical(estimates)
    assert abs(sum(out.posterior.values()) - 1.0) < 1e-6
    for name in estimates:
        fam = estimates[name].geometry_family
        assert fam is not None
        expected = out.family_posterior[fam] * out.family_conditional[fam][name]
        assert abs(out.posterior[name] - expected) < 1e-12
    for fam, cond in out.family_conditional.items():
        assert abs(sum(cond.values()) - 1.0) < 1e-6
    # 胜者仍取 score 最小者 (composite_attach residual 0.3)
    assert out.estimate.structure_id == "composite_attach"


def test_hierarchical_degenerates_to_flat_for_singletons() -> None:
    """所有专家各成一组时, 层级后验退化为平铺后验。"""
    estimates = {
        "a": _est("a", 0.2),
        "b": _est("b", 0.8),
        "c": _est("c", 1.5),
    }
    gate = GenericStructureGate()
    flat = gate.decide(estimates)
    hier = gate.decide_hierarchical(estimates)
    for name in estimates:
        assert abs(flat.posterior[name] - hier.posterior[name]) < 1e-12


def test_temperature_scale_sharpens_posterior() -> None:
    """temperature_scale<1 应锐化后验 (winner 置信更高)。"""
    estimates = {
        "a": _est("a", 0.3),
        "b": _est("b", 0.5),
    }
    sharp = GenericStructureGate(temperature_scale=0.5).decide_hierarchical(
        estimates
    )
    flat = GenericStructureGate(temperature_scale=1.0).decide_hierarchical(
        estimates
    )
    assert sharp.posterior["a"] > flat.posterior["a"]
    assert sharp.posterior["b"] < flat.posterior["b"]
