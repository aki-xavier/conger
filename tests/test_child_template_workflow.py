"""ChildTemplateWorkflow 集成测试: 渲染提案 → 子模板注册。"""

import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_template_proposer import CompositeTemplateProposer
from expert_registry import ExpertRegistry
from inverse_app import InverseApp
from layered_codebook import LayeredCodebook
from mixture_spn import MixtureSPN
from structure_birth import StructureBirthRequest, StructureCase
from structure_gate import StructureGate
from structured_hypothesis import StructuredHypothesis


class _ParentExpert:
    def lineage(self):
        return LayeredCodebook.TEMPLATE_LINEAGE


def _rendered_request() -> tuple[StructureBirthRequest, tuple[float, ...]]:
    """真实渲染一个 attach 样本, 并让提案器产生 parent/delta 证据。"""
    base = (0.0, 72.0, 90.0, 0.45, 3.2, 1.0, 0.0, 1.0)
    proposer = CompositeTemplateProposer(
        ratios=(0.45,),
        lateral_ratios=(0.0,),
        part_kinds=(1,),
        part_hues=(2,),
        max_proposals=2,
    )
    gt = proposer._attach(base, 1, 2, 0.45, 0.0)
    cb = CompositeCodebook(proposer.codebook.cfg)
    renderer, cam_l, cam_r = Codebook.make_renderer()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    case = StructureCase(
        fl=fl,
        fr=fr,
        residuals={"layered": 1000.0},
        posterior={"layered": 1.0},
        params=base,
        structure_id="layered",
    )
    return (
        StructureBirthRequest(
            cases=(case, case),
            residual_mean=1000.0,
            best_posterior_mean=1.0,
            reason="test",
            proposals=proposer.propose((case, case)),
        ),
        gt,
    )


def test_child_template_workflow_end_to_end(monkeypatch, tmp_path) -> None:
    """真实渲染提案应生成子模板, 并通过显式训练接口注册。"""
    request, _ = _rendered_request()
    registry = ExpertRegistry({"layered": _ParentExpert()})

    def run(app: InverseApp, artifacts=None) -> None:
        path = app.default_model_path(artifacts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setattr(InverseApp, "run", run)
    monkeypatch.setattr(
        MixtureSPN, "load", classmethod(lambda cls, path: object())
    )
    registry.enable_child_template_learning()
    pending = registry.observe_birth_request(request)
    assert len(pending) == 1
    reg = registry.confirm_child_template(
        pending[0].name, artifacts=tmp_path
    )
    assert reg.spec.parent_family == "layered"
    assert reg.spec.operation == "attach"
    assert reg.spec.evidence_count == 2
    assert reg.expert.lineage().parent_family == "layered"
    assert reg.spec.name in registry.lineages()


def test_dynamic_child_uses_composite_geometry_family(monkeypatch) -> None:
    """动态子模板门控时应继承 composite 几何证据, 而不是落空为 0。"""
    monkeypatch.setattr(
        "structure_gate.StructureGeometry.costs",
        lambda fl, fr: {"single": 0.0, "layered": 0.0, "composite": 0.25},
    )
    monkeypatch.setattr(StructureGate, "residual", lambda self, e, fl, fr: 1.0)
    estimate = StructuredHypothesis(
        structure_id="child",
        geometry_family="composite",
        params=(0.0,),
        residual=1.0,
    )
    out = StructureGate().decide(
        {"child": estimate}, mx.zeros((1, 1, 4)), mx.zeros((1, 1, 4))
    )
    assert out.estimate.geometry_cost == 0.25
