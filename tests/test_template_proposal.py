"""模板提案测试: 残差驱动组合候选与出生请求集成。"""

import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_template_proposer import CompositeTemplateProposer
from generic_structure_gate import GenericStructureDecision
from structure_birth import StructureBirthController, StructureCase
from structured_hypothesis import StructuredHypothesis
from template_proposal import TemplateProposal


def test_composite_template_proposer_recovers_attached_part() -> None:
    """正确组合候选应通过左右图重渲染残差排到 top1。"""
    base = (1.0, 72.0, 88.0, 0.45, 3.2, 2.0, 1.0, 2.0)
    proposer = CompositeTemplateProposer(
        ratios=(0.45,),
        lateral_ratios=(0.0,),
        part_kinds=(2,),
        part_hues=(4,),
        max_proposals=3,
    )
    gt = proposer._attach(base, part_kind=2, part_hue=4, ratio=0.45, lateral_ratio=0.0)
    cb = CompositeCodebook(proposer.codebook.cfg)
    renderer, cam_l, cam_r = Codebook.make_renderer()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    case = StructureCase(
        fl=fl,
        fr=fr,
        residuals={"single": 1000.0},
        posterior={"single": 1.0},
        params=base,
    )
    proposals = proposer.propose((case,))
    assert proposals
    best = proposals[0]
    assert best.family == "composite"
    assert best.operation == "attach"
    assert best.parent_family == "layered"
    assert best.delta["relation"] == "attach"
    assert best.metadata["part_kind"] == 2
    assert best.metadata["part_hue"] == 4
    assert best.residual < 1e-6
    assert best.metadata["residual_gain"] > 999.0


class _StaticProposer:
    def propose(self, cases: tuple[StructureCase, ...]) -> tuple[TemplateProposal, ...]:
        return (
            TemplateProposal(
                family="composite",
                operation="attach",
                params=tuple(float(x) for x in range(14)),
                residual=1.0,
                complexity=1.5,
                score=2.5,
                parent_family="layered",
                delta={"relation": "attach"},
                metadata={"n_cases": len(cases)},
            ),
        )


def test_birth_request_carries_template_proposals() -> None:
    """结构出生请求应携带提案, 但训练/注册仍由调用方显式决定。"""
    estimate = StructuredHypothesis(
        structure_id="single", params=(0.0,), residual=10.0
    )
    decision = GenericStructureDecision(
        estimate=estimate,
        posterior={"single": 0.4},
        residuals={"single": 10.0},
        scores={"single": 11.0},
        needs_new_structure=True,
    )
    controller = StructureBirthController(min_cases=2, proposer=_StaticProposer())
    assert controller.observe(decision, mx.zeros(2), mx.zeros(2)) is None
    request = controller.observe(decision, mx.zeros(2), mx.zeros(2))
    assert request is not None
    assert len(request.proposals) == 1
    assert request.cases[0].structure_id == "single"
    assert request.proposals[0].parent_family == "layered"
    assert request.proposals[0].delta["relation"] == "attach"
    assert request.proposals[0].metadata["n_cases"] == 2
    assert "1 个模板提案" in request.reason
