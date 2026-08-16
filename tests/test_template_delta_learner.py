"""TemplateDeltaLearner 与动态子 Codebook 工厂测试。"""

from child_codebook_factory import ChildCodebookFactory
from codebook import Codebook
from composite_codebook import CompositeCodebook
from expert_registry import ExpertRegistry
from inverse_app import InverseApp
from inverse_config import InverseConfig
from layered_codebook import LayeredCodebook
from mixture_spn import MixtureSPN
from structure_birth import StructureBirthRequest
from template_delta_learner import TemplateDeltaLearner
from template_lineage import ChildTemplateSpec
from template_proposal import TemplateProposal


class _OldExpert:
    def lineage(self):
        return LayeredCodebook.TEMPLATE_LINEAGE


def _proposal(ratio: float, lateral: float) -> TemplateProposal:
    return TemplateProposal(
        family="composite",
        operation="attach",
        params=tuple(float(x) for x in range(14)),
        residual=10.0 + ratio,
        complexity=1.5,
        score=11.5 + ratio,
        parent_family="layered",
        delta={
            "relation": "attach",
            "ratio": ratio,
            "lateral_ratio": lateral,
            "part_kind": 1,
            "part_hue": 2,
        },
    )


def _request() -> StructureBirthRequest:
    return StructureBirthRequest(
        cases=(),
        residual_mean=10.0,
        best_posterior_mean=0.4,
        reason="test",
        proposals=(_proposal(0.4, -0.1), _proposal(0.6, 0.1)),
    )


def test_template_delta_learning_groups_and_ranges() -> None:
    """相似 attach 提案应聚合成一个带边距的 ChildTemplateSpec。"""
    specs = TemplateDeltaLearner(min_evidence=2).learn(
        (_request(),),
        lineages={"layered": LayeredCodebook.TEMPLATE_LINEAGE},
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.parent_family == "layered"
    assert spec.operation == "attach"
    assert spec.generation == 2
    assert spec.evidence_count == 2
    assert spec.constraints["scale_ratio"] == (0.38, 0.62)
    lateral = spec.constraints["lateral_ratio"]
    assert abs(lateral[0] + 0.12) < 1e-12
    assert abs(lateral[1] - 0.12) < 1e-12
    assert spec.constraints["part_kinds"] == (1,)


def test_child_codebook_factory_and_cache_variant() -> None:
    """attach spec 应物化为受限 CompositeCodebook 并进入缓存指纹。"""
    spec = TemplateDeltaLearner(min_evidence=2).learn((_request(),))[0]
    child_cls = ChildCodebookFactory.build(spec)
    assert issubclass(child_cls, CompositeCodebook)
    assert child_cls.SCALE_RATIO == (0.38, 0.62)
    assert abs(child_cls.LATERAL_RANGE[0] + 0.12) < 1e-12
    assert abs(child_cls.LATERAL_RANGE[1] - 0.12) < 1e-12
    assert child_cls.PART_KINDS == (1,)
    assert child_cls.N_COMBO == 3 * 1 * 6 * 1 * 3 * 3
    assert child_cls.TEMPLATE_LINEAGE.parent_family == "layered"

    cfg = InverseConfig(scene_family="composite")
    app = InverseApp(cfg, codebook=child_cls(cfg))
    assert isinstance(app.codebook, child_cls)
    assert spec.name in app.data.cache_tag()


def test_child_template_train_and_register(monkeypatch, tmp_path) -> None:
    """动态子 Codebook 应能通过显式 train_and_register 成为专家。"""
    spec = TemplateDeltaLearner(min_evidence=2).learn((_request(),))[0]
    child_cls = ChildCodebookFactory.build(spec)
    registry = ExpertRegistry({"old": _OldExpert()})

    def run(app: InverseApp, artifacts=None) -> None:
        path = app.default_model_path(artifacts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setattr(InverseApp, "run", run)
    monkeypatch.setattr(
        MixtureSPN, "load", classmethod(lambda cls, path: object())
    )
    cfg = InverseConfig(scene_family="composite")
    expert = registry.train_and_register(
        spec.name, cfg, artifacts=tmp_path, codebook_cls=child_cls
    )
    assert expert.lineage().family == spec.name
    assert registry.children_of("layered") == (spec.name,)


def test_layer_child_lateral_guarantees_back_visibility() -> None:
    """layer 子模板近对齐会被强制加宽, 保证后层投影越出前层。"""
    spec = ChildTemplateSpec(
        name="layered_layer_test",
        family="layered",
        parent_family="layered",
        operation="layer",
        constraints={
            "relation": "layer",
            "scale_ratio": (0.43, 0.62),
            "lateral_ratio": (-0.02, 0.02),  # 近对齐 → 后层被完全遮挡
            "depth_gap": (0.78, 0.82),
            "part_kinds": (1,),
            "part_hues": (2,),
        },
        complexity=2.0,
        generation=2,
        evidence_count=2,
        residual_mean=0.0,
        score_mean=0.0,
    )
    child_cls = ChildCodebookFactory.build(spec)
    assert child_cls.PART_LATERAL_RANGE == (0.35, 0.7)
    cb = child_cls(InverseConfig(scene_family="layered"))
    for seed in (1, 2, 3, 4, 5):
        row = tuple(float(x) for x in cb.sample(1, seed)[0].tolist())
        u0, s0, z0 = row[1], row[3], row[4]
        u1, s1, z1 = row[7], row[9], row[10]
        r0 = s0 * Codebook.FX / (Codebook.CAM_Z - z0)
        r1 = s1 * Codebook.FX / (Codebook.CAM_Z - z1)
        # 后层投影至少越出前层 0.5px (可见, 不退化成单物体)
        assert abs(u1 - u0) + r1 > r0 + 0.5
