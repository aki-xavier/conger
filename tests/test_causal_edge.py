"""结构级因果发现 (路线 ③) 的黑盒测试: 跨环境一致度区分因果边/伪相关。"""

from causal_edge import CausalDeltaLearner
from template_proposal import TemplateProposal


def _proposal(env: int, ratio: float, lateral: float) -> TemplateProposal:
    return TemplateProposal(
        family="composite_attach_xyz",
        operation="attach",
        params=(0.0, 70.0, 70.0, 0.4, 3.0, 1.0, 0.0, 1.0),
        residual=100.0,
        complexity=1.5,
        score=100.0,
        parent_family="composite",
        delta={"ratio": ratio, "lateral_ratio": lateral},
        metadata={"env": env},
    )


def _learn(proposals):
    return {
        e.target: e
        for e in CausalDeltaLearner().learn(
            proposals, env_key=lambda p: p.metadata["env"]
        )
    }


def test_stable_delta_is_causal_edge() -> None:
    """scale_ratio 跨 3 环境稳定 → 一致度高, 判因果。"""
    proposals = [
        _proposal(0, 0.44, 0.0),
        _proposal(0, 0.46, 0.0),
        _proposal(1, 0.45, 0.0),
        _proposal(2, 0.45, 0.0),
    ]
    edges = _learn(proposals)
    scale = edges["scale_ratio"]
    assert scale.agreement == 1.0  # 各环境中点 0.45 一致, 漂移 0
    assert scale.is_causal


def test_drifting_delta_is_not_causal_edge() -> None:
    """lateral_ratio 跨环境漂移 (0→0.5→1) → 一致度低, 判伪相关。"""
    proposals = [
        _proposal(0, 0.45, 0.0),
        _proposal(1, 0.45, 0.5),
        _proposal(2, 0.45, 1.0),
    ]
    edges = _learn(proposals)
    lateral = edges["lateral_ratio"]
    assert lateral.agreement == 0.0  # 中点极差 1 = 池化展宽 1
    assert not lateral.is_causal


def test_single_env_is_not_causal_despite_trivial_agreement() -> None:
    """单环境无跨环境证据, 一致度虽为 1 但不可判因果。"""
    proposals = [_proposal(0, 0.45, 0.0), _proposal(0, 0.46, 0.0)]
    edges = _learn(proposals)
    scale = edges["scale_ratio"]
    assert scale.agreement == 1.0
    assert not scale.is_causal  # n_envs == 1


def test_mirror_maps_lateral_to_period_ratio() -> None:
    """mirror/repeat 的 lateral_ratio 映射为 period_ratio 目标。"""
    p = _proposal(0, 0.45, 0.6)
    mirror = TemplateProposal(
        family="composite_mirror_x",
        operation="mirror",
        params=p.params,
        residual=p.residual,
        complexity=p.complexity,
        score=p.score,
        parent_family="composite",
        delta={"ratio": 0.45, "lateral_ratio": 0.6},
        metadata={"env": 0},
    )
    edges = {
        e.target: e
        for e in CausalDeltaLearner().learn([mirror, p], env_key=lambda q: q.metadata["env"])
    }
    assert "period_ratio" in edges
    assert "lateral_ratio" in edges  # attach 提案的 lateral 仍独立成边
