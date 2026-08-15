"""ExpertRegistry 黑盒测试: 专家加载失败语义与门控调用。"""

import mlx.core as mx
import pytest

from codebook import Codebook
from expert_registry import ExpertRegistry, SceneExpert
from inverse_config import InverseConfig
from scene_reconstructor import SceneReconstructor
from structured_hypothesis import StructuredHypothesis


class _DummyExpert:
    def __init__(self, estimate: StructuredHypothesis):
        self.estimate = estimate
        self.calls = 0

    def reconstruct(self, fl: mx.array, fr: mx.array) -> StructuredHypothesis:
        self.calls += 1
        return self.estimate


def _estimate(params: tuple[float, ...], cb: Codebook) -> StructuredHypothesis:
    return StructuredHypothesis(cb.to_scene(params), params, mx.zeros(15))


def test_registry_calls_all_experts_and_gates() -> None:
    """注册表应把同一帧对交给全部专家, 并返回门控决策。"""
    cb = Codebook(InverseConfig())
    good = (0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 1.0)
    bad = (2.0, 20.0, 30.0, 0.55, 2.5, 5.0, 2.0, 0.0)
    renderer, cam_l, cam_r = SceneReconstructor.rig()
    scene = cb.to_scene(good)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    experts = {"good": _DummyExpert(_estimate(good, cb)),
               "bad": _DummyExpert(_estimate(bad, cb))}
    out = ExpertRegistry(experts).decide(fl, fr)
    assert out.estimate.structure_id == "good"
    assert out.posterior["good"] > 0.99
    assert experts["good"].calls == experts["bad"].calls == 1


def test_missing_expert_model_fails_closed(tmp_path) -> None:
    """缺模型时默认 fail closed; missing_ok 时才跳过。"""
    cfg = InverseConfig(n_objects=1, model_path=tmp_path / "none.safetensors")
    with pytest.raises(FileNotFoundError):
        SceneExpert.from_config("missing", cfg, tmp_path)


def test_train_and_register_workflow(monkeypatch, tmp_path) -> None:
    """train_and_register: 显式候选训练 → 加载 → 加入注册表。"""
    from inverse_app import InverseApp

    cb = Codebook(InverseConfig())
    prm = (0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 1.0)
    registry = ExpertRegistry({"old": _DummyExpert(_estimate(prm, cb))})
    calls = []
    monkeypatch.setattr(
        InverseApp, "run", lambda app: calls.append(app.cfg.n_objects)
    )
    monkeypatch.setattr(
        SceneExpert,
        "from_config",
        classmethod(
            lambda cls, name, cfg, artifacts=None: _DummyExpert(
                _estimate(prm, cb)
            )
        ),
    )
    cfg = InverseConfig(n_objects=1, model_path=tmp_path / "new.safetensors")
    registry.train_and_register("new", cfg, tmp_path)
    assert calls == [1]
    assert "new" in registry.experts


def test_birth_queue_and_dynamic_registration() -> None:
    """未知结构证据达到阈值 → 出生请求; 注册后可加入后续门控。"""
    from structure_birth import StructureBirthController

    cb = Codebook(InverseConfig())
    gt = (0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 1.0)
    bad_a = (1.0, 30.0, 40.0, 0.55, 3.8, 4.0, 1.0, 2.0)
    bad_b = (2.0, 110.0, 100.0, 0.35, 2.8, 5.0, 2.0, 0.0)
    renderer, cam_l, cam_r = SceneReconstructor.rig()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    registry = ExpertRegistry(
        {
            "bad_a": _DummyExpert(_estimate(bad_a, cb)),
            "bad_b": _DummyExpert(_estimate(bad_b, cb)),
        },
        birth_controller=StructureBirthController(min_cases=2),
    )
    registry.decide(fl, fr)
    assert registry.last_birth_request is None
    registry.decide(fl, fr)
    req = registry.last_birth_request
    assert req is not None and len(req.cases) == 2
    assert registry.birth_controller is not None
    assert registry.birth_controller.cases == []

    good_expert = _DummyExpert(_estimate(gt, cb))
    registry.register("born", expert=good_expert)
    out = registry.decide(fl, fr)
    assert "born" in out.posterior
