"""ExpertRegistry 黑盒测试: 专家加载失败语义与门控调用。"""

import pytest
import mlx.core as mx

from codebook import Codebook
from expert_registry import ExpertRegistry, SceneExpert
from inverse_config import InverseConfig
from scene_estimate import SceneEstimate
from scene_reconstructor import SceneReconstructor


class _DummyExpert:
    def __init__(self, estimate: SceneEstimate):
        self.estimate = estimate
        self.calls = 0

    def reconstruct(self, fl: mx.array, fr: mx.array) -> SceneEstimate:
        self.calls += 1
        return self.estimate


def _estimate(params: tuple[float, ...], cb: Codebook) -> SceneEstimate:
    return SceneEstimate(cb.to_scene(params), params, mx.zeros(15))


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
