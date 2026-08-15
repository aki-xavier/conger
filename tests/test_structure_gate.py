"""StructureGate 黑盒测试: 结构后验与未知结构出生检测。"""

import mlx.core as mx

from codebook import Codebook
from inverse_config import InverseConfig
from scene_reconstructor import SceneReconstructor
from structure_gate import StructureGate
from structured_hypothesis import StructuredHypothesis


def _estimate(params: tuple[float, ...], cb: Codebook) -> StructuredHypothesis:
    return StructuredHypothesis(
        scene=cb.to_scene(params),
        params=params,
        spn_posterior=mx.zeros(15),
    )


def test_structure_gate_posterior_and_birth() -> None:
    """正确结构应胜出; 多个都不兼容时触发新结构信号。"""
    cb = Codebook(InverseConfig())
    good_prm = (0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 1.0)
    bad_a = (1.0, 30.0, 40.0, 0.55, 3.8, 4.0, 1.0, 2.0)
    bad_b = (2.0, 110.0, 100.0, 0.35, 2.8, 5.0, 2.0, 0.0)
    renderer, cam_l, cam_r = SceneReconstructor.rig()
    gt = cb.to_scene(good_prm)
    fl = renderer.render(gt, cam_l)
    fr = renderer.render(gt, cam_r)
    gate = StructureGate()
    good = _estimate(good_prm, cb)
    bad = _estimate(bad_a, cb)
    out = gate.decide({"good": good, "bad": bad}, fl, fr)
    assert out.estimate.structure_id == "good"
    assert out.posterior["good"] > 0.99
    assert not out.needs_new_structure

    born = gate.decide(
        {"bad_a": _estimate(bad_a, cb), "bad_b": _estimate(bad_b, cb)}, fl, fr
    )
    assert born.needs_new_structure
    assert max(born.posterior.values()) < 0.8
