"""完整 cga.Scene 重建器测试。

拆分场景后验、残差反参数化、Scene 构造、渲染残差光照精炼和评估
契约; InverseApp 全流程由 slow 集成测试覆盖。
"""

import mlx.core as mx

from codebook import Codebook
from evaluator import Evaluator
from inverse_config import InverseConfig
from scene_reconstructor import SceneReconstructor


def test_scene_param_decoding() -> None:
    """模型输出 → 完整 Scene 参数: 离散头 MAP + s,z 物理反参数化。"""
    cat_p = mx.zeros((1, 15))
    cat_p = mx.concatenate(
        [
            mx.array([[0.0, 3.0, 1.0]]),  # kind 1
            mx.array([[1.0, 0.0, 4.0, 0.0, 0.0, 0.0]]),  # hue 2
            mx.array([[1.0, 5.0, 0.0]]),  # lcol 1
            mx.array([[0.0, 0.0, 2.0]]),  # ldir 2
        ],
        axis=1,
    )
    t = mx.array([[72.0, 70.0, 0.01, 0.02]])
    stats = mx.array([[3.25, 5.0, 1000.0]])
    prm = SceneReconstructor.params(t, cat_p, stats)[0]
    assert prm[0] == 1.0 and prm[5:] == (2.0, 1.0, 2.0)
    assert abs(prm[1] - 72.0) < 1e-6 and abs(prm[2] - 70.0) < 1e-6
    assert abs(prm[4] - 3.27) < 1e-6  # 0.02 + ẑ
    expected_s = 0.01 + SceneReconstructor.s_proxy(stats)[0]
    assert abs(prm[3] - float(expected_s)) < 1e-6


def test_scene_reconstruction_contains_light() -> None:
    """完整 Scene 参数 → cga.Scene: 光照类别进入场景对象。"""
    cb = Codebook(InverseConfig())
    scene = SceneReconstructor.scenes(
        ((1, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0),), cb
    )[0]
    kinds = {type(x).__name__ for x in scene.lights + scene.objects}
    assert "DirectionalLight" in kinds
    assert "AmbientLight" in kinds
    assert "Mesh" in kinds


def test_render_residual_recovers_appearance() -> None:
    """候选渲染残差: 固定正确几何/kind 时恢复 hue/lcol/ldir。"""
    cb = Codebook(InverseConfig())
    gt = (1.0, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0)
    wrong_appearance = gt[:5] + (0.0, 0.0, 0.0)
    renderer, cam_l, cam_r = SceneReconstructor.rig()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    pred, score = SceneReconstructor.refine_appearance(
        cb, wrong_appearance, fl, fr, renderer, cam_l, cam_r
    )
    assert pred == gt
    assert score < 1e-6


def test_evaluator_full_scene_contract() -> None:
    """Evaluator: 4 个离散场景因子与 4 个连续目标全部入指标。"""
    p_gt = mx.array(
        [
            [0, 10, 20, 0.4, 3.0, 1, 0, 2],
            [2, 30, 40, 0.5, 3.5, 5, 2, 1],
        ],
        dtype=mx.float32,
    )
    t_pred = p_gt[:, 1:5]
    scene_pred = tuple(tuple(map(float, r)) for r in p_gt.tolist())
    out = Evaluator.report("合成", p_gt, t_pred, scene_pred, p_gt)
    for k in ("kind", "hue", "lcol", "ldir"):
        assert out[k] == 1.0
    assert out["u_rmse"] == 0.0 and out["z_r2"] == 1.0
