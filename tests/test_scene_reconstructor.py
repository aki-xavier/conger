"""完整 cga.Scene 重建器测试。

拆分场景后验、残差反参数化、Scene 构造、渲染残差光照精炼和评估
契约; InverseApp 全流程由 slow 集成测试覆盖。
"""

import math

import mlx.core as mx

from codebook import Codebook
from evaluator import Evaluator
from inverse_config import InverseConfig
from scene_estimate import SceneEstimate
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth


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
    expected_s = 0.01 + SceneReconstructor.s_proxy(1, stats)[0]
    assert abs(prm[3] - float(expected_s)) < 1e-6


def test_kind_conditioned_size_proxy() -> None:
    """面积→尺寸代理按结构因子变化: box 正面与圆盘几何不同。"""
    stats = mx.array([[3.0, 6.0, 1600.0]])
    round_s = SceneReconstructor.s_proxy(0, stats)[0]
    box_s = SceneReconstructor.s_proxy(2, stats)[0]
    q = (40.0 * (Codebook.CAM_Z - 3.0) / Codebook.FX)
    assert abs(float(round_s) - q / math.sqrt(math.pi)) < 1e-6
    assert abs(float(box_s) - q * 0.5) < 1e-6


def test_scene_reconstruction_contains_light() -> None:
    """完整 Scene 参数 → cga.Scene: 光照类别进入场景对象。"""
    cb = Codebook(InverseConfig())
    scene = SceneReconstructor.scenes(((1, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0),), cb)[
        0
    ]
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
    pred, score, scores = SceneReconstructor.refine_appearance(
        cb, wrong_appearance, fl, fr, renderer, cam_l, cam_r
    )
    assert pred == gt
    assert score < 1e-6
    assert scores.shape == (54,)


def test_topk_structure_refinement_and_marginals() -> None:
    """top-k kind 结构候选: SPN 错选 kind 时仍可由渲染残差纠正。"""
    cb = Codebook(InverseConfig())
    gt = (2.0, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0)
    renderer, cam_l, cam_r = SceneReconstructor.rig()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    z_hat, d, area = StereoDepth().estimate(fl, fr)
    stats = mx.array([[z_hat, d, area]])
    s_resid = gt[3] - float(SceneReconstructor.s_proxy(2, stats)[0])
    wrong_s = float(SceneReconstructor.s_proxy(0, stats)[0]) + s_resid
    wrong = (0.0, gt[1], gt[2], wrong_s, gt[4], 0.0, 0.0, 0.0)
    kind_p = mx.array([0.4, 0.1, 0.5])
    pred, candidates, scores, posterior, temperature = SceneReconstructor.refine_scene(
        cb,
        wrong,
        kind_p,
        stats,
        fl,
        fr,
        kind_topk=2,
        renderer=renderer,
        cam_l=cam_l,
        cam_r=cam_r,
    )
    assert pred[:3] == gt[:3]
    assert abs(pred[3] - gt[3]) < 1e-6
    assert pred[4:] == gt[4:]
    assert len(candidates) == 108
    assert scores.shape == posterior.shape == (108,)
    assert temperature > 0.0
    assert abs(float(mx.sum(posterior)) - 1.0) < 1e-5
    estimate = SceneEstimate(
        scene=cb.to_scene(pred),
        params=pred,
        spn_posterior=mx.zeros(15),
        candidate_params=candidates,
        candidate_scores=scores,
        candidate_posterior=posterior,
        candidate_temperature=temperature,
    )
    kind_m, hue_m, lcol_m, ldir_m = estimate.factor_marginals()
    assert float(kind_m[2]) > 0.9
    assert float(hue_m[2]) > 0.9
    assert float(lcol_m[1]) > 0.9
    assert float(ldir_m[2]) > 0.9


def test_novelty_metrics_contract() -> None:
    """新颖性证据: δ责任度+δ后验低, 均匀后验熵高, 残差入综合分。"""
    cat = mx.zeros(15)
    cat[0] = cat[3] = cat[9] = cat[12] = 1.0
    r0 = mx.array([[1.0, 0.0, 0.0]])
    rn0, ent0, nov0 = SceneReconstructor.novelty_metrics(
        cat, r0, SceneReconstructor.CAT_SIZES, None
    )
    assert rn0 < 1e-6 and ent0 < 1e-6 and nov0 < 1e-6
    r1 = mx.full((1, 4), 0.25)
    _, ent1, nov1 = SceneReconstructor.novelty_metrics(
        cat, r1, SceneReconstructor.CAT_SIZES, 9.0
    )
    assert ent1 == 0.0  # 类目仍确定, 分量责任度才模糊
    assert nov1 > nov0


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
