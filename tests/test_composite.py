"""CompositeCodebook 测试: 显式附着组合模板与部分感知几何契约。"""

import math
from typing import cast

import mlx.core as mx
import pytest

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_geometry import CompositeGeometry
from composite_reconstructor import CompositeReconstructor
from data_builder import DataBuilder
from inverse_app import InverseApp
from inverse_config import InverseConfig
from scene_reconstructor import SceneReconstructor
from structured_hypothesis import StructuredHypothesis


@pytest.fixture(scope="module")
def composite_block() -> mx.array:
    return CompositeCodebook.sample(1, 123)


def test_composite_sampling_and_scene(composite_block: mx.array) -> None:
    """组合件由底座导出: 顶部附着、尺度相关、深度接近且取景合法。"""
    p = composite_block
    assert p.shape == (CompositeCodebook.N_COMBO, 14)
    assert bool(mx.all(p[:, 2] > p[:, 8]))  # 图像 v 向下, 附着件在上方
    ratio = p[:, 9] / p[:, 3]
    assert float(mx.min(ratio)) >= CompositeCodebook.SCALE_RATIO[0] - 1e-6
    assert float(mx.max(ratio)) <= CompositeCodebook.SCALE_RATIO[1] + 1e-6
    assert float(mx.max(mx.abs(p[:, 10] - p[:, 4]))) <= 0.060001

    combos = p[:, list(CompositeCodebook.CLASS_IDX)].astype(mx.int32)
    assert len({tuple(row) for row in cast(list, combos.tolist())}) == (
        CompositeCodebook.N_COMBO
    )
    scene = CompositeCodebook(InverseConfig()).to_scene(
        tuple(float(x) for x in cast(list, p[0].tolist()))
    )
    assert len(scene.objects) == 2
    assert len(scene.lights) == 2

    pe = CompositeCodebook.sample(1, 124, extrap=True)
    assert pe.shape == (CompositeCodebook.N_COMBO, 14)
    assert bool(mx.all(pe[:, 2] > pe[:, 8]))


def test_composite_geometry_recovers_parts(composite_block: mx.array) -> None:
    """部分模板锚点应恢复 base/part 中心顺序和物理深度范围。"""
    cfg = InverseConfig(scene_family="composite")
    cb = CompositeCodebook(cfg)
    prm = tuple(float(x) for x in cast(list, composite_block[0].tolist()))
    renderer, cam_l, cam_r = Codebook.make_renderer()
    scene = cb.to_scene(prm)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    st = CompositeGeometry.estimate(fl, fr)
    assert abs(st[0] - prm[1]) < 8.0
    assert abs(st[4] - prm[7]) < 8.0
    assert st[1] > st[5]
    assert st[3] > st[7]
    assert 2.0 < st[2] < 4.5
    assert 2.0 < st[6] < 4.5


def test_composite_decoding_roundtrip(composite_block: mx.array) -> None:
    """8 连续目标 + 6 离散头 → 14 维组合场景参数。"""
    p = composite_block[:1]
    t = DataBuilder.targets(p)
    c = DataBuilder.scene_classes(p)
    cat_p = mx.zeros((1, sum(CompositeReconstructor.CAT_SIZES)))
    off = 0
    for nc, val in zip(CompositeReconstructor.CAT_SIZES, c[0].tolist(), strict=True):
        cat_p[0, off + val] = 1.0
        off += nc
    prm = CompositeReconstructor.params(t, cat_p)[0]
    assert bool(mx.allclose(mx.array(prm), p[0], atol=1e-5))

    est = StructuredHypothesis(
        scene=CompositeCodebook(InverseConfig()).to_scene(prm),
        params=prm,
        spn_posterior=cat_p[0],
        candidate_params=(prm,),
        factor_sizes=CompositeCodebook.CAT_SIZES,
        factor_indices=CompositeCodebook.CLASS_IDX,
    )
    assert len(est.factor_marginals()) == 6


def test_composite_residual_roundtrip(composite_block: mx.array) -> None:
    """base/part 锚点残差目标可加回物理目标。"""
    p = composite_block[:4]
    t = DataBuilder.targets(p)
    c = DataBuilder.scene_classes(p)
    stats = []
    for row, cls_row in zip(p.tolist(), c.tolist(), strict=True):
        layers = []
        for ti, ki in (((1, 2, 3, 4), 0), ((7, 8, 9, 10), 1)):
            u, v, s, z = (row[j] for j in ti)
            ratio = 0.5 if cls_row[ki] == 2 else 1.0 / math.sqrt(math.pi)
            area = (s * 90.0 / (ratio * (5.5 - z))) ** 2
            layers.extend([u, v, z, area])
        stats.append(layers)
    stats_mx = mx.array(stats, dtype=mx.float32)
    rt = CompositeReconstructor.residual_targets(t, c, stats_mx)
    cat_p = mx.zeros((p.shape[0], sum(CompositeReconstructor.CAT_SIZES)))
    for i, row in enumerate(c.tolist()):
        off = 0
        for nc, val in zip(CompositeReconstructor.CAT_SIZES, row, strict=True):
            cat_p[i, off + val] = 1.0
            off += nc
    got = CompositeReconstructor.params(rt, cat_p, stats_mx)
    got_t = CompositeReconstructor.targets_from_params(got)
    assert bool(mx.allclose(got_t, t, atol=1e-5))


def test_composite_render_refinement(composite_block: mx.array) -> None:
    """固定组合几何, top-k 结构候选可由左右图渲染残差纠正。"""
    cfg = InverseConfig(scene_family="composite", refine_composite=True)
    cb = CompositeCodebook(cfg)
    gt = tuple(float(x) for x in cast(list, composite_block[0].tolist()))
    renderer, cam_l, cam_r = Codebook.make_renderer()
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)

    heads = []
    classes = DataBuilder.scene_classes(composite_block[:1])[0].tolist()
    for nc, val in zip(
        CompositeReconstructor.CAT_SIZES, classes, strict=True
    ):
        p = mx.zeros(nc)
        p[val] = 0.7
        p[(val + 1) % nc] = 0.3
        heads.append(p)
    cat_p = mx.concatenate(heads)
    wrong = (
        float((int(gt[0]) + 1) % 3), *gt[1:6],
        float((int(gt[6]) + 1) % 3), *gt[7:14],
    )
    pred, candidates, scores, posterior, temperature = (
        CompositeReconstructor.refine_scene(
            cb, wrong, cat_p, fl, fr, kind_topk=2, hue_topk=1, light_topk=1
        )
    )
    assert pred == gt
    assert len(candidates) == 4
    assert scores.shape == posterior.shape == (4,)
    assert float(mx.min(scores)) < 1e-6
    assert temperature > 0.0


def test_composite_app_contract_and_frame_features(
    composite_block: mx.array,
) -> None:
    """组合族走 base/part 模板统计, 不复用遮挡逐层统计。"""
    cfg = InverseConfig(scene_family="composite")
    app = InverseApp(cfg)
    assert cfg.n_objects == 2
    assert isinstance(app.codebook, CompositeCodebook)
    assert app.default_model_path().name.startswith("spn_composite_")
    assert "cp2" in app.data.cache_tag()

    renderer, cam_l, cam_r = Codebook.make_renderer()
    prm = tuple(float(x) for x in cast(list, composite_block[0].tolist()))
    scene = app.codebook.to_scene(prm)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    vec, stats, _ = SceneReconstructor.frame_features(app, fl, fr)
    assert vec.shape == (1, cfg.n_feat)
    assert stats.shape == (1, 8)
    assert stats[0, 5] < stats[0, 1]  # part 在 base 上方
