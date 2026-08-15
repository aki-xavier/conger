"""CompositeCodebook 测试: 显式附着组合模板与全局统计契约。"""

import mlx.core as mx
import pytest

from codebook import Codebook
from composite_codebook import CompositeCodebook
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
    assert len({tuple(row) for row in combos.tolist()}) == CompositeCodebook.N_COMBO
    scene = CompositeCodebook(InverseConfig()).to_scene(
        tuple(float(x) for x in p[0].tolist())
    )
    assert len(scene.objects) == 2
    assert len(scene.lights) == 2

    pe = CompositeCodebook.sample(1, 124, extrap=True)
    assert pe.shape == (CompositeCodebook.N_COMBO, 14)
    assert bool(mx.all(pe[:, 2] > pe[:, 8]))


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


def test_composite_app_contract_and_frame_features(
    composite_block: mx.array,
) -> None:
    """组合族走全局 [ẑ,area] 统计, 不复用遮挡逐层统计。"""
    cfg = InverseConfig(scene_family="composite")
    app = InverseApp(cfg)
    assert cfg.n_objects == 2
    assert isinstance(app.codebook, CompositeCodebook)
    assert app.default_model_path().name.startswith("spn_composite_")
    assert "cp1" in app.data.cache_tag()

    renderer, cam_l, cam_r = Codebook.make_renderer()
    prm = tuple(float(x) for x in composite_block[0].tolist())
    scene = app.codebook.to_scene(prm)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    vec, stats, _ = SceneReconstructor.frame_features(app, fl, fr)
    assert vec.shape == (1, cfg.n_feat)
    assert stats.shape == (1, 3)
