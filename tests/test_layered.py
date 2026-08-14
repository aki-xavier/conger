"""双层遮挡场景族测试: 采样、Scene 构造、目标/类目契约。"""

import mlx.core as mx

from data_builder import DataBuilder
from inverse_config import InverseConfig
from layered_codebook import LayeredCodebook
from layered_reconstructor import LayeredReconstructor
from scene_estimate import SceneEstimate


def test_layered_sampling_and_scene() -> None:
    """2916 组合全因子覆盖, 深度规范排序, 双 Mesh + 双灯。"""
    cb = LayeredCodebook(InverseConfig(n_objects=2))
    p = cb.sample(1, 123)
    assert p.shape == (LayeredCodebook.N_COMBO, 14)
    assert bool(mx.all(p[:, 4] < p[:, 10]))
    combos = p[:, [0, 6, 5, 11, 12, 13]].astype(mx.int32)
    got = {tuple(row) for row in combos.tolist()}
    assert len(got) == LayeredCodebook.N_COMBO
    scene = cb.to_scene(tuple(float(x) for x in p[0].tolist()))
    assert len(scene.objects) == 2
    assert len(scene.lights) == 2


def test_layered_targets_and_decoding() -> None:
    """双层参数 → 8 连续目标 + 6 离散因子 → 完整参数。"""
    cb = LayeredCodebook(InverseConfig(n_objects=2))
    p = cb.sample(1, 7)
    t = DataBuilder.targets(p)
    c = DataBuilder.scene_classes(p)
    assert t.shape == (p.shape[0], 8)
    assert c.shape == (p.shape[0], 6)
    cat_p = mx.zeros((1, 24))
    for lo, (nc, val) in enumerate(zip(LayeredReconstructor.CAT_SIZES, c[0].tolist())):
        col = sum(LayeredReconstructor.CAT_SIZES[:lo]) + val
        cat_p[0, col] = 1.0
    prm = LayeredReconstructor.params(t[:1], cat_p)[0]
    got = mx.array(prm, dtype=mx.float32)
    assert bool(mx.allclose(got, p[0], atol=1e-5))
    est = SceneEstimate(
        scene=cb.to_scene(prm),
        params=prm,
        spn_posterior=cat_p[0],
        candidate_params=(prm,),
        factor_sizes=LayeredCodebook.CAT_SIZES,
        factor_indices=LayeredCodebook.CLASS_IDX,
    )
    marginals = est.factor_marginals()
    assert len(marginals) == 6
    assert all(abs(float(mx.sum(m)) - 1.0) < 1e-6 for m in marginals)
