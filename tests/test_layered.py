"""双层遮挡场景族测试: 采样、Scene 构造、目标/类目契约。"""

import math
from typing import cast

import mlx.core as mx

from data_builder import DataBuilder
from inverse_config import InverseConfig
from layered_codebook import LayeredCodebook
from layered_reconstructor import LayeredReconstructor
from structured_hypothesis import StructuredHypothesis


def test_layered_sampling_and_scene() -> None:
    """2916 组合全因子覆盖, 深度规范排序, 双 Mesh + 双灯。"""
    cb = LayeredCodebook(InverseConfig(n_objects=2))
    p = cb.sample(1, 123)
    assert p.shape == (LayeredCodebook.N_COMBO, 14)
    assert bool(mx.all(p[:, 4] > p[:, 10]))
    combos = p[:, [0, 6, 5, 11, 12, 13]].astype(mx.int32)
    got = {tuple(row) for row in cast(list[list[int]], combos.tolist())}
    assert len(got) == LayeredCodebook.N_COMBO
    scene = cb.to_scene(tuple(float(x) for x in cast(list[float], p[0].tolist())))
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
    sizes = LayeredReconstructor.CAT_SIZES
    cats = cast(list[int], c[0].tolist())
    for lo, (nc, val) in enumerate(zip(sizes, cats)):
        col = sum(LayeredReconstructor.CAT_SIZES[:lo]) + val
        cat_p[0, col] = 1.0
    prm = LayeredReconstructor.params(t[:1], cat_p)[0]
    got = mx.array(prm, dtype=mx.float32)
    assert bool(mx.allclose(got, p[0], atol=1e-5))
    est = StructuredHypothesis(
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


def test_layered_residual_roundtrip() -> None:
    """逐层几何锚点: 残差训练目标可无损加回物理目标。"""
    cb = LayeredCodebook(InverseConfig(n_objects=2))
    p = cb.sample(1, 11)[:4]
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
    stats = mx.array(stats, dtype=mx.float32)
    rt = LayeredReconstructor.residual_targets(t, c, stats)
    cat_p = mx.zeros((p.shape[0], 24))
    for i, row in enumerate(c.tolist()):
        off = 0
        for j, (nc, val) in enumerate(zip(LayeredReconstructor.CAT_SIZES, row)):
            cat_p[i, off + val] = 1.0
            off += nc
    got = LayeredReconstructor.params(rt, cat_p, stats)
    got_t = LayeredReconstructor.targets_from_params(got)
    assert bool(mx.allclose(got_t, t, atol=1e-5))
