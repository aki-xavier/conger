"""模型内存/遗忘机制 (model_memory) 的黑盒测试。"""

from typing import cast

import mlx.core as mx
import pytest

from mixture_spn import MixtureSPN
from model_memory import (
    _coreset,
    assemble,
    forget_components,
    load_components,
    model_size_mb,
    split_save,
    truncate_basis,
)


def _tiny_model(n: int = 60, v: int = 16, seed: int = 0) -> MixtureSPN:
    rng = mx.random.key(seed)
    f = mx.random.normal(shape=(n, v), key=rng)
    t = mx.random.normal(shape=(n, 3), key=rng)
    stratum = mx.random.randint(0, 3, shape=(n,), key=rng)
    return MixtureSPN.fit(
        f, t, stratum, rel_floor=1e-2,
        scene_classes=stratum[:, None].astype(mx.int32), cat_sizes=(3,),
    )


def test_split_assemble_roundtrip(tmp_path) -> None:
    m = _tiny_model()
    split_save(m, tmp_path / "m")
    m2 = assemble(tmp_path / "m")
    assert m2.basis is not None and m.basis is not None
    assert m2.f_mu.shape == m.f_mu.shape
    assert float(mx.max(mx.abs(m2.f_mu - m.f_mu))) < 1e-6
    assert float(mx.max(mx.abs(m2.basis - m.basis))) < 1e-6
    assert m2.cat_sizes_tuple == m.cat_sizes_tuple
    assert m2.rel_floor == m.rel_floor


def test_load_components_excludes_basis(tmp_path) -> None:
    m = _tiny_model()
    split_save(m, tmp_path / "m")
    comp, meta = load_components(tmp_path / "m")
    assert "f_mu" in comp and "basis" not in comp
    assert meta["cat_sizes"] == (3,)
    assert meta["rel_floor"] == 0.01


def test_truncate_basis_keeps_high_variance_columns() -> None:
    m = _tiny_model()
    m2 = truncate_basis(m, 8)
    assert m2.basis is not None and m.basis is not None
    assert m2.basis.shape[1] == 8
    assert m2.f_mu.shape[1] == 8
    # 保留尾部 (最高方差) 列
    assert float(mx.max(mx.abs(m2.basis - m.basis[:, -8:]))) < 1e-6
    assert model_size_mb(m2) < model_size_mb(m)


def test_truncate_basis_noop_when_d_max_too_large() -> None:
    m = _tiny_model()
    assert truncate_basis(m, 999) is m


def test_forget_components_bounds_k_and_uniform_weights() -> None:
    m = _tiny_model()
    m2 = forget_components(m, 30, "coreset")
    assert m2.f_mu.shape[0] == 30
    assert float(mx.sum(mx.exp(m2.log_w))) == pytest.approx(1.0)
    # 三个 stratum 都保留
    strat = mx.argmax(m2.cat_logp[:, :3], axis=1)
    assert len({int(x) for x in cast(list, strat.tolist())}) == 3


def test_forget_components_noop_when_k_max_large() -> None:
    m = _tiny_model()
    assert forget_components(m, 999) is m


def test_coreset_returns_distinct_indices() -> None:
    Z = mx.random.normal(shape=(100, 5), key=mx.random.key(0))
    idx = _coreset(Z, 10, mx.random.key(1))
    vals = [int(i) for i in cast(list, idx.tolist())]
    assert len(set(vals)) == 10
    assert all(0 <= i < 100 for i in vals)
