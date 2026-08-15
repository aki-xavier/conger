"""MixtureSPN 黑盒测试 (公理性质/实例回归/白化相关病理/序列化)。

契约全部来自第一性原理 (归一化/δ证据/Product 结构/白化等价类),
阈值依据见各测试注释。从 src/mixture_spn.py 内嵌自检迁移。
"""

import math

import mlx.core as mx
import pytest

from mixture_spn import MixtureSPN


def _manual_model() -> tuple[MixtureSPN, mx.array]:
    """手工模型 (D=4, T=2, K=3; 恒等白化基)。

    分量 0,1 同属 kind0 —— 检验类目跨分量聚合。"""
    f_mu = mx.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]]
    )
    f_var = mx.full((3, 4), 0.01)  # σ=0.1, 分量间距 10 → 近似可分
    t_mu = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cat_logp = mx.log(
        mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    m = MixtureSPN(
        mx.full((3,), -math.log(3.0)),
        f_mu,
        f_var,
        t_mu,
        cat_logp,
        0.1,
        mx.zeros(4),
        mx.eye(4),
    )
    return m, t_mu


def test_axioms() -> None:
    """公理性质: 归一化 / δ选择 / 类目聚合 / 单分量证据无关。"""
    m, t_mu = _manual_model()
    cat_logp = m.cat_logp

    # 归一化公理: 责任度与 kind 概率行和为 1
    # (容差 1e-4: float32 exp/log 复合往返误差 ~数 ulp @ O(1) 值, 实测 3e-5)
    xs = mx.array([[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    tm, kp, r = m.predict(xs)
    assert mx.allclose(mx.sum(r, axis=1), mx.ones(2), atol=1e-4)
    assert mx.allclose(mx.sum(kp, axis=1), mx.ones(2), atol=1e-4)

    # δ 证据选择性: x 精确落在分量 0 质心 → E[t] = t_mu_0 (分量被唯一选中)
    assert mx.allclose(tm[0], t_mu[0], atol=1e-3), f"δ证据: {tm[0]}"

    # 类目聚合: x 在分量 0/1 中点 (分量归属模糊) 但两者同 kind0
    # → P(kind0) ≈ 1: 类目后验对分量置换对称, 不受分量模糊影响
    assert float(kp[1, 0]) > 0.999, f"类目聚合: {kp[1]}"

    # Product 结构性质: 单分量模型对任意证据 E[t|x] = t_mu
    # (块内特征⊥目标 → 证据不影响目标期望; 插值全靠多分量混合)
    m1 = MixtureSPN(
        mx.zeros(1),
        m.f_mu[:1],
        m.f_var[:1],
        t_mu[:1],
        cat_logp[:1],
        0.1,
        mx.zeros(4),
        mx.eye(4),
    )
    tm1, _, _ = m1.predict(mx.array([[99.0, 99.0, 99.0, 99.0]]))
    assert mx.allclose(tm1[0], t_mu[0]), f"单分量证据无关性: {tm1[0]}"


@pytest.fixture(scope="module")
def separable() -> tuple[mx.array, mx.array, mx.array]:
    """可分离合成混合 (3 簇 × 200 样本, 簇间距 8.0)。"""
    keys = mx.random.split(mx.random.key(0), 4)
    n_per, d_f, d_t = 200, 6, 2
    true_fmu = mx.random.normal(shape=(3, d_f), key=keys[0]) * 3.0  # 分量可分
    true_tmu = mx.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    fs, ts, ks = [], [], []
    for c in range(3):
        fs.append(
            true_fmu[c] + 0.3 * mx.random.normal(shape=(n_per, d_f), key=keys[1])
        )
        ts.append(
            true_tmu[c] + 0.1 * mx.random.normal(shape=(n_per, d_t), key=keys[2])
        )
        ks.append(mx.full((n_per,), c % 3))
    return mx.concatenate(fs), mx.concatenate(ts), mx.concatenate(ks)


@pytest.fixture(scope="module")
def fitted(
    separable: tuple[mx.array, mx.array, mx.array],
) -> tuple[MixtureSPN, mx.array, mx.array, mx.array]:
    f_all, t_all, k_all = separable
    return MixtureSPN.fit(f_all, t_all, k_all), f_all, t_all, k_all


def test_instance_regression(
    fitted: tuple[MixtureSPN, mx.array, mx.array, mx.array],
) -> None:
    """实例级回归: 可分数据上 ≈ 精确插值, RMSE 远小于簇间距。"""
    model, f_all, t_all, k_all = fitted
    # 断 0.5 = 簇间距 (8.0) 的 1/16, 目标噪声 σ=0.1 的 5 倍
    tm, kp, _ = model.predict(f_all)
    rmse = float(mx.sqrt(mx.mean((tm - t_all) ** 2)))
    assert rmse < 0.5, f"实例级回归 RMSE {rmse}"
    # kind: 簇间可分 → kind 后验应近完美
    acc = float(mx.mean((mx.argmax(kp, axis=1) == k_all).astype(mx.float32)))
    assert acc > 0.99, f"可分混合 kind {acc:.3f}"


def test_full_scene_heads() -> None:
    """完整 Scene 离散头: kind/hue/lcol/ldir 各自归一且逐头命中。"""
    f = mx.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )
    t = mx.zeros((3, 1))
    scene = mx.array(
        [
            [0, 1, 0, 2],
            [1, 5, 1, 0],
            [2, 3, 2, 1],
        ],
        dtype=mx.int32,
    )
    m = MixtureSPN.fit(
        f,
        t,
        scene[:, 0],
        rel_floor=1e-5,
        scene_classes=scene,
        cat_sizes=(3, 6, 3, 3),
    )
    _, cp, _ = m.predict(f)
    lo = 0
    for nc in (3, 6, 3, 3):
        p = cp[:, lo : lo + nc]
        # one-hot 经 log/exp 往返是 float32 近似 (δ头允许 ~1e-3 残差)
        assert mx.allclose(mx.sum(p, axis=1), mx.ones(3), atol=1e-3)
        lo += nc
    got = mx.concatenate(
        [
            mx.argmax(cp[:, 0:3], axis=1)[:, None],
            mx.argmax(cp[:, 3:9], axis=1)[:, None],
            mx.argmax(cp[:, 9:12], axis=1)[:, None],
            mx.argmax(cp[:, 12:15], axis=1)[:, None],
        ],
        axis=1,
    )
    assert mx.all(got == scene)


def test_incremental_add(
    separable: tuple[mx.array, mx.array, mx.array],
) -> None:
    """增量训练: fit 半量 + add 半量 → 与全量 fit 同契约。

    交错对半分 (两半均含全部 kind); 白化基冻结是唯一与全量 fit 的
    差异 (注释见 MixtureSPN.add), 可分数据上应达到同一标准。"""
    f_all, t_all, k_all = separable
    even, odd = mx.arange(0, 600, 2), mx.arange(1, 600, 2)
    m = MixtureSPN.fit(f_all[even], t_all[even], k_all[even])
    m.add(f_all[odd], t_all[odd], k_all[odd], k_all[odd][:, None])
    assert m.f_mu.shape[0] == 600, "增量后 K 应翻倍"
    tm, kp, _ = m.predict(f_all)
    rmse = float(mx.sqrt(mx.mean((tm - t_all) ** 2)))
    assert rmse < 0.5, f"增量回归 RMSE {rmse}"
    acc = float(mx.mean((mx.argmax(kp, axis=1) == k_all).astype(mx.float32)))
    assert acc > 0.99, f"增量 kind {acc:.3f}"


def test_correlation_pathology() -> None:
    """相关性病理 (白化的存在理由): 类分离沿正交低方差方向。

    两类样本沿对角线拉长 (强相关), 原空间对角高斯的最坏情形
    (逐维方差被拉长方向污染, 类间逐维重叠); 模型契约: 白化后应
    正确分离 (白化对角 ≡ 原空间全协方差)。"""
    n = 120
    k4a, k4b, _ = mx.random.split(mx.random.key(5), 3)
    direction = mx.array([1.0, 1.0, 1.0, 1.0]) / 2.0  # 相关方向 (单位向量)
    perp = mx.array([1.0, -1.0, 1.0, -1.0]) / 2.0  # 正交方向
    lo = mx.random.normal(shape=(n, 1), key=k4a) * 3.0  # 沿相关方向大散
    pe = mx.random.normal(shape=(n, 1), key=k4b) * 0.1  # 正交小噪声
    # 类均值沿【正交】(低方差) 方向差 0.6 (正交 σ=0.1 → 全协方差 d'=6,
    # 完全可分); 逐维看: 每轴方差被相关方向污染 σ²≈4.5, 均值差仅 0.3
    # → 原空间对角高斯 d'²≈0.08 不可分。模型契约: 白化后应正确分离。
    off = mx.where(mx.arange(n) < n // 2, 0.0, 0.6)[:, None]
    f = lo * direction[None, :] + pe * perp[None, :] + off * perp[None, :]
    k = (mx.arange(n) >= n // 2).astype(mx.int32)
    t = mx.zeros((n, 1))  # 目标不参与本组断言
    m = MixtureSPN.fit(f, t, k)
    _, kp, _ = m.predict(f)
    acc = float(mx.mean((mx.argmax(kp, axis=1) == k).astype(mx.float32)))
    # 白化后两类在正交方向上 d'=6 应完全可分; 断 0.95
    assert acc > 0.95, f"相关特征类分离失败 {acc:.3f}"


def test_serialization_roundtrip(
    fitted: tuple[MixtureSPN, mx.array, mx.array, mx.array], tmp_path
) -> None:
    """序列化 roundtrip: 存盘后预测逐位一致。"""
    model, f_all, _, _ = fitted
    tm, kp, _ = model.predict(f_all)
    p = tmp_path / "m.safetensors"
    model.save(p)
    tm2, kp2, _ = MixtureSPN.load(p).predict(f_all)
    assert bool(mx.all(mx.equal(tm, tm2))) and bool(mx.all(mx.equal(kp, kp2)))


def test_category_contract_expansion(tmp_path) -> None:
    """类别动态扩展: 契约序列化, 旧分量 padding, 新类别可增量学习。"""
    f = mx.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    t = mx.zeros((3, 1))
    cls = mx.array([[0, 0], [1, 0], [2, 1]], dtype=mx.int32)
    m = MixtureSPN.fit(
        f, t, cls[:, 0], scene_classes=cls, cat_sizes=(3, 2)
    )
    path = tmp_path / "m.safetensors"
    m.save(path)
    loaded = MixtureSPN.load(path)
    assert loaded.cat_sizes_tuple == (3, 2)

    loaded.expand_categories((4, 3))
    assert loaded.cat_sizes_tuple == (4, 3)
    assert loaded.cat_logp.shape == (3, 7)
    _, cp, _ = loaded.predict(f)
    # 旧样本对新类别概率为 0; 各因子边缘仍归一
    assert float(cp[0, 3]) == 0.0
    assert mx.allclose(mx.sum(cp[:, 4:], axis=1), mx.ones(3), atol=1e-4)

    f_new = mx.array([[30.0, 30.0]])
    c_new = mx.array([[3, 2]], dtype=mx.int32)
    loaded.add(f_new, mx.zeros((1, 1)), c_new[:, 0], c_new)
    _, cp_new, _ = loaded.predict(f_new)
    assert int(mx.argmax(cp_new[0, :4])) == 3
    assert int(mx.argmax(cp_new[0, 4:])) == 2
