"""外观机制代理 (路线 ②) 的黑盒测试: 乘法分解/反事实/不变性校验。"""

import mlx.core as mx
import pytest

from scm_proxy import AppearanceMechanism


def _synthetic(n_hue=6, n_lcol=3, n_ldir=3, noise=0.0, seed=0):
    """a_true[h]⊙g_true[l,d] 的干预数据 (无交互项 = 完美模块机制)。"""
    rng = mx.random.key(seed)
    a = mx.random.uniform(shape=(n_hue, 3), key=rng) + 0.2  # 正反照率
    k1, k2 = mx.random.split(rng)
    g = mx.random.uniform(shape=(n_lcol, n_ldir, 3), key=k1) + 0.2
    rgb = a[:, None, None, :] * g[None, :, :, :]
    if noise > 0.0:
        rgb = rgb + mx.random.normal(shape=rgb.shape, key=k2) * noise
    return rgb, a, g


def test_fit_recovers_albedo_up_to_per_channel_scale() -> None:
    rgb, a_true, _ = _synthetic()
    m = AppearanceMechanism().fit(rgb)
    # a / a_true 应只随通道变, 不随 hue 变 (每通道常数比)
    ratio = m.albedo / a_true  # (n_hue, 3)
    spread = float(mx.max(mx.max(ratio, axis=0) - mx.min(ratio, axis=0)))
    assert spread < 1e-4


def test_perfect_modularity_has_invariance_near_one() -> None:
    rgb, _, _ = _synthetic()
    m = AppearanceMechanism().fit(rgb)
    assert m.albedo_invariance(rgb) == pytest.approx(1.0, abs=1e-4)


def test_noise_reduces_invariance_but_not_catastrophically() -> None:
    rgb, _, _ = _synthetic(noise=0.02)
    m = AppearanceMechanism().fit(rgb)
    score = m.albedo_invariance(rgb)
    assert 0.9 < score < 1.0


def test_do_lighting_counterfactual() -> None:
    rgb, _, _ = _synthetic()
    m = AppearanceMechanism().fit(rgb)
    got = m.do_lighting(2, 0, 0, 1, 2)
    expect = m.predict(2, 1, 2)
    assert float(mx.max(mx.abs(got - expect))) < 1e-5
    # 反事实 = 同反照率在新光照下的颜色 = 真实干预色 rgb[2,1,2]
    # (不随原光照 (0,0) 变; 归一化把尺度归入 albedo, 故用干预色校验)
    assert float(mx.max(mx.abs(got - rgb[2, 1, 2]))) < 1e-5


def test_unfitted_query_raises() -> None:
    with pytest.raises(RuntimeError):
        AppearanceMechanism().predict(0, 0, 0)


def test_foreground_mean_rgb() -> None:
    frame = mx.array([[[10.0, 20.0, 30.0], [30.0, 60.0, 90.0]]])
    weights = mx.array([[1.0, 3.0]])
    mean = AppearanceMechanism.foreground_mean_rgb(frame, weights)
    # (10*1+30*3)/4=25, (20*1+60*3)/4=50, (30*1+90*3)/4=75
    assert float(mean[0]) == pytest.approx(25.0)
    assert float(mean[1]) == pytest.approx(50.0)
    assert float(mean[2]) == pytest.approx(75.0)
