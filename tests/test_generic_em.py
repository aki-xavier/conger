"""GenericEM 与透明层叠加 EM 探针的黑盒测试。"""

import numpy as np

from generic_em import EMLoop
from transparent_layers import TransparentLayerModel


def test_transparent_layer_em_recovers_layers() -> None:
    """EM 应从软归属混合中恢复两层强度。"""
    n = 256
    rng = np.random.default_rng(0)
    alpha = np.linspace(0.08, 0.92, n)  # 空间变化的混合权重
    true = (0.4, -0.6)
    model = TransparentLayerModel(alpha, sigma=0.05)
    obs = model.sample(true, rng)

    loop = EMLoop(model, max_iters=200, tol=1e-10)
    result = loop.run(obs, (0.0, 0.0))  # 朴素初始化

    got = np.array(result.params)
    assert np.allclose(got, np.array(true), atol=0.03)
    # 对数似然应单调不减
    for a, b in zip(result.trajectory, result.trajectory[1:], strict=False):
        assert b >= a - 1e-6


def test_em_loop_temperature_damping_and_convergence() -> None:
    """EMLoop 收敛; temperature 只锐化、damping 只稳定, 均不改接口。"""
    n = 128
    rng = np.random.default_rng(1)
    alpha = np.linspace(0.1, 0.9, n)
    true = (0.7, -0.3)
    model = TransparentLayerModel(alpha, sigma=0.05)
    obs = model.sample(true, rng)

    result = EMLoop(model, max_iters=200, tol=1e-10).run(obs, (0.0, 0.0))
    assert result.iterations < 200
    assert np.isfinite(result.log_likelihood)

    sharp = EMLoop(model, max_iters=200, tol=1e-10, temperature=0.7).run(obs, (0.0, 0.0))
    assert np.allclose(np.array(sharp.params), np.array(true), atol=0.05)

    damped = EMLoop(model, max_iters=200, tol=1e-10, damping=0.3).run(obs, (0.0, 0.0))
    assert np.allclose(np.array(damped.params), np.array(true), atol=0.05)
