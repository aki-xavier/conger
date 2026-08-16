"""OcclusionLayerModel (遮挡 + 深度序) 的黑盒测试。"""

import numpy as np

from generic_em import EMLoop
from occlusion_layers import OcclusionLayerModel


def test_occlusion_em_recovers_layers_and_depth_order() -> None:
    """EM 应从遮挡观测中恢复两层线性系数 (深度序由重叠区决定)。"""
    rng = np.random.default_rng(0)
    model = OcclusionLayerModel(n=96, a=0.4, b=0.6, sigma=0.02)
    true = (0.5, 1.0, -0.3, -1.2)  # L1=a0+a1·x, L2=b0+b1·x
    obs = model.render(true, front=0, rng=rng)  # L1 在前

    loop = EMLoop(model, max_iters=40, tol=1e-10)
    result = loop.run(obs, (0.0, 0.0, 0.0, 0.0))

    got = np.array(result.params)
    assert np.allclose(got, np.array(true), atol=0.15)
    # 深度序应判 L1 在前 (P(D=0) > 0.5)
    assert float(result.responsibilities[0]) > 0.5


def test_occlusion_em_reverses_depth_order() -> None:
    """L2 在前时, 深度序后验应翻转到 P(D=1) 占优。"""
    rng = np.random.default_rng(1)
    model = OcclusionLayerModel(n=96, a=0.4, b=0.6, sigma=0.02)
    true = (0.4, 0.8, -0.6, 1.0)
    obs = model.render(true, front=1, rng=rng)  # L2 在前

    result = EMLoop(model, max_iters=40, tol=1e-10).run(obs, (0.0, 0.0, 0.0, 0.0))
    got = np.array(result.params)
    assert np.allclose(got, np.array(true), atol=0.15)
    assert float(result.responsibilities[0]) < 0.5
