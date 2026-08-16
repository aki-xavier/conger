"""SoftICPModel (EM-ICP) 的黑盒测试。"""

import numpy as np

from generic_em import EMLoop
from soft_icp import SoftICPModel


def test_soft_icp_recovers_rigid_transform() -> None:
    """EM-ICP 应从身份初始化恢复旋转 + 平移。"""
    rng = np.random.default_rng(0)
    # 非对称点云 (避免旋转歧义)
    source = rng.normal(0.0, 0.6, (20, 2))
    source[0] = [0.0, 0.0]  # 一个显著锚点破坏对称
    true = (0.4, 0.5, -0.3)  # (θ, tx, ty)
    model = SoftICPModel(source, sigma=0.03)
    obs = model.sample(true, rng)

    loop = EMLoop(model, max_iters=40, tol=1e-10)
    result = loop.run(obs, (0.0, 0.0, 0.0))  # 身份初始化

    got = np.array(result.params)
    assert np.allclose(got, np.array(true), atol=0.05)
    # 对数似然单调不减
    for a, b in zip(result.trajectory, result.trajectory[1:], strict=False):
        assert b >= a - 1e-6


def test_soft_icp_improves_from_perturbed_init() -> None:
    """从扰动变换出发, EM 应拉近真值。"""
    rng = np.random.default_rng(1)
    source = rng.normal(0.0, 0.5, (16, 2))
    true = (-0.25, -0.4, 0.6)
    model = SoftICPModel(source, sigma=0.02)
    obs = model.sample(true, rng)

    init = (0.1, 0.1, -0.1)
    result = EMLoop(model, max_iters=30, tol=1e-10).run(obs, init)
    got = np.array(result.params)
    assert np.linalg.norm(got - np.array(true)) < np.linalg.norm(
        np.array(init) - np.array(true)
    )
