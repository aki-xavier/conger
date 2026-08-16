"""MotionLayersModel (运动分割↔光流) 的黑盒测试。"""

import numpy as np

from generic_em import EMLoop
from motion_layers import MotionLayersModel


def test_motion_layers_recovers_velocities() -> None:
    """EM 应从分段常数光流场恢复各层速度。"""
    rng = np.random.default_rng(0)
    model = MotionLayersModel(k=2, n=64, sigma=0.03, smooth=4)
    true = (0.2, 0.8)
    obs = model.sample(true, rng)

    loop = EMLoop(model, max_iters=30, tol=1e-10)
    result = loop.run(obs, (0.3, 0.7))  # 非对称初始化 (对称 0.5/0.5 是退化不动点)

    got = np.array(result.params)
    # 两速度应分别接近两个真值 (顺序可能交换)
    assert min(np.abs(got - np.array(true))) < 0.05
    assert max(np.abs(got - np.array(true))) < 0.05
