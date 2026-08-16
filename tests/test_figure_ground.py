"""FigureGroundModel (分割↔位姿) 的黑盒测试。"""

import numpy as np

from figure_ground import FigureGroundModel
from generic_em import EMLoop


def test_figure_ground_recovers_pose_and_intensities() -> None:
    """EM 应同时恢复前景位姿 (c,r) 与前景/背景强度 (f,b)。"""
    rng = np.random.default_rng(0)
    model = FigureGroundModel(n=100, sigma=0.03)
    true = (0.5, 0.2, 2.0, 0.5)  # (c, r, f, b)
    obs = model.sample(true, rng)

    init = (0.35, 0.3, 1.0, 1.0)  # 扰动初始化
    loop = EMLoop(model, max_iters=30, tol=1e-10)
    result = loop.run(obs, init)

    got = np.array(result.params)
    assert np.allclose(got, np.array(true), atol=0.05)
