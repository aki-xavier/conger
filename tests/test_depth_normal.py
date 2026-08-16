"""DepthNormalModel (深度↔法向) 的黑盒测试。"""

import numpy as np

from depth_normal import DepthNormalModel
from generic_em import EMLoop


def test_depth_normal_denoises_depth() -> None:
    """深度↔法向交替应把噪声深度拉向真值 (精确法向给高频结构)。"""
    rng = np.random.default_rng(0)
    n = 64
    x = np.linspace(0.0, 1.0, n)
    z_true = np.sin(3.0 * x) + 0.5 * x  # 光滑真值
    s_true = np.diff(z_true) * (n - 1)  # 真值斜率 (按单位区间重标)
    z_obs = z_true + rng.normal(0.0, 0.3, n)  # 噪声深度
    s_obs = s_true + rng.normal(0.0, 0.02, n - 1)  # 较准法向

    model = DepthNormalModel(z_obs, s_obs, lam=0.5)
    loop = EMLoop(model, max_iters=30, tol=1e-8)
    result = loop.run(None, tuple(z_obs))  # 初始化 = 噪声深度

    got = np.array(result.params)
    assert float(np.sqrt(np.mean((got - z_true) ** 2))) < float(
        np.sqrt(np.mean((z_obs - z_true) ** 2))
    )
