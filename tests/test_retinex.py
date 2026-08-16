"""RetinexModel (反照率↔光照) 的黑盒测试。"""

import numpy as np

from generic_em import EMLoop
from retinex import RetinexModel


def test_retinex_decomposes_albedo_and_light() -> None:
    """EM 应把 I=A·L 分解回分段常数反照率 + 平滑光照。"""
    rng = np.random.default_rng(0)
    n = 96
    segments = (np.arange(n) > n // 2).astype(int)  # 两段
    model = RetinexModel(segments, sigma=0.02)
    true_log_a = (0.3, -0.4)  # 两段反照率 (log)
    true_l = (0.2, 0.5)  # 光照 l0 + l1·x
    obs = model.render(true_log_a, true_l, rng)

    loop = EMLoop(model, max_iters=20, tol=1e-10)
    result = loop.run(obs, (0.0, 0.0))  # 光照初始化为 0

    log_i = np.log(obs + 1e-12)
    log_a = result.responsibilities  # 中心化后的反照率
    l0, l1 = result.params
    recon = log_a + l0 + l1 * model.x
    assert float(np.sqrt(np.mean((recon - log_i) ** 2))) < 0.05

    # 反照率对比度 (段均值差) 应匹配真值
    contrast = float(log_a[segments == 1].mean() - log_a[segments == 0].mean())
    assert abs(contrast - (true_log_a[1] - true_log_a[0])) < 0.05
