"""逆渲染管线集成自检 (slow: 渲染 648 帧, 缓存热时秒级)。

端到端走 InverseApp: 渲染 → 特征 → 实例级组装 → 推理 → 评估 →
阈值断言 (阈值依据见 InverseApp.self_check 注释)。默认跳过,
`pytest -m slow` 运行; 等价于 `cd src && python inverse.py --quick`。
"""

import pytest

from inverse_app import InverseApp
from inverse_config import InverseConfig


@pytest.mark.slow
def test_quick_pipeline() -> None:
    InverseApp(InverseConfig(quick=True)).run()
