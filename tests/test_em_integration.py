"""几何↔光照 ECM 接入主链路的契约测试 (默认关闭)。"""

from inverse_config import InverseConfig
from structured_hypothesis import StructuredHypothesis


def test_em_refine_defaults_off() -> None:
    """ECM 默认关闭, 且开关/参数都在 InverseConfig。"""
    cfg = InverseConfig()
    assert cfg.em_refine is False
    assert cfg.em_max_iters == 2
    assert cfg.em_appearance_topk == 3

    cfg_on = InverseConfig(em_refine=True, em_max_iters=4, em_appearance_topk=2)
    assert cfg_on.em_refine is True
    assert cfg_on.em_max_iters == 4
    assert cfg_on.em_appearance_topk == 2


def test_hypothesis_carries_em_trajectory() -> None:
    """StructuredHypothesis 应记录 ECM 每轮 log-likelihood 轨迹。"""
    h = StructuredHypothesis(structure_id="single", em_trajectory=(1.0, 2.0))
    assert h.em_trajectory == (1.0, 2.0)
    assert StructuredHypothesis(structure_id="single").em_trajectory is None
