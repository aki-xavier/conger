"""因果不变性探针 (路线 ①) 的黑盒测试: holdout 划分/边缘化/不变性打分。"""

import mlx.core as mx
import pytest

from causal_invariance import (
    InvarianceProbe,
    InvarianceReport,
    LightingHoldout,
    invariance_score,
)
from scene_reconstructor import SceneReconstructor


def test_lighting_holdout_partitions_grid() -> None:
    """holdout 应把 9 个光照组合分成 4 训练 / 5 池外。"""
    h = LightingHoldout.split(n_colors=3, n_dirs=3, holdout_color=2, holdout_dir=2)
    n_in = sum(
        1
        for lc in range(3)
        for ld in range(3)
        if h.in_support(lc, ld)
    )
    n_out = sum(
        1
        for lc in range(3)
        for ld in range(3)
        if h.holdout(lc, ld)
    )
    assert n_in == 4
    assert n_out == 5
    # 两者互斥且覆盖全域
    assert n_in + n_out == 9


def test_marginal_appearance_sums_to_one() -> None:
    """边缘化应是合法分布: 各因子边缘和为 1。"""
    rng = mx.random.key(0)
    logp = mx.random.normal(shape=(6 * 3 * 3,), key=rng)
    posterior = mx.exp(logp - mx.logsumexp(logp))
    for factor in ("hue", "lcol", "ldir"):
        m = SceneReconstructor.marginal_appearance(posterior, factor)
        assert float(mx.sum(m)) == pytest.approx(1.0, abs=1e-5)


def test_marginal_appearance_recovers_dominant_hue() -> None:
    """把后验质量集中在一个 (hue,lcol,ldir) 上, 各因子边缘应指向该水平。"""
    posterior = mx.zeros((6 * 3 * 3,))
    posterior[1 * 9 + 2 * 3 + 1] = 1.0  # hue=1, lcol=2, ldir=1
    assert int(mx.argmax(SceneReconstructor.marginal_appearance(posterior, "hue"))) == 1
    assert int(mx.argmax(SceneReconstructor.marginal_appearance(posterior, "lcol"))) == 2
    assert int(mx.argmax(SceneReconstructor.marginal_appearance(posterior, "ldir"))) == 1


def test_marginal_appearance_rejects_unknown_factor() -> None:
    with pytest.raises(ValueError):
        SceneReconstructor.marginal_appearance(mx.zeros((54,)), "kind")


def test_marginal_joint_sums_to_one() -> None:
    """kind×hue×lcol×ldir 联合后验的各因子边缘和应为 1。"""
    rng = mx.random.key(1)
    logp = mx.random.normal(shape=(2 * 6 * 3 * 3,), key=rng)
    posterior = mx.exp(logp - mx.logsumexp(logp))
    for factor in ("kind", "hue", "lcol", "ldir"):
        m = SceneReconstructor.marginal_joint(posterior, factor, n_kind=2)
        assert float(mx.sum(m)) == pytest.approx(1.0, abs=1e-5)


def test_decoupled_map_matches_joint_on_sharp_posterior() -> None:
    """单峰尖锐后验 → 解耦 MAP 与联合 argmax 一致 (支持集内无回归)。"""
    posterior = mx.zeros((2 * 6 * 3 * 3,))
    posterior[1 * 54 + 2 * 9 + 1 * 3 + 2] = 1.0  # kind=1,hue=2,lcol=1,ldir=2
    assert int(mx.argmax(posterior)) == 1 * 54 + 2 * 9 + 1 * 3 + 2
    assert SceneReconstructor.decoupled_map(posterior, n_kind=2) == (1, 2, 1, 2)


def test_decoupled_map_prefers_hue_consistent_across_lighting() -> None:
    """反照率×光照歧义: 单一 (hue0,lcol0,ldir0) 联合略胜, 但 hue1 与
    更多光照组合一致 → 边缘化后 hue1 胜出 (因果不变估计的鲁棒性)。"""
    posterior = mx.zeros((1, 6, 3, 3))
    posterior[0, 0, 0, 0] = 0.30  # 联合 argmax → hue 0
    posterior[0, 1, 0, 1] = 0.25
    posterior[0, 1, 1, 0] = 0.25
    posterior[0, 1, 1, 1] = 0.20
    flat = mx.reshape(posterior, (-1,))
    assert int(mx.argmax(flat)) == 0  # 联合 argmax 落在 hue0 组合
    ki, hi, ci, di = SceneReconstructor.decoupled_map(flat, n_kind=1)
    assert ki == 0
    assert hi == 1  # 边缘化后 hue1 总证据 0.70 > hue0 的 0.30


def test_invariance_score_is_worst_group() -> None:
    assert invariance_score([1.0, 1.0, 0.7]) == pytest.approx(0.7)
    assert invariance_score([]) == 0.0


def _report(groups: dict, holdout: LightingHoldout) -> InvarianceReport:
    return InvarianceProbe.summarize(groups, "hue", holdout)


def test_summarize_computes_gap_and_invariance() -> None:
    """池内全准 + 池外崩塌 → gap>0 且不变性=池外准确率。"""
    h = LightingHoldout.split(3, 3, 2, 2)
    # 池内光照 (0/1 color × 0/1 dir) 全对; 池外光照 hue 全错
    groups = {
        (0, 0): [(0, 0), (1, 1)],
        (1, 1): [(2, 2), (3, 3)],
        (2, 0): [(4, 0), (5, 1)],  # 池外光色 → 预测错
        (0, 2): [(0, 3), (1, 4)],  # 池外光向 → 预测错
    }
    rep = _report(groups, h)
    assert rep.in_support_accuracy == pytest.approx(1.0)
    assert rep.holdout_accuracy == pytest.approx(0.0)
    assert rep.gap == pytest.approx(1.0)
    assert rep.invariance_score == pytest.approx(0.0)
    assert rep.n_groups == 4


def test_summarize_fully_invariant() -> None:
    """所有光照分组全对 → 不变性 1, gap 0。"""
    h = LightingHoldout.split(3, 3, 2, 2)
    groups = {(0, 0): [(0, 0)], (1, 1): [(1, 1)], (2, 2): [(2, 2)]}
    rep = _report(groups, h)
    assert rep.invariance_score == pytest.approx(1.0)
    assert rep.gap == pytest.approx(0.0)
