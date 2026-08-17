"""CausalEdge: 结构级因果发现 —— 把模板 delta 边升级为候选因果边 (路线 ③)。

`TemplateDeltaLearner` 只把提案聚合为参数约束范围 (条件密度); 这里问的是
更强的因果问题: 「操作 → 参数 delta」这条边是不是**稳定机制**, 还是只在
当前样本内成立的相关?

不变性因果发现 (invariant prediction / ICP 的精神): 因果机制在环境间不变,
伪相关随环境漂移。`CausalDeltaLearner` 把提案按环境 (seed / 父几何配置)
分组, 每组估 delta 位置 (中位/中点), 跨环境漂移越少 → 边越像因果。配合
路线 ① 的不变性判据, 这是项目里唯一「从数据发现因果结构」的位置。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

from template_proposal import TemplateProposal

# TemplateDeltaLearner 产出的标量 delta 目标名 (与约束名对齐)
_TARGET_KEYS = (
    "scale_ratio",
    "lateral_ratio",
    "period_ratio",
    "depth_gap",
    "depth_jitter",
)


@dataclass(frozen=True)
class CausalEdge:
    """一条候选因果边: (parent, operation) → target, 附跨环境一致度证据。"""

    parent_family: str
    operation: str
    target: str
    env_midpoints: tuple[float, ...]
    env_ranges: tuple[tuple[float, float], ...]
    pooled_range: tuple[float, float]
    agreement: float
    n_envs: int

    @property
    def is_causal(self) -> bool:
        """因果需要 ≥2 个环境的跨环境一致证据 + 低漂移。"""
        return self.n_envs >= 2 and self.agreement >= 0.5


class CausalDeltaLearner:
    """提案 → 候选因果边 (按环境分组的不变性验证)。"""

    def __init__(self, agreement_threshold: float = 0.5):
        self.agreement_threshold = agreement_threshold

    @staticmethod
    def _targets(p: TemplateProposal) -> dict[str, float]:
        """提案 → 标量目标 (对齐 TemplateDeltaLearner 的约束命名)。

        优先 metadata["observed"] 的实测 delta (来自观测帧几何证据, 是
        因果边验证的观测), 缺失时回退网格 delta (搜索点)。
        """
        meta = p.metadata or {}
        observed = meta.get("observed", {}) or {}
        out: dict[str, float] = {}
        if "scale_ratio" in observed:
            out["scale_ratio"] = float(observed["scale_ratio"])
        elif "ratio" in p.delta:
            out["scale_ratio"] = float(p.delta["ratio"])
        if "period_ratio" in observed:
            out["period_ratio"] = float(observed["period_ratio"])
        elif "lateral_ratio" in observed:
            key = (
                "period_ratio"
                if p.operation in {"mirror", "repeat"}
                else "lateral_ratio"
            )
            out[key] = float(observed["lateral_ratio"])
        elif "lateral_ratio" in p.delta:
            key = (
                "period_ratio"
                if p.operation in {"mirror", "repeat"}
                else "lateral_ratio"
            )
            out[key] = float(p.delta["lateral_ratio"])
        if "depth_gap" in observed:
            out["depth_gap"] = float(observed["depth_gap"])
        elif "depth_gap" in p.delta:
            out["depth_gap"] = float(p.delta["depth_gap"])
        if "depth_jitter" in p.delta:
            lo, hi = (float(x) for x in p.delta["depth_jitter"])
            out["depth_jitter"] = 0.5 * (lo + hi)
        return out

    @staticmethod
    def default_env_key(p: TemplateProposal) -> Hashable:
        """默认环境键: metadata 的 env → seed → case_index → 0。

        真实出生提案没有显式 env/seed, 用 case_index (同一 propose() 调用
        内的样本序号) 区分不同数据生成条件。
        """
        meta = p.metadata or {}
        return meta.get("env", meta.get("seed", meta.get("case_index", 0)))

    def _agreement(self, mids: list[float], ranges: list[tuple[float, float]]) -> float:
        """跨环境一致度 = 1 − 中点漂移 / 总展宽 (∈[0,1])。

        中点漂移 = 各环境中点跨环境的极差; 总展宽 = 池化范围的宽度。
        漂移 0 → 1 (稳定机制); 漂移 ≈ 展宽 → 0 (伪相关随环境漂移)。
        环境内采样噪声只进展宽, 不伤一致度 (位置稳定才算因果)。
        """
        if len(mids) <= 1:
            return 1.0
        width = max(r[1] for r in ranges) - min(r[0] for r in ranges)
        if width < 1e-12:
            return 1.0
        drift = max(mids) - min(mids)
        return max(0.0, min(1.0, 1.0 - drift / width))

    def learn(
        self,
        proposals: Iterable[TemplateProposal],
        env_key: Callable[[TemplateProposal], Hashable] | None = None,
    ) -> tuple[CausalEdge, ...]:
        """按 (parent, operation, target) × 环境分组, 输出候选因果边。"""
        key_fn = env_key or self.default_env_key
        groups: dict[tuple[str, str, str], dict[Hashable, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for p in proposals:
            if p.parent_family is None:
                continue
            env = key_fn(p)
            for target, val in self._targets(p).items():
                groups[(p.parent_family, p.operation, target)][env].append(val)

        edges: list[CausalEdge] = []
        for (parent, op, target), env_vals in groups.items():
            ranges = [(min(v), max(v)) for v in env_vals.values()]
            mids = [0.5 * (lo + hi) for lo, hi in ranges]
            pooled = (min(r[0] for r in ranges), max(r[1] for r in ranges))
            agreement = self._agreement(mids, ranges)
            edges.append(
                CausalEdge(
                    parent_family=parent,
                    operation=op,
                    target=target,
                    env_midpoints=tuple(mids),
                    env_ranges=tuple(ranges),
                    pooled_range=pooled,
                    agreement=agreement,
                    n_envs=len(mids),
                )
            )
        edges.sort(key=lambda e: (-e.agreement, e.operation, e.target))
        return tuple(edges)
