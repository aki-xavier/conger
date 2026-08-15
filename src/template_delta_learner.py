"""TemplateDeltaLearner: 从出生提案分布估计子模板约束。

第一阶段只做可审计的约束学习: 按 parent_family+operation 聚合提案,
对 ratio/lateral/depth 等数值 delta 取范围并留边距, 生成 ChildTemplateSpec。
是否真正训练和注册仍由调用方显式决定。
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from structure_birth import StructureBirthRequest
from template_lineage import ChildTemplateSpec, TemplateLineage
from template_proposal import TemplateProposal


class TemplateDeltaLearner:
    """StructureBirthRequest.proposals → ChildTemplateSpec 列表。"""

    def __init__(self, min_evidence: int = 2, range_margin: float = 0.10):
        if min_evidence < 1:
            raise ValueError("min_evidence 必须 >=1")
        if range_margin < 0:
            raise ValueError("range_margin 必须 >=0")
        self.min_evidence = min_evidence
        self.range_margin = range_margin

    @staticmethod
    def _group_key(p: TemplateProposal) -> tuple[str, str] | None:
        if p.parent_family is None:
            return None
        return p.parent_family, p.operation

    def _range(self, values: list[float]) -> tuple[float, float]:
        """数值证据 → 带相对/绝对边距的范围。"""
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * self.range_margin, 0.02)
        return lo - pad, hi + pad

    @staticmethod
    def _hash(constraints: Mapping[str, object]) -> str:
        text = repr(sorted(constraints.items()))
        return hashlib.sha1(text.encode("utf8")).hexdigest()[:8]

    def _spec(
        self,
        parent: str,
        operation: str,
        proposals: list[TemplateProposal],
        lineages: Mapping[str, TemplateLineage],
    ) -> ChildTemplateSpec:
        ratios = [float(p.delta["ratio"]) for p in proposals if "ratio" in p.delta]
        laterals = [
            float(p.delta["lateral_ratio"])
            for p in proposals
            if "lateral_ratio" in p.delta
        ]
        constraints: dict[str, object] = {"relation": operation}
        if ratios:
            constraints["scale_ratio"] = self._range(ratios)
        if laterals:
            if operation in {"mirror", "repeat"}:
                constraints["period_ratio"] = self._range(
                    [abs(x) for x in laterals]
                )
            else:
                constraints["lateral_ratio"] = self._range(laterals)
        depth_gaps = [
            float(p.delta["depth_gap"])
            for p in proposals
            if "depth_gap" in p.delta
        ]
        if depth_gaps:
            constraints["depth_gap"] = self._range(depth_gaps)
        depth = [
            tuple(p.delta["depth_jitter"])
            for p in proposals
            if "depth_jitter" in p.delta
        ]
        if depth:
            constraints["depth_jitter"] = (
                min(d[0] for d in depth), max(d[1] for d in depth)
            )
        constraints["part_kinds"] = tuple(
            sorted(
                {
                    int(p.delta["part_kind"])
                    for p in proposals
                    if "part_kind" in p.delta
                }
            )
        )
        constraints["part_hues"] = tuple(
            sorted(
                {
                    int(p.delta["part_hue"])
                    for p in proposals
                    if "part_hue" in p.delta
                }
            )
        )
        digest = self._hash(constraints)
        parent_lineage = lineages.get(parent)
        generation = 1 if parent_lineage is None else parent_lineage.generation + 1
        complexity = sum(p.complexity for p in proposals) / len(proposals)
        residual = sum(p.residual for p in proposals) / len(proposals)
        score = sum(p.score for p in proposals) / len(proposals)
        family = proposals[0].family
        return ChildTemplateSpec(
            name=f"{parent}_{operation}_{digest}",
            family=family,
            parent_family=parent,
            operation=operation,
            constraints=constraints,
            complexity=complexity,
            generation=generation,
            evidence_count=len(proposals),
            residual_mean=residual,
            score_mean=score,
        )

    def learn(
        self,
        requests: Iterable[StructureBirthRequest],
        lineages: Mapping[str, TemplateLineage] | None = None,
    ) -> tuple[ChildTemplateSpec, ...]:
        """多个出生请求 → 达到证据阈值的子模板规格。"""
        groups: dict[tuple[str, str], list[TemplateProposal]] = defaultdict(list)
        for request in requests:
            for proposal in request.proposals:
                key = self._group_key(proposal)
                if key is not None and all(
                    math.isfinite(v) for v in (proposal.residual, proposal.score)
                ):
                    groups[key].append(proposal)
        lineage_map = lineages or {}
        specs = [
            self._spec(parent, op, props, lineage_map)
            for (parent, op), props in groups.items()
            if len(props) >= self.min_evidence
        ]
        specs.sort(key=lambda s: (s.score_mean, -s.evidence_count, s.name))
        return tuple(specs)
