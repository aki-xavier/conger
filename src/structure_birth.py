"""StructureBirth: 未知结构样本队列、出生请求与候选训练注册。

安全边界: 本模块只聚合“所有现有结构专家都不兼容”的证据并生成
请求; 新结构必须由可渲染的 SceneFamily/Codebook 显式提供, 不自动
发明 renderer 不支持的几何模板。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from template_proposal import TemplateProposal, TemplateProposer


class GateDecision(Protocol):
    """StructureBirthController 所需的最小门控结果协议。"""

    needs_new_structure: bool
    residuals: Mapping[str, float]
    posterior: Mapping[str, float]

    @property
    def estimate(self): ...


@dataclass(frozen=True)
class StructureCase:
    """一个未知结构样本及其各专家残差/后验证据。"""

    fl: object
    fr: object
    residuals: Mapping[str, float]
    posterior: Mapping[str, float]
    params: tuple[float, ...]


@dataclass(frozen=True)
class StructureBirthRequest:
    """达到证据阈值后的结构出生请求 (供人工/自动化注册候选)"""

    cases: tuple[StructureCase, ...]
    residual_mean: float
    best_posterior_mean: float
    reason: str
    proposals: tuple[TemplateProposal, ...] = field(default_factory=tuple)


class StructureBirthController:
    """聚合未知结构信号, 达到 min_cases 后产出一次出生请求。"""

    def __init__(
        self,
        min_cases: int = 3,
        max_cases: int = 128,
        proposer: TemplateProposer | None = None,
    ):
        if min_cases < 1:
            raise ValueError("min_cases 必须 >=1")
        self.min_cases = min_cases
        self.max_cases = max_cases
        self.proposer = proposer
        self.cases: list[StructureCase] = []

    def observe(
        self, decision: GateDecision, fl: object, fr: object
    ) -> StructureBirthRequest | None:
        """记录一次结构门控结果; 未触发或证据不足时返回 None。"""
        if not decision.needs_new_structure:
            return None
        self.cases.append(
            StructureCase(
                fl=fl,
                fr=fr,
                residuals=dict(decision.residuals),
                posterior=dict(decision.posterior),
                params=decision.estimate.params,
            )
        )
        if len(self.cases) > self.max_cases:
            self.cases = self.cases[-self.max_cases :]
        if len(self.cases) < self.min_cases:
            return None
        cases = tuple(self.cases)
        residual_mean = sum(
            min(case.residuals.values()) for case in cases
        ) / len(cases)
        best_posterior_mean = sum(
            max(case.posterior.values()) for case in cases
        ) / len(cases)
        self.cases.clear()
        proposals = self.proposer.propose(cases) if self.proposer is not None else ()
        return StructureBirthRequest(
            cases=cases,
            residual_mean=residual_mean,
            best_posterior_mean=best_posterior_mean,
            reason=(
                f"{len(cases)} 个样本在所有结构专家中均不兼容; "
                f"已生成 {len(proposals)} 个模板提案; "
                "请提供新的可渲染结构族并训练注册"
            ),
            proposals=proposals,
        )
