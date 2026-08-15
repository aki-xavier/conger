"""TemplateProposal: 结构出生候选的统一描述。

提案器只做“候选生成 + 正向模型评分”; 是否训练/注册仍由
StructureBirthRequest 的调用方显式决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TemplateProposal:
    """一个由现有模板组合得到的新结构候选。"""

    family: str
    operation: str
    params: tuple[float, ...]
    residual: float
    complexity: float
    score: float
    parent_family: str | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TemplateProposer(Protocol):
    """从结构出生证据生成可评分的新模板候选。"""

    def propose(self, cases: tuple[object, ...]) -> tuple[TemplateProposal, ...]:
        """聚合一个或多个未知结构样本, 返回按 score 升序的候选。"""
