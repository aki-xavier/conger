"""TemplateLineage: 结构模板的血缘/继承契约。

血缘只描述“从哪个父模板、通过什么 delta 得到当前模板”, 不要求子模板
复用父模型参数。它用于结构专家树、模板提案聚类和后续数据驱动子模板
学习。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateLineage:
    """一个结构模板的父模板与生成差异。"""

    family: str
    parent_family: str | None
    operation: str
    complexity: float
    generation: int = 0
    delta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent_family is None

    def signature(self) -> str:
        """稳定血缘签名 (日志/提案聚类键)。"""
        parent = self.parent_family or "root"
        return f"{parent}->{self.family}:{self.operation}"

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化表示。"""
        return {
            "family": self.family,
            "parent_family": self.parent_family,
            "operation": self.operation,
            "complexity": self.complexity,
            "generation": self.generation,
            "delta": self.delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateLineage:
        """JSON 表示 → TemplateLineage。"""
        return cls(
            family=str(data["family"]),
            parent_family=data.get("parent_family"),
            operation=str(data["operation"]),
            complexity=float(data["complexity"]),
            generation=int(data.get("generation", 0)),
            delta=dict(data.get("delta", {})),
        )


@dataclass(frozen=True)
class ChildTemplateSpec:
    """从多个模板提案估计得到的候选子模板约束。"""

    name: str
    family: str
    parent_family: str
    operation: str
    constraints: dict[str, Any]
    complexity: float
    generation: int
    evidence_count: int
    residual_mean: float
    score_mean: float

    def lineage(self) -> TemplateLineage:
        """子模板规格 → 可注册血缘对象。"""
        return TemplateLineage(
            family=self.name,
            parent_family=self.parent_family,
            operation=self.operation,
            complexity=self.complexity,
            generation=self.generation,
            delta=dict(self.constraints),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化表示。"""
        return {
            "name": self.name,
            "family": self.family,
            "parent_family": self.parent_family,
            "operation": self.operation,
            "constraints": self.constraints,
            "complexity": self.complexity,
            "generation": self.generation,
            "evidence_count": self.evidence_count,
            "residual_mean": self.residual_mean,
            "score_mean": self.score_mean,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChildTemplateSpec:
        """JSON 表示 → ChildTemplateSpec。"""
        return cls(
            name=str(data["name"]),
            family=str(data["family"]),
            parent_family=str(data["parent_family"]),
            operation=str(data["operation"]),
            constraints=dict(data.get("constraints", {})),
            complexity=float(data["complexity"]),
            generation=int(data["generation"]),
            evidence_count=int(data["evidence_count"]),
            residual_mean=float(data["residual_mean"]),
            score_mean=float(data["score_mean"]),
        )
