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
