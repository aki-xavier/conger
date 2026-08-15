"""TemplateGrammar: 有界几何模板文法。

当前可渲染组合操作: attach / layer / mirror / repeat。文法只负责
生成有限候选空间; 每个候选仍必须由正向 renderer 评分, 训练与注册
保持显式。
"""

from __future__ import annotations

from dataclasses import dataclass

from codebook import Codebook


@dataclass(frozen=True)
class TemplateRule:
    """一条模板文法派生规则 (depth≤2 的模板组合)。"""

    operation: str
    base_kind: int
    part_kind: int | None = None
    complexity: float = 1.0
    depth: int = 1

    def signature(self) -> str:
        """稳定字符串签名 (提案元数据/缓存用)。"""
        if self.part_kind is None:
            return f"primitive:{self.base_kind}"
        return f"{self.operation}:{self.base_kind}:{self.part_kind}"


class TemplateGrammar:
    """生成 depth 受限的 primitive 组合规则。"""

    # operation → (复杂度, 是否要求同 kind)
    OPERATORS = {
        "attach": (1.5, False),
        "layer": (2.0, False),
        "mirror": (1.4, True),
        "repeat": (1.3, True),
    }

    def __init__(
        self,
        operations: tuple[str, ...] = ("attach",),
        max_depth: int = 2,
        kinds: tuple[int, ...] | None = None,
    ):
        unknown = set(operations) - set(self.OPERATORS)
        if unknown:
            raise ValueError(f"未知模板操作: {sorted(unknown)}")
        if max_depth < 1:
            raise ValueError("max_depth 必须 >=1")
        self.operations = operations
        self.max_depth = max_depth
        self.kinds = kinds or tuple(range(Codebook.N_KIND))

    def primitives(self) -> tuple[TemplateRule, ...]:
        """depth=1: 已有 primitive 模板。"""
        return tuple(
            TemplateRule("primitive", k, complexity=1.0, depth=1)
            for k in self.kinds
        )

    def composites(self) -> tuple[TemplateRule, ...]:
        """depth=2: primitive ∘ primitive 组合规则。"""
        out = []
        for op in self.operations:
            complexity, same_kind = self.OPERATORS[op]
            for base in self.kinds:
                parts = (base,) if same_kind else self.kinds
                for part in parts:
                    out.append(
                        TemplateRule(
                            operation=op,
                            base_kind=base,
                            part_kind=part,
                            complexity=complexity,
                            depth=2,
                        )
                    )
        return tuple(out)

    def rules(self) -> tuple[TemplateRule, ...]:
        """有界搜索空间: depth1 primitives + 可选 depth2 组合。"""
        rules = list(self.primitives())
        if self.max_depth >= 2:
            rules.extend(self.composites())
        return tuple(rules)
