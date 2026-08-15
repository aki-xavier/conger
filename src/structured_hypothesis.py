"""StructuredHypothesis: 领域无关的结构化假设/后验返回对象。

视觉路径直接把 `scene` 字段用作 `cga.Scene` 表示; 非视觉域可改用
`representation`。候选参数/因子后验/新颖性证据均为通用字段, 领域可以
按需要留空。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class HypothesisCandidate:
    """一个完整结构化假设及其概率/正向残差。"""

    params: tuple[float, ...]
    probability: float
    residual: float | None = None


@dataclass(frozen=True)
class StructuredHypothesis:
    """一个结构专家对一个观测的参数化解释与不确定性。"""

    scene: Any = None  # 视觉领域对象; 与 representation 互为镜像
    params: tuple[float, ...] = ()
    spn_posterior: mx.array | None = None
    structure_id: str = "unknown"
    geometry_family: str | None = None  # 门控几何证据所属基础结构族
    template_delta: dict[str, Any] | None = None  # 子模板约束 (门控证据)
    representation: Any = None
    candidate_params: tuple[tuple[float, ...], ...] = ()
    candidate_scores: mx.array | None = None
    candidate_posterior: mx.array | None = None
    candidate_temperature: float | None = None
    hypotheses: tuple[HypothesisCandidate, ...] = field(default_factory=tuple)
    factor_sizes: tuple[int, ...] = (3, 6, 3, 3)
    factor_indices: tuple[int, ...] = (0, 5, 6, 7)
    responsibility_max: float | None = None
    posterior_entropy: float | None = None
    residual: float | None = None
    complexity: float | None = None  # 模板描述复杂度 (结构门控惩罚项)
    geometry_cost: float | None = None  # 观测级结构几何不兼容度
    novelty_score: float | None = None
    structure_posterior: float | None = None
    structure_posteriors: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.representation is None and self.scene is not None:
            object.__setattr__(self, "representation", self.scene)
        if self.scene is None and self.representation is not None:
            object.__setattr__(self, "scene", self.representation)

    def factor_marginals(self) -> tuple[mx.array, ...]:
        """候选后验 → 离散因子边缘后验 (视觉单/双层及其他领域通用)。"""
        vals = [[0.0] * n for n in self.factor_sizes]
        if self.candidate_posterior is None or not self.candidate_params:
            cols = tuple(int(self.params[j]) for j in self.factor_indices)
            for p, j in zip(vals, cols, strict=True):
                p[j] = 1.0
            return tuple(mx.array(p) for p in vals)
        for prm, prob in zip(
            self.candidate_params, self.candidate_posterior.tolist(), strict=True
        ):
            cols = tuple(int(prm[j]) for j in self.factor_indices)
            for p, j in zip(vals, cols, strict=True):
                p[j] += float(prob)
        return tuple(mx.array(p) for p in vals)
