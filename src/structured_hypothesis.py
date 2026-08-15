"""StructuredHypothesis: 领域无关的结构化假设返回对象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class StructuredHypothesis:
    """一个结构专家对一个观测的参数化解释。

    `representation` 是领域对象 (视觉中为 cga.Scene, 玩具域中为预测序列);
    residual 由该专家的正向模型/模拟器计算。结构字段由门控填写。"""
    structure_id: str
    params: tuple[float, ...]
    representation: Any
    spn_posterior: mx.array | None = None
    responsibility_max: float | None = None
    posterior_entropy: float | None = None
    residual: float | None = None
    novelty_score: float | None = None
    structure_posterior: float | None = None
    structure_posteriors: dict[str, float] | None = None
