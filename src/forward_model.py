"""ForwardModel: 领域无关的正向模型/模拟器协议。"""

from __future__ import annotations

from typing import Protocol


class ForwardModel(Protocol):
    """给定结构化潜变量, 生成或评分观测解释。"""

    def residual(self, observation: object, params: tuple[float, ...]) -> float:
        """观测与参数化假设之间的不匹配度 (越小越好)。"""
        ...
