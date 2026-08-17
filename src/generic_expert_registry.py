"""GenericExpertRegistry: 领域无关结构专家注册与统一调用。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from generic_structure_gate import GenericStructureDecision, GenericStructureGate
from structure_birth import StructureBirthController, StructureBirthRequest
from structured_hypothesis import StructuredHypothesis


class GenericExpert(Protocol):
    """一个结构专家: observation → StructuredHypothesis。"""

    def estimate(self, observation: object) -> StructuredHypothesis:
        """返回带正向模型残差的结构化假设。"""
        ...


class GenericExpertRegistry:
    """专家集合 + 结构门控 + 可选出生控制。"""

    def __init__(
        self,
        experts: Mapping[str, GenericExpert],
        gate: GenericStructureGate | None = None,
        birth_controller: StructureBirthController | None = None,
    ):
        assert experts, "至少注册一个结构专家"
        self.experts = dict(experts)
        self.gate = gate or GenericStructureGate()
        self.birth_controller = birth_controller
        self.last_birth_request: StructureBirthRequest | None = None

    def register(self, name: str, expert: GenericExpert) -> GenericExpert:
        self.experts[name] = expert
        return expert

    def train_and_register(
        self, name: str, trainer: Callable[[], GenericExpert]
    ) -> GenericExpert:
        """显式候选训练: trainer() → 新专家 → 注册。"""
        return self.register(name, trainer())

    def decide(self, observation: object) -> GenericStructureDecision:
        estimates = {
            name: expert.estimate(observation)
            for name, expert in self.experts.items()
        }
        decision = self.gate.decide(estimates)
        self.last_birth_request = (
            self.birth_controller.observe(decision, observation, observation)
            if self.birth_controller is not None
            else None
        )
        return decision
