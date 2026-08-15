"""ChildTemplateWorkflow: 出生提案 → 子模板规格 → 显式训练注册。

这是数据驱动子模板学习的编排层。它不自动触发训练; 调用方显式调用
`run`/`train_and_register`, 由 TemplateDeltaLearner 聚合证据,
ChildCodebookFactory 物化场景族, ExpertRegistry 完成训练和注册。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from child_codebook_factory import ChildCodebookFactory
from expert_registry import ExpertRegistry, SceneExpert
from inverse_config import InverseConfig
from structure_birth import StructureBirthRequest
from template_delta_learner import TemplateDeltaLearner
from template_lineage import ChildTemplateSpec


@dataclass(frozen=True)
class ChildTemplateRegistration:
    """一个已完成训练注册的子模板及其生成规格。"""

    spec: ChildTemplateSpec
    codebook_cls: type
    expert: SceneExpert


class ChildTemplateWorkflow:
    """数据驱动子模板学习/注册编排。"""

    def __init__(self, learner: TemplateDeltaLearner | None = None):
        self.learner = learner or TemplateDeltaLearner()

    def learn(
        self,
        requests: Iterable[StructureBirthRequest],
        registry: ExpertRegistry,
    ) -> tuple[ChildTemplateSpec, ...]:
        """从出生请求学习候选子模板规格 (血缘来自当前注册表)。"""
        return self.learner.learn(requests, lineages=registry.lineages())

    def materialize(self, spec: ChildTemplateSpec) -> type:
        """子模板规格 → 可训练 Codebook 类。"""
        return ChildCodebookFactory.build(spec)

    def train_and_register(
        self,
        registry: ExpertRegistry,
        spec: ChildTemplateSpec,
        cfg: InverseConfig | None = None,
        artifacts: Path | None = None,
    ) -> ChildTemplateRegistration:
        """显式训练并注册一个子模板专家。"""
        codebook_cls = self.materialize(spec)
        cfg = cfg or InverseConfig(scene_family=spec.family, replicates=1)
        expert = registry.train_and_register(
            spec.name,
            cfg,
            artifacts=artifacts,
            codebook_cls=codebook_cls,
        )
        return ChildTemplateRegistration(spec, codebook_cls, expert)

    def run(
        self,
        requests: Iterable[StructureBirthRequest],
        registry: ExpertRegistry,
        cfg: InverseConfig | None = None,
        artifacts: Path | None = None,
        max_children: int = 1,
    ) -> tuple[ChildTemplateRegistration, ...]:
        """学习 top 子模板规格并显式训练注册 (默认最多一个)。"""
        specs = self.learn(requests, registry)[:max_children]
        return tuple(
            self.train_and_register(registry, spec, cfg, artifacts)
            for spec in specs
        )
