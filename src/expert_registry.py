"""ExpertRegistry: 场景结构专家注册、加载与门控入口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from codebook import Codebook
from generic_structure_gate import GenericStructureDecision
from inverse_app import InverseApp
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from structure_birth import StructureBirthController, StructureBirthRequest
from structure_gate import StructureGate
from structured_hypothesis import StructuredHypothesis
from template_lineage import ChildTemplateSpec, TemplateLineage

if TYPE_CHECKING:
    from child_template_workflow import (
        ChildTemplateRegistration,
        ChildTemplateWorkflow,
    )


@dataclass
class SceneExpert:
    """一个结构专家: 固定场景族 + 对应 MixtureSPN + 重建器。"""

    name: str
    app: InverseApp
    net: MixtureSPN

    @classmethod
    def from_config(
        cls,
        name: str,
        cfg: InverseConfig,
        artifacts: Path | None = None,
    ) -> SceneExpert:
        """按配置加载默认模型; 缺模型时显式失败, 不静默降级。"""
        app = InverseApp(cfg)
        path = cfg.model_path or app.default_model_path(artifacts)
        if not path.exists():
            raise FileNotFoundError(f"结构专家 {name} 缺模型: {path}")
        return cls(name=name, app=app, net=MixtureSPN.load(path))

    def reconstruct(self, fl: mx.array, fr: mx.array) -> StructuredHypothesis:
        """左右图 → 该结构下的 StructuredHypothesis。"""
        return self.app.reconstruct_scene(self.net, fl, fr)

    def lineage(self) -> TemplateLineage:
        """该专家对应场景族的血缘元数据。"""
        return self.app.codebook.TEMPLATE_LINEAGE


class ExpertRegistry:
    """结构专家集合 + 渲染残差结构门控。"""

    def __init__(
        self,
        experts: Mapping[str, SceneExpert],
        gate: StructureGate | None = None,
        birth_controller: StructureBirthController | None = None,
        child_workflow: ChildTemplateWorkflow | None = None,
    ):
        assert experts, "至少注册一个结构专家"
        self.experts = dict(experts)
        self.gate = gate or StructureGate()
        self.birth_controller = birth_controller
        self.child_workflow = child_workflow
        self.last_birth_request: StructureBirthRequest | None = None
        self.birth_requests: list[StructureBirthRequest] = []
        self.pending_child_specs: dict[str, ChildTemplateSpec] = {}

    def lineages(self) -> dict[str, TemplateLineage]:
        """当前专家树/森林的血缘表。"""
        return {name: expert.lineage() for name, expert in self.experts.items()}

    def children_of(self, parent_family: str) -> tuple[str, ...]:
        """返回直接继承自 parent_family 的已注册专家名。"""
        return tuple(
            name
            for name, lineage in self.lineages().items()
            if lineage.parent_family == parent_family
        )

    def enable_child_template_learning(
        self, workflow: ChildTemplateWorkflow | None = None
    ) -> ChildTemplateWorkflow:
        """启用出生请求 → pending 子模板规格学习 (不自动训练)。"""
        from child_template_workflow import ChildTemplateWorkflow

        self.child_workflow = workflow or ChildTemplateWorkflow()
        return self.child_workflow

    def observe_birth_request(
        self, request: StructureBirthRequest
    ) -> tuple[ChildTemplateSpec, ...]:
        """记录出生请求, 并用已启用 workflow 更新 pending 子模板规格。"""
        self.last_birth_request = request
        self.birth_requests.append(request)
        if self.child_workflow is None:
            return ()
        specs = self.child_workflow.learn(self.birth_requests, self)
        new = []
        for spec in specs:
            if spec.name not in self.experts and (
                spec.name not in self.pending_child_specs
            ):
                self.pending_child_specs[spec.name] = spec
                new.append(spec)
        return tuple(new)

    def confirm_child_template(
        self,
        name: str,
        cfg: InverseConfig | None = None,
        artifacts: Path | None = None,
    ) -> ChildTemplateRegistration:
        """显式确认 pending 子模板: 物化、训练并注册。"""
        if self.child_workflow is None:
            raise RuntimeError("尚未启用 child template learning")
        if name not in self.pending_child_specs:
            raise KeyError(f"没有 pending 子模板: {name}")
        spec = self.pending_child_specs[name]
        registration = self.child_workflow.train_and_register(
            self, spec, cfg=cfg, artifacts=artifacts
        )
        del self.pending_child_specs[name]
        return registration

    @classmethod
    def from_configs(
        cls,
        configs: Mapping[str, InverseConfig],
        artifacts: Path | None = None,
        priors: Mapping[str, float] | None = None,
        complexity_weight: float = 1.0,
        geometry_weight: float = 5000.0,
        missing_ok: bool = False,
    ) -> ExpertRegistry:
        """按名称 → 配置注册专家; missing_ok=True 时跳过缺模型专家。"""
        experts = {}
        for name, cfg in configs.items():
            try:
                experts[name] = SceneExpert.from_config(name, cfg, artifacts)
            except FileNotFoundError:
                if not missing_ok:
                    raise
        if not experts:
            raise FileNotFoundError("所有结构专家模型都缺失")
        return cls(
            experts,
            StructureGate(
                priors=priors,
                complexity_weight=complexity_weight,
                geometry_weight=geometry_weight,
            ),
        )

    @classmethod
    def default(
        cls,
        artifacts: Path | None = None,
        include_layered: bool = True,
        include_composite: bool = True,
    ) -> ExpertRegistry:
        """默认注册结构专家 (要求对应模型已训练)。"""
        configs = {"single": InverseConfig(n_objects=1)}
        if include_layered:
            configs["layered"] = InverseConfig(n_objects=2, replicates=1)
        if include_composite:
            configs["composite"] = InverseConfig(
                n_objects=2,
                scene_family="composite",
                replicates=1,
            )
        return cls.from_configs(configs, artifacts=artifacts)

    def register(
        self,
        name: str,
        cfg: InverseConfig | None = None,
        expert: SceneExpert | None = None,
        artifacts: Path | None = None,
    ) -> SceneExpert:
        """注册已训练专家; 传 expert 或 cfg 之一。"""
        if expert is None:
            if cfg is None:
                raise ValueError("register 需要 expert 或 cfg")
            expert = SceneExpert.from_config(name, cfg, artifacts)
        self.experts[name] = expert
        return expert

    def train_and_register(
        self,
        name: str,
        cfg: InverseConfig,
        artifacts: Path | None = None,
        codebook_cls: type[Codebook] | None = None,
    ) -> SceneExpert:
        """结构出生后的显式候选训练: InverseApp.run() → 加载 → 注册。"""
        if codebook_cls is None:
            InverseApp(cfg).run(artifacts)
            return self.register(name, cfg=cfg, artifacts=artifacts)
        app = InverseApp(cfg, codebook=codebook_cls(cfg))
        app.run(artifacts)
        path = cfg.model_path or app.default_model_path(artifacts)
        expert = SceneExpert(name=name, app=app, net=MixtureSPN.load(path))
        self.experts[name] = expert
        return expert

    def decide(self, fl: mx.array, fr: mx.array) -> GenericStructureDecision:
        """同一左右图交给全部专家, 再按重建残差门控结构。"""
        estimates = {
            name: expert.reconstruct(fl, fr)
            for name, expert in self.experts.items()
        }
        decision = self.gate.decide(estimates, fl, fr)
        request = (
            self.birth_controller.observe(decision, fl, fr)
            if self.birth_controller is not None
            else None
        )
        if request is not None:
            self.observe_birth_request(request)
        return decision
