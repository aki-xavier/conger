"""ExpertRegistry: 场景结构专家注册、加载与门控入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from inverse_app import InverseApp
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from structure_gate import StructureDecision, StructureGate


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

    def reconstruct(self, fl: mx.array, fr: mx.array):
        """左右图 → 该结构下的 SceneEstimate。"""
        return self.app.reconstruct_scene(self.net, fl, fr)


class ExpertRegistry:
    """结构专家集合 + 渲染残差结构门控。"""

    def __init__(
        self,
        experts: Mapping[str, SceneExpert],
        gate: StructureGate | None = None,
    ):
        assert experts, "至少注册一个结构专家"
        self.experts = dict(experts)
        self.gate = gate or StructureGate()

    @classmethod
    def from_configs(
        cls,
        configs: Mapping[str, InverseConfig],
        artifacts: Path | None = None,
        priors: Mapping[str, float] | None = None,
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
        return cls(experts, StructureGate(priors=priors))

    @classmethod
    def default(
        cls,
        artifacts: Path | None = None,
        include_layered: bool = True,
    ) -> ExpertRegistry:
        """默认注册单物体与双层专家 (要求对应模型已训练)。"""
        configs = {"single": InverseConfig(n_objects=1)}
        if include_layered:
            configs["layered"] = InverseConfig(n_objects=2, replicates=1)
        return cls.from_configs(configs, artifacts=artifacts)

    def decide(self, fl: mx.array, fr: mx.array) -> StructureDecision:
        """同一左右图交给全部专家, 再按重建残差门控结构。"""
        estimates = {
            name: expert.reconstruct(fl, fr)
            for name, expert in self.experts.items()
        }
        return self.gate.decide(estimates, fl, fr)
