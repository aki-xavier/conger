"""StructureGate: 视觉结构专家门控 (GenericStructureGate 适配层)。

视觉特有部分只有左右图重渲染残差; 结构后验/出生检测由
`GenericStructureGate` 提供。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import mlx.core as mx

from generic_structure_gate import GenericStructureDecision, GenericStructureGate
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth
from structure_geometry import StructureGeometry
from structured_hypothesis import StructuredHypothesis


class StructureGate(GenericStructureGate):
    """按左右图渲染残差融合多个视觉结构专家。"""

    def __init__(
        self,
        birth_residual: float = 10000.0,
        posterior_floor: float | None = 0.6,
        priors: Mapping[str, float] | None = None,
        complexity_weight: float = 1.0,
        geometry_weight: float = 5000.0,
        temperature_scale: float = 1.0,
    ):
        super().__init__(
            birth_residual=birth_residual,
            posterior_floor=posterior_floor,
            priors=priors,
            complexity_weight=complexity_weight,
            geometry_weight=geometry_weight,
            temperature_scale=temperature_scale,
        )

    @staticmethod
    def residual(
        estimate: StructuredHypothesis,
        fl: mx.array,
        fr: mx.array,
    ) -> float:
        """候选 Scene 重渲染左右视图 → 前景加权平均 MSE。"""
        renderer, cam_l, cam_r = SceneReconstructor.rig()
        cl = renderer.render(estimate.scene, cam_l)
        cr = renderer.render(estimate.scene, cam_r)
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        return 0.5 * (
            SceneReconstructor._masked_mse(fl, cl, wl)
            + SceneReconstructor._masked_mse(fr, cr, wr)
        )

    def decide(
        self,
        estimates: Mapping[str, StructuredHypothesis],
        fl: mx.array,
        fr: mx.array,
    ) -> GenericStructureDecision:
        """多结构 StructuredHypothesis → 结构后验 + 最佳估计 + 出生信号。"""
        geometry_costs = StructureGeometry.costs(fl, fr)
        stats_cache = {}
        with_residual = {}
        for name, est in estimates.items():
            family = est.geometry_family or name
            geometry_cost = geometry_costs.get(family, 0.0)
            if est.template_delta:
                if family not in stats_cache:
                    stats_cache[family] = StructureGeometry.geometry_stats(
                        family, fl, fr
                    )
                geometry_cost += StructureGeometry.delta_cost(
                    family, est.template_delta, stats_cache[family], fl, fr
                )
            with_residual[name] = replace(
                est,
                residual=self.residual(est, fl, fr),
                geometry_cost=geometry_cost,
            )
        return self.decide_hierarchical(with_residual)
