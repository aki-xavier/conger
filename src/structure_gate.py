"""StructureGate: 视觉结构专家门控 (GenericStructureGate 适配层)。

视觉特有部分只有左右图重渲染残差; 结构后验/出生检测由
`GenericStructureGate` 提供。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import mlx.core as mx

from generic_structure_gate import GenericStructureDecision, GenericStructureGate
from scene_estimate import SceneEstimate
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth

StructureDecision = GenericStructureDecision


class StructureGate(GenericStructureGate):
    """按左右图渲染残差融合多个视觉结构专家。"""

    def __init__(
        self,
        birth_residual: float = 10000.0,
        posterior_floor: float | None = 0.6,
        priors: Mapping[str, float] | None = None,
    ):
        super().__init__(
            birth_residual=birth_residual,
            posterior_floor=posterior_floor,
            priors=priors,
        )

    @staticmethod
    def residual(
        estimate: SceneEstimate,
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
        estimates: Mapping[str, SceneEstimate],
        fl: mx.array,
        fr: mx.array,
    ) -> StructureDecision:
        """多结构 SceneEstimate → 结构后验 + 最佳估计 + 出生信号。"""
        with_residual = {
            name: replace(est, residual=self.residual(est, fl, fr))
            for name, est in estimates.items()
        }
        return super().decide(with_residual)
