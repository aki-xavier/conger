"""StructureGate: 结构专家混合与未知结构出生检测。

每个结构专家先独立给出 SceneEstimate; 门控用候选 Scene 的左右图渲染
残差计算 p(structure|images)。残差含未知绝对尺度, 因此温度仍采用
T=max(2·best,1) 的相对校准, 出生检测另看绝对残差阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import mlx.core as mx

from scene_estimate import SceneEstimate
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth


@dataclass(frozen=True)
class StructureDecision:
    """结构门控结果: 最佳专家、结构后验、残差和出生信号。"""

    estimate: SceneEstimate
    posterior: dict[str, float]
    residuals: dict[str, float]
    needs_new_structure: bool


class StructureGate:
    """按渲染残差融合多个结构专家。"""

    def __init__(
        self,
        birth_residual: float = 25.0,
        posterior_floor: float = 0.8,
        priors: Mapping[str, float] | None = None,
    ):
        self.birth_residual = birth_residual
        self.posterior_floor = posterior_floor
        self.priors = dict(priors or {})

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
        assert estimates, "至少需要一个结构专家"
        residuals = {
            name: self.residual(est, fl, fr) for name, est in estimates.items()
        }
        best_score = min(residuals.values())
        temperature = max(2.0 * best_score, 1.0)
        logp = {}
        for name, score in residuals.items():
            prior = self.priors.get(name, 1.0)
            logp[name] = -score / temperature + mx.log(prior).item()
        mxp = mx.array(list(logp.values()))
        p = mx.exp(mxp - mx.logsumexp(mxp)).tolist()
        posterior = dict(zip(logp.keys(), map(float, p), strict=True))
        best_name = min(residuals, key=residuals.get)
        best = replace(
            estimates[best_name],
            structure_id=best_name,
            structure_posterior=posterior[best_name],
            structure_posteriors=posterior,
        )
        needs_new = (
            best_score > self.birth_residual
            and posterior[best_name] < self.posterior_floor
        )
        return StructureDecision(best, posterior, residuals, needs_new)
