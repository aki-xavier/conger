"""GenericStructureGate: 领域无关的结构后验门控。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import mlx.core as mx

from structured_hypothesis import StructuredHypothesis


@dataclass(frozen=True)
class GenericStructureDecision:
    estimate: StructuredHypothesis
    posterior: dict[str, float]
    residuals: dict[str, float]
    scores: dict[str, float]
    needs_new_structure: bool
    family_posterior: dict[str, float] = field(default_factory=dict)
    family_conditional: dict[str, dict[str, float]] = field(default_factory=dict)


class GenericStructureGate:
    """用正向残差 + 模板复杂度计算 p(structure|observation)。

    `residuals` 保持原始正向模型误差, 供未知结构出生判断; `scores`
    是结构选择分数:
    score = residual + complexity_weight × C + geometry_weight × G。

    平铺 `decide` 对所有专家做一次 softmax; `decide_hierarchical` 先按
    `geometry_family` 分族 (父族级 softmax), 再在族内做父子级 softmax,
    联合后验 = p(family) × p(expert|family)。两级各自用本级最低分标定
    温度, 避免平铺混合在父子模板并存时的过置信。`temperature_scale`
    是全局温度缩放 (校准旋钮, >1 摊平、<1 锐化)。
    """

    def __init__(
        self,
        birth_residual: float = 1.0,
        posterior_floor: float | None = None,
        priors: Mapping[str, float] | None = None,
        complexity_weight: float = 0.0,
        geometry_weight: float = 0.0,
        temperature_scale: float = 1.0,
    ):
        self.birth_residual = birth_residual
        self.posterior_floor = posterior_floor
        self.priors = dict(priors or {})
        self.complexity_weight = complexity_weight
        self.geometry_weight = geometry_weight
        self.temperature_scale = temperature_scale

    def _scores(
        self, estimates: Mapping[str, StructuredHypothesis]
    ) -> tuple[dict[str, float], dict[str, float]]:
        residuals: dict[str, float] = {}
        scores: dict[str, float] = {}
        for name, est in estimates.items():
            assert est.residual is not None, f"专家 {name} 未提供正向残差"
            residuals[name] = est.residual
            scores[name] = (
                est.residual
                + self.complexity_weight * (est.complexity or 0.0)
                + self.geometry_weight * (est.geometry_cost or 0.0)
            )
        return residuals, scores

    @staticmethod
    def _softmax(
        scores: Mapping[str, float],
        temperature: float,
        priors: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        names = list(scores)
        pr = dict(priors or {})
        logp = [
            -scores[n] / temperature + float(mx.log(pr.get(n, 1.0)))
            for n in names
        ]
        arr = mx.array(logp)
        probs = mx.exp(arr - mx.logsumexp(arr)).tolist()
        return dict(zip(names, map(float, probs), strict=True))

    def decide(
        self, estimates: Mapping[str, StructuredHypothesis]
    ) -> GenericStructureDecision:
        """平铺门控: 所有专家一次 softmax (单层/无血缘退化形态)。"""
        assert estimates, "至少需要一个结构专家"
        residuals, scores = self._scores(estimates)
        best_raw = min(residuals.values())
        best_score = min(scores.values())
        # 温度用胜者 score 的幅度 (abs): score 含负几何奖励 (geometry_weight
        # ×geometry_cost 可为负), 直接 max(2·best_score, 1e-8) 会在负分时
        # 钳到 1e-8 → 后验退化成近 one-hot, 破坏 posterior_floor 出生判据。
        temperature = max(2.0 * abs(best_score), 1e-8) * self.temperature_scale
        posterior = self._softmax(scores, temperature, self.priors)
        best_name = min(scores, key=lambda n: scores[n])
        best = replace(
            estimates[best_name],
            structure_id=best_name,
            structure_posterior=posterior[best_name],
            structure_posteriors=posterior,
        )
        needs_new = best_raw > self.birth_residual and (
            self.posterior_floor is None
            or posterior[best_name] < self.posterior_floor
        )
        return GenericStructureDecision(
            estimate=best,
            posterior=posterior,
            residuals=residuals,
            scores=scores,
            needs_new_structure=needs_new,
        )

    def decide_hierarchical(
        self, estimates: Mapping[str, StructuredHypothesis]
    ) -> GenericStructureDecision:
        """两级门控: 父族级 softmax → 族内父子级 softmax。

        联合后验 p(expert) = p(family) × p(expert|family)。无血缘
        (geometry_family 均为 None/唯一) 时退化为平铺后验。胜者仍取
        score 全局最小者, 与平铺一致; 差别只在后验的标定方式。
        """
        assert estimates, "至少需要一个结构专家"
        residuals, scores = self._scores(estimates)
        best_raw = min(residuals.values())
        family_of = {
            name: (estimates[name].geometry_family or name)
            for name in estimates
        }
        groups: dict[str, list[str]] = {}
        for name in estimates:
            groups.setdefault(family_of[name], []).append(name)
        family_scores = {
            fam: min(scores[n] for n in names) for fam, names in groups.items()
        }
        # 温度用本级胜者 score 的幅度 (abs), 见 decide 的注释
        fam_temp = (
            max(2.0 * abs(min(family_scores.values())), 1e-8) * self.temperature_scale
        )
        family_posterior = self._softmax(family_scores, fam_temp, self.priors)
        family_conditional: dict[str, dict[str, float]] = {}
        posterior: dict[str, float] = {}
        for fam, names in groups.items():
            member_scores = {n: scores[n] for n in names}
            mem_temp = (
                max(2.0 * abs(min(member_scores.values())), 1e-8)
                * self.temperature_scale
            )
            cond = self._softmax(member_scores, mem_temp, self.priors)
            family_conditional[fam] = cond
            for n in names:
                posterior[n] = family_posterior[fam] * cond[n]
        best_name = min(scores, key=lambda n: scores[n])
        best = replace(
            estimates[best_name],
            structure_id=best_name,
            structure_posterior=posterior[best_name],
            structure_posteriors=posterior,
        )
        needs_new = best_raw > self.birth_residual and (
            self.posterior_floor is None
            or posterior[best_name] < self.posterior_floor
        )
        return GenericStructureDecision(
            estimate=best,
            posterior=posterior,
            residuals=residuals,
            scores=scores,
            needs_new_structure=needs_new,
            family_posterior=family_posterior,
            family_conditional=family_conditional,
        )
