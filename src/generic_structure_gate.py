"""GenericStructureGate: 领域无关的结构后验门控。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import mlx.core as mx

from structured_hypothesis import StructuredHypothesis


@dataclass(frozen=True)
class GenericStructureDecision:
    estimate: StructuredHypothesis
    posterior: dict[str, float]
    residuals: dict[str, float]
    scores: dict[str, float]
    needs_new_structure: bool


class GenericStructureGate:
    """用正向残差 + 模板复杂度计算 p(structure|observation)。

    `residuals` 保持原始正向模型误差, 供未知结构出生判断; `scores`
    是结构选择用的复杂度惩罚分数:
    score = residual + complexity_weight × template_complexity。
    """

    def __init__(
        self,
        birth_residual: float = 1.0,
        posterior_floor: float | None = None,
        priors: Mapping[str, float] | None = None,
        complexity_weight: float = 0.0,
    ):
        self.birth_residual = birth_residual
        self.posterior_floor = posterior_floor
        self.priors = dict(priors or {})
        self.complexity_weight = complexity_weight

    def decide(
        self, estimates: Mapping[str, StructuredHypothesis]
    ) -> GenericStructureDecision:
        assert estimates, "至少需要一个结构专家"
        residuals = {}
        scores = {}
        for name, est in estimates.items():
            assert est.residual is not None, f"专家 {name} 未提供正向残差"
            residuals[name] = est.residual
            scores[name] = (
                est.residual + self.complexity_weight * (est.complexity or 0.0)
            )
        best_raw = min(residuals.values())
        best_score = min(scores.values())
        temperature = max(2.0 * best_score, 1e-8)
        logp = []
        names = list(estimates)
        for name in names:
            prior = self.priors.get(name, 1.0)
            logp.append(-scores[name] / temperature + float(mx.log(prior)))
        arr = mx.array(logp)
        probs = mx.exp(arr - mx.logsumexp(arr)).tolist()
        posterior = dict(zip(names, map(float, probs), strict=True))
        best_name = min(scores, key=scores.get)
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
