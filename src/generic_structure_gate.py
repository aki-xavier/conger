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
    needs_new_structure: bool


class GenericStructureGate:
    """用各专家正向模型残差计算 p(structure|observation)。"""

    def __init__(
        self,
        birth_residual: float = 1.0,
        posterior_floor: float | None = None,
        priors: Mapping[str, float] | None = None,
    ):
        self.birth_residual = birth_residual
        self.posterior_floor = posterior_floor
        self.priors = dict(priors or {})

    def decide(
        self, estimates: Mapping[str, StructuredHypothesis]
    ) -> GenericStructureDecision:
        assert estimates, "至少需要一个结构专家"
        residuals = {}
        for name, est in estimates.items():
            assert est.residual is not None, f"专家 {name} 未提供正向残差"
            residuals[name] = est.residual
        best_score = min(residuals.values())
        temperature = max(2.0 * best_score, 1e-8)
        logp = []
        names = list(estimates)
        for name in names:
            prior = self.priors.get(name, 1.0)
            logp.append(-residuals[name] / temperature + float(mx.log(prior)))
        arr = mx.array(logp)
        probs = mx.exp(arr - mx.logsumexp(arr)).tolist()
        posterior = dict(zip(names, map(float, probs), strict=True))
        best_name = min(residuals, key=residuals.get)
        best = replace(
            estimates[best_name],
            structure_id=best_name,
            structure_posterior=posterior[best_name],
            structure_posteriors=posterior,
        )
        needs_new = best_score > self.birth_residual and (
            self.posterior_floor is None
            or posterior[best_name] < self.posterior_floor
        )
        return GenericStructureDecision(best, posterior, residuals, needs_new)
