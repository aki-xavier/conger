"""SceneEstimate: 完整 cga.Scene 推理的结构化后验返回值。"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
from cga.engine import Scene

from codebook import Codebook


@dataclass(frozen=True)
class SceneHypothesis:
    """一个完整场景假设及其后验/渲染残差。"""

    params: tuple[float, ...]
    probability: float
    residual: float | None = None


@dataclass(frozen=True)
class SceneEstimate:
    """MAP Scene + 候选后验 (不早丢弃逆渲染歧义)。

    `spn_posterior` 是 MixtureSPN 的拼接因子后验; 候选数组来自渲染
    残差精炼 (可含 top-k kind × 54 外观)。`scene/params` 是 MAP 假设,
    `hypotheses` 是概率降序的完整场景候选。"""

    scene: Scene
    params: tuple[float, ...]
    spn_posterior: mx.array
    candidate_params: tuple[tuple[float, ...], ...] = ()
    candidate_scores: mx.array | None = None
    candidate_posterior: mx.array | None = None
    candidate_temperature: float | None = None
    hypotheses: tuple[SceneHypothesis, ...] = field(default_factory=tuple)

    def factor_marginals(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """候选后验 → kind/hue/lcol/ldir 边缘后验。"""
        sizes = (
            Codebook.N_KIND,
            Codebook.N_HUE,
            len(Codebook.LIGHT_COLORS),
            len(Codebook.LIGHT_DIRS),
        )
        vals = [[0.0] * n for n in sizes]
        if self.candidate_posterior is None or not self.candidate_params:
            cols = (
                int(self.params[0]),
                int(self.params[5]),
                int(self.params[6]),
                int(self.params[7]),
            )
            for p, j in zip(vals, cols, strict=True):
                p[j] = 1.0
            return tuple(mx.array(p) for p in vals)
        for prm, prob in zip(
            self.candidate_params, self.candidate_posterior.tolist(), strict=True
        ):
            cols = (int(prm[0]), int(prm[5]), int(prm[6]), int(prm[7]))
            for p, j in zip(vals, cols, strict=True):
                p[j] += float(prob)
        return tuple(mx.array(p) for p in vals)
