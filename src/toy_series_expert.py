"""ToySeriesExpert: MixtureSPN 包装的非视觉结构专家。"""

from __future__ import annotations

import math

import mlx.core as mx

from mixture_spn import MixtureSPN
from structured_hypothesis import StructuredHypothesis
from toy_series_family import ToySeriesFamily


class ToySeriesExpert:
    """固定机制族 + 实例级 MixtureSPN + 正向序列残差。"""

    def __init__(self, family: ToySeriesFamily, net: MixtureSPN):
        self.family = family
        self.net = net

    @classmethod
    def train(
        cls, mechanism: str, n: int = 256, seed: int = 0
    ) -> ToySeriesExpert:
        family = ToySeriesFamily(mechanism)
        p = family.sample(n, seed)
        y = family.simulate(p)
        f = mx.stack([family.encode(row) for row in y])
        zeros = mx.zeros(n, dtype=mx.int32)
        classes = zeros[:, None]
        net = MixtureSPN.fit(
            f,
            p,
            zeros,
            scene_classes=classes,
            cat_sizes=(1,),
            rel_floor=1e-3,
        )
        return cls(family, net)

    def estimate(self, observation: mx.array) -> StructuredHypothesis:
        f = self.family.encode(observation)[None, :]
        t, _, r = self.net.predict(f)
        params = tuple(float(x) for x in t[0].tolist())
        pred = self.family.simulate(mx.array(params)[None, :])[0]
        residual = float(mx.sqrt(mx.mean((observation - pred) ** 2)))
        max_r = float(mx.max(r)) + 1e-12
        novelty = -math.log(max_r) / math.log(r.shape[1]) + math.log1p(residual)
        return StructuredHypothesis(
            structure_id=self.family.mechanism,
            params=params,
            representation=pred,
            responsibility_max=max_r,
            posterior_entropy=0.0,
            residual=residual,
            novelty_score=novelty,
        )
