"""GaussLeaf: 单变量高斯叶 (连续列), 含在线充分统计量 (n, m2)。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from leaf import Leaf


@dataclass(frozen=True)
class GaussLeaf(Leaf):
    """单变量高斯叶 (连续列)。

    n/m2: 在线学习的可加充分统计量 (计数 / 二阶中心矩×n)。
    mu 兼作运行均值, 合并用 Chan 并行公式 (批内两遍法算 m2) ——
    单遍 E[x²]−E[x]² 在 float32 下对近零方差维度灾难性抵消
    (确定性渲染特征 σ→σ_floor, 实测毁后验), 禁用。
    """

    var: int
    mu: float
    sigma: float
    n: float = 0.0
    m2: float = 0.0

    def eval_log(self, x: mx.array) -> mx.array:
        v = x[:, self.var]
        z = (v - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - 0.5 * math.log(2.0 * math.pi)

    def flatten(self, acc: dict[str, list]) -> int:
        idx = len(acc["type"])
        acc["type"].append(0)
        acc["kids"].append([])
        for k, v in (
            ("node", idx), ("var", self.var), ("mu", self.mu),
            ("sigma", self.sigma), ("n", self.n), ("m2", self.m2),
        ):
            acc[f"gauss.{k}"].append(v)
        return idx
