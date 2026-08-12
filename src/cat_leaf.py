"""CatLeaf: 单变量分类叶 (离散列), Laplace 平滑 + 原始计数。"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from leaf import Leaf


@dataclass(frozen=True)
class CatLeaf(Leaf):
    """单变量分类叶 (离散列), logp (K,) 经 Laplace 平滑。

    counts (K,) 原始计数 (可选): 增量学习的叶支撑判定用
    count>0 (Laplace 平滑后 logp 无法反推零计数值)。
    """

    var: int
    logp: mx.array
    counts: mx.array | None = None

    def eval_log(self, x: mx.array) -> mx.array:
        idx = x[:, self.var].astype(mx.int32)
        return mx.take(self.logp, idx)

    def flatten(self, acc: dict[str, list]) -> int:
        idx = len(acc["type"])
        acc["type"].append(1)
        acc["kids"].append([])
        lp = self.logp.tolist()
        acc["cat.node"].append(idx)
        acc["cat.var"].append(self.var)
        acc["cat.k"].append(len(lp))
        acc["cat.logp"].append(lp)
        acc["cat.counts"].append(
            self.counts.tolist() if self.counts is not None else [0.0] * len(lp)
        )
        acc["cat.has_counts"].append(self.counts is not None)
        return idx
