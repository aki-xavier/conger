"""Sum: 行混合节点 (logsumexp(log_w + 子节点))。

文件名 sum_node 而非 sum: 防与内建函数混淆。
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from node import Node


@dataclass(frozen=True)
class Sum(Node):
    """行混合: log 密度 = logsumexp(log_w + 子节点 log 密度)。

    counts (K,) 子节点累计质量 (可选): 在线学习的权重更新用,
    log_w 由平滑计数重建 log((counts+1)/(Σ+K))。"""

    children: tuple[Node, ...]
    log_w: mx.array
    counts: mx.array | None = None

    def eval_log(self, x: mx.array) -> mx.array:
        evals = mx.stack([c.eval_log(x) for c in self.children])
        return mx.logsumexp(evals + self.log_w[:, None], axis=0)

    def is_leaf_block(self) -> bool:
        return False  # Sum 永不属叶块

    def leaf_blocks(
        self, path: tuple[int, ...] = ()
    ) -> list[tuple[Node, tuple[int, ...]]]:
        out: list[tuple[Node, tuple[int, ...]]] = []
        for i, c in enumerate(self.children):
            out.extend(c.leaf_blocks(path + (i,)))
        return out

    def replace_leaf(self, path: tuple[int, ...], new: Node) -> Node:
        assert path, "替换目标非叶块: Sum"
        kids = list(self.children)
        kids[path[0]] = kids[path[0]].replace_leaf(path[1:], new)
        return Sum(tuple(kids), self.log_w, self.counts)

    def node_at(self, path: tuple[int, ...]) -> Node:
        if not path:
            return self
        return self.children[path[0]].node_at(path[1:])

    def flatten(self, acc: dict[str, list]) -> int:
        idx = len(acc["type"])
        acc["type"].append(3)
        acc["kids"].append([])
        lw = self.log_w.tolist()
        acc["sum.node"].append(idx)
        acc["sum.nch"].append(len(lw))
        acc["sum.logw"].append(lw)
        acc["sum.counts"].append(
            self.counts.tolist() if self.counts is not None else [0.0] * len(lw)
        )
        acc["sum.has_counts"].append(self.counts is not None)
        acc["kids"][idx] = [c.flatten(acc) for c in self.children]
        return idx
