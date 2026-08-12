"""Product: 变量分解节点 (log 密度 = 子节点之和, 变量条件独立)。"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from leaf import Leaf
from node import Node


@dataclass(frozen=True)
class Product(Node):
    """变量分解: log 密度 = 子节点 log 密度之和 (变量条件独立)。"""

    children: tuple[Node, ...]

    def eval_log(self, x: mx.array) -> mx.array:
        acc = self.children[0].eval_log(x)
        for i, c in enumerate(self.children[1:], 1):
            acc = acc + c.eval_log(x)
            if i % 128 == 0:
                # 图截断: 长加法链 (533 叶/块) 整图惰性执行会超
                # Metal 显存峰值, 每 128 步强制求值释放中间量
                mx.eval(acc)
        return acc

    def is_leaf_block(self) -> bool:
        return all(isinstance(c, Leaf) for c in self.children)

    def leaf_blocks(
        self, path: tuple[int, ...] = ()
    ) -> list[tuple[Node, tuple[int, ...]]]:
        if self.is_leaf_block():
            return [(self, path)]
        out: list[tuple[Node, tuple[int, ...]]] = []
        for i, c in enumerate(self.children):
            out.extend(c.leaf_blocks(path + (i,)))
        return out

    def replace_leaf(self, path: tuple[int, ...], new: Node) -> Node:
        if not path:
            # 替换目标必须是叶块: 防路径漂移 (树中间层重排导致替换到
            # 错误节点会静默腐蚀结构)
            assert self.is_leaf_block(), "替换目标非叶块: Product 含非叶子节点"
            return new
        kids = list(self.children)
        kids[path[0]] = kids[path[0]].replace_leaf(path[1:], new)
        return Product(tuple(kids))

    def node_at(self, path: tuple[int, ...]) -> Node:
        if not path:
            return self
        return self.children[path[0]].node_at(path[1:])

    def flatten(self, acc: dict[str, list]) -> int:
        idx = len(acc["type"])
        acc["type"].append(2)
        acc["kids"].append([])
        acc["kids"][idx] = [c.flatten(acc) for c in self.children]
        return idx
