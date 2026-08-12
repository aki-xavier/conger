"""Leaf: 单叶 (GaussLeaf/CatLeaf) 共享的结构行为 —— 自身即叶块。"""

from __future__ import annotations

from node import Node


class Leaf(Node):
    """单叶 (GaussLeaf/CatLeaf) 共享的结构行为: 自身即叶块。"""

    def is_leaf_block(self) -> bool:
        return True

    def leaf_blocks(
        self, path: tuple[int, ...] = ()
    ) -> list[tuple[Node, tuple[int, ...]]]:
        return [(self, path)]

    def replace_leaf(self, path: tuple[int, ...], new: Node) -> Node:
        assert not path, f"叶节点无子路径 {path}"
        return new

    def node_at(self, path: tuple[int, ...]) -> Node:
        assert not path, f"叶节点无子路径 {path}"
        return self
