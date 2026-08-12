"""SPN 节点基类 (eval_log + 结构操作多态契约)。

结构操作 (is_leaf_block/leaf_blocks/replace_leaf/node_at/flatten)
由各子类多态实现, 不用 isinstance 分派; 反序列化工厂在 SPN
(node_from_records) —— 若放在 Node 会与叶类模块循环导入。
"""

from __future__ import annotations

import mlx.core as mx


class Node:
    """SPN 节点基类。eval_log(x) 对证据批 (M, V) 求 log 密度 (M,)。
    结构操作 (叶块收集/替换/定位/序列化) 为多态方法, 不用 isinstance 分派。"""

    def eval_log(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def is_leaf_block(self) -> bool:
        """叶块 (单叶或全叶 Product): 可独立生长/替换的结构单元。"""
        raise NotImplementedError

    def leaf_blocks(
        self, path: tuple[int, ...] = ()
    ) -> list[tuple[Node, tuple[int, ...]]]:
        """收集所有叶块及其从本节点的路径 (子索引序列)。"""
        raise NotImplementedError

    def replace_leaf(self, path: tuple[int, ...], new: Node) -> Node:
        """沿 path 重建, 把目标叶块替换为 new (frozen → 路径逐层重建)。"""
        raise NotImplementedError

    def node_at(self, path: tuple[int, ...]) -> Node:
        """沿路径取子节点。"""
        raise NotImplementedError

    def flatten(self, acc: dict[str, list]) -> int:
        """DFS 先序序列化到记录累加器 acc, 返回本节点下标 (SPN.save 用)。"""
        raise NotImplementedError

