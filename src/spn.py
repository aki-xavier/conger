"""Sum-Product Network (SPN, Gens & Domingos 2013 风格) — 可求和边缘化的生成混合模型。

结构: 叶 (单变量高斯 / 分类) / Product (变量分解) / Sum (行混合)。

学习 (SPNLearner) 贪心递归:
  * 行太少 / 单变量 / 达深度上限 → 叶 (或对角乘积);
  * 节点含离散列且行码混杂 → Sum, 直接按码空间 k-means 分裂
    (查询变量驱动: 混合分量按码同质, 后验不再被高维特征边缘
    淹没 —— 等价 class-conditional 分量、留在单网络里);
  * 变量两两独立 (G 检验, 连续变量分位数离散化) → Product,
    按依赖图连通分量拆分变量子集, 各分量用全部行递归;
  * 否则 k-means 行分裂 → Sum (权重 = 子集数据占比)。

不含离散列的纯连续数据退化为经典 learnSPN (Gens & Domingos)。

推理: 证据沿叶评估 → log 空间自底向上 (数值稳定)。posterior()
对离散列全枚举, 求 log-softmax 后验 —— 这是 SPN 相对贝叶斯网络/
GMM 的卖点: 边缘化天然可求和, 无推理 NP 问题。

契约 (demo_inverse.py 消费):
    X 列布局 = [连续特征列 | 离散码列]; disc_cols = 离散列下标集合;
    card[col] = 离散列基数 (缺省从数据取 max+1)。
    spn.posterior(feats, codes): feats (M, Vf) 连续观测, codes (K, C)
    全枚举离散码 → (M, K) log 后验, 行和归一。

本文件自检: `python src/spn.py` (G 检验 / 独立结构 / 混合后验)。

ponytail: 参数=硬分裂 MLE (k-means 子集内拟合), 未做 EM 精修 ——
结构对、后验单调性对的前提下, EM 只提升密度估计不改变 MAP; 加
EM 当 demo 出明显欠拟合时。
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from utils import Utils

# ── 节点 ──────────────────────────────────────────────────────────


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

    @staticmethod
    def from_records(d: dict[str, mx.array]) -> Node:
        """扁平记录数组 → 树 (类型标签分派: 反序列化的唯一工厂点)。"""
        types = d["node.type"].tolist()
        starts = d["node.start"].tolist()
        children = d["node.child"].tolist()
        g_i = {n: i for i, n in enumerate(d["gauss.node"].tolist())}
        gv = d["gauss.var"].tolist()
        gmu = d["gauss.mu"].tolist()
        gs = d["gauss.sigma"].tolist()
        gn = d["gauss.n"].tolist()
        gm2 = d["gauss.m2"].tolist()
        c_i = {n: i for i, n in enumerate(d["cat.node"].tolist())}
        cv = d["cat.var"].tolist()
        ck = d["cat.k"].tolist()
        clp = d["cat.logp"].tolist()
        ccnt = d["cat.counts"].tolist()
        chas = d["cat.has_counts"].tolist()
        s_i = {n: i for i, n in enumerate(d["sum.node"].tolist())}
        snch = d["sum.nch"].tolist()
        slw = d["sum.logw"].tolist()
        scnt = d["sum.counts"].tolist()
        shas = d["sum.has_counts"].tolist()

        def build(idx: int) -> Node:
            t = types[idx]
            if t == 0:
                j = g_i[idx]
                return GaussLeaf(
                    int(gv[j]), float(gmu[j]), float(gs[j]),
                    float(gn[j]), float(gm2[j]),
                )
            if t == 1:
                j = c_i[idx]
                k = int(ck[j])
                counts = mx.array(ccnt[j][:k], dtype=mx.float32) if chas[j] else None
                return CatLeaf(
                    int(cv[j]), mx.array(clp[j][:k], dtype=mx.float32), counts
                )
            kids = tuple(
                build(children[p]) for p in range(starts[idx], starts[idx + 1])
            )
            if t == 2:
                return Product(kids)
            j = s_i[idx]
            m = int(snch[j])
            counts = mx.array(scnt[j][:m], dtype=mx.float32) if shas[j] else None
            return Sum(kids, mx.array(slw[j][:m], dtype=mx.float32), counts)

        return build(0)


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


class SPN:
    """学习好的 SPN: 根节点 + 变量布局 (连续列 | 离散码列)。"""

    def __init__(self, root: Node, n_vars: int):
        self.root = root
        self.n_vars = n_vars

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """safetensors 存盘: 树扁平化为记录数组 (DFS 先序 + CSR 子索引),
        张量二进制体 + JSON 明文头 (Utils.st_metadata 可查, 无代码执行
        风险, 与类路径解耦)。.pkl 后缀 → 旧 pickle 格式 (向后兼容)。

        extra: 与模型配套的非 SPN 状态 (如特征 z-score 统计 mu/sd),
        mx 数组以 extra.* 键入文件, 标量 JSON 化进头。
        """
        if str(path).endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump({"spn": self, "extra": extra or {}}, f)
            return
        acc: dict[str, list] = {
            k: []
            for k in (
                "type", "kids",
                "gauss.node", "gauss.var", "gauss.mu", "gauss.sigma",
                "gauss.n", "gauss.m2",
                "cat.node", "cat.var", "cat.k", "cat.logp", "cat.counts",
                "cat.has_counts",
                "sum.node", "sum.nch", "sum.logw", "sum.counts",
                "sum.has_counts",
            )
        }
        self.root.flatten(acc)
        arrs = SPN.records_to_arrays(acc)
        meta = {"config": json.dumps({"n_vars": self.n_vars})}
        for k, v in (extra or {}).items():
            if isinstance(v, mx.array):
                arrs[f"extra.{k}"] = v
            else:
                meta[f"extra.{k}"] = json.dumps(v)
        mx.save_safetensors(str(path), arrs, meta)

    @staticmethod
    def records_to_arrays(acc: dict[str, list]) -> dict[str, mx.array]:
        """记录累加器 → 扁平数组: CSR 子索引 (按下标序拼接 → starts 单调)
        + 变长负载零填充。"""
        children: list[int] = []
        starts: list[int] = []
        for kids in acc["kids"]:
            starts.append(len(children))
            children.extend(kids)
        max_k = max(acc["cat.k"], default=1)
        max_ch = max(acc["sum.nch"], default=1)

        def pad(rows: list, width: int) -> list:
            return [r + [0.0] * (width - len(r)) for r in rows]

        return {
            "node.type": mx.array(acc["type"], dtype=mx.int32),
            "node.start": mx.array(starts + [len(children)], dtype=mx.int32),
            "node.child": mx.array(children, dtype=mx.int32),
            "gauss.node": mx.array(acc["gauss.node"], dtype=mx.int32),
            "gauss.var": mx.array(acc["gauss.var"], dtype=mx.int32),
            "gauss.mu": mx.array(acc["gauss.mu"], dtype=mx.float32),
            "gauss.sigma": mx.array(acc["gauss.sigma"], dtype=mx.float32),
            "gauss.n": mx.array(acc["gauss.n"], dtype=mx.float32),
            "gauss.m2": mx.array(acc["gauss.m2"], dtype=mx.float32),
            "cat.node": mx.array(acc["cat.node"], dtype=mx.int32),
            "cat.var": mx.array(acc["cat.var"], dtype=mx.int32),
            "cat.k": mx.array(acc["cat.k"], dtype=mx.int32),
            "cat.logp": mx.array(pad(acc["cat.logp"], max_k), dtype=mx.float32),
            "cat.counts": mx.array(
                pad(acc["cat.counts"], max_k), dtype=mx.float32
            ),
            "cat.has_counts": mx.array(acc["cat.has_counts"], dtype=mx.int32),
            "sum.node": mx.array(acc["sum.node"], dtype=mx.int32),
            "sum.nch": mx.array(acc["sum.nch"], dtype=mx.int32),
            "sum.logw": mx.array(pad(acc["sum.logw"], max_ch), dtype=mx.float32),
            "sum.counts": mx.array(
                pad(acc["sum.counts"], max_ch), dtype=mx.float32
            ),
            "sum.has_counts": mx.array(acc["sum.has_counts"], dtype=mx.int32),
        }

    @staticmethod
    def load(path: str | Path) -> tuple[SPN, dict[str, Any]]:
        """save 的逆操作 (按扩展名识别 safetensors/pickle) → (SPN, extra)。"""
        if str(path).endswith(".pkl"):
            with open(path, "rb") as f:
                d = pickle.load(f)
            return d["spn"], d["extra"]
        d = mx.load(str(path))
        hd = Utils.st_metadata(path).get("__metadata__", {})
        cfg = json.loads(hd["config"])
        extra: dict[str, Any] = {
            k[6:]: v for k, v in d.items() if k.startswith("extra.")
        }
        extra.update(
            {k[6:]: json.loads(v) for k, v in hd.items() if k.startswith("extra.")}
        )
        return SPN(Node.from_records(d), int(cfg["n_vars"])), extra
    def tree_str(
        self,
        labels: dict[int, str] | None = None,
        code_names: dict[int, dict[int, str]] | None = None,
    ) -> str:
        """树结构文本可视化 (缩进层级), 带节点语义解释。

        labels: 列号→名称 (如 "log_mag@(3,2)" / "kind")。
        code_names: 码列→{值:语义名} (如 {144: {0:"sphere",1:"cylinder",
        2:"box"}}, 145: {i:f"gx={i}"}})。给出后: 叶块追加主码组合
        (≈ box gx=3 gy=0 s=0.6), Sum 追加分裂轴 (两子代表码首次分歧
        的码列与值), Product 标注变量独立分解。
        """
        lines: list[str] = []
        code_cols = sorted(code_names) if code_names else []

        def leaf_block(node: Node) -> str | None:
            """叶块 → 摘要行; 非叶块 (含 Sum) → None。"""
            if isinstance(node, (GaussLeaf, CatLeaf)):
                leaves = [node]
            elif isinstance(node, Product) and all(
                isinstance(c, (GaussLeaf, CatLeaf)) for c in node.children
            ):
                leaves = list(node.children)
            else:
                return None
            gs = [n for n in leaves if isinstance(n, GaussLeaf)]
            cs = [n for n in leaves if isinstance(n, CatLeaf)]
            parts = [f"Gauss×{len(gs)}"]
            if gs:
                sig = sorted(n.sigma for n in gs)
                parts.append(f"σmed={sig[len(sig) // 2]:.3f}")
            for c in cs:
                name = labels.get(c.var, str(c.var)) if labels else str(c.var)
                lp = c.logp.tolist()
                top = sorted(range(len(lp)), key=lambda i: -lp[i])[:4]
                dist = " ".join(f"{v}:{math.exp(lp[v]):.2f}" for v in top)
                parts.append(f"Cat({name}) {dist}")
            return "LeafBlock " + " | ".join(parts)

        def block_rep(node: Node) -> tuple[int, ...] | None:
            """叶块主码: 每个码列取 argmax 值; 无码叶 → None。"""
            if not code_cols:
                return None
            if isinstance(node, (GaussLeaf, CatLeaf)):
                leaves = [node]
            elif isinstance(node, Product) and all(
                isinstance(c, (GaussLeaf, CatLeaf)) for c in node.children
            ):
                leaves = list(node.children)
            else:
                return None
            cats = {c.var: c for c in leaves if isinstance(c, CatLeaf)}
            if not cats:
                return None
            return tuple(int(mx.argmax(cats[col].logp)) for col in code_cols)

        def human(rep: tuple[int, ...] | None) -> str:
            """主码 → 语义串, 如 "box gx=3 gy=0 s=0.6"。"""
            if rep is None or not code_names:
                return ""
            return " ".join(
                code_names[col].get(v, str(v)) for col, v in zip(code_cols, rep)
            )

        def split_axis(reps: list[tuple[int, ...] | None]) -> str:
            """Sum 分裂轴: 两子代表码首次分歧的码列与值对比。"""
            if len(reps) < 2 or reps[0] is None or reps[1] is None or not code_names:
                return ""
            for col, (va, vb) in zip(code_cols, zip(reps[0], reps[1])):
                if va != vb:
                    na = labels.get(col, str(col)) if labels else str(col)
                    lhs = code_names[col].get(va, va)
                    rhs = code_names[col].get(vb, vb)
                    return f"| 分裂轴 {na}: {lhs} ↔ {rhs}"
            return "| 分裂轴: 码分布相近 (主要靠特征)"

        def rec(node: Node, depth: int) -> tuple[int, ...] | None:
            pad = "  " * depth
            blk = leaf_block(node)
            if blk is not None:
                rep = block_rep(node)
                lines.append(pad + blk + ("  ≈ " + human(rep) if rep else ""))
                return rep
            if isinstance(node, Sum):
                reps = [rec(c, depth + 1) for c in node.children]
                w = mx.exp(node.log_w).tolist()
                lines.append(
                    pad
                    + "Sum w="
                    + ",".join(f"{x:.3f}" for x in w)
                    + "  "
                    + split_axis(reps)
                )
                best = max(range(len(w)), key=lambda i: w[i])
                return reps[best]
            assert isinstance(node, Product)
            reps = [rec(c, depth + 1) for c in node.children]
            lines.append(pad + "Product  | 变量独立分解")
            return next((r for r in reps if r is not None), None)

        rec(self.root, 0)
        return "\n".join(lines)

    def eval_log(self, x: mx.array) -> mx.array:
        """证据批 (M, V) → log 密度 (M,)。"""
        return self.root.eval_log(x)

    def posterior(
        self,
        feats: mx.array,
        codes: mx.array,
        log_prior: mx.array | None = None,
    ) -> mx.array:
        """贝叶斯反演: P(码 | 特征) ∝ P(特征 | 码)·P(码)。

        feats (M, Vf) 连续观测 (列布局的连续部分); codes (K, C) 离散码
        全枚举 → (M, K) log 后验, 行归一。列布局: [feats | codes]。

        log_prior (K,) 或 (M, K): 码先验 log P(c) (外部知识注入, 如一般视角/
        熟悉尺寸/视平线)。None = 均匀先验 (纯数据似然)。实现即
        贝叶斯公式的 P(S) 项: logp += log_prior 再行归一。
        """
        m, vf = feats.shape
        k, c = codes.shape
        fe = mx.tile(feats[:, None, :], (1, k, 1)).reshape(m * k, vf)
        co = mx.tile(codes[None, :, :], (m, 1, 1)).reshape(m * k, c)
        x = mx.concatenate([fe, co], axis=1)
        logp = self.root.eval_log(x).reshape(m, k)
        if log_prior is not None:
            if log_prior.ndim == 1:
                logp = logp + log_prior[None, :]
            else:
                logp = logp + log_prior
        return logp - mx.logsumexp(logp, axis=1, keepdims=True)


# ── 结构学习 ──────────────────────────────────────────────────────


class SPNLearner:
    """learnSPN 结构学习器 (贪心递归): 配置入构造器, learn(X) → SPN。

    算法 (详见模块 docstring):
      * 行太少 / 单变量 / 达深度上限 → 叶 (对角乘积);
      * 节点含离散列且行码混杂 → Sum, 码空间 k-means 分裂 (查询变量驱动:
        混合分量按码同质, 后验不再被高维特征边缘淹没);
      * 变量两两独立 (G 检验, 连续变量分位数离散化) → Product,
        依赖图连通分量拆分变量子集;
      * 否则 k-means 行分裂 → Sum (权重 = 子集数据占比)。
    """

    NBINS: ClassVar[int] = 8  # one-hot 宽度 = 最大离散基数
    NBINS_CONT: ClassVar[int] = 5  # 连续变量分位数 bin 数
    # χ² 临界值 (α=0.05), dof 1..16 精确表; dof>16 用 Wilson-Hilferty
    # 近似 (z=1.6449, 误差 <2%) —— 离散-离散对 dof 可达 (8−1)²=49。
    # 硬编码避免 scipy 依赖。
    CHI2_05: ClassVar[tuple[float, ...]] = (
        3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507,
        16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
    )

    def __init__(
        self,
        disc_cols: set[int],
        card: dict[int, int] | None = None,
        min_n: int = 24,
        max_depth: int = 12,
        alpha: float = 0.05,
        sigma_floor: float = 1e-6,
    ):
        """disc_cols: 离散列 (分类叶/查询变量); card: 离散列基数 (缺省
        learn 时从数据取 max+1)。sigma_floor: 高斯叶 σ 下限 (平滑性先验,
        见 prior.md 紧凑性/平滑性) —— 防确定性渲染 σ→0 过拟合,
        是 MAP 正则 (隐式 Gaussian prior on σ)。"""
        self.disc_cols = disc_cols
        self.card = card
        self.min_n = min_n
        self.max_depth = max_depth
        self.alpha = alpha
        self.sigma_floor = sigma_floor

    def learn(self, X: mx.array) -> SPN:
        """X (N, V) float32 → 学好的 SPN。"""
        n, v = X.shape
        if self.card is None:
            self.card = {c: int(mx.max(X[:, c])) + 1 for c in self.disc_cols}
        root = self.learn_node(X, list(range(n)), list(range(v)), 0)
        return SPN(root, v)

    def learn_node(
        self, X: mx.array, rows: list[int], cols: list[int], depth: int
    ) -> Node:
        n, c = len(rows), len(cols)
        if c == 1 or n < self.min_n or depth >= self.max_depth:
            return self.diag_leaves(X, rows, cols)
        xr = X[mx.array(rows, dtype=mx.int32)]

        # 查询变量驱动: 节点含离散列且行码混杂 → Sum 按码空间分裂
        code_cols = [cc for cc in cols if cc in self.disc_cols]
        if code_cols and self.rows_code_mixed(xr, code_cols):
            r0, r1 = self.split_rows(xr, code_cols)
            if len(r0) < self.min_n or len(r1) < self.min_n:
                return self.diag_leaves(X, rows, cols)
            return self.make_sum(X, rows, r0, r1, cols, depth)

        dep = self.gtest(self.binarize(xr, cols, self.disc_cols), self.alpha)
        comps = self.dep_components(dep)
        if len(comps) > 1:
            # 变量可分: 依赖图连通分量各自建模, 全部行共享
            return Product(
                tuple(
                    self.learn_node(X, rows, [cols[i] for i in comp], depth + 1)
                    for comp in comps
                )
            )
        r0, r1 = self.split_rows(xr, cols)
        if len(r0) < self.min_n or len(r1) < self.min_n:
            return self.diag_leaves(X, rows, cols)
        return self.make_sum(X, rows, r0, r1, cols, depth)

    def make_sum(
        self,
        X: mx.array,
        rows: list[int],
        r0: list[int],
        r1: list[int],
        cols: list[int],
        depth: int,
    ) -> Node:
        """行子集 → Sum(两个递归子树), 权重 = 子集数据占比。"""
        g0 = [rows[i] for i in r0]  # 局部下标 → 全局行号
        g1 = [rows[i] for i in r1]
        n = len(rows)
        log_w = mx.log(mx.array([len(g0) / n, len(g1) / n], dtype=mx.float32))
        return Sum(
            (
                self.learn_node(X, g0, cols, depth + 1),
                self.learn_node(X, g1, cols, depth + 1),
            ),
            log_w,
        )

    def diag_leaves(self, X: mx.array, rows: list[int], cols: list[int]) -> Node:
        """基例: 每列一个叶 (对角)。离散列分类叶, 连续列高斯叶。"""
        assert self.card is not None  # learn() 已填
        xr = X[mx.array(rows, dtype=mx.int32)]
        leaves: list[Node] = []
        for c in cols:
            v = xr[:, c]
            if c in self.disc_cols:
                k = self.card[c]
                cnt = mx.sum(
                    mx.equal(v[:, None], mx.arange(k)[None, :]).astype(mx.float32),
                    axis=0,
                )
                # Laplace: 未见类别保底, 防后验硬零
                logp = mx.log((cnt + 1.0) / (float(mx.sum(cnt)) + k))
                leaves.append(CatLeaf(c, logp, cnt))
            else:
                sd = float(mx.maximum(mx.std(v), self.sigma_floor))
                leaves.append(GaussLeaf(c, float(mx.mean(v)), sd))
        if len(leaves) == 1:
            return leaves[0]
        return Product(tuple(leaves))

    @staticmethod
    def rows_code_mixed(xr: mx.array, code_cols: list[int]) -> bool:
        """行集在码列上是否混杂 (存在 >1 个不同码)。"""
        for j in code_cols:
            distinct = len({int(v) for v in xr[:, j].tolist()})
            if distinct > 1:
                return True
        return False

    # ── 独立性检验 (G 检验) ────────────────────────────────────────

    @staticmethod
    def binarize(xr: mx.array, cols: list[int], disc_cols: set[int]) -> mx.array:
        """列子集 → (n, c) int32 bin id: 连续列分位数分档, 离散列取原值。"""
        parts = []
        for c in cols:
            v = xr[:, c]
            if c in disc_cols:
                parts.append(v.astype(mx.int32))
            else:
                parts.append(SPNLearner.quant_bins(v, SPNLearner.NBINS_CONT))
        return mx.stack(parts, axis=1)

    @staticmethod
    def quant_bins(v: mx.array, nbins: int) -> mx.array:
        """分位数分箱: 阈值取排序值 i·(n−1)/nbins 处, bin = Σ (v > t_i)。"""
        n = v.shape[0]
        sv = mx.sort(v)
        thr = [sv[min(n - 1, int(round(i * (n - 1) / nbins)))] for i in range(1, nbins)]
        t = mx.stack(thr)[None, :]
        return mx.sum(v[:, None] > t, axis=1).astype(mx.int32)

    @staticmethod
    def gtest(binned: mx.array, alpha: float = 0.05) -> mx.array:
        """两两 G 检验 → (c, c) bool 依赖矩阵 (对称, 对角线 False)。

        向量化: one-hot (n,c,B) → 联合计数 J = AᵀA (cB,cB) → (c,c,B,B);
        G = 2·Σ n_ab·ln(n_ab·N/(n_a·n_b)), dof = (k_i−1)(k_j−1) (k = 实际
        非空 bin 数), 超 χ² 临界值 (α) 即依赖。零计数用掩码归零。
        """
        n, c = binned.shape
        b = SPNLearner.NBINS
        oh = mx.equal(binned[:, :, None], mx.arange(b)[None, None, :]).astype(
            mx.float32
        )
        a = oh.reshape(n, c * b)
        jp = (a.T @ a).reshape(c, b, c, b).transpose(0, 2, 1, 3)  # (c,c,B,B)
        m = mx.sum(oh, axis=0)  # (c, B) 边缘计数
        logn = math.log(n)
        inner = (
            mx.log(mx.maximum(jp, 1.0))
            + logn
            - mx.log(mx.maximum(m[:, None, :, None], 1.0))
            - mx.log(mx.maximum(m[None, :, None, :], 1.0))
        )
        term = mx.where(jp > 0, jp * inner, 0.0)
        g = 2.0 * mx.sum(term, axis=(2, 3))  # (c, c)
        k = mx.sum(m > 0, axis=1)  # (c,) 每列非空 bin 数
        dof = (k[:, None] - 1) * (k[None, :] - 1)  # (c,c) float
        dof = mx.maximum(dof, 0.0)
        thr_tab = mx.take(
            mx.array(SPNLearner.CHI2_05, dtype=mx.float32),
            mx.clip(dof - 1, 0, 15).astype(mx.int32),
        )
        # dof>16: Wilson-Hilferty χ²_0.95(d) ≈ d·(1−2/9d+z·√(2/9d))³
        # (ddof 钳 1 防 dof=0 除零 —— 该分支会被 where 丢弃但 NaN 不雅)
        z = 1.6448536
        ddof = mx.maximum(dof, 1.0)
        thr_wh = (
            ddof * (1.0 - 2.0 / (9.0 * ddof) + z * mx.sqrt(2.0 / (9.0 * ddof))) ** 3
        )
        thr = mx.where(dof <= 16, thr_tab, thr_wh)
        return (g > thr) & (dof >= 1)

    @staticmethod
    def dep_components(dep: mx.array) -> list[list[int]]:
        """依赖图连通分量 (union-find), 按大小降序。"""
        c = dep.shape[0]
        parent = list(range(c))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        edges: list[list[bool]] = dep.tolist()
        for i in range(c):
            for j in range(i + 1, c):
                if edges[i][j]:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        comps: dict[int, list[int]] = {}
        for i in range(c):
            comps.setdefault(find(i), []).append(i)
        return sorted(comps.values(), key=len, reverse=True)

    # ── 行分裂 (k-means) ───────────────────────────────────────────

    @staticmethod
    def split_rows(
        xr: mx.array, cols: list[int], iters: int = 12
    ) -> tuple[list[int], list[int]]:
        """k=2 均值聚类 (z-score, 最远对初始化) → 两簇行下标。

        cols 由调用方选: 码列 (查询变量驱动, 分裂按码空间 Voronoi) 或全部列
        (经典 learnSPN)。退化保护: 一簇为空 → 返回 ([], all) 由调用方转基例。
        """
        data = xr[:, mx.array(cols, dtype=mx.int32)]
        mu = mx.mean(data, axis=0, keepdims=True)
        sd = mx.maximum(mx.std(data, axis=0, keepdims=True), 1e-6)
        z = (data - mu) / sd
        i0 = int(mx.argmax(mx.sum(z * z, axis=1)))
        i1 = int(mx.argmax(mx.sum((z - z[i0]) ** 2, axis=1)))
        means = mx.stack([z[i0], z[i1]])  # (2, c)
        for _ in range(iters):
            d = mx.sum((z[:, None, :] - means[None, :, :]) ** 2, axis=2)  # (n, 2)
            a = mx.argmin(d, axis=1)
            c0 = a == 0
            c1 = a == 1
            n0 = int(mx.sum(c0))
            n1 = int(mx.sum(c1))
            if n0 == 0 or n1 == 0:
                break
            means = mx.stack(
                [
                    mx.sum(z * c0[:, None], axis=0) / n0,
                    mx.sum(z * c1[:, None], axis=0) / n1,
                ]
            )
        d = mx.sum((z[:, None, :] - means[None, :, :]) ** 2, axis=2)
        a = mx.argmin(d, axis=1)
        r0 = Utils.nonzero(a == 0).tolist()
        r1 = Utils.nonzero(a == 1).tolist()
        return r0, r1

# ── 在线学习 (软路由吸收 + 显著性延迟生长) ────────────────────────


class OnlineSPN:
    """在线 SPN 学习器: 软路由吸收 + 显著性延迟生长 (追平全量的架构)。

    与批量的理论关系:
      * 参数层精确 —— 叶 (n,μ,M2 中心矩, Chan 合并) / 码联合计数 /
        Sum 计数全是可加充分统计量, 同结构同数据下在线 = 批量 MLE
        (自检 6 验证);
      * 软路由 —— 每行按叶后验 (路径先验 × 叶似然) 分配 responsibility,
        EM 语义: 分组/分裂错误被后续数据自校正, 无硬路由反馈环;
      * 延迟生长 —— 码计数达 N_SPLIT 才分裂 (分组稳定性), 检查点按计数
        翻倍 (VFDT grace period 对应物); 子叶继承父叶高斯参数作伪计数
        先验, 抗早承诺过拟合。
    边界 (scope 一致性使在线不可局部执行, 留给周期全局修订):
      Product (变量独立) 分裂改 scope; 特征驱动 Sum 分裂需行级 k-means。
    契约: 所有 CatLeaf 属于 code_cols (查询变量驱动设定, 见 demo_inverse)。
    """

    N_SPLIT: ClassVar[float] = 1.0  # 码计数下限: 码分裂语义上无风险
    # (查询变量驱动: 不同码本来就是不同分量), 早分裂由软路由自校正
    N_FIRST: ClassVar[float] = 8.0  # 首次生长检查的总计数下限
    PSEUDO: ClassVar[float] = 0.05  # 子叶继承父叶高斯统计的伪计数比例

    def __init__(
        self,
        root: Node,
        n_vars: int,
        code_cols: tuple[int, ...],
        cards: tuple[int, ...],
        sigma_floor: float = 1e-6,
    ):
        self.n_vars = n_vars
        self.code_cols = code_cols
        self.cards = cards
        self.ncodes = math.prod(cards)
        self.sigma_floor = sigma_floor
        self.root = self.init_stats(root)
        self.tables: dict[tuple[int, ...], mx.array] = {}  # 叶路径 → 码联合计数
        self.last_check: dict[tuple[int, ...], float] = {}
        for node, path in self.root.leaf_blocks():
            for lf in self.leaves_of(node):
                assert not isinstance(lf, CatLeaf) or lf.var in code_cols, (
                    f"CatLeaf var {lf.var} 非码列 (契约外)"
                )
            self.tables[path] = mx.zeros(self.ncodes)
            self.last_check[path] = 0.0

    @staticmethod
    def leaves_of(node: Node) -> list[Node]:
        """叶块 → 组成叶列表。"""
        if isinstance(node, (GaussLeaf, CatLeaf)):
            return [node]
        assert isinstance(node, Product)
        return list(node.children)

    @staticmethod
    def init_stats(node: Node) -> Node:
        """Sum 补零计数 (叶统计字段有零默认值), 递归重建。"""
        if node.is_leaf_block():
            return node
        if isinstance(node, Sum):
            children = tuple(OnlineSPN.init_stats(c) for c in node.children)
            counts = node.counts
            if counts is None:
                counts = mx.zeros(len(children))
            return Sum(children, node.log_w, counts)
        assert isinstance(node, Product)
        return Product(tuple(OnlineSPN.init_stats(c) for c in node.children))

    def to_spn(self) -> SPN:
        """当前树 → 不可变推理对象。"""
        return SPN(self.root, self.n_vars)

    def code_index(self, X: mx.array) -> mx.array:
        """行 → 联合码下标 (字典序, 与 cards 顺序一致)。"""
        idx = mx.zeros(X.shape[0], dtype=mx.int32)
        for c, k in zip(self.code_cols, self.cards):
            idx = idx * k + X[:, c].astype(mx.int32)
        return idx

    def path_logpi(self) -> dict[tuple[int, ...], float]:
        """每叶块的路径先验 log π = 沿途 Sum 权重之积 (log)。"""
        out: dict[tuple[int, ...], float] = {}

        def walk(node: Node, path: tuple[int, ...], acc: float) -> None:
            if node.is_leaf_block():
                out[path] = acc
                return
            if isinstance(node, Sum):
                w = node.log_w.tolist()
                for i, c in enumerate(node.children):
                    walk(c, path + (i,), acc + w[i])
            else:
                assert isinstance(node, Product)
                for i, c in enumerate(node.children):
                    walk(c, path + (i,), acc)

        walk(self.root, (), 0.0)
        return out

    def absorb(self, X: mx.array, grow: bool = True) -> None:
        """吸收一批样本 (含码列的全列布局): 软分配 → 统计累加 → 参数刷新
        → 显著性生长。X (B, V) 列布局与 learn_spn 输入一致。"""
        cidx = self.code_index(X)
        leaves = self.root.leaf_blocks()
        pis = self.path_logpi()
        cols = []
        for node, path in leaves:
            lp = node.eval_log(X) + pis[path]
            mx.eval(lp)  # 逐叶求值, 防长惰性图超 Metal 显存
            cols.append(lp)
        logr = mx.stack(cols, axis=1)  # (B, L) log 责任度
        r = mx.exp(logr - mx.logsumexp(logr, axis=1, keepdims=True))
        r = mx.where(r < 1e-6, 0.0, r)  # 截断: 稀疏化, 防全图稀释
        r = r / mx.sum(r, axis=1, keepdims=True)
        mx.eval(r)
        deltas: dict[
            tuple[int, ...], tuple[float, list[float] | None, list[float] | None]
        ] = {}
        rmap: dict[tuple[int, ...], mx.array] = {}  # 叶路径 → 本批责任度 (生长播种用)
        for j, (node, path) in enumerate(leaves):
            rj = r[:, j]
            dn = float(mx.sum(rj))
            if dn < 1e-8:
                continue
            rmap[path] = rj
            self.tables[path] = self.tables[path].at[cidx].add(rj)
            gs = [lf for lf in self.leaves_of(node) if isinstance(lf, GaussLeaf)]
            if not gs:
                deltas[path] = (dn, None, None)
                continue
            xf = X[:, mx.array([g.var for g in gs], dtype=mx.int32)]
            w = rj[:, None]
            bmu = mx.sum(w * xf, axis=0) / dn
            bvar = mx.sum(w * (xf - bmu) * (xf - bmu), axis=0) / dn  # 两遍法
            mx.eval(bmu, bvar)
            deltas[path] = (dn, bmu.tolist(), bvar.tolist())
        self.root = self.refresh(self.root, (), deltas)[0]
        if grow:
            self.maybe_grow(X, cidx, rmap)

    def refresh(
        self,
        node: Node,
        path: tuple[int, ...],
        deltas: dict[
            tuple[int, ...], tuple[float, list[float] | None, list[float] | None]
        ],
    ) -> tuple[Node, float]:
        """自底向上用累积统计量重建参数, 返回 (新节点, 本批流入质量)。"""
        if node.is_leaf_block():
            d = deltas.get(path)
            if d is None:
                return node, 0.0
            dn, ds1, ds2 = d
            tm = self.tables[path].reshape(self.cards)
            new_leaves: list[Node] = []
            gi = 0
            for lf in self.leaves_of(node):
                if isinstance(lf, GaussLeaf):
                    assert ds1 is not None and ds2 is not None
                    # Chan 并行合并: (n,μ,M2) ⊕ (dn,bμ,bvar) — 数值稳定
                    n = lf.n + dn
                    delta = ds1[gi] - lf.mu
                    mu = lf.mu + delta * dn / n
                    m2 = lf.m2 + dn * ds2[gi] + delta * delta * lf.n * dn / n
                    gi += 1
                    var = max(m2 / n, 0.0)
                    sig = max(math.sqrt(var), self.sigma_floor)
                    new_leaves.append(GaussLeaf(lf.var, mu, sig, n, m2))
                else:
                    assert isinstance(lf, CatLeaf)
                    col = self.code_cols.index(lf.var)
                    k = self.cards[col]
                    axes = tuple(i for i in range(len(self.cards)) if i != col)
                    cnt = mx.sum(tm, axis=axes)  # 码联合计数边缘化
                    logp = mx.log((cnt + 1.0) / (float(mx.sum(cnt)) + k))
                    new_leaves.append(CatLeaf(lf.var, logp, cnt))
            out = new_leaves[0] if len(new_leaves) == 1 else Product(tuple(new_leaves))
            return out, dn
        if isinstance(node, Sum):
            children = []
            masses = []
            for i, c in enumerate(node.children):
                nc, m = self.refresh(c, path + (i,), deltas)
                children.append(nc)
                masses.append(m)
            assert node.counts is not None
            counts = node.counts + mx.array(masses, dtype=mx.float32)
            total = float(mx.sum(counts))
            # Laplace 平滑: 软路由下饿死子节点不永久归零
            log_w = mx.log((counts + 1.0) / (total + len(masses)))
            return Sum(tuple(children), log_w, counts), sum(masses)
        assert isinstance(node, Product)
        children = []
        masses = []
        for i, c in enumerate(node.children):
            nc, m = self.refresh(c, path + (i,), deltas)
            children.append(nc)
            masses.append(m)
        # Product 全子节点行集相同 → 质量相等, 取其一
        return Product(tuple(children)), (masses[0] if masses else 0.0)

    def maybe_grow(
        self,
        X: mx.array,
        cidx: mx.array,
        rmap: dict[tuple[int, ...], mx.array],
    ) -> None:
        """显著性延迟生长: 计数翻倍检查点, 码支撑 ≥2 → 码空间加权分裂。
        子叶高斯用本批行按码分组当即播种 (伪先验 + 组内加权统计),
        避免分裂后子叶参数雷同、需再等一批才分化。"""
        for path in sorted(list(self.tables)):
            t = self.tables[path]
            n_tot = float(mx.sum(t))
            # 1e-3 质量裕量: float32 表和与 float64 last_check 的 rounding
            # 尘会误触发 "n_tot > last_check" (实测 KeyError 根因)
            if n_tot < self.N_FIRST or n_tot <= self.last_check[path] + 1e-3:
                continue
            self.last_check[path] = n_tot
            node = self.root.node_at(path)
            if not any(isinstance(lf, CatLeaf) for lf in self.leaves_of(node)):
                continue  # 纯特征叶: 无码可判 (特征分裂需行级数据, 不做)
            support = Utils.nonzero(t >= self.N_SPLIT)
            if len(support) < 2:
                continue
            assign = self.code_kmeans2(t)
            if assign is None:
                continue
            mask0 = mx.zeros(self.ncodes).at[Utils.nonzero(assign == 0)].add(1.0)
            n0 = float(mx.sum(t * mask0))
            n1 = n_tot - n0
            if min(n0, n1) < self.N_SPLIT:
                continue  # 退化分组: 数据更多时再试
            sub = self.split_block(
                node, t, mask0, n0, n1, X, cidx, assign, rmap.get(path)
            )
            self.root = self.root.replace_leaf(path, sub)
            del self.tables[path]
            del self.last_check[path]
            self.tables[path + (0,)] = t * mask0
            self.tables[path + (1,)] = t * (1.0 - mask0)
            self.last_check[path + (0,)] = n0
            self.last_check[path + (1,)] = n1

    def code_coords(self, codes: mx.array) -> mx.array:
        """码下标 → 网格坐标 (各列按均匀网格均值/标准差 z-score, 与数据无关)。"""
        cols = []
        rem = codes
        for k in reversed(self.cards):
            cols.append((rem % k).astype(mx.float32))
            rem = rem // k
        cols.reverse()
        pts = mx.stack(cols, axis=1)
        mu = mx.array([(k - 1) / 2 for k in self.cards])
        sd = mx.array([math.sqrt((k * k - 1) / 12) for k in self.cards])
        return (pts - mu) / sd

    def code_kmeans2(self, t: mx.array) -> mx.array | None:
        """码空间加权 k=2 聚类 (最远对初始化) → (ncodes,) 0/1 分配
        (零计数码给 0); 退化 (单点/空簇) → None。"""
        codes = Utils.nonzero(t > 0)
        if len(codes) < 2:
            return None
        pts = self.code_coords(codes)  # (K, 5)
        w = t[codes]
        i0 = int(mx.argmax(mx.sum(pts * pts, axis=1)))
        i1 = int(mx.argmax(mx.sum((pts - pts[i0]) ** 2, axis=1)))
        means = mx.stack([pts[i0], pts[i1]])
        a = mx.zeros(len(codes), dtype=mx.int32)
        for _ in range(12):
            d = mx.sum((pts[:, None, :] - means[None, :, :]) ** 2, axis=2)
            a = mx.argmin(d, axis=1)
            new_means = []
            for g in (0, 1):
                mg = mx.where(a == g, w, 0.0)
                sg = float(mx.sum(mg))
                if sg <= 0:
                    return None  # 空簇退化
                new_means.append(mx.sum(mg[:, None] * pts, axis=0) / sg)
            means = mx.stack(new_means)
        return mx.zeros(self.ncodes, dtype=mx.int32).at[codes].add(a)

    def split_block(
        self,
        node: Node,
        t: mx.array,
        mask0: mx.array,
        n0: float,
        n1: float,
        X: mx.array,
        cidx: mx.array,
        assign: mx.array,
        rj: mx.array | None,
    ) -> Sum:
        """叶块按码分组劈成 Sum(两个新叶块): 码计数从联合表精确分割;
        高斯 = 父叶伪计数先验 (PSEUDO × 组计数) + 本批行按码分组的
        加权统计当即播种 —— 分裂前的混合特征证据按方向保留、按量折扣,
        防陈旧混合量拖拽; 播种让子叶当批即分化。"""
        leaves = self.leaves_of(node)
        gs = [lf for lf in leaves if isinstance(lf, GaussLeaf)]
        grow = assign[cidx]  # (B,) 本批行的组分配
        children: list[Node] = []
        for g, (mask, ng) in enumerate(((mask0, n0), (1.0 - mask0, n1))):
            tg = (t * mask).reshape(self.cards)
            pseudo = max(1.0, self.PSEUDO * ng)
            # rj=None: 本批无流入 (rounding 尘触发的检查) → 纯先验
            wg = rj * (grow == g) if rj is not None else None
            dn = float(mx.sum(wg)) if wg is not None else 0.0
            ds1: list[float] = []
            ds2: list[float] = []
            if gs and wg is not None and dn > 1e-8:
                xf = X[:, mx.array([lf.var for lf in gs], dtype=mx.int32)]
                w2 = wg[:, None]
                bmu = mx.sum(w2 * xf, axis=0) / dn
                bvar = mx.sum(w2 * (xf - bmu) * (xf - bmu), axis=0) / dn
                mx.eval(bmu, bvar)
                ds1, ds2 = bmu.tolist(), bvar.tolist()
            gi = 0
            new_leaves: list[Node] = []
            for lf in leaves:
                if isinstance(lf, GaussLeaf):
                    if dn > 1e-8:
                        # 先验 (pseudo, μ父, σ父²) ⊕ 本批组内统计 (Chan)
                        n = pseudo + dn
                        delta = ds1[gi] - lf.mu
                        mu = lf.mu + delta * dn / n
                        m2 = (
                            pseudo * lf.sigma * lf.sigma
                            + dn * ds2[gi]
                            + delta * delta * pseudo * dn / n
                        )
                    else:  # 本批无本组播种行: 纯先验
                        n, mu, m2 = pseudo, lf.mu, pseudo * lf.sigma * lf.sigma
                    gi += 1
                    var = max(m2 / n, 0.0)
                    sig = max(math.sqrt(var), self.sigma_floor)
                    new_leaves.append(GaussLeaf(lf.var, mu, sig, n, m2))
                else:
                    assert isinstance(lf, CatLeaf)
                    col = self.code_cols.index(lf.var)
                    k = self.cards[col]
                    axes = tuple(i for i in range(len(self.cards)) if i != col)
                    cnt = mx.sum(tg, axis=axes)
                    logp = mx.log((cnt + 1.0) / (float(mx.sum(cnt)) + k))
                    new_leaves.append(CatLeaf(lf.var, logp, cnt))
            children.append(
                new_leaves[0] if len(new_leaves) == 1 else Product(tuple(new_leaves))
            )
        counts = mx.array([n0, n1], dtype=mx.float32)
        log_w = mx.log((counts + 1.0) / (n0 + n1 + 2.0))
        return Sum((children[0], children[1]), log_w, counts)


# ── 自检 ──────────────────────────────────────────────────────────


class SelfTest:
    """spn.py 自检 (python src/spn.py): 7 组。"""

    @staticmethod
    def run() -> None:
        key = mx.random.key(7)

        # 1) G 检验: 独立对 → 不相关; 相关对 → 相关 (α=0.05)
        n = 500
        a = mx.random.normal(shape=(n,), key=key)
        b = mx.random.normal(shape=(n,), key=mx.random.key(8))
        dep = SPNLearner.gtest(
            SPNLearner.binarize(mx.stack([a, b], axis=1), [0, 1], set())
        )
        assert not bool(dep[0, 1].item()), "G: 独立变量误判相关"
        c = a + 0.15 * mx.random.normal(shape=(n,), key=mx.random.key(9))
        dep2 = SPNLearner.gtest(
            SPNLearner.binarize(mx.stack([a, c], axis=1), [0, 1], set())
        )
        assert bool(dep2[0, 1].item()), "G: 相关变量误判独立"
        print("  ok  G 检验: 独立→独立, 相关→相关")

        # 2) 独立变量 → 根为 Product, log 密度 = 边缘对数之和
        x = mx.random.normal(shape=(4000, 3), key=key)
        spn = SPNLearner(set(), min_n=64).learn(x)
        assert isinstance(spn.root, Product), (
            f"根应为 Product, 实际 {type(spn.root).__name__}"
        )
        pt = mx.array([[0.3, -0.7, 1.1]])
        got = float(spn.eval_log(pt)[0])
        want = sum(
            -0.5 * v * v - 0.5 * math.log(2.0 * math.pi) for v in (0.3, -0.7, 1.1)
        )
        # n=4000 → 叶参数 MLE 误差 ~1/sqrt(2n)≈0.011, 容差 0.05 富余
        assert abs(got - want) < 0.05, (got, want)
        print("  ok  独立结构: 根 Product, log 密度 = 边缘乘积")

        # 3) 混合 + 离散标签 → 后验从连续证据恢复类
        n = 400
        lab = mx.concatenate([mx.zeros((n // 2,)), mx.ones((n // 2,))])
        f = lab * 4.0 + mx.random.normal(shape=(n,), key=key)
        spn3 = SPNLearner({1}, card={1: 2}, min_n=16).learn(mx.stack([f, lab], axis=1))
        feats = mx.array([[-4.0], [0.0], [4.0]])
        codes = mx.array([[0.0], [1.0]])
        post = spn3.posterior(feats, codes)  # (3, 2)
        assert float(post[0, 1]) < math.log(0.01), "x=−4 应属类 0"
        assert float(post[2, 1]) > math.log(0.99), "x=+4 应属类 1"
        assert abs(float(mx.exp(post[1]).sum()) - 1.0) < 1e-5, "后验行未归一"
        print("  ok  混合后验: 类标签从连续证据恢复, 行归一")

        # 4) 码先验注入: P(c|x) ∝ P(x|c)·P(c), 先验改变后验但保持归一
        prior = mx.array([math.log(0.9), math.log(0.1)])  # 强偏好类 0
        post_p = spn3.posterior(feats, codes, log_prior=prior)
        norm = float(mx.exp(post_p).sum(axis=1)[1])
        assert abs(norm - 1.0) < 1e-5, "先验注入后未归一"
        # x=0 处似然两分类相近, 先验应把后验推向类 0
        assert float(post_p[1, 0]) > float(post[1, 0]), "先验未提高类 0 后验"
        print("  ok  码先验: P(c|x) ∝ P(x|c)·P(c), 注入后行归一仍成立")

        # 5) 序列化 roundtrip (safetensors): save → load → eval 逐位一致
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            spn3.save(tmp, {"mu": mx.array([0.5]), "sd": mx.array([1.0])})
            spn4, extra = SPN.load(tmp)
            xq = mx.array([[-4.0, 0.0], [4.0, 1.0]])
            a = spn3.eval_log(xq)
            b = spn4.eval_log(xq)
            assert mx.all(mx.abs(a - b) < 1e-6), "roundtrip 后 eval 不一致"
            assert float(extra["mu"][0]) == 0.5, "extra 未随模型保存"
        finally:
            os.unlink(tmp)
        print("  ok  序列化: save → load → eval 一致, extra 随存")

        # 6) 在线参数等价: 同结构同数据, OnlineSPN 吸收 ≈ learn_spn MLE
        on = OnlineSPN(spn3.root, n_vars=2, code_cols=(1,), cards=(2,))
        x6 = mx.stack([f, lab], axis=1)
        on.absorb(x6, grow=False)
        xq = mx.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 1.0]])
        d = float(mx.max(mx.abs(spn3.eval_log(xq) - on.to_spn().eval_log(xq))))
        assert d < 0.05, f"在线参数应≈批量 MLE: {d}"
        print(f"  ok  在线等价: 同结构同数据, |Δeval| = {d:.2e}")

        # 7) 生长: 码混合叶 + 计数显著 → 分裂, 后验仍恢复类
        # (打乱行序: 两批都含两类, 模拟真实增量; 有序喂入会让后分裂的
        # 子叶当批无本类播种行, 继承的混合高斯残留 —— 实验数据本就打乱)
        base7 = SPNLearner({1}, card={1: 2}, min_n=300).learn(x6)  # 强制浅树
        assert len(base7.root.leaf_blocks()) == 1, "应为单叶块 (码混合)"
        perm = mx.random.permutation(x6.shape[0], key=mx.random.key(5))
        x7 = x6[perm]
        on7 = OnlineSPN(base7.root, n_vars=2, code_cols=(1,), cards=(2,))
        on7.absorb(x7[:200])
        on7.absorb(x7[200:])
        assert len(on7.root.leaf_blocks()) == 2, "码混合叶应已分裂"
        post7 = on7.to_spn().posterior(feats, codes)
        assert float(post7[0, 1]) < math.log(0.01), "生长后 x=−4 应属类 0"
        assert float(post7[2, 1]) > math.log(0.99), "生长后 x=+4 应属类 1"
        print("  ok  在线生长: 码混合叶分裂, 后验仍恢复类")


if __name__ == "__main__":
    SelfTest.run()
    print("spn.py: 7 组自检 ✓")
