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

契约 (inverse_app.py 消费):
    X 列布局 = [连续特征列 | 离散码列]; disc_cols = 离散列下标集合;
    card[col] = 离散列基数 (缺省从数据取 max+1)。
    spn.posterior(feats, codes): feats (M, Vf) 连续观测, codes (K, C)
    全枚举离散码 → (M, K) log 后验, 行和归一。

本文件自检: `python src/spn_selftest.py` (7 组, 见 spn_selftest.py)。

ponytail: 参数=硬分裂 MLE (k-means 子集内拟合), 未做 EM 精修 ——
结构对、后验单调性对的前提下, EM 只提升密度估计不改变 MAP; 加
EM 当 demo 出明显欠拟合时。
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import mlx.core as mx

from cat_leaf import CatLeaf
from gauss_leaf import GaussLeaf
from node import Node
from product import Product
from sum_node import Sum
from utils import Utils


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
    def node_from_records(d: dict[str, mx.array]) -> Node:
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
        return SPN(SPN.node_from_records(d), int(cfg["n_vars"])), extra
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
