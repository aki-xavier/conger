"""Sum-Product Network (SPN, Gens & Domingos 2013 风格) — 可求和边缘化的生成混合模型。

结构: 叶 (单变量高斯 / 分类) / Product (变量分解) / Sum (行混合)。

学习 (learn_spn) 贪心递归:
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

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from utils import Utils

# 连续变量分位数 bin 数; 与离散变量共用 one-hot 宽度 = 最大离散基数
_NBINS = 8
_NBINS_CONT = 5

# χ² 临界值 (α=0.05), dof 1..16 精确表; dof>16 用 Wilson-Hilferty
# 近似 (z=1.6449, 误差 <2%) —— 离散-离散对 dof 可达 (8−1)²=49。
# 硬编码避免 scipy 依赖。
_CHI2_05: tuple[float, ...] = (
    3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507,
    16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
)


# ── 节点 ──────────────────────────────────────────────────────────


class Node:
    """SPN 节点基类。eval_log(x) 对证据批 (M, V) 求 log 密度 (M,)。"""

    def eval_log(self, x: mx.array) -> mx.array:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussLeaf(Node):
    """单变量高斯叶 (连续列)。"""

    var: int
    mu: float
    sigma: float

    def eval_log(self, x: mx.array) -> mx.array:
        v = x[:, self.var]
        z = (v - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class CatLeaf(Node):
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


@dataclass(frozen=True)
class Sum(Node):
    """行混合: log 密度 = logsumexp(log_w + 子节点 log 密度)。"""

    children: tuple[Node, ...]
    log_w: mx.array

    def eval_log(self, x: mx.array) -> mx.array:
        evals = mx.stack([c.eval_log(x) for c in self.children])
        return mx.logsumexp(evals + self.log_w[:, None], axis=0)


class SPN:
    """学习好的 SPN: 根节点 + 变量布局 (连续列 | 离散码列)。"""

    def __init__(self, root: Node, n_vars: int):
        self.root = root
        self.n_vars = n_vars

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """pickle 存盘 (节点全是 dataclass, mx.array 可序列化)。

        extra: 与模型配套的非 SPN 状态 (如特征 z-score 统计 mu/sd),
        加载时原样返回 —— 推理与训练必须用同一预处理。
        """
        with open(path, "wb") as f:
            pickle.dump({"spn": self, "extra": extra or {}}, f)

    @staticmethod
    def load(path: str | Path) -> tuple[SPN, dict[str, Any]]:
        """save 的逆操作 → (SPN, extra)。"""
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["spn"], d["extra"]

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
                parts.append(f"σmed={sig[len(sig)//2]:.3f}")
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
                    pad + "Sum w=" + ",".join(f"{x:.3f}" for x in w)
                    + "  " + split_axis(reps)
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


def learn_spn(
    X: mx.array,
    disc_cols: set[int],
    card: dict[int, int] | None = None,
    min_n: int = 24,
    max_depth: int = 12,
    alpha: float = 0.05,
    sigma_floor: float = 1e-6,
) -> SPN:
    """X (N, V) float32。disc_cols: 离散列 (分类叶/查询变量); card:
    离散列基数 (缺省从数据取 max+1)。sigma_floor: 高斯叶 σ 下限
    (平滑性先验, 见 prior.md 紧凑性/平滑性) —— 防确定性渲染 σ→0
    过拟合, 是 MAP 正则 (隐式 Gaussian prior on σ)。返回 SPN。"""
    n, v = X.shape
    if card is None:
        card = {c: int(mx.max(X[:, c])) + 1 for c in disc_cols}
    rows = list(range(n))
    cols = list(range(v))
    root = _learn(
        X, rows, cols, 0, disc_cols, card, min_n, max_depth, alpha, sigma_floor
    )
    return SPN(root, v)


def _learn(
    X: mx.array,
    rows: list[int],
    cols: list[int],
    depth: int,
    disc_cols: set[int],
    card: dict[int, int],
    min_n: int,
    max_depth: int,
    alpha: float,
    sigma_floor: float,
) -> Node:
    n, c = len(rows), len(cols)
    if c == 1 or n < min_n or depth >= max_depth:
        return _diag(X, rows, cols, disc_cols, card, sigma_floor)
    xr = X[mx.array(rows, dtype=mx.int32)]

    # 查询变量驱动: 节点含离散列且行码混杂 → Sum 按码空间分裂
    code_cols = [cc for cc in cols if cc in disc_cols]
    if code_cols and _rows_code_mixed(xr, code_cols):
        r0, r1 = _split_rows(xr, code_cols)
        r0 = [rows[i] for i in r0]  # 局部下标 → 全局行号
        r1 = [rows[i] for i in r1]
        if len(r0) < min_n or len(r1) < min_n:
            return _diag(X, rows, cols, disc_cols, card, sigma_floor)
        n0, n1 = len(r0), len(r1)
        log_w = mx.log(mx.array([n0 / n, n1 / n], dtype=mx.float32))
        return Sum(
            (
                _learn(
                    X, r0, cols, depth + 1,
                    disc_cols, card, min_n, max_depth, alpha, sigma_floor,
                ),
                _learn(
                    X, r1, cols, depth + 1,
                    disc_cols, card, min_n, max_depth, alpha, sigma_floor,
                ),
            ),
            log_w,
        )

    dep = _gtest(_bin(xr, cols, disc_cols), alpha)
    comps = _components(dep)
    if len(comps) > 1:
        # 变量可分: 依赖图连通分量各自建模, 全部行共享
        return Product(
            tuple(
                _learn(
                    X, rows, [cols[i] for i in comp], depth + 1,
                    disc_cols, card, min_n, max_depth, alpha, sigma_floor,
                )
                for comp in comps
            )
        )
    r0, r1 = _split_rows(xr, cols)
    r0 = [rows[i] for i in r0]  # 局部下标 → 全局行号
    r1 = [rows[i] for i in r1]
    if len(r0) < min_n or len(r1) < min_n:
        return _diag(X, rows, cols, disc_cols, card, sigma_floor)
    n0, n1 = len(r0), len(r1)
    log_w = mx.log(mx.array([n0 / n, n1 / n], dtype=mx.float32))
    return Sum(
        (
            _learn(
                X, r0, cols, depth + 1, disc_cols, card, min_n, max_depth,
                alpha, sigma_floor,
            ),
            _learn(
                X, r1, cols, depth + 1, disc_cols, card, min_n, max_depth,
                alpha, sigma_floor,
            ),
        ),
        log_w,
    )


def _rows_code_mixed(xr: mx.array, code_cols: list[int]) -> bool:
    """行集在码列上是否混杂 (存在 >1 个不同码)。"""
    for j in code_cols:
        vals = xr[:, j]
        distinct = len({int(v) for v in vals.tolist()})
        if distinct > 1:
            return True
    return False


# ── 结构增量辅助 (叶收集 / 叶替换) ──────────────────────────────


def leaf_blocks(
    node: Node, path: tuple[int, ...] = ()
) -> list[tuple[Node, tuple[int, ...]]]:
    """收集所有叶块 (对角乘积) 及其从根的路径 (子索引序列)。

    用于增量学习: 叶块 = 可独立生长的结构单元。"""
    if isinstance(node, (GaussLeaf, CatLeaf)):
        return [(node, path)]
    if isinstance(node, Product) and all(
        isinstance(c, (GaussLeaf, CatLeaf)) for c in node.children
    ):
        return [(node, path)]
    out: list[tuple[Node, tuple[int, ...]]] = []
    if isinstance(node, Sum):
        for i, c in enumerate(node.children):
            out.extend(leaf_blocks(c, path + (i,)))
    else:
        assert isinstance(node, Product)  # 非叶 Product: 全部分量
        for i, c in enumerate(node.children):
            out.extend(leaf_blocks(c, path + (i,)))
    return out


def replace_leaf(
    root: Node, path: tuple[int, ...], new: Node
) -> Node:
    """沿 path 重建树, 把叶替换为 new (frozen dataclass 不可变 →
    路径上节点逐层重建)。结构增量: 叶缓冲满 → 局部子树生长。"""
    if not path:
        # 被替换目标必须是叶块 (叶或全叶 Product): 防路径漂移
        # (树中间层重排导致替换到错误节点会静默腐蚀结构)
        is_leaf_block = isinstance(root, (GaussLeaf, CatLeaf)) or (
            isinstance(root, Product)
            and all(isinstance(c, (GaussLeaf, CatLeaf)) for c in root.children)
        )
        assert is_leaf_block, f"替换目标非叶块: {type(root).__name__}"
        return new
    i, *rest = path
    rest_t = tuple(rest)
    if isinstance(root, Sum):
        children = list(root.children)
        children[i] = replace_leaf(children[i], rest_t, new)
        return Sum(tuple(children), root.log_w)
    if isinstance(root, Product):
        children = list(root.children)
        children[i] = replace_leaf(children[i], rest_t, new)
        return Product(tuple(children))
    return root


def _diag(
    X: mx.array,
    rows: list[int],
    cols: list[int],
    disc_cols: set[int],
    card: dict[int, int],
    sigma_floor: float,
) -> Node:
    """基例: 每列一个叶 (对角)。离散列分类叶, 连续列高斯叶。"""
    xr = X[mx.array(rows, dtype=mx.int32)]
    leaves: list[Node] = []
    for c in cols:
        v = xr[:, c]
        if c in disc_cols:
            k = card[c]
            cnt = mx.sum(
                mx.equal(v[:, None], mx.arange(k)[None, :]).astype(mx.float32), axis=0
            )
            # Laplace: 未见类别保底, 防后验硬零
            logp = mx.log((cnt + 1.0) / (float(mx.sum(cnt)) + k))
            leaves.append(CatLeaf(c, logp, cnt))
        else:
            sd = float(mx.maximum(mx.std(v), sigma_floor))
            leaves.append(GaussLeaf(c, float(mx.mean(v)), sd))
    if len(leaves) == 1:
        return leaves[0]
    return Product(tuple(leaves))


# ── 独立性检验 (G 检验) ──────────────────────────────────────────


def _bin(xr: mx.array, cols: list[int], disc_cols: set[int]) -> mx.array:
    """列子集 → (n, c) int32 bin id: 连续列分位数分 5 档, 离散列取原值。"""
    parts = []
    for c in cols:
        v = xr[:, c]
        if c in disc_cols:
            parts.append(v.astype(mx.int32))
        else:
            parts.append(_quant_bins(v, _NBINS_CONT))
    return mx.stack(parts, axis=1)


def _quant_bins(v: mx.array, nbins: int) -> mx.array:
    """分位数分箱: 阈值取排序值 i·(n−1)/nbins 处, bin = Σ (v > t_i)。"""
    n = v.shape[0]
    sv = mx.sort(v)
    thr = [sv[min(n - 1, int(round(i * (n - 1) / nbins)))] for i in range(1, nbins)]
    t = mx.stack(thr)[None, :]
    return mx.sum(v[:, None] > t, axis=1).astype(mx.int32)


def _gtest(binned: mx.array, alpha: float = 0.05) -> mx.array:
    """两两 G 检验 → (c, c) bool 依赖矩阵 (对称, 对角线 False)。

    向量化: one-hot (n,c,B) → 联合计数 J = AᵀA (cB,cB) → (c,c,B,B);
    G = 2·Σ n_ab·ln(n_ab·N/(n_a·n_b)), dof = (k_i−1)(k_j−1) (k = 实际
    非空 bin 数), 超 χ² 临界值 (α) 即依赖。零计数用掩码归零。
    """
    n, c = binned.shape
    b = _NBINS
    oh = mx.equal(binned[:, :, None], mx.arange(b)[None, None, :]).astype(mx.float32)
    a = oh.reshape(n, c * b)
    jp = (a.T @ a).reshape(c, b, c, b).transpose(0, 2, 1, 3)  # (c,c,B,B)
    m = mx.sum(oh, axis=0)  # (c, B) 边缘计数
    logn = math.log(n)
    inner = mx.log(mx.maximum(jp, 1.0)) + logn - mx.log(
        mx.maximum(m[:, None, :, None], 1.0)
    ) - mx.log(mx.maximum(m[None, :, None, :], 1.0))
    term = mx.where(jp > 0, jp * inner, 0.0)
    g = 2.0 * mx.sum(term, axis=(2, 3))  # (c, c)
    k = mx.sum(m > 0, axis=1)  # (c,) 每列非空 bin 数
    dof = (k[:, None] - 1) * (k[None, :] - 1)  # (c,c) float
    dof = mx.maximum(dof, 0.0)
    thr_tab = mx.take(
        mx.array(_CHI2_05, dtype=mx.float32),
        mx.clip(dof - 1, 0, 15).astype(mx.int32),
    )
    # dof>16: Wilson-Hilferty χ²_0.95(d) ≈ d·(1−2/9d+z·√(2/9d))³
    # (ddof 钳 1 防 dof=0 除零 —— 该分支会被 where 丢弃但 NaN 不雅)
    z = 1.6448536
    ddof = mx.maximum(dof, 1.0)
    thr_wh = ddof * (1.0 - 2.0 / (9.0 * ddof) + z * mx.sqrt(2.0 / (9.0 * ddof))) ** 3
    thr = mx.where(dof <= 16, thr_tab, thr_wh)
    return (g > thr) & (dof >= 1)


def _components(dep: mx.array) -> list[list[int]]:
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


# ── 行分裂 (k-means) ─────────────────────────────────────────────


def _split_rows(
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


# ── 自检 ──────────────────────────────────────────────────────────


def _selftest() -> None:
    key = mx.random.key(7)

    # 1) G 检验: 独立对 → 不相关; 相关对 → 相关 (α=0.05)
    n = 500
    a = mx.random.normal(shape=(n,), key=key)
    b = mx.random.normal(shape=(n,), key=mx.random.key(8))
    dep = _gtest(_bin(mx.stack([a, b], axis=1), [0, 1], set()))
    assert not bool(dep[0, 1].item()), "G: 独立变量误判相关"
    c = a + 0.15 * mx.random.normal(shape=(n,), key=mx.random.key(9))
    dep2 = _gtest(_bin(mx.stack([a, c], axis=1), [0, 1], set()))
    assert bool(dep2[0, 1].item()), "G: 相关变量误判独立"
    print("  ok  G 检验: 独立→独立, 相关→相关")

    # 2) 独立变量 → 根为 Product, log 密度 = 边缘对数之和
    x = mx.random.normal(shape=(4000, 3), key=key)
    spn = learn_spn(x, set(), min_n=64)
    assert isinstance(spn.root, Product), (
        f"根应为 Product, 实际 {type(spn.root).__name__}"
    )
    pt = mx.array([[0.3, -0.7, 1.1]])
    got = float(spn.eval_log(pt)[0])
    want = sum(-0.5 * v * v - 0.5 * math.log(2.0 * math.pi) for v in (0.3, -0.7, 1.1))
    # n=4000 → 叶参数 MLE 误差 ~1/sqrt(2n)≈0.011, 容差 0.05 富余
    assert abs(got - want) < 0.05, (got, want)
    print("  ok  独立结构: 根 Product, log 密度 = 边缘乘积")

    # 3) 混合 + 离散标签 → 后验从连续证据恢复类
    n = 400
    lab = mx.concatenate([mx.zeros((n // 2,)), mx.ones((n // 2,))])
    f = lab * 4.0 + mx.random.normal(shape=(n,), key=key)
    spn3 = learn_spn(mx.stack([f, lab], axis=1), disc_cols={1}, card={1: 2}, min_n=16)
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
    assert abs(float(mx.exp(post_p).sum(axis=1)[1]) - 1.0) < 1e-5, "先验注入后未归一"
    # x=0 处似然两分类相近, 先验应把后验推向类 0
    assert float(post_p[1, 0]) > float(post[1, 0]), "先验未提高类 0 后验"
    print("  ok  码先验: P(c|x) ∝ P(x|c)·P(c), 注入后行归一仍成立")

    # 5) 序列化 roundtrip: save → load → eval 逐位一致
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".pkl")
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


if __name__ == "__main__":
    _selftest()
    print("spn.py: 5 组自检 ✓")
