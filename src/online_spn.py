"""OnlineSPN: 在线 SPN 学习器 (软路由吸收 + 显著性延迟生长)。

理论与边界说明见类 docstring 与 spn.py 模块 docstring。
"""

from __future__ import annotations

import math
from typing import ClassVar

import mlx.core as mx

from cat_leaf import CatLeaf
from gauss_leaf import GaussLeaf
from node import Node
from product import Product
from spn import SPN
from sum_node import Sum
from utils import Utils


class OnlineSPN:
    """在线 SPN 学习器: 软路由吸收 + 显著性延迟生长 (追平全量的架构)。

    与批量的理论关系:
      * 参数层精确 —— 叶 (n,μ,M2 中心矩, Chan 合并) / 码联合计数 /
        Sum 计数全是可加充分统计量, 同结构同数据下在线 = 批量 MLE
        (spn_selftest 自检 6 验证);
      * 软路由 —— 每行按叶后验 (路径先验 × 叶似然) 分配 responsibility,
        EM 语义: 分组/分裂错误被后续数据自校正, 无硬路由反馈环;
      * 延迟生长 —— 每批检查, 码计数达 N_SPLIT 才分裂; 码分裂语义上
        无统计风险 (查询变量驱动: 不同码本来不同分量), 无需 VFDT 式
        高阈值/翻倍节流 (套错对象会压死生长, 实测); 子叶继承父叶
        高斯参数作伪计数先验 + 当批行按码分组播种, 抗早承诺过拟合。
    边界 (scope 一致性使在线不可局部执行, 留给周期全局修订):
      Product (变量独立) 分裂改 scope; 特征驱动 Sum 分裂需行级 k-means。
    契约: 所有 CatLeaf 属于 code_cols (查询变量驱动设定, 见 inverse_app)。
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
        → 显著性生长。X (B, V) 列布局与 SPNLearner.learn 输入一致。"""
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
        """显著性延迟生长: 每批检查 (计数有新质量才查), 码支撑 ≥2 → 分裂。
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
