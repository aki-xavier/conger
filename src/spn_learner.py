"""SPNLearner: learnSPN 结构学习器 (贪心递归, 配置入构造器)。

算法与先验说明见 spn.py 模块 docstring。
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
