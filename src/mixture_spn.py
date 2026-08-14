"""MixtureSPN: 全分辨率浅混合 SPN (实例级对角高斯块的 Sum)。

结构: Sum(K) × Block; Block = Product(特征对角高斯 × D | 连续目标 × T |
场景类目 × C), K = 训练样本数 (实例级, 每样本一个分量)。深度结构学习
在 V=186K 列上是 O(V²) 列对检验, 不可行 —— 浅混合是可落地的全
分辨率形态 (SPN 的退化深度形态, 推理仍精确)。

学习 = 组装 (无 EM, 确定性): 逐 kind 分层 (kind 是形状因子, 决定方差
度量), 分量 = 全部训练样本, 类内 tied 对角方差 (全子集估计), 均匀权重。
kind / 图元色相 / 光色 / 光向作为离散场景因子共享同一责任度。为什么
不做 EM/质心压缩: 小数据 (本项目设计目标) + 弯曲流形时, K-means 质心
把流形上的点平均到流形外, 实测 R² 0.50 vs 实例级 0.94; EM 的四条
实测退化通道 (目标 razor 门控 winner-take-all / 死分量均值爆炸 /
方差无上限大方差吃一切 / 每分量方差噪声淹没距离项) 随压缩层一起
删除。EM 压缩是大数据优化, N≫K 且流形平直时再加 (ponytail: 此刻
YAGNI)。

推理: r = softmax(log_w + Σ_d logN(x_d; μ,σ)) → E[t|x] = r @ t_mu,
P(场景因子|x) = r @ exp(cat_logp)。数学上等价逐 kind 分层的
Nadaraya-Watson 核回归/核分类: 样本 = 核中心, tied σ = 带宽。

数组实现而非 Node 对象树: K 块 × 186K 叶的对象递归不可行。

度量基础: PCA 白化。原始特征相邻像素强相关 + 边缘像素双峰, 对角
高斯把相关维当独立选票 —— 62K 亮度维的相关性夸大有效证据两个
数量级, 会淹没色度维的 kind 信号 (实测 kind 0.47 vs 色度 1-NN
0.95)。白化空间里的对角高斯 ≡ 原空间全协方差高斯, 相关性病理
连根拔掉; 且 N 样本的秩 ≤ N−1, 白化同时是无损降维 (186K→N)。
基/均值随模型序列化 (推理必须用同一变换)。

数值纪律: E 步用直接 (x−μ)² 分块形式, 不用展开式 matmul
(E[x²]−E[x]² 在 float32 下对近零方差维灾难性抵消, code_bayes
教训); 分块 + 逐块 mx.eval: 惰性图全量累积会超 Metal 显存上限。
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from utils import Utils

_NC = 64  # E 步样本块行数
_KC = 8  # E 步分量块数: (64,8,·) 中间量有界, Metal 可承受


class MixtureSPN:
    """浅混合 SPN: 实例级对角高斯块混合 (白化空间) + 连续头 + 多分类头。

    场景因子分层: kind / 图元色相 / 光色 / 光向各是一个类目头。分层决定
    特征方差度量的仍用 kind (形状因子); 其余离散因子沿维度复用同一
    责任度, 得到完整场景参数的条件后验。"""

    N_STRATUM = 3

    def __init__(
        self,
        log_w: mx.array,  # (K,)
        f_mu: mx.array,  # (K, D) 白化空间特征均值 (实例级 = 样本)
        f_var: mx.array,  # (K, D) 白化空间特征方差 (类内 tied)
        t_mu: mx.array,  # (K, T) 连续目标 (= 样本目标)
        cat_logp: mx.array,  # (K, 21) 场景因子类目 log 概率 (行 one-hot)
        rel_floor: float,
        f_mean: mx.array | None = None,  # (V,) 白化中心
        basis: mx.array | None = None,  # (V, D) 白化基 (随模型序列化)
    ):
        self.log_w, self.f_mu, self.f_var = log_w, f_mu, f_var
        self.t_mu, self.cat_logp = t_mu, cat_logp
        self.f_mean, self.basis = f_mean, basis
        self.rel_floor = rel_floor
        # 预计算特征侧归一常数 (K,): log_w − ½·Σ_d(log var + log2π)
        self._norm = log_w - 0.5 * mx.sum(
            mx.log(f_var) + math.log(2.0 * math.pi), axis=1
        )

    def _z(self, f: mx.array) -> mx.array:
        """原空间特征 (N,V) → 白化坐标 (N,D)。"""
        assert self.f_mean is not None and self.basis is not None, (
            "模型缺白化基, 不可 predict"
        )
        return (f - self.f_mean[None, :]) @ self.basis

    # ── 特征侧似然 ──────────────────────────────────────────────────

    def _logq_feat(self, z: mx.array) -> mx.array:
        """白化空间特征侧未归一 log 联合 (N,K), 分块直接 (x−μ)² 形式。"""
        out = []
        for i in range(0, z.shape[0], _NC):
            xb = z[i : i + _NC]
            parts = []
            for j in range(0, self.f_mu.shape[0], _KC):
                d = xb[:, None, :] - self.f_mu[None, j : j + _KC, :]
                q = self._norm[None, j : j + _KC] - 0.5 * mx.sum(
                    d * d / self.f_var[None, j : j + _KC, :], axis=2
                )
                mx.eval(q)  # 逐块求值, 防惰性图累积爆显存
                parts.append(q)
            out.append(mx.concatenate(parts, axis=1))
        return mx.concatenate(out, axis=0)

    # ── 组装 (学习) ─────────────────────────────────────────────────

    @staticmethod
    def _tied_vars(z: mx.array, stratum: mx.array, rel_floor: float) -> mx.array:
        """逐分层 tied 对角方差 → (N_STRATUM, D) (类内全子集估计 = 核带宽,
        地板防零; 缺场分层行无引用, 填 1 占位)。"""
        out = []
        for j in range(MixtureSPN.N_STRATUM):
            sel = Utils.nonzero(stratum == j)
            if sel.shape[0] == 0:
                out.append(mx.ones((1, z.shape[1])))
                continue
            zj = z[sel]
            out.append(
                mx.maximum(
                    mx.var(zj, axis=0, keepdims=True),
                    (rel_floor * zj.std(axis=0, keepdims=True)) ** 2 + 1e-8,
                )
            )
        return mx.concatenate(out)

    @staticmethod
    def _cat_logp(classes: mx.array, sizes: tuple[int, ...]) -> mx.array:
        """场景离散因子 (N,len(sizes)) → 拼接 one-hot log 概率 (N,Σsizes)。"""
        cols = []
        for j, nc in enumerate(sizes):
            eye = mx.arange(nc)[None, :] == classes[:, j, None]
            cols.append(mx.log(mx.zeros(classes.shape[:1] + (nc,)) + eye))
        return mx.concatenate(cols, axis=1)

    @staticmethod
    def cat_sizes(cat_logp: mx.array) -> tuple[int, ...]:
        """拼接类目头的宽度 → 各场景因子类目数。"""
        if cat_logp.shape[1] == MixtureSPN.N_STRATUM:
            return (MixtureSPN.N_STRATUM,)
        if cat_logp.shape[1] == 24:
            return (3, 3, 6, 6, 3, 3)  # 双层 k0,k1,h0,h1,lcol,ldir
        return (3, 6, 3, 3)

    @classmethod
    def fit(
        cls,
        f: mx.array,  # (N, V) 特征
        t: mx.array,  # (N, T) 连续目标
        stratum: mx.array,  # (N,) 分层因子 int (kind)
        rel_floor: float = 1e-2,
        scene_classes: mx.array | None = None,  # (N,4) kind,hue,lcol,ldir
        cat_sizes: tuple[int, ...] | None = None,
    ) -> MixtureSPN:
        """逐分层实例级组装 (P(stratum)·P(f,t,场景因子|stratum)), 确定性。"""
        f_mean, basis, z = cls._whiten(f)
        mus, vars_, tmus, ws = [], [], [], []
        n = z.shape[0]
        gvar = cls._tied_vars(z, stratum, rel_floor)
        clps = []
        if scene_classes is None:
            scene_classes = stratum[:, None].astype(mx.int32)
            cat_sizes = (cls.N_STRATUM,)
        assert cat_sizes is not None
        assert scene_classes.shape[1] == len(cat_sizes)
        for j in range(cls.N_STRATUM):
            sel = Utils.nonzero(stratum == j)
            nj = sel.shape[0]
            if nj == 0:
                continue  # 缺场分层 (合成测试/子集) 不建分量
            zj, tj = z[sel], t[sel]
            mus.append(zj)
            vars_.append(mx.tile(gvar[j : j + 1], (nj, 1)))
            tmus.append(tj)
            ws.append(mx.full((nj,), -math.log(n)))  # 均匀 (含类频率)
            clps.append(cls._cat_logp(scene_classes[sel], cat_sizes))
        m = cls(
            mx.concatenate(ws), mx.concatenate(mus), mx.concatenate(vars_),
            mx.concatenate(tmus), mx.concatenate(clps), rel_floor, f_mean,
            basis,
        )
        mx.eval(m.log_w, m.f_mu, m.f_var, m.t_mu, m.cat_logp)
        return m

    def add(
        self,
        f: mx.array,
        t: mx.array,
        stratum: mx.array,
        scene_classes: mx.array,
    ) -> None:
        """增量训练: 追加新样本分量 + tied 方差/均匀权重全量重估。

        精确性: 实例级模型的 f_mu 即全部训练样本, 重估与"全量 fit 的
        方差步"是同估计量 —— 唯一冻结的是白化基 (度量): 新样本跑出
        旧主子空间的方向不可表示 (ponytail: 分布漂移大时重新 fit,
        基扩展要做正交化+坐标迁移, YAGNI 直到漂移实测发生)。"""
        assert self.f_mean is not None and self.basis is not None
        z_new = self._z(f)  # 冻结基白化
        self.f_mu = mx.concatenate([self.f_mu, z_new])
        self.t_mu = mx.concatenate([self.t_mu, t])
        cat_sizes = self.cat_sizes(self.cat_logp)
        assert scene_classes.shape[1] == len(cat_sizes)
        self.cat_logp = mx.concatenate(
            [self.cat_logp, self._cat_logp(scene_classes, cat_sizes)]
        )
        n = self.f_mu.shape[0]
        n_new = scene_classes.shape[0]
        s_all = mx.concatenate(
            [
                mx.argmax(
                    self.cat_logp[:-n_new, : self.N_STRATUM], axis=1
                ),
                stratum.astype(mx.int32),
            ]
        )
        gvar = self._tied_vars(self.f_mu, s_all, self.rel_floor)
        self.f_var = gvar[s_all]
        self.log_w = mx.full((n,), -math.log(n))
        self._norm = self.log_w - 0.5 * mx.sum(
            mx.log(self.f_var) + math.log(2.0 * math.pi), axis=1
        )
        mx.eval(self.log_w, self.f_mu, self.f_var, self.t_mu, self.cat_logp)

    @staticmethod
    def _whiten(f: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """PCA 白化: → (均值 (V,), 基 (V,D), 白化训练坐标 (N,D))。

        N 样本秩 ≤ N: Gram 技巧 (N×N eigh) 避免 V×V 协方差。截断
        λ > λmax·1e-6: 合成数据近零方差方向是数值噪声, 非信号。"""
        f_mean = mx.mean(f, axis=0)
        xc = f - f_mean[None, :]
        g = xc @ xc.T  # (N,N) Gram
        lam, u = mx.linalg.eigh(g, stream=mx.cpu)  # 升序; GPU 无 eigh
        keep = Utils.nonzero(lam > float(mx.max(lam)) * 1e-6)
        lam, u = lam[keep], u[:, keep]
        basis = (xc.T @ (u / mx.sqrt(lam)[None, :])).astype(mx.float32)
        z = xc @ basis
        mx.eval(f_mean, basis, z)
        return f_mean, basis, z

    # ── 推理 (特征证据 → 条件期望/条件后验) ─────────────────────────

    def predict(
        self, f: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """特征 (N,V) → (E[t|x] (N,T), P(场景因子|x) (N,C), 责任度 (N,K))。"""
        logq = self._logq_feat(self._z(f))
        r = mx.exp(logq - mx.logsumexp(logq, axis=1, keepdims=True))
        mx.eval(r)
        t_mean = r @ self.t_mu
        cat_p = r @ mx.exp(self.cat_logp)
        mx.eval(t_mean, cat_p)
        return t_mean, cat_p, r

    # ── 序列化 (safetensors, 标量入 JSON 头) ─────────────────────────

    def save(self, path: str | Path) -> None:
        import json

        mx.save_safetensors(
            str(path),
            {
                "log_w": self.log_w,
                "f_mu": self.f_mu,
                "f_var": self.f_var,
                "t_mu": self.t_mu,
                "cat_logp": self.cat_logp,
                "f_mean": self.f_mean,
                "basis": self.basis,
            },
            {"rel_floor": json.dumps(self.rel_floor)},
        )

    @staticmethod
    def load(path: str | Path) -> MixtureSPN:
        import json

        d = mx.load(str(path))
        hd = Utils.st_metadata(path).get("__metadata__", {})
        return MixtureSPN(
            d["log_w"], d["f_mu"], d["f_var"], d["t_mu"], d["cat_logp"],
            float(json.loads(hd["rel_floor"])), d["f_mean"], d["basis"],
        )


