"""MixtureSPN: 全分辨率浅混合 SPN (实例级对角高斯块的 Sum)。

结构: Sum(K) × Block; Block = Product(特征对角高斯 × D | 目标 × T |
kind 类目), K = 训练样本数 (实例级, 每样本一个分量)。深度结构学习
在 V=186K 列上是 O(V²) 列对检验, 不可行 —— 浅混合是可落地的全
分辨率形态 (SPN 的退化深度形态, 推理仍精确)。

学习 = 组装 (无 EM, 确定性): 逐 kind 分层, 分量 = 全部训练样本,
类内 tied 对角方差 (全子集估计), 均匀权重。为什么不做 EM/质心
压缩: 小数据 (本项目设计目标) + 弯曲流形时, K-means 质心把流形
上的点平均到流形外, 实测 R² 0.50 vs 实例级 0.94; EM 的四条实测
退化通道 (目标 razor 门控 winner-take-all / 死分量均值爆炸 /
方差无上限大方差吃一切 / 每分量方差噪声淹没距离项) 随压缩层
一起删除。EM 压缩是大数据优化, N≫K 且流形平直时再加 (ponytail:
此刻 YAGNI)。

推理: r = softmax(log_w + Σ_d logN(x_d; μ,σ)) → E[t|x] = r @ t_mu,
P(kind|x) = r @ exp(k_logp)。数学上等价逐 kind 分层的 Nadaraya-
Watson 核回归: 样本 = 核中心, tied σ = 带宽。

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
    """浅混合 SPN: 实例级对角高斯块混合 (白化空间) + 目标头 + kind 头。"""

    def __init__(
        self,
        log_w: mx.array,  # (K,)
        f_mu: mx.array,  # (K, D) 白化空间特征均值 (实例级 = 样本)
        f_var: mx.array,  # (K, D) 白化空间特征方差 (类内 tied)
        t_mu: mx.array,  # (K, T) 连续目标 (= 样本目标)
        k_logp: mx.array,  # (K, 3) kind 类目 log 概率 (行 one-hot)
        rel_floor: float,
        f_mean: mx.array | None = None,  # (V,) 白化中心
        basis: mx.array | None = None,  # (V, D) 白化基 (随模型序列化)
    ):
        self.log_w, self.f_mu, self.f_var = log_w, f_mu, f_var
        self.t_mu, self.k_logp = t_mu, k_logp
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

    @classmethod
    def fit(
        cls,
        f: mx.array,  # (N, V) 特征
        t: mx.array,  # (N, T) 连续目标
        kind: mx.array,  # (N,) int
        rel_floor: float = 1e-2,
    ) -> MixtureSPN:
        """逐 kind 分层实例级组装 (P(kind)·P(f,t|kind)), 确定性。"""
        f_mean, basis, z = cls._whiten(f)
        mus, vars_, tmus, ws, klp = [], [], [], [], []
        n = z.shape[0]
        for j in range(3):
            sel = Utils.nonzero(kind == j)
            nj = sel.shape[0]
            if nj == 0:
                continue  # 缺场 kind (合成测试/子集) 不建分量
            zj, tj = z[sel], t[sel]
            # 类内 tied 方差: 全子集对角方差 (核带宽), 地板防零
            gvar = mx.maximum(
                mx.var(zj, axis=0, keepdims=True),
                (rel_floor * zj.std(axis=0, keepdims=True)) ** 2 + 1e-8,
            )
            mus.append(zj)
            vars_.append(mx.tile(gvar, (nj, 1)))
            tmus.append(tj)
            ws.append(mx.full((nj,), -math.log(n)))  # 均匀 (含类频率)
            onehot = mx.zeros((nj, 3))
            klp.append(mx.log(onehot + (mx.arange(3) == j)[None, :]))
        m = cls(
            mx.concatenate(ws), mx.concatenate(mus), mx.concatenate(vars_),
            mx.concatenate(tmus), mx.concatenate(klp), rel_floor, f_mean, basis,
        )
        mx.eval(m.log_w, m.f_mu, m.f_var, m.t_mu, m.k_logp)
        return m

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

    # ── 推理 (特征证据 → 条件期望) ──────────────────────────────────

    def predict(
        self, f: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """特征 (N,V) → (E[t|x] (N,T), P(kind|x) (N,3), 责任度 (N,K))。"""
        logq = self._logq_feat(self._z(f))
        r = mx.exp(logq - mx.logsumexp(logq, axis=1, keepdims=True))
        mx.eval(r)
        t_mean = r @ self.t_mu
        kind_p = r @ mx.exp(self.k_logp)
        mx.eval(t_mean, kind_p)
        return t_mean, kind_p, r

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
                "k_logp": self.k_logp,
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
            d["log_w"], d["f_mu"], d["f_var"], d["t_mu"], d["k_logp"],
            float(json.loads(hd["rel_floor"])), d["f_mean"], d["basis"],
        )


def _selftest() -> None:
    """黑盒自检: 只用公开构造器/fit/predict/save/load, 断言全部来自
    概率公理与混合模型的对外契约, 不触碰内部实现。"""
    import tempfile

    # ── 组 1: 手工模型, 公理性质 (D=4, T=2, K=3; 恒等白化基) ─────
    f_mu = mx.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]]
    )
    f_var = mx.full((3, 4), 0.01)  # σ=0.1, 分量间距 10 → 近似可分
    t_mu = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    k_logp = mx.log(
        mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )  # 分量 0,1 同属 kind0 —— 检验类目跨分量聚合
    eye = mx.eye(4)
    zero4 = mx.zeros(4)
    m = MixtureSPN(
        mx.full((3,), -math.log(3.0)), f_mu, f_var, t_mu, k_logp,
        0.1, zero4, eye,
    )

    # 归一化公理: 责任度与 kind 概率行和为 1
    # (容差 1e-4: float32 exp/log 复合往返误差 ~数 ulp @ O(1) 值, 实测 3e-5)
    xs = mx.array([[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    tm, kp, r = m.predict(xs)
    assert mx.allclose(mx.sum(r, axis=1), mx.ones(2), atol=1e-4)
    assert mx.allclose(mx.sum(kp, axis=1), mx.ones(2), atol=1e-4)

    # δ 证据选择性: x 精确落在分量 0 质心 → E[t] = t_mu_0 (分量被唯一选中)
    assert mx.allclose(tm[0], t_mu[0], atol=1e-3), f"δ证据: {tm[0]}"

    # 类目聚合: x 在分量 0/1 中点 (分量归属模糊) 但两者同 kind0
    # → P(kind0) ≈ 1: 类目后验对分量置换对称, 不受分量模糊影响
    assert float(kp[1, 0]) > 0.999, f"类目聚合: {kp[1]}"

    # Product 结构性质: 单分量模型对任意证据 E[t|x] = t_mu
    # (块内特征⊥目标 → 证据不影响目标期望; 插值全靠多分量混合)
    m1 = MixtureSPN(
        mx.zeros(1), f_mu[:1], f_var[:1], t_mu[:1], k_logp[:1],
        0.1, zero4, eye,
    )
    tm1, _, _ = m1.predict(mx.array([[99.0, 99.0, 99.0, 99.0]]))
    assert mx.allclose(tm1[0], t_mu[0]), f"单分量证据无关性: {tm1[0]}"
    print("组 1 ✓ 公理性质 (归一化/δ选择/类目聚合/单分量证据无关)")

    # ── 组 2: 实例级回归精度 (可分离合成混合) ──────────────────────
    key = mx.random.key(0)
    keys = mx.random.split(key, 4)
    n_per, d_f, d_t = 200, 6, 2
    true_fmu = mx.random.normal(shape=(3, d_f), key=keys[0]) * 3.0  # 分量可分
    true_tmu = mx.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    fs, ts, ks = [], [], []
    for c in range(3):
        fs.append(
            true_fmu[c] + 0.3 * mx.random.normal(shape=(n_per, d_f), key=keys[1])
        )
        ts.append(
            true_tmu[c] + 0.1 * mx.random.normal(shape=(n_per, d_t), key=keys[2])
        )
        ks.append(mx.full((n_per,), c % 3))
    f_all = mx.concatenate(fs)
    t_all = mx.concatenate(ts)
    k_all = mx.concatenate(ks)
    fitted = MixtureSPN.fit(f_all, t_all, k_all)
    # 实例级模型在可分数据上 ≈ 精确插值: 预测 RMSE 应远小于簇间距
    # (8.0); 断 0.5 = 间距的 1/16, 目标噪声 σ=0.1 的 5 倍
    tm, kp, _ = fitted.predict(f_all)
    rmse = float(mx.sqrt(mx.mean((tm - t_all) ** 2)))
    assert rmse < 0.5, f"实例级回归 RMSE {rmse}"
    # kind: 簇间可分 → kind 后验应近完美
    acc2 = float(
        mx.mean((mx.argmax(kp, axis=1) == k_all).astype(mx.float32))
    )
    assert acc2 > 0.99, f"可分混合 kind {acc2:.3f}"
    print(f"组 2 ✓ 实例级回归 (RMSE {rmse:.3f}, kind {acc2:.3f}, 簇间距 8.0)")

    # ── 组 4: 相关性病理 (白化的存在理由) ─────────────────────────
    # 两类样本沿对角线拉长 (强相关), 类分离方向 = 正交低方差方向 ——
    # 原空间对角高斯的最坏情形 (逐维方差被拉长方向污染, 类间逐维重叠);
    # 模型契约: 白化后应正确分离 (白化对角 ≡ 原空间全协方差)。
    n4 = 120
    k4a, k4b, k4c = mx.random.split(mx.random.key(5), 3)
    direction = mx.array([1.0, 1.0, 1.0, 1.0]) / 2.0  # 相关方向 (单位向量)
    perp = mx.array([1.0, -1.0, 1.0, -1.0]) / 2.0  # 正交方向
    lo = mx.random.normal(shape=(n4, 1), key=k4a) * 3.0  # 沿相关方向大散
    pe = mx.random.normal(shape=(n4, 1), key=k4b) * 0.1  # 正交小噪声
    # 类均值沿【正交】(低方差) 方向差 0.6 (正交 σ=0.1 → 全协方差 d'=6,
    # 完全可分); 逐维看: 每轴方差被相关方向污染 σ²≈4.5, 均值差仅 0.3
    # → 原空间对角高斯 d'²≈0.08 不可分。模型契约: 白化后应正确分离。
    off = mx.where(mx.arange(n4) < n4 // 2, 0.0, 0.6)[:, None]
    f4 = lo * direction[None, :] + pe * perp[None, :] + off * perp[None, :]
    k4 = (mx.arange(n4) >= n4 // 2).astype(mx.int32)
    t4 = mx.zeros((n4, 1))  # 目标不参与本组断言
    m4 = MixtureSPN.fit(f4, t4, k4)
    _, kp4, _ = m4.predict(f4)
    acc4 = float(
        mx.mean((mx.argmax(kp4, axis=1) == k4).astype(mx.float32))
    )
    # 白化后两类在正交方向上 d'=6 应完全可分; 断 0.95
    assert acc4 > 0.95, f"相关特征类分离失败 {acc4:.3f}"
    print(f"组 4 ✓ 相关性病理 (白化后类准确率 {acc4:.3f})")

    # ── 组 3: 序列化 roundtrip (预测逐位一致) ─────────────────────
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        fitted.save(p)
        tm2, kp2, _ = MixtureSPN.load(p).predict(f_all)
    assert bool(mx.all(mx.equal(tm, tm2))) and bool(mx.all(mx.equal(kp, kp2)))
    print("组 3 ✓ 序列化 roundtrip (预测逐位一致)")


if __name__ == "__main__":
    _selftest()
    print("MixtureSPN 自检全过 ✓")
