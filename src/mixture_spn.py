"""MixtureSPN: 全分辨率浅混合 SPN (K 个对角高斯块的 Sum) + 联合 EM 学习。

结构: Sum(K) × Block; Block = Product(特征对角高斯 × V | 目标高斯 × T |
kind 类目)。深度结构学习在 V=186K 列上是 O(V²) 列对检验, 不可行 ——
浅混合是唯一可落地的全分辨率形态 (SPN 的退化深度形态, 推理仍精确)。

学习: GMR (Gaussian Mixture Regression) 式联合 EM, 按生成结构分解
P(f,t,kind) = P(kind)·P(f,t|kind) —— 逐 kind 分层拟合 K/3 个分量
的联合 (特征,目标) 混合, 再按类频率合成全模型。为什么必须分层:
kind 与连续因子独立采样, 连续流形上任何局部 patch 天然混合三种
kind, 无约束 EM 按位置相似度聚类 → 分量结构性混色 (局部最优);
分层后 kind 纯度由构造保证, P(kind|x) 成为三类条件混合的似然比。
推理责任度只由特征给出, 再条件期望。

推理: r = softmax(log_w + Σ_d logN(x_d; μ,σ)) → E[t|x] = r @ t_mu,
P(kind|x) = r @ exp(k_logp)。数学上等价 Nadaraya-Watson 核回归:
分量质心 = 核中心, σ = 带宽。sigma_rel_floor 即带宽下限, 是插值
平滑度的原理旋钮 (非纯数值保护)。

数组实现而非 Node 对象树: K 块 × 186K 叶的对象递归不可行, EM 直接
操作 (K,V) 参数矩阵。

度量基础: PCA 白化。原始特征相邻像素强相关 + 边缘像素双峰, 对角
高斯把相关维当独立选票 —— 62K 亮度维的相关性夸大有效证据两个
数量级, 会淹没色度维的 kind 信号 (实测 kind 0.47 vs 色度 1-NN
0.95)。白化空间里的对角高斯 ≡ 原空间全协方差高斯, 相关性病理
连根拔掉; 且 N 样本的秩 ≤ N−1, 白化同时是无损降维 (186K→N)。
基/均值随模型序列化 (推理必须用同一变换)。

数值纪律 (code_bayes 教训): 方差 M 步两遍法 (先 μ 后 Σr(x−μ)²) ——
E[x²]−E[x]² 在 float32 下对近零方差维灾难性抵消; E 步同样用直接
(x−μ)² 分块形式, 不用展开式 matmul (同因)。分块 + 逐块 mx.eval:
惰性图全量累积会超 Metal 显存上限。
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from utils import Utils

_NC = 64  # E/M 步样本块行数
_KC = 8  # E/M 步分量块数: (64,8,186624) 中间量 ≈ 383MB, Metal 可承受


class MixtureSPN:
    """浅混合 SPN: 对角高斯块混合 (白化空间) + 连续目标头 + kind 类目头。"""

    def __init__(
        self,
        log_w: mx.array,  # (K,)
        f_mu: mx.array,  # (K, D) 白化空间特征均值
        f_var: mx.array,  # (K, D) 白化空间特征方差 (对角)
        t_mu: mx.array,  # (K, T) 连续目标均值
        k_logp: mx.array,  # (K, 3) kind 类目 log 概率 (行归一)
        rel_floor: float,
        f_mean: mx.array | None = None,  # (V,) 白化中心; None = EM 内部态
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
            "EM 内部态无白化基, 不可 predict"
        )
        return (f - self.f_mean[None, :]) @ self.basis

    # ── E 步 ────────────────────────────────────────────────────────

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

    def _logq_joint(
        self, f: mx.array, t: mx.array, t_var: mx.array
    ) -> mx.array:
        """训练 E 步: 特征 + 目标联合 (N,K) (kind 已由分层处理)。"""
        logq = self._logq_feat(f)
        # 目标项: −½·Σ_t[(t−μ)²/var + log var + log2π] (T 很小, 不分块)
        d = t[:, None, :] - self.t_mu[None, :, :]  # (N,K,T)
        return logq - 0.5 * mx.sum(
            d * d / t_var[None, :, :]
            + mx.log(t_var)[None, :, :]
            + math.log(2.0 * math.pi),
            axis=2,
        )

    # ── 学习 ────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        f: mx.array,  # (N, V) 特征
        t: mx.array,  # (N, T) 连续目标
        kind: mx.array,  # (N,) int
        k: int,
        iters: int = 20,
        rel_floor: float = 1e-2,
        key: mx.array | None = None,
    ) -> MixtureSPN:
        """逐 kind 分层联合 EM (生成结构 P(kind)·P(f,t|kind)), 再合成。
        k 为总分量预算, 逐 kind 各 k//3 个。"""
        f_mean, basis, z = cls._whiten(f)
        ks = mx.random.split(key, 4) if key is not None else [None] * 4
        mus, vars_, tmus, ws, klp = [], [], [], [], []
        n = z.shape[0]
        for j in range(3):
            sel = Utils.nonzero(kind == j)
            nj = sel.shape[0]
            if nj == 0:
                continue  # 缺场 kind (合成测试/子集) 不建分量
            sub = cls._fit_em(
                z[sel], t[sel], max(1, k // 3), iters, rel_floor, ks[j]
            )
            mus.append(sub.f_mu)
            vars_.append(sub.f_var)
            tmus.append(sub.t_mu)
            # 合成权重: 类频率 × 类内权重
            ws.append(sub.log_w + math.log(nj / n))
            onehot = mx.zeros((sub.f_mu.shape[0], 3))
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

    @classmethod
    def _fit_em(
        cls, f, t, k, iters, rel_floor, key, var_prior=20.0
    ) -> MixtureSPN:
        """单 kind 子集的 (特征,目标) 联合 EM。init: 随机 K 样本为质心
        + 全局方差 (流形平铺靠 EM 收敛)。

        var_prior: 方差的逆伽马先验等效样本数 (Ledoit-Wolf 收缩) ——
        高维小样本下分量内样本方差在零空间维上撞地板, 责任度会被
        零空间抖动主导 (实测); 向类内全局方差收缩, nk≫var_prior 时
        纯数据。先验强度是架构选择, 非调参。"""
        n, v = f.shape
        # 无放随机 K 样本 (argsort 洗牌, 比 permutation 的 key 支持稳)
        perm = mx.argsort(mx.random.uniform(shape=(n,), key=key))[:k]
        f_mu, t_mu = f[perm], t[perm]
        f_var = mx.tile(mx.var(f, axis=0, keepdims=True), (k, 1))
        t_var = mx.tile(mx.var(t, axis=0, keepdims=True), (k, 1))
        k_logp = mx.zeros((k, 3))  # 占位 (分层拟合不用, 由 fit 合成时覆写)
        log_w = mx.full((k,), -math.log(k))
        # 带宽下限: 各维全局 std 的相对比例 (绝对地板防 std=0 维除零)
        f_floor = mx.maximum((rel_floor * f.std(axis=0)) ** 2, 1e-8)
        t_floor = mx.maximum((rel_floor * t.std(axis=0)) ** 2, 1e-8)
        # 初始化即施地板: 首轮 E 步就用 var (零方差目标维 0/0 = NaN 实测)
        f_var = mx.maximum(f_var, f_floor[None, :])
        t_var = mx.maximum(t_var, t_floor[None, :])
        # 收缩锚点: 类内 (本子集) 全局方差
        f_gvar = mx.var(f, axis=0)
        t_gvar = mx.var(t, axis=0)
        mx.eval(f_mu, t_mu, f_var, t_var, k_logp)
        it = -1
        ll_new = float("nan")
        ll = float("-inf")
        for it in range(iters):
            m = cls(log_w, f_mu, f_var, t_mu, k_logp, rel_floor)
            logq = m._logq_joint(f, t, t_var)
            ll_new = float(mx.mean(mx.logsumexp(logq, axis=1)))
            r = mx.exp(logq - mx.logsumexp(logq, axis=1, keepdims=True))
            mx.eval(r)
            # M 步: nk 防零 (死分量保持原参数, 等下轮 E 步自然复活/淘汰)
            nk = mx.sum(r, axis=0) + 1e-6
            f_mu = (r.T @ f) / nk[:, None]
            t_mu = (r.T @ t) / nk[:, None]
            mx.eval(f_mu, t_mu)
            # 两遍法方差 (先 μ 后 Σr(x−μ)²), 分块
            parts = []
            for j in range(0, k, _KC):
                acc = mx.zeros_like(f_mu[j : j + _KC])
                for i in range(0, n, _NC):
                    d = f[i : i + _NC, None, :] - f_mu[None, j : j + _KC, :]
                    c = mx.sum(r[i : i + _NC, j : j + _KC, None] * d * d, axis=0)
                    mx.eval(c)
                    acc = acc + c
                parts.append(acc / nk[j : j + _KC, None])
            f_var = mx.maximum(
                (nk[:, None] * mx.concatenate(parts) + var_prior * f_gvar[None, :])
                / (nk + var_prior)[:, None],
                f_floor[None, :],
            )
            d = t[:, None, :] - t_mu[None, :, :]  # (N,K,T), T 小不分块
            t_var = mx.maximum(
                (
                    nk[:, None] * (mx.sum(r[:, :, None] * d * d, axis=0) / nk[:, None])
                    + var_prior * t_gvar[None, :]
                )
                / (nk + var_prior)[:, None],
                t_floor[None, :],
            )
            log_w = mx.log(nk / mx.sum(nk))
            mx.eval(f_var, t_var, log_w)
            if it > 0 and abs(ll_new - ll) < 1e-4 * abs(ll):
                break  # 对数似然相对收敛
            ll = ll_new
        print(f"      EM 收敛: {it + 1} 轮, 平均 log 联合 {ll_new:.1f}")
        return cls(log_w, f_mu, f_var, t_mu, k_logp, rel_floor)

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

    # ── 组 2: EM 恢复 (可分离合成混合, 标签置换意义下) ────────────
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
    fitted = MixtureSPN.fit(f_all, t_all, k_all, k=3, iters=30, key=keys[3])
    # 分量匹配: 按 t_mu 最近邻 (EM 标签任意置换)
    for c in range(3):
        j = int(mx.argmin(mx.sum((fitted.t_mu - true_tmu[c]) ** 2, axis=1)))
        # 恢复精度基准: 均值的标准误 ≈ σ/√n = 0.1/√200 ≈ 0.007 (目标维);
        # 断 0.1 = 14σ 裕量
        assert float(mx.max(mx.abs(fitted.t_mu[j] - true_tmu[c]))) < 0.1, (
            f"分量 {c} 目标质心未恢复: {fitted.t_mu[j]} vs {true_tmu[c]}"
        )
    # 预测 RMSE 应远小于分量间距 (8.0): 断 0.5 = 间距的 1/16
    tm, kp, _ = fitted.predict(f_all)
    rmse = float(mx.sqrt(mx.mean((tm - t_all) ** 2)))
    assert rmse < 0.5, f"EM 恢复后预测 RMSE {rmse}"
    print(f"组 2 ✓ EM 恢复 (预测 RMSE {rmse:.3f}, 分量间距 8.0)")

    # ── 组 4: 相关性病理 (白化的存在理由) ─────────────────────────
    # 两类样本沿对角线拉长 (强相关), 类分离方向 = 相关方向 —— 原空间
    # 对角高斯的最坏情形 (逐维方差被拉长方向污染, 类间逐维重叠);
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
    m4 = MixtureSPN.fit(f4, t4, k4, k=2, iters=20, key=k4c)
    _, kp4, _ = m4.predict(f4)
    acc4 = float(
        mx.mean((mx.argmax(kp4, axis=1) == k4).astype(mx.float32))
    )
    # 白化后两类在相关方向上距离 0.6/类内白化σ 应完全可分; 断 0.95
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
