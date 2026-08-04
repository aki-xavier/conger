import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx

from color import Color
from riesz import RieszFeatures, RieszWavelet
from utils import Utils

# ── MLX 缺的特殊函数: 递推 + 渐近展开 ─────────────────────────────


def digamma(x: mx.array) -> mx.array:
    """ψ(x): 递推 ψ(x)=ψ(x+1)−1/x 推到 x≥8, 再用渐近级数。"""
    y = mx.zeros_like(x)
    for _ in range(16):
        small = x < 8.0
        y = mx.where(small, y - 1.0 / x, y)
        x = mx.where(small, x + 1.0, x)
    return y + mx.log(x) - 0.5 / x - 1.0 / (12.0 * x**2) + 1.0 / (120.0 * x**4)


def lgamma(x: mx.array) -> mx.array:
    """ln Γ(x): 递推 lnΓ(x)=lnΓ(x+1)−ln x 推到 x≥8, 再 Stirling。"""
    y = mx.zeros_like(x)
    for _ in range(16):
        small = x < 8.0
        y = mx.where(small, y - mx.log(x), y)
        x = mx.where(small, x + 1.0, x)
    stir = (x - 0.5) * mx.log(x) - x + 0.5 * math.log(2.0 * math.pi)
    return y + stir + 1.0 / (12.0 * x) - 1.0 / (360.0 * x**3)


def mvlgamma(a: mx.array, d: int) -> mx.array:
    """多元 ln Γ_d(a) = d(d−1)/4·lnπ + Σᵢ lnΓ(a+(1−i)/2)。"""
    i = mx.arange(d, dtype=mx.float32)
    off = a[..., None] + 0.5 * (1.0 - i)
    return d * (d - 1) / 4.0 * math.log(math.pi) + mx.sum(lgamma(off), axis=-1)


def logdet_spd(a: mx.array) -> float:
    """ln|A|, A 对称正定。MLX cholesky 只有 CPU stream。"""
    d = a.shape[-1]
    jitter = mx.eye(d) * 1e-6
    L = mx.linalg.cholesky(a + jitter, stream=mx.cpu)
    return float(2.0 * mx.sum(mx.log(mx.diagonal(L))))


# ── Riesz 特征 → 特征矩阵 ─────────────────────────────────────────

FEAT_NAMES = ["log_mag", "slope", "resid", "bump", "spread", "ori_R", "phase_coh"]


def feature_matrix(feat: RieszFeatures) -> mx.array:
    """(H,W) 特征图栈 → (N,7) 特征矩阵 (未标准化)。"""
    cols = [
        feat.log_mag,
        feat.slope,
        feat.residual,
        feat.bump,
        feat.spread,
        feat.ori_R,
        feat.phase_coh,
    ]
    return mx.stack([c.reshape(-1) for c in cols], axis=-1)


# ── 变分贝叶斯 GMM (全协方差, NIW 先验) ────────────────────────────


def _nu0(d: int) -> float:
    """Wishart 先验自由度。"""
    return float(d + 2)


def _lnb0(nu0: float, d: int) -> float:
    """ln B(W0, ν0), W0 = I (Bishop B.79)。"""
    lnb = -(nu0 * d / 2.0) * math.log(2.0)
    return lnb - float(mvlgamma(mx.array([nu0 / 2.0]), d)[0])


@dataclass(slots=True)
class Posterior:
    """一轮 M 步后的变分后验参数及期望充分量 (Bishop 10.2.1)。"""

    alpha: mx.array  # (K,) Dirichlet 后验浓度
    beta: mx.array  # (K,) NIW 均值强度
    nu: mx.array  # (K,) Wishart 自由度
    m: mx.array  # (K,D) 后验均值
    w: mx.array  # (K,D,D) Wishart 尺度矩阵
    logdet_w: mx.array  # (K,) ln|W_k|
    tr_w: mx.array  # (K,) Tr(W_k)
    log_pi: mx.array  # (K,) E[ln π_k]
    log_lt: mx.array  # (K,) E[ln|Λ_k|]


@dataclass(slots=True)
class VBGMM:
    """Bishop PRML 10.2 的 VB-GMM。

    边缘似然的第一性原理位置: ELBO 是 log p(X) (贝叶斯意义下的
    边缘似然) 的下界, 逐像素责任 r_nk 是离散隐变量的后验 —
    "像素 n 属于边缘类" 的概率 = 边缘类分量的责任之和,
    不是人为门控, 是生成模型下的后验推断。

    α0 取小值 → 空分量被剪枝, 有效 K 由数据决定 (ARD 式模型选择)。
    先验取 m0 = 0, W0 = I (数据已 z-score 标准化)。
    """

    x_orig: mx.array  # (N, D) 原始特征
    k_max: int = 8
    alpha0: float = 1e-2  # Dirichlet 浓度: 小 → 稀疏混合
    beta0: float = 1.0  # NIW 均值先验强度
    max_iter: int = 100
    tol: float = 1e-5
    subsample: int = 0  # >0 时 M 步只用这么多随机像素 (E 步始终全图)
    warm: Posterior | None = None  # 暖启动: 上一帧的后验 (online VB 简化版)
    elbo_every: int = 1  # 每几轮算一次 ELBO (标量同步是固定开销)
    mu: mx.array | None = None  # 特征均值 (标准化用)
    sd: mx.array | None = None
    r: mx.array | None = None  # (N, K) 责任
    alpha: mx.array | None = None  # (K,) Dirichlet 后验
    posterior: Posterior | None = None  # 拟合后的完整后验, 供 infer 复用
    weights: mx.array | None = None  # (K,) 混合权重
    means_orig: mx.array | None = None  # (K, D) 分量均值 (原始空间)
    elbo: list[float] = field(default_factory=list)

    def __post_init__(self):
        x = self.x_orig
        self.mu = mx.mean(x, axis=0)
        self.sd = mx.maximum(mx.sqrt(mx.var(x, axis=0)), 1e-6)
        self._fit((x - self.mu) / self.sd)
        nk = mx.sum(self.r, axis=0)
        self.weights = self.alpha / mx.sum(self.alpha)
        self.means_orig = (self.r.T @ x) / mx.maximum(nk[:, None], 1e-12)

    # ── VB-EM 主循环 ──────────────────────────────────────────────

    def _fit(self, z: mx.array):
        # M 步拟合可以只用子采样 (聚类统计量对子采样稳健),
        # 最终责任始终对全图重算。
        if 0 < self.subsample < z.shape[0]:
            idx = mx.random.permutation(z.shape[0], key=mx.random.key(1))
            z_fit = z[idx[: self.subsample]]
        else:
            z_fit = z

        r = self._init_resp(z_fit)
        q = None
        prev = -math.inf
        for it in range(self.max_iter):
            q = self._m_step(z_fit, r)
            r = self._e_step(z_fit, q)
            if it % self.elbo_every == 0:
                bound = self._elbo(z_fit, r, q)
                self.elbo.append(bound)
                # 只在正的微小增益时收敛; ELBO 下降是 bug 信号, 不能当收敛
                gain = bound - prev
                if it > 0 and 0.0 <= gain < self.tol * max(abs(prev), 1.0):
                    break
                prev = bound
        self.posterior = q
        self.r = r if z_fit is z else self._e_step(z, q)
        self.alpha = q.alpha

    def _init_resp(self, z: mx.array) -> mx.array:
        """初始责任: 有暖启动后验则先做一次 E 步 (跟踪模式只需
        再迭代 1–5 轮), 否则随机中心硬分配 + 平滑。"""
        if self.warm is not None:
            return self._e_step(z, self.warm)
        n, _ = z.shape
        k = self.k_max
        idx = mx.random.permutation(n, key=mx.random.key(0))[:k]
        centers = z[idx]
        d2 = mx.sum(z**2, axis=1)[:, None] + mx.sum(centers**2, axis=1)[None, :]
        d2 = d2 - 2.0 * (z @ centers.T)
        assign = mx.argmin(d2, axis=1)
        return mx.eye(k)[assign] * 0.9 + 0.1 / k

    @staticmethod
    def _stats(z: mx.array, r: mx.array):
        """责任加权的充分统计量 N_k / x̄_k / S_k。"""
        nk = mx.sum(r, axis=0)
        safe = mx.maximum(nk, 1e-12)
        xbar = (r.T @ z) / safe[:, None]
        m2 = mx.einsum("nk,nd,ne->kde", r, z, z) / safe[:, None, None]
        s = m2 - xbar[:, :, None] @ xbar[:, None, :]
        return nk, xbar, s

    def _m_step(self, z: mx.array, r: mx.array) -> Posterior:
        d = z.shape[1]
        nk, xbar, s = self._stats(z, r)
        alpha = self.alpha0 + nk
        beta = self.beta0 + nk
        nu = _nu0(d) + nk
        m = (nk / beta)[:, None] * xbar  # m0 = 0 的收缩

        # W_k⁻¹ = W0⁻¹ + N_k·S_k + (β0·N_k/β_k)·x̄x̄ᵀ (m0=0, W0=I)。
        # S_k 加 1e-3 正则: ori_R/phase_coh 会饱和在 1.0、bump 是量化值,
        # 零方差方向让 W 特征值爆炸, float32 下 logdet 与 ν·tr(W)
        # 的抵消失效 → ELBO 发散。等价 sklearn reg_covar。
        eye = mx.eye(d)
        winv = eye + nk[:, None, None] * (s + 1e-3 * eye)
        winv = winv + (self.beta0 * nk / beta)[:, None, None] * (
            xbar[:, :, None] @ xbar[:, None, :]
        )

        # MLX 的 inv/cholesky 只有 CPU stream, 逐分量算
        w_list, logdet_w, tr_w = [], [], []
        for j in range(self.k_max):
            wj = mx.linalg.inv(winv[j], stream=mx.cpu)
            w_list.append(wj)
            logdet_w.append(logdet_spd(wj))
            tr_w.append(float(mx.trace(wj)))
        w = mx.stack(w_list)
        logdet_w = mx.array(logdet_w)
        tr_w = mx.array(tr_w)
        mx.eval(w, logdet_w, tr_w)

        log_pi = digamma(alpha) - digamma(mx.sum(alpha))  # E[ln π_k]
        i_off = mx.arange(d, dtype=mx.float32)
        log_lt = mx.sum(digamma((nu[:, None] + 1.0 - i_off) / 2.0), axis=1)
        log_lt = log_lt + d * math.log(2.0) + logdet_w  # E[ln|Λ_k|]

        return Posterior(alpha, beta, nu, m, w, logdet_w, tr_w, log_pi, log_lt)

    def _e_step(self, z: mx.array, q: Posterior) -> mx.array:
        """r_nk ∝ π̃_k·|Λ̃_k|^{1/2}·exp(−D/2β_k − ν_k/2·maha_nk)。"""
        d = z.shape[1]
        # (z−m)ᵀW(z−m): 两步收缩 —— 单步 einsum("nd,kde,ne") 会
        # 物化 (N,K,D,E) 中间量 (462×490 图 ≈ 700MB), 慢 3 倍
        zw = mx.einsum("nd,kde->nke", z, q.w)
        zz = mx.sum(zw * z[:, None, :], axis=-1)
        zm = mx.sum(zw * q.m[None, :, :], axis=-1)
        mm = mx.einsum("kd,kde,ke->k", q.m, q.w, q.m)
        maha = zz - 2.0 * zm + mm[None, :]

        log_rho = q.log_pi + 0.5 * q.log_lt - 0.5 * d * math.log(2.0 * math.pi)
        log_rho = log_rho - 0.5 * (d / q.beta + q.nu * maha)
        log_rho = log_rho - mx.max(log_rho, axis=1, keepdims=True)
        rho = mx.exp(log_rho)
        return rho / mx.sum(rho, axis=1, keepdims=True)

    def _elbo(self, z: mx.array, r: mx.array, q: Posterior) -> float:
        """Bishop 10.70–10.77。统计量必须用 E 步后的新 r 重算,
        否则算出来的不是当前 q 的真实下界, 会出现假性的非单调。"""
        d = z.shape[1]
        k = self.k_max
        nu0 = _nu0(d)
        nk, xbar, s = self._stats(z, r)

        def qform(a: mx.array) -> mx.array:
            """aᵀ W_k a, (K,D) → (K,)"""
            return mx.einsum("kd,kde,ke->k", a, q.w, a)

        # E[ln p(X|Z,μ,Λ)] (10.71)
        tr_sw = mx.einsum("kde,ked->k", s, q.w)  # Tr(S_k W_k)
        t_x = q.log_lt - d / q.beta - q.nu * tr_sw - q.nu * qform(xbar - q.m)
        e_x = 0.5 * float(mx.sum(nk * t_x) - mx.sum(nk) * d * math.log(2.0 * math.pi))

        # E[ln p(Z|π)] (10.72) 与 E[ln p(π)] (10.73)
        e_z = float(mx.sum(nk * q.log_pi))
        e_pi = float(lgamma(mx.array([k * self.alpha0]))[0])
        e_pi -= k * float(lgamma(mx.array([self.alpha0]))[0])
        e_pi += float((self.alpha0 - 1.0) * mx.sum(q.log_pi))

        # E[ln p(μ,Λ)] (10.74): 注意马氏项是 β0·ν_k 不是 β_k·ν_k
        # (后者 ≈ nk², 随聚类收紧爆炸 → 假发散)
        t_mu = q.log_lt - d * self.beta0 / q.beta - self.beta0 * q.nu * qform(q.m)
        e_mula = 0.5 * k * d * math.log(self.beta0 / (2.0 * math.pi))
        e_mula += 0.5 * float(mx.sum(t_mu)) + k * _lnb0(nu0, d)
        e_mula += 0.5 * (nu0 - d - 1.0) * float(mx.sum(q.log_lt))
        e_mula -= 0.5 * float(mx.sum(q.nu * q.tr_w))  # W0⁻¹ = I → Tr(W)

        # E[ln q(Z)] (10.75) 与 E[ln q(π)] (10.76)
        e_qz = float(mx.sum(r * mx.log(mx.maximum(r, 1e-12))))
        e_qpi = float(mx.sum((q.alpha - 1.0) * q.log_pi))
        e_qpi += float(lgamma(mx.sum(q.alpha))) - float(mx.sum(lgamma(q.alpha)))

        # E[ln q(μ,Λ)] (10.77); Wishart 熵 H[Λ] (Bishop B.82)
        neg_lnb = 0.5 * q.nu * q.logdet_w
        neg_lnb = neg_lnb + (q.nu * d / 2.0) * math.log(2.0) + mvlgamma(q.nu / 2.0, d)
        h = neg_lnb - 0.5 * (q.nu - d - 1.0) * q.log_lt + q.nu * d / 2.0
        t_qmla = 0.5 * q.log_lt + 0.5 * d * mx.log(q.beta / (2.0 * math.pi))
        e_qmla = float(mx.sum(t_qmla - d / 2.0 - h))

        return e_x + e_z + e_pi + e_mula - e_qz - e_qpi - e_qmla

    # ── 聚类结果: 类证据 → 后验似然 ───────────────────────────────

    def infer(self, x_new: mx.array) -> mx.array:
        """逐帧推断: 固定已拟合的后验参数, 对新特征只做一次 E 步。
        这是实时模式每帧的唯一贝叶斯计算 (毫秒级)。"""
        z = (x_new - self.mu) / self.sd
        return self._e_step(z, self.posterior)

    def k_eff(self, min_weight: float = 0.01) -> int:
        """有效分量数: 混合权重超过阈值的分量个数。"""
        return int(mx.sum(self.weights > min_weight))

    def pixel_evidence(self, cls: str, x: mx.array | None = None) -> mx.array:
        """逐像素类证据 (N,) ∈ {0,1}, 规则来自合成真值验证 (增益控制
        后的特征语义): 边缘 = 宽带幂律谱 (resid 小, spread 大) + 跨
        尺度相位对齐 (phase_coh 高); 周期纹理 = 窄带单峰 (resid 大,
        bump 居中, spread 小)。注意 ori_R 不能用: 平坦区的能量是
        边缘泄漏, 方向天然相干, ori_R 分不开; phase_coh 在增益控制
        后才是最强判别 (边缘 ≈0.55, 平坦 ≈0.17, 纹理 ≈0.34)。
        逐像素特征不会被责任混合稀释, 分量均值才会。"""
        x = self.x_orig if x is None else x
        f = {name: x[:, i] for i, name in enumerate(FEAT_NAMES)}  # type: ignore
        if cls == "edge":
            mask = (f["resid"] < 1.0) & (f["spread"] > 1.0) & (f["phase_coh"] > 0.4)
        elif cls == "texture":
            mask = (
                (f["resid"] > 1.5)
                & (f["bump"] > 0.05)
                & (f["bump"] < 0.95)
                & (f["spread"] < 0.9)
            )
        else:
            raise ValueError(f"unknown class: {cls}")
        return mask.astype(mx.float32)

    def class_fraction(
        self, cls: str, x: mx.array | None = None, r: mx.array | None = None
    ) -> mx.array:
        """每个分量内, 该类证据像素占分量质量的比例 (K,)。"""
        r = self.r if r is None else r
        e = self.pixel_evidence(cls, x)
        nk = mx.maximum(mx.sum(r, axis=0), 1e-12)
        return (r.T @ e) / nk

    def class_likelihood(
        self, cls: str, x: mx.array | None = None, r: mx.array | None = None
    ) -> mx.array:
        """某类的后验似然图 (N,): 分量类占比经责任的后验期望
        Σ_k r_nk·frac_k —— 特征证据被相似性聚类平滑: 与边缘像素
        同分量的像素分得高似然, 等价于证据在低维聚类流形上的扩散。
        x/r 缺省用拟合数据; 逐帧模式传 infer() 的新数据。"""
        r = self.r if r is None else r
        return r @ self.class_fraction(cls, x, r)

    def labels(self) -> mx.array:
        return mx.argmax(self.r, axis=1)


def neighbor_similarity(r: mx.array, shape: tuple[int, int]) -> mx.array:
    """共分配相似度: r_i·r_j 是后验意义下两像素同类的概率。
    对 4 邻域取均值 → 区域内部 ≈1, 边界 ≈0, 即相似性聚类的
    软边界图 (不需要 N×N 全相似度矩阵)。"""
    h, w = shape
    rr = r.reshape(h, w, -1)
    sim = mx.zeros((h, w))
    right = mx.sum(rr[:, :-1] * rr[:, 1:], axis=-1)
    sim = sim.at[:, :-1].add(right).at[:, 1:].add(right)
    down = mx.sum(rr[:-1] * rr[1:], axis=-1)
    sim = sim.at[:-1].add(down).at[1:].add(down)
    return sim / 4.0


def visualize_maps(img, gm: VBGMM, shape, out_path: str | Path):
    h, w = shape
    edge = gm.class_likelihood("edge").reshape(h, w)
    tex = gm.class_likelihood("texture").reshape(h, w)
    labs = gm.labels().reshape(h, w).astype(mx.float32)
    bnd = 1.0 - neighbor_similarity(gm.r, shape)
    plots = [
        ("original", "gray", img),
        ("edge likelihood", "gray", edge),
        ("texture likelihood", "gray", tex),
        ("labels", "tab10", labs),
        ("1 − neighbor sim", "gray", bnd),
    ]
    fig = Utils.visualize(plots)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    from PIL import Image

    # ── synthetic validation ─────────────────────────────────────────
    # 弱边缘 @x=64 (Δ=0.05), 强边缘 @x=128 (Δ=0.55), 纹理区 x≥192
    H, W = 128, 256
    img = mx.full((H, W), 0.2)
    img[:, 64:128] = 0.25
    img[:, 128:192] = 0.8
    img[:, 192:] = Utils.make_grating((H, 64), 8.0, 0.0)[:, :64]
    img = img + mx.random.normal((H, W), key=mx.random.key(3)) * 0.01

    feat = RieszFeatures(RieszWavelet(img))
    gm = VBGMM(feature_matrix(feat), k_max=48)

    e = gm.elbo
    mono = max((e[i] - e[i + 1] for i in range(len(e) - 1)), default=0.0)
    print(f"ELBO: {e[0]:.1f} → {e[-1]:.1f} ({len(e)} iters, 最大回退 {mono:.2f})")
    print(f"effective K = {gm.k_eff()} / {gm.k_max}")
    hdr = "w      " + " ".join(f"{n:>9s}" for n in FEAT_NAMES) + "  e_frac t_frac"
    print(hdr)
    e_frac = gm.class_fraction("edge")
    t_frac = gm.class_fraction("texture")
    for j in range(gm.k_max):
        if float(gm.weights[j]) < 0.005:  # type: ignore
            continue
        vals = " ".join(f"{float(v):9.2f}" for v in gm.means_orig[j])  # type: ignore
        print(
            f"{float(gm.weights[j]):.3f}  {vals}  "  # type: ignore
            f"{float(e_frac[j]):.2f}   {float(t_frac[j]):.2f}"
        )

    edge = gm.class_likelihood("edge").reshape(H, W)
    tex = gm.class_likelihood("texture").reshape(H, W)
    for name, sl in [
        ("weak edge @64   ", slice(62, 66)),
        ("strong edge @128", slice(126, 130)),
        ("tex border @192 ", slice(190, 194)),
        ("flat interior   ", slice(90, 120)),
        ("tex interior    ", slice(200, 250)),
    ]:
        print(
            f"{name}: edge={float(edge[:, sl].mean()):.2f} "
            f"tex={float(tex[:, sl].mean()):.2f}"
        )
    path = Utils.project_root() / "artifacts/vbgmm_synth.png"
    visualize_maps(img, gm, (H, W), path)
    print(path)

    # ── natural image ────────────────────────────────────────────────
    im = Image.open(Utils.project_root() / "images/12.png").convert("L")
    arr = Color.image_to_mlx(im)
    feat2 = RieszFeatures(RieszWavelet(arr))
    gm2 = VBGMM(feature_matrix(feat2), k_max=48)
    print(f"12.png: ELBO {gm2.elbo[0]:.1f} → {gm2.elbo[-1]:.1f}, K_eff={gm2.k_eff()}")
    path2 = Utils.project_root() / "artifacts/vbgmm_12.png"
    visualize_maps(arr, gm2, arr.shape, path2)
    print(path2)

    # ── 实时模式: 相邻帧 (边界移动) 冷启动 vs 暖启动 ────────────────
    import time

    img2 = mx.full((H, W), 0.2)
    img2[:, 72:136] = 0.25  # 弱边缘 64→72, 强边缘 128→136, 纹理区 192→200
    img2[:, 136:200] = 0.8
    img2[:, 200:] = Utils.make_grating((H, 56), 8.0, 0.0)[:, :56]
    img2 = img2 + mx.random.normal((H, W), key=mx.random.key(4)) * 0.01

    t0 = time.perf_counter()
    X2 = feature_matrix(RieszFeatures(RieszWavelet(img2)))
    mx.eval(X2)
    t1 = time.perf_counter()
    gm_cold = VBGMM(X2, k_max=48)
    t2 = time.perf_counter()
    gm_warm = VBGMM(X2, k_max=48, warm=gm.posterior, subsample=8192)
    t3 = time.perf_counter()
    print(
        f"frame2 特征提取 {1000 * (t1 - t0):.0f}ms | "
        f"冷启动 {len(gm_cold.elbo)} iters {t2 - t1:.2f}s | "
        f"暖启动+子采样 {len(gm_warm.elbo)} iters {t3 - t2:.2f}s"
    )

    t4 = time.perf_counter()
    r2 = gm.infer(X2)  # 逐帧模式: 固定帧 1 后验, 只做一次 E 步
    edge2 = gm.class_likelihood("edge", x=X2, r=r2).reshape(H, W)
    mx.eval(edge2)
    t5 = time.perf_counter()
    warm_edge = gm_warm.class_likelihood("edge").reshape(H, W)
    print(f"infer() 逐帧边缘似然: {1000 * (t5 - t4):.0f} ms")
    print(
        f"  暖启动 frame2: weak@72={float(warm_edge[:, 70:74].mean()):.2f}"
        f" strong@136={float(warm_edge[:, 134:138].mean()):.2f}"
        f" | infer(帧1后验): weak@72={float(edge2[:, 70:74].mean()):.2f}"
        f" strong@136={float(edge2[:, 134:138].mean()):.2f}"
    )
