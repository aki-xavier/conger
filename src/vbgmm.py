import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import mlx.core as mx

from riesz import FeatureMaps, RieszWavelet
from utils import Utils

# from color import Color  # 随下方 natural image 块一起注释, 放开时恢复

# ── 变分贝叶斯 GMM (全协方差, NIW 先验) ────────────────────────────
#
# 模块流程:
#
#   RieszWavelet.features() (7 维特征图)
#        │  feature_matrix(): 摊平成 (N,7), 列序由 FEAT_NAMES 定;
#        │  双通路给 hs 复数色相通道 → 追加三列 (N,10)
#        ▼
#   __post_init__: z-score 标准化 (mu/sd)
#        │
#        └─ fit(): VB-EM 主循环 (Bishop 10.2)
#             init_resp(): 暖启动后验先做一次 E 步; 否则随机中心硬分配
#             loop ≤ max_iter: online_update(ρ=1) —— 一次调用即经典
#                VB-EM 一次迭代 (E 步 → 记 bound → 累积量出后验)
#                冷启动: 随机中心 → 一次 M 步; 暖启动: 旧后验直入
#             子采样只作用 M 步, 最终 r 始终对全图重算
#        │
#        ▼
#   输出契约 (两个协同量, 同一后验的两个边际):
#     软聚类 r (N,K) ──→ labels() / neighbor_similarity() (软边界)
#     边缘似然 ──→ pixel_evidence() 逐像素硬规则 → class_fraction()
#                  簇内证据占比 → class_likelihood() r@frac 后验期望
#                  → edge_likelihood() (H,W) ──→ EdgePrior (edgemap.py)
#        ▲                                            │
#        └──────── feedback_round(enh) ◄───────────────┘
#           虚线反馈边: prior=μ·margin·resid·frac 注入 e_step,
#           阻尼混合, 不改 self.r (frac 锚定), 协议上限 2 轮
#
#   逐帧模式: infer() 固定后验只做一次 E 步; online_update() 每帧
#   一次 E 步 + 统计量 EWMA (常数 ρ 跟踪; monitor=True 记 ELBO)


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
    subsample: int = 0  # >0: M 步子采样数 (E 步始终全图)
    warm: Posterior | None = None  # 暖启动: 上一帧的后验 (online VB 简化版)
    mu: mx.array | None = None  # 特征均值 (标准化用)
    sd: mx.array | None = None
    r: mx.array | None = None  # (N, K) 责任
    alpha: mx.array | None = None  # (K,) Dirichlet 后验
    posterior: Posterior | None = None  # 拟合后的完整后验, 供 infer 复用
    weights: mx.array | None = None  # (K,) 混合权重
    means_orig: mx.array | None = None  # (K, D) 分量均值 (原始空间)
    acc: tuple[mx.array, mx.array, mx.array] | None = None  # online VB 状态
    elbo: list[float] = field(default_factory=list)

    # 特征矩阵列名 (本模块按需组装特征; 列名 resid 对应特征图 residual)
    FEAT_NAMES: ClassVar[list[str]] = [
        "log_mag",
        "slope",
        "resid",
        "bump",
        "spread",
        "ori_R",
        "phase_coh",
    ]

    # ── MLX 缺的特殊函数: 递推 + 渐近展开 ─────────────────────────

    @staticmethod
    def digamma(x: mx.array) -> mx.array:
        """ψ(x): 递推 ψ(x)=ψ(x+1)−1/x 推到 x≥8, 再用渐近级数。"""
        y = mx.zeros_like(x)
        for _ in range(16):
            small = x < 8.0
            y = mx.where(small, y - 1.0 / x, y)
            x = mx.where(small, x + 1.0, x)
        return y + mx.log(x) - 0.5 / x - 1.0 / (12.0 * x**2) + 1.0 / (120.0 * x**4)

    @staticmethod
    def lgamma(x: mx.array) -> mx.array:
        """ln Γ(x): 递推 lnΓ(x)=lnΓ(x+1)−ln x 推到 x≥8, 再 Stirling。"""
        y = mx.zeros_like(x)
        for _ in range(16):
            small = x < 8.0
            y = mx.where(small, y - mx.log(x), y)
            x = mx.where(small, x + 1.0, x)
        stir = (x - 0.5) * mx.log(x) - x + 0.5 * math.log(2.0 * math.pi)
        return y + stir + 1.0 / (12.0 * x) - 1.0 / (360.0 * x**3)

    @staticmethod
    def mvlgamma(a: mx.array, d: int) -> mx.array:
        """多元 ln Γ_d(a) = d(d−1)/4·lnπ + Σᵢ lnΓ(a+(1−i)/2)。"""
        i = mx.arange(d, dtype=mx.float32)
        off = a[..., None] + 0.5 * (1.0 - i)
        return d * (d - 1) / 4.0 * math.log(math.pi) + mx.sum(
            VBGMM.lgamma(off), axis=-1
        )

    @staticmethod
    def logdet_spd(a: mx.array) -> float:
        """ln|A|, A 对称正定。MLX cholesky 只有 CPU stream。"""
        d = a.shape[-1]
        jitter = mx.eye(d) * 1e-6
        L = mx.linalg.cholesky(a + jitter, stream=mx.cpu)
        return float(2.0 * mx.sum(mx.log(mx.diagonal(L))))

    @staticmethod
    def nu0(d: int) -> float:
        """Wishart 先验自由度。"""
        return float(d + 2)

    @staticmethod
    def lnb0(nu0: float, d: int) -> float:
        """ln B(W0, ν0), W0 = I (Bishop B.79)。"""
        lnb = -(nu0 * d / 2.0) * math.log(2.0)
        return lnb - float(VBGMM.mvlgamma(mx.array([nu0 / 2.0]), d)[0])

    # ── 级联快速冷启动 (B1+B2+C1) ──────────────────────────────────

    @classmethod
    def fast_fit(
        cls,
        x: mx.array,
        shape: tuple[int, int],
        k_max: int = 48,
        ds: int = 8,
        refine: int = 1,
        coarse_iter: int = 25,
    ):
        """多分辨率级联冷启动 (全量冷拟合 ~16s → ~3-4s, ~5×):
        ① 1/ds² 分辨率拟合 coarse_iter 轮 —— 剖析显示成本结构是
        轮数×固定开销 (ELBO 同步) 而非像素数, 低分辨率只是粗种子
        (快初始化由级联吸收), 轮数才是要压的;
        ② 暖启动 + 全分辨率精化 refine 轮 (每轮 ~1.6s 全数据 E 步)。
        500ms 级还需 E 步 kernel 级优化 (同步消除), 属实现层工程,
        记录在 roadmap。"""
        h, w = shape
        sub = x.reshape(h, w, x.shape[1])[::ds, ::ds].reshape(-1, x.shape[1])
        gm_small = cls(sub, k_max=k_max, max_iter=coarse_iter)
        return cls(x, k_max=k_max, warm=gm_small.posterior,
                   max_iter=refine)

    # ── 特征图 → 特征矩阵 (本模块按需组装) ─────────────────────────

    HS_FEAT_NAMES: ClassVar[list[str]] = list(
        ["log_mag", "slope", "resid", "bump", "spread", "ori_R", "phase_coh"]
    )  # HS 支路与 L 同构 (同名谱特征, 证据规则直接复用)

    @staticmethod
    def hs_feature_matrix(hs: mx.array) -> mx.array:
        """HS 复数色相通道 → (H,W,7) 谱特征 (与 L 支路同构)。
        Re/Im 各跑一遍 riesz, 逐特征图按通道能量加权合并 ——
        判别实验 (2026-08-08): 原始值+梯度 3 列版把等亮度色度
        光栅误报为边缘 (0.493) 且无纹理证据机制; 7 列谱特征版
        纹理分类 1.000 / 边缘 0.000。色度支路需要同样的谱描述。
        成本: 两次 riesz (色度分辨率本可降采样, 留作优化钩)。"""
        f_re = RieszWavelet(mx.real(hs)).features()
        f_im = RieszWavelet(mx.imag(hs)).features()
        w_re = mx.exp(f_re.log_mag)
        w_im = mx.exp(f_im.log_mag)
        wsum = w_re + w_im + 1e-12
        cols = []
        for name in VBGMM.HS_FEAT_NAMES:
            key = "residual" if name == "resid" else name
            a = getattr(f_re, key)
            b = getattr(f_im, key)
            cols.append((a * w_re + b * w_im) / wsum)
        return mx.stack(cols, axis=-1)

    @staticmethod
    def feature_matrix(feat: FeatureMaps) -> mx.array:
        """rw.features() 的逐像素特征图 → (N,7) 特征矩阵 (未标准化)。
        选哪几列、什么顺序由本模块按 FEAT_NAMES 决定; 列名 resid
        对应特征图 residual。
        HS 色度支路不并入此矩阵 (特征级合并稀释色相反差且丢通路
        出身, 2026-08-08 实验定) —— 用 hs_feature_matrix 建独立
        模型, 似然级融合。"""
        cols = [
            feat.log_mag,
            feat.slope,
            feat.residual,
            feat.bump,
            feat.spread,
            feat.ori_R,
            feat.phase_coh,
        ]
        return mx.stack(cols, axis=-1).reshape(feat.log_mag.size, 7)

    def __post_init__(self):
        """标准化并拟合 (冷启动可选 α0 网格), 再导出混合权重与
        原始空间的分量均值。"""
        x = self.x_orig
        self.mu = mx.mean(x, axis=0)
        self.sd = mx.maximum(mx.sqrt(mx.var(x, axis=0)), 1e-6)
        z = (x - self.mu) / self.sd
        self.fit(x, 0)
        nk = mx.sum(self.r, axis=0)
        self.weights = self.alpha / mx.sum(self.alpha)
        self.means_orig = (self.r.T @ x) / mx.maximum(nk[:, None], 1e-12)
        # online VB 统计量状态的初值 (online_update 的 EWMA 起点)
        nks, xbars, ss = self.stats(z, self.r)
        self.acc = self.accumulate(nks, xbars, ss)

    # ── VB-EM 主循环 ──────────────────────────────────────────────

    def fit(self, x: mx.array, key: int = 0):
        """离线拟合 = 同一数据上的 online_update(ρ=1) 迭代至 ELBO 收敛。

        ρ=1 时 EWMA 退化为全量替换, online_update 的一次调用精确等于
        经典 VB-EM 的一次迭代 (E 步 → 记 bound → M 步), 故离线/在线
        共享同一条代码路径。子采样只作用迭代过程, 最终责任对全图重算。
        """
        z = (x - self.mu) / self.sd
        sub = self.subsample
        if 0 < sub < z.shape[0]:
            idx = mx.random.permutation(z.shape[0], key=mx.random.key(1))
            x_fit = x[idx[:sub]]
            z_fit = z[idx[:sub]]
        else:
            x_fit, z_fit = x, z

        if self.warm is not None:
            self.posterior = self.warm  # 暖启动: 以上帧后验进入循环
        else:
            # 冷启动: 随机中心硬分配的初始责任直接做一次 M 步
            # (后验直出, 不走累积量往返 —— float32 相消会偏移轨迹)
            r0 = self.init_resp(z_fit, key)
            nk, xbar, s = self.stats(z_fit, r0)
            self.acc = self.accumulate(nk, xbar, s)
            self.posterior = self._posterior(nk, xbar, s)

        prev = -math.inf
        # 记录点对齐经典 VB-EM (q_i, r_i 紧致 bound): 冷启动 q1 在循环
        # 外, 循环少跑一轮; 暖启动第一次调用只 E+M 不记 (old 不记录
        # 入口 bound); 收尾补最后一步 E + 记录 (未收敛时)
        n_iter = self.max_iter if self.warm is not None else self.max_iter - 1
        done = False
        for it in range(n_iter):
            mon = it > 0 or self.warm is None
            r = self.online_update(x_fit, rho=1.0, monitor=mon)
            if mon:
                bound = self.elbo[-1]
                # 只在正的微小增益时收敛; ELBO 下降是 bug 信号, 不能当收敛
                gain = bound - prev
                if prev > -math.inf and 0.0 <= gain < self.tol * max(abs(prev), 1.0):
                    done = True
                    break
                prev = bound
        if not done:
            r = self.e_step(z_fit, self.posterior)  # type: ignore
            self.elbo.append(self.compute_elbo(z_fit, r, self.posterior))  # type: ignore
        self.r = r if z_fit is z else self.e_step(z, self.posterior)  # type: ignore
        self.alpha = self.posterior.alpha  # type: ignore

    def init_resp(self, z: mx.array, key: int = 0) -> mx.array:
        """冷启动初始责任: 随机中心硬分配 + 平滑 (key 固定 → 可复现)。
        暖启动不需要 —— 直接以旧后验进入 EM 循环。"""
        n, _ = z.shape
        k = self.k_max
        idx = mx.random.permutation(n, key=mx.random.key(key))[:k]
        centers = z[idx]
        d2 = mx.sum(z**2, axis=1)[:, None] + mx.sum(centers**2, axis=1)[None, :]
        d2 = d2 - 2.0 * (z @ centers.T)
        assign = mx.argmin(d2, axis=1)
        return mx.eye(k)[assign] * 0.9 + 0.1 / k

    @staticmethod
    def stats(z: mx.array, r: mx.array):
        """责任加权的充分统计量 N_k / x̄_k / S_k。"""
        nk = mx.sum(r, axis=0)
        safe = mx.maximum(nk, 1e-12)
        xbar = (r.T @ z) / safe[:, None]
        m2 = mx.einsum("nk,nd,ne->kde", r, z, z) / safe[:, None, None]
        s = m2 - xbar[:, :, None] @ xbar[:, None, :]
        return nk, xbar, s

    @staticmethod
    def accumulate(nk: mx.array, xbar: mx.array, s: mx.array):
        """责任加权统计量 → 充分统计累积量 (A,B,C) = (Σr, Σr·z, Σr·zzᵀ)。
        后验的全部输入; online VB 的 EWMA 作用在累积量上。"""
        a = nk
        b = nk[:, None] * xbar
        c = nk[:, None, None] * (s + xbar[:, :, None] @ xbar[:, None, :])
        return a, b, c

    def posterior_from_acc(self, a: mx.array, b: mx.array, c: mx.array) -> Posterior:
        """由充分统计累积量出后验 (online VB 路径)。恢复 S 有
        m2−x̄x̄ᵀ 的 float32 相消 (比 _posterior 直出多一次往返),
        由逐分量特征值钳底兑住 —— 离线路径不走这里。"""
        safe = mx.maximum(a, 1e-12)
        xbar = b / safe[:, None]
        m2 = c / safe[:, None, None]
        s = m2 - xbar[:, :, None] @ xbar[:, None, :]
        return self._posterior(a, xbar, s)

    def _posterior(self, nk: mx.array, xbar: mx.array, s: mx.array) -> Posterior:
        """NIW/Dirichlet 后验更新 (Bishop 10.57–10.63 式),
        含 S_k 特征值钳底 (防 W⁻¹ 不定, 详见内联注释)。"""
        d = xbar.shape[1]
        alpha = self.alpha0 + nk
        beta = self.beta0 + nk
        nu = self.nu0(d) + nk
        m = (nk / beta)[:, None] * xbar  # m0 = 0 的收缩

        # W_k⁻¹ = W0⁻¹ + N_k·Ŝ_k + (β0·N_k/β_k)·x̄x̄ᵀ (m0=0, W0=I)。
        # Ŝ_k = S_k 的特征值地板 (替代原 1e-3 固定正则, 曾试 LW 收缩,
        # 见下行)。S_k = m2 − x̄x̄ᵀ 的相消在 float32 下产生 O(1e-4)
        # 的负特征值; 大分量 N_k·λ < −1 时 W_k⁻¹ 不定 → inv 出不定的
        # W → maha 变负 → 整行 log_rho −inf → 责任 NaN (实测 nat10
        # 第 32 轮爆发)。Ledoit-Wolf 收缩对这类分量恰好估计 λ≈0
        # (Frobenius 意义良态), 救不了单方向的负特征值, 且收缩整体
        # 偏离精确 M 步打破 ELBO 单调性, 故弃用。改为逐分量特征值
        # 钳底 ε·μ (μ = 该分量平均特征值, ε=1e-3): 只修退化方向,
        # 不动良态方向, 地板随分量尺度自适应 (饱和/量化特征的近零
        # 方差方向同样被兜住, 覆盖原 1e-3 正则的防爆职能)。
        eye = mx.eye(d)
        tr = mx.sum(s * eye, axis=(1, 2))  # Tr(S_k), 取 μ 用

        # MLX 的 eigh/inv/cholesky 只有 CPU stream, 逐分量算
        w_list, logdet_w, tr_w = [], [], []
        for j in range(self.k_max):
            ev, q_vec = mx.linalg.eigh(s[j], stream=mx.cpu)
            mu_j = max(float(tr[j]) / d, 1e-6)
            ev = mx.maximum(ev, 1e-3 * mu_j)
            s_floor = (q_vec * ev) @ q_vec.T
            winv_j = eye + float(nk[j]) * s_floor
            winv_j = winv_j + float(self.beta0 * nk[j] / beta[j]) * (
                xbar[j][:, None] @ xbar[j][None, :]
            )
            wj = mx.linalg.inv(winv_j, stream=mx.cpu)
            w_list.append(wj)
            logdet_w.append(self.logdet_spd(wj))
            tr_w.append(float(mx.trace(wj)))
        w = mx.stack(w_list)
        logdet_w = mx.array(logdet_w)
        tr_w = mx.array(tr_w)
        mx.eval(w, logdet_w, tr_w)

        log_pi = self.digamma(alpha) - self.digamma(mx.sum(alpha))  # E[ln π_k]
        i_off = mx.arange(d, dtype=mx.float32)
        log_lt = mx.sum(self.digamma((nu[:, None] + 1.0 - i_off) / 2.0), axis=1)
        log_lt = log_lt + d * math.log(2.0) + logdet_w  # E[ln|Λ_k|]

        return Posterior(alpha, beta, nu, m, w, logdet_w, tr_w, log_pi, log_lt)

    def e_step(
        self, z: mx.array, q: Posterior, prior: mx.array | None = None
    ) -> mx.array:
        """r_nk ∝ π̃_k·|Λ̃_k|^{1/2}·exp(−D/2β_k − ν_k/2·maha_nk)。
        prior: 可选 (N,K) 外加对数证据 (反馈边, 见 feedback_round)。"""
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
        if prior is not None:
            log_rho = log_rho + prior
        log_rho = log_rho - mx.max(log_rho, axis=1, keepdims=True)
        rho = mx.exp(log_rho)
        return rho / mx.sum(rho, axis=1, keepdims=True)

    def compute_elbo(self, z: mx.array, r: mx.array, q: Posterior) -> float:
        """Bishop 10.70–10.77。统计量必须用 E 步后的新 r 重算,
        否则算出来的不是当前 q 的真实下界, 会出现假性的非单调。"""
        d = z.shape[1]
        k = self.k_max
        nu0 = self.nu0(d)
        nk, xbar, s = self.stats(z, r)

        def qform(a: mx.array) -> mx.array:
            """二次型 aᵀW_ka, (K,D) → (K,)。"""
            return mx.einsum("kd,kde,ke->k", a, q.w, a)

        # E[ln p(X|Z,μ,Λ)] (10.71)
        tr_sw = mx.einsum("kde,ked->k", s, q.w)  # Tr(S_k W_k)
        t_x = q.log_lt - d / q.beta - q.nu * tr_sw - q.nu * qform(xbar - q.m)
        e_x = 0.5 * float(mx.sum(nk * t_x) - mx.sum(nk) * d * math.log(2.0 * math.pi))

        # E[ln p(Z|π)] (10.72) 与 E[ln p(π)] (10.73)
        e_z = float(mx.sum(nk * q.log_pi))
        e_pi = float(self.lgamma(mx.array([k * self.alpha0]))[0])
        e_pi -= k * float(self.lgamma(mx.array([self.alpha0]))[0])
        e_pi += float((self.alpha0 - 1.0) * mx.sum(q.log_pi))

        # E[ln p(μ,Λ)] (10.74): 注意马氏项是 β0·ν_k 不是 β_k·ν_k
        # (后者 ≈ nk², 随聚类收紧爆炸 → 假发散)
        t_mu = q.log_lt - d * self.beta0 / q.beta - self.beta0 * q.nu * qform(q.m)
        e_mula = 0.5 * k * d * math.log(self.beta0 / (2.0 * math.pi))
        e_mula += 0.5 * float(mx.sum(t_mu)) + k * self.lnb0(nu0, d)
        e_mula += 0.5 * (nu0 - d - 1.0) * float(mx.sum(q.log_lt))
        e_mula -= 0.5 * float(mx.sum(q.nu * q.tr_w))  # W0⁻¹ = I → Tr(W)

        # E[ln q(Z)] (10.75) 与 E[ln q(π)] (10.76)
        e_qz = float(mx.sum(r * mx.log(mx.maximum(r, 1e-12))))
        e_qpi = float(mx.sum((q.alpha - 1.0) * q.log_pi))
        e_qpi += float(self.lgamma(mx.sum(q.alpha))) - float(
            mx.sum(self.lgamma(q.alpha))
        )

        # E[ln q(μ,Λ)] (10.77); Wishart 熵 H[Λ] (Bishop B.82)
        neg_lnb = 0.5 * q.nu * q.logdet_w
        neg_lnb = (
            neg_lnb + (q.nu * d / 2.0) * math.log(2.0) + self.mvlgamma(q.nu / 2.0, d)
        )
        h = neg_lnb - 0.5 * (q.nu - d - 1.0) * q.log_lt + q.nu * d / 2.0
        t_qmla = 0.5 * q.log_lt + 0.5 * d * mx.log(q.beta / (2.0 * math.pi))
        e_qmla = float(mx.sum(t_qmla - d / 2.0 - h))

        return e_x + e_z + e_pi + e_mula - e_qz - e_qpi - e_qmla

    # ── 聚类结果: 类证据 → 后验似然 ───────────────────────────────

    def infer(self, x_new: mx.array) -> mx.array:
        """逐帧推断: 固定已拟合的后验参数, 对新特征只做一次 E 步。
        这是实时模式每帧的唯一贝叶斯计算 (毫秒级)。"""
        z = (x_new - self.mu) / self.sd
        return self.e_step(z, self.posterior)

    def online_update(
        self, x_new: mx.array, rho: float = 0.05, monitor: bool = False
    ) -> mx.array:
        """一帧在线 VB 更新 (Sato 式, 常数遗忘因子 = 跟踪而非收敛)。

        E 步 (当前后验) → monitor 时记该后验的紧致 ELBO → 帧充分
        统计量 → EWMA 进 acc → 重出后验。μ/sd 冻结于冷启动 (z 空间
        不漂移是新旧统计量可混的前提)。死分量 (A_k→0) 自动剪枝;
        新结构的"生"未实现 —— 被最近分量硬吞 (重播种留作后续)。
        ρ=1 时就是经典 VB-EM 的一次迭代 (离线 fit 即反复调此)。
        有效记忆窗口 ≈ 1/ρ 帧。返回新帧责任 (N,K)。"""
        z = (x_new - self.mu) / self.sd  # type: ignore
        r = self.e_step(z, self.posterior)  # type: ignore
        if monitor:
            # E 步刚结束, r 对当前后验紧致 —— 这是收敛判据的正确取点
            self.elbo.append(self.compute_elbo(z, r, self.posterior))  # type: ignore
        nk, xbar, s = self.stats(z, r)
        a, b, c = self.accumulate(nk, xbar, s)
        if rho >= 1.0 or self.acc is None:
            # ρ=1: 全量替换 (离线 EM 语义), 旧 acc 无关; 后验直出,
            # 不走累积量往返 (float32 相消会偏移轨迹, 见 posterior_from_acc)
            self.acc = (a, b, c)
            self.posterior = self._posterior(nk, xbar, s)
        else:
            A, B, C = self.acc
            self.acc = (
                (1.0 - rho) * A + rho * a,
                (1.0 - rho) * B + rho * b,
                (1.0 - rho) * C + rho * c,
            )
            self.posterior = self.posterior_from_acc(*self.acc)
        self.alpha = self.posterior.alpha
        self.r = r
        self.weights = self.alpha / mx.sum(self.alpha)
        nk2 = mx.maximum(mx.sum(r, axis=0), 1e-12)
        self.means_orig = (r.T @ x_new) / nk2[:, None]
        return r

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
        逐像素特征不会被责任混合稀释, 分量均值才会。
        L/HS 两通路各自独立建模 (separate), 证据规则两模型通用。"""
        x = self.x_orig if x is None else x
        assert x is not None, "未拟合且未传 x"
        f = {name: x[:, i] for i, name in enumerate(self.FEAT_NAMES)}  # type: ignore
        if cls == "edge":
            thr = self.phase_coh_thr(f["phase_coh"])
            mask = (f["resid"] < 1.0) & (f["spread"] > 1.0) & (f["phase_coh"] > thr)
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

    @staticmethod
    def phase_coh_thr(v: mx.array) -> float:
        """phase_coh 阈值的锚点化: 按全图分布的 20%/90% 分位取 lo/hi,
        thr = lo + 0.6·(hi−lo), 再夹在 [0.3, 0.5]。合成锚点上
        lo≈0.17 (平坦), hi≈0.55 (边缘), 0.6 相对位 ≈ 原固定阈值 0.4;
        分位自适应让阈值随图内增益/噪声整体漂移, 夹取防极端分布退化。"""
        s = mx.sort(v)
        n = s.shape[0]
        lo = float(s[int(0.2 * (n - 1))])
        hi = float(s[int(0.9 * (n - 1))])
        return float(mx.clip(mx.array(lo + 0.6 * (hi - lo)), 0.3, 0.5))

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
        """硬标签: argmax_k r_nk。"""
        return mx.argmax(self.r, axis=1)

    def feedback_round(
        self,
        enh: mx.array,
        lam: float = 0.3,
        mu: float = 10.0,
        x: mx.array | None = None,
        r: mx.array | None = None,
    ) -> mx.array:
        """EdgePrior 增强图的阻尼反馈 (虚线边, flow.md 迭代协议)。

        注入是不确定度门控的残差驱动:
            prior[n,k] = μ · margin_n · resid_n · frac_k
        margin_n = 1 − max_k r_nk (聚类自身的不确定度), resid_n =
        max(enh−like, 0) (传播与似然的分歧)。三重门控各司其职:
        frac_k 让证据只流向边缘类分量; resid 使一致区 (enh≈like)
        零注入, 堵死自确认; margin 使自信像素 (纹理/平坦区 r 近
        one-hot) 零注入 —— 没有 margin 时残差注入会把传播渗漏
        放大回聚类 (实测纹理区 like +58%), margin≈0 恰好关上
        这扇侧门。三项都有界且随收敛归零, 天然符合定点协议。

        返回新责任, 不更新 self.r —— frac 锚定因此自然成立:
        调用方先取 frac = class_fraction("edge") (读前馈 r), 再用
        r_new @ frac 算反馈似然, 无重算漂移。多轮反馈由调用方
        显式传 r=r_new (协议上限 2 轮, 未收敛分歧保留)。

        诚实的经验天花板: VB 平均场后验系统性偏尖 (flow.md §2),
        合成图上 margin 几乎处处 ≈0, 故本反馈的实际调整量 ≈0。
        这不是失效而是结论 —— gap 处边缘分量在 log 证据上差数百
        纳特, 分歧不该由聚类吸收: 桥接归 EdgePrior 的增强图, 分歧
        按 flow.md 保留交下游仲裁。让反馈真正有分量需先温度缩放
        r 软化后验 (单独改造, 见下)。
        """
        # ponytail: μ 由合成验证手工定; 当前数据上 margin≈0 使注入≈0,
        # 反馈要起作用依赖温度缩放 r 的改造 (flow.md §2 已背书偏尖诊断)
        x = self.x_orig if x is None else x
        r = self.r if r is None else r
        frac = self.class_fraction("edge", x, r)
        e = enh.reshape(-1).astype(mx.float32)
        resid = mx.maximum(e - r @ frac, 0.0)
        margin = 1.0 - mx.max(r, axis=1)
        prior = mu * (margin * resid)[:, None] * frac[None, :]
        z = (x - self.mu) / self.sd
        r_inj = self.e_step(z, self.posterior, prior)  # type: ignore
        return (1.0 - lam) * r + lam * r_inj  # type: ignore

    def edge_likelihood(
        self,
        shape: tuple[int, int],
        x: mx.array | None = None,
        r: mx.array | None = None,
    ) -> mx.array:
        """逐像素边缘似然 (H,W)。x/r 缺省用拟合数据,
        逐帧模式传 infer() 的结果。"""
        return self.class_likelihood("edge", x, r).reshape(shape)

    @staticmethod
    def neighbor_similarity(r: mx.array, shape: tuple[int, int]) -> mx.array:
        """共分配相似度: r_i·r_j 是后验意义下两像素同类的概率。
        对 4 邻域取均值 → 区域内部 ≈1, 边界 ≈0, 即相似性聚类的
        软边界图 (不需要 N×N 全相似度矩阵)。
        按实际邻居数归一 (边框 3、角 2; 恒除 4 会让边框相似度
        系统性偏低出伪边框)。"""
        h, w = shape
        rr = r.reshape(h, w, -1)
        sim = mx.zeros((h, w))
        cnt = mx.zeros((h, w))
        right = mx.sum(rr[:, :-1] * rr[:, 1:], axis=-1)
        sim = sim.at[:, :-1].add(right).at[:, 1:].add(right)
        cnt = cnt.at[:, :-1].add(1.0).at[:, 1:].add(1.0)
        down = mx.sum(rr[:-1] * rr[1:], axis=-1)
        sim = sim.at[:-1].add(down).at[1:].add(down)
        cnt = cnt.at[:-1].add(1.0).at[1:].add(1.0)
        return sim / mx.maximum(cnt, 1.0)

    @staticmethod
    def _hsv_palette(k: int, s: float = 0.75, v: float = 0.95) -> mx.array:
        """黄金比散色调色板 (K,3): h = (i·φ)%1, HSV→RGB 标准分段。"""
        h = (mx.arange(k, dtype=mx.float32) * 0.6180339887) % 1.0
        f = h * 6.0 % 1.0
        sec = (h * 6.0).astype(mx.int32) % 6
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        def pick(vals: list) -> mx.array:
            """按扇区从 6 个候选值中选择 (标量广播, 数组逐像素)。"""
            out = mx.full_like(h, vals[0]) if isinstance(vals[0], float) else vals[0]
            for si in range(1, 6):
                vi = vals[si]
                vi = mx.full_like(h, vi) if isinstance(vi, float) else vi
                out = mx.where(sec == si, vi, out)
            return out

        r = pick([v, q, p, p, t, v])
        g = pick([t, v, v, q, p, p])
        b = pick([p, p, t, v, v, q])
        return mx.stack([r, g, b], axis=-1)

    def macro_labels(self, k_macro: int = 8) -> mx.array:
        """分量级凝聚合并 → 宏簇标签 (K,) int。

        距离用 Bhattacharyya (z 空间, 含协方差 —— 椭球形状参与合并
        判决, 取代纯均值欧氏; 曾用均值二级 VBGMM, 已弃):
            d_B = ⅛·ΔμᵀΣ̄⁻¹Δμ + ½·ln(detΣ̄ / √(detΣᵢ·detΣⱼ))
        每轮合并质量加权的池化高斯, 直到剩 k_macro 个。死分量
        (weight ≤ 1e-3) 按 d_B 归入最近宏簇。不改变 like = r@frac
        主管线; 用途: 去碎片化显示 / 分割层的语义先验。"""
        z = (self.x_orig - self.mu) / self.sd  # type: ignore
        nk, xbar, s = self.stats(z, self.r)  # type: ignore
        w = nk / mx.sum(nk)
        alive = Utils.nonzero(w > 1e-3).tolist()

        # 活跃分量的 (μ, Σ, 质量) —— z 空间
        mus = [xbar[j] for j in alive]
        covs = [s[j] for j in alive]
        mass = [float(nk[j]) for j in alive]
        groups = [[i] for i in range(len(alive))]  # 宏簇 → 活跃下标

        def bhatt(m1, c1, m2, c2) -> float:
            """两高斯的 Bhattacharyya 距离 (utils 共享实现)。"""
            return float(
                Utils.bhatt(
                    m1[None], c1[None], m2[None], c2[None]
                )[0]
            )

        while len(groups) > k_macro:
            # 找 d_B 最小的一对
            best, bp = math.inf, (0, 1)
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    d = bhatt(mus[i], covs[i], mus[j], covs[j])
                    if d < best:
                        best, bp = d, (i, j)
            i, j = bp
            # 质量加权池化: μ, Σ 合并 (Σ 经二阶矩合成)
            mi, mj = mass[i], mass[j]
            m12 = mi + mj
            mu = (mus[i] * mi + mus[j] * mj) / m12
            m2i = covs[i] + mus[i][:, None] @ mus[i][None, :]
            m2j = covs[j] + mus[j][:, None] @ mus[j][None, :]
            cov = (m2i * mi + m2j * mj) / m12 - mu[:, None] @ mu[None, :]
            merged = groups[i] + groups[j]
            for k in sorted(bp, reverse=True):
                del groups[k], mus[k], covs[k], mass[k]
            groups.append(merged)
            mus.append(mu)
            covs.append(cov)
            mass.append(m12)

        # 活跃分量按成员关系打标, 死分量按 d_B 归入最近宏簇
        alive_set = set(alive)
        out = [0] * self.k_max
        for g_id, members in enumerate(groups):
            for t in members:
                out[alive[t]] = g_id
        for j in range(self.k_max):
            if j in alive_set:
                continue
            out[j] = min(
                range(len(groups)),
                key=lambda g: bhatt(xbar[j], s[j], mus[g], covs[g]),
            )
        return mx.array(out, dtype=mx.int32)

    def soft_colors(self, shape: tuple[int, int]):
        """软聚类混色图 (H,W,3): 每个分量一个黄金比散色 (相邻下标
        色相不相邻, K 大时仍可辨), 像素色 = Σ_k r_nk·color_k ——
        r 混合处 (边缘/边界像素) 自然呈现多簇颜色的混合。"""
        pal = self._hsv_palette(self.k_max)  # (K,3)
        rgb = self.r @ pal  # type: ignore  # (N,3)
        return rgb.reshape(shape[0], shape[1], 3)

    def visualize_maps(self, img, shape, out_path: str | Path):
        """原图/边缘似然/纹理似然/硬标签/软聚类混色/软边界 六联图保存。"""
        h, w = shape
        edge = self.class_likelihood("edge").reshape(h, w)
        tex = self.class_likelihood("texture").reshape(h, w)
        labs = self.labels().reshape(h, w).astype(mx.float32)
        soft = self.soft_colors(shape)
        bnd = 1.0 - self.neighbor_similarity(self.r, shape)
        plots = [
            ("original", "gray", img),
            ("edge likelihood", "gray", edge),
            ("texture likelihood", "gray", tex),
            ("labels", "tab10", labs),
            ("soft clusters (r@palette)", None, soft),
            ("1 − neighbor sim", "gray", bnd),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    # from PIL import Image  # 随 natural image 块一起注释, 放开时恢复

    # ── synthetic validation ─────────────────────────────────────────
    # 弱边缘 @x=64 (Δ=0.05), 强边缘 @x=128 (Δ=0.55), 纹理区 x≥192
    H, W = 128, 256
    img = mx.full((H, W), 0.2)
    img[:, 64:128] = 0.25
    img[:, 128:192] = 0.8
    img[:, 192:] = Utils.make_grating((H, 64), 8.0, 0.0)[:, :64]
    img = img + mx.random.normal((H, W), key=mx.random.key(3)) * 0.01

    rw = RieszWavelet(img)
    feat = rw.features()
    gm = VBGMM(VBGMM.feature_matrix(feat), k_max=48)

    e = gm.elbo
    mono = max((e[i] - e[i + 1] for i in range(len(e) - 1)), default=0.0)
    print(f"ELBO: {e[0]:.1f} → {e[-1]:.1f} ({len(e)} iters, 最大回退 {mono:.2f})")
    print(f"effective K = {gm.k_eff()} / {gm.k_max}")
    hdr = "w      " + " ".join(f"{n:>9s}" for n in VBGMM.FEAT_NAMES) + "  e_frac t_frac"
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
    gm.visualize_maps(img, (H, W), path)
    print(path)

    # ── 双通路: 等亮度红绿边 (L 支路不可见, HS 支路检出) ──────────
    from color import Color

    red = mx.array([1.0, 0.0, 0.0])
    grn = mx.array([0.0, 0.5094, 0.0])  # Rec601 亮度均 0.299 → 等亮度边
    rgb_c = mx.concatenate(
        [mx.broadcast_to(red, (H, W // 2, 3)),
         mx.broadcast_to(grn, (H, W // 2, 3))],
        axis=1,
    )
    lum, hs_c = Color.split_dual_path(rgb_c)
    # 微噪声防奇异 (恒定特征 → 协方差奇异 → eigh C++ 层 terminate)
    lum = lum + mx.random.normal((H, W), key=mx.random.key(5)) * 0.003
    assert float(mx.max(mx.abs(lum - 0.299))) < 0.02, "等亮度前提"
    feat_c = RieszWavelet(lum).features()
    gm_g = VBGMM(VBGMM.feature_matrix(feat_c), k_max=8)
    like_g = gm_g.edge_likelihood((H, W))
    # separate 模式: HS 独立模型 + 似然级概率 OR 融合
    gm_h = VBGMM(VBGMM.hs_feature_matrix(hs_c).reshape(-1, 7), k_max=32)
    # k_max 不能小: 边界带只有 ~3% 质量, 分量不足会被红/绿簇
    # 稀释 (k=8 → 0.04, k=32 → 0.50, 实测)
    like_h = gm_h.edge_likelihood((H, W))
    like_d = 1 - (1 - like_g) * (1 - like_h)
    bnd_g = float(like_g[:, 126:130].mean())
    bnd_d = float(like_d[:, 126:130].mean())
    int_d = float(like_d[:, :100].mean())
    assert bnd_g < 0.1, f"灰度支路应看不见等亮度边: {bnd_g:.3f}"
    assert bnd_d > 5 * max(int_d, 1e-3) and bnd_d > 0.1, (
        f"HS 支路应检出: 边界 {bnd_d:.3f} vs 内部 {int_d:.3f}"
    )
    print(f"双通路: 等亮度边 灰度={bnd_g:.3f}(盲) 双通路={bnd_d:.3f} "
          f"(内部 {int_d:.3f}) ✓")

    # ── 色度纹理: 等亮度色度光栅应被分类为纹理 (HS 谱特征的意义) ──
    phase_g = 0.5 + 0.5 * mx.sin(2 * mx.pi * mx.arange(W)[None, :] / 8.0)
    hue_g = phase_g * (1.0 / 3.0) * mx.ones((H, 1))
    inb_g = ((mx.arange(W)[None, :] >= 64) & (mx.arange(W)[None, :] < 192))
    sat_g = inb_g.astype(mx.float32) * 0.9 * mx.ones((H, 1))
    hsl_g = mx.stack([hue_g, sat_g, mx.full((H, W), 0.5)], axis=-1)
    rgb_g = Color.hsl_to_rgb(hsl_g)
    rgb_g = mx.clip(
        rgb_g + mx.random.normal((H, W, 3), key=mx.random.key(11)) * 0.005,
        0.0, 1.0,
    )
    _, hs_g = Color.split_dual_path(rgb_g)
    x7 = VBGMM.hs_feature_matrix(hs_g).reshape(-1, 7)
    gm_7 = VBGMM(x7, k_max=16)
    tex7 = float(gm_7.class_likelihood("texture").reshape(H, W)[:, 100:150].mean())
    ed7 = float(gm_7.class_likelihood("edge").reshape(H, W)[:, 100:150].mean())
    assert tex7 > 0.5 and ed7 < 0.2, (
        f"色度光栅应判为纹理: tex={tex7:.2f} edge={ed7:.2f}"
    )
    print(f"色度纹理: 等亮度光栅 tex={tex7:.2f} edge={ed7:.2f} "
          f"(3列原始值版曾误报 edge=0.49) ✓")

    # ── natural image (慢: 冷启动拟合 ~10-25s, 默认注释; 全量验证时放开) ──
    # im = Image.open(Utils.project_root() / "images/12.png").convert("L")
    # arr = Color.image_to_mlx(im)
    # feat2 = RieszWavelet(arr).features()
    # gm2 = VBGMM(VBGMM.feature_matrix(feat2), k_max=48)
    # print(f"12.png: ELBO {gm2.elbo[0]:.1f} → {gm2.elbo[-1]:.1f}, K_eff={gm2.k_eff()}")
    # path2 = Utils.project_root() / "artifacts/vbgmm_12.png"
    # gm2.visualize_maps(arr, arr.shape, path2)
    # print(path2)

    # ── 实时模式: 相邻帧 (边界移动) 冷启动 vs 暖启动 ────────────────
    import time

    img2 = mx.full((H, W), 0.2)
    img2[:, 72:136] = 0.25  # 弱边缘 64→72, 强边缘 128→136, 纹理区 192→200
    img2[:, 136:200] = 0.8
    img2[:, 200:] = Utils.make_grating((H, 56), 8.0, 0.0)[:, :56]
    img2 = img2 + mx.random.normal((H, W), key=mx.random.key(4)) * 0.01

    t0 = time.perf_counter()
    rw.update(img2)
    feat = rw.features()
    X2 = VBGMM.feature_matrix(feat)
    mx.eval(X2)
    t1 = time.perf_counter()
    # 冷启动对照 (慢, ~26s, 默认注释; 全量验证时放开)
    # gm_cold = VBGMM(X2, k_max=48)
    # t2 = time.perf_counter()
    gm_warm = VBGMM(X2, k_max=48, warm=gm.posterior, subsample=8192)
    t3 = time.perf_counter()
    print(
        f"frame2 特征提取 {1000 * (t1 - t0):.0f}ms | "
        f"暖启动+子采样 {len(gm_warm.elbo)} iters {t3 - t1:.2f}s"
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

    # ── online VB: 移动边界序列的跟踪 ─────────────────────────────
    # 边界每帧移 2px, 共 6 帧; online_update (ρ=0.3) 跟踪, 与末帧
    # 冷启动拟合对比稳态偏差
    def frame(f: int) -> mx.array:
        """第 f 帧: 三条边界整体右移 2f px。"""
        im = mx.full((H, W), 0.2)
        im[:, 64 + 2 * f : 128 + 2 * f] = 0.25
        im[:, 128 + 2 * f : 192 + 2 * f] = 0.8
        im[:, 192 + 2 * f :] = Utils.make_grating((H, 64 - 2 * f), 8.0, 0.0)
        return im + mx.random.normal((H, W), key=mx.random.key(10 + f)) * 0.01

    gm_on = VBGMM(VBGMM.feature_matrix(RieszWavelet(frame(0)).features()), k_max=48)
    X_f = VBGMM.feature_matrix(RieszWavelet(frame(0)).features())
    t0 = time.perf_counter()
    for f in range(1, 6):
        rw.update(frame(f))
        X_f = VBGMM.feature_matrix(rw.features())
        gm_on.online_update(X_f, rho=0.3)
    mx.eval(gm_on.r)
    t1 = time.perf_counter()
    # 末帧冷启动对照 (慢, ~26s, 默认注释); 断言用实测记录值
    # (2026-08-07: weak=0.43, strong=0.73), 全量验证时放开对照拟合
    # gm_c5 = VBGMM(X_f, k_max=48)
    # like_c5 = gm_c5.edge_likelihood((H, W))
    # weak_c5 = float(like_c5[:, 72:77].mean())
    # strong_c5 = float(like_c5[:, 136:141].mean())
    weak_c5, strong_c5 = 0.43, 0.73
    like_on = gm_on.class_likelihood("edge", x=X_f, r=gm_on.r).reshape(H, W)
    fw = 74 + 10  # 末帧弱边缘 ≈ 64+2·5=74, 强边缘 ≈ 138
    weak_on = float(like_on[:, 72:77].mean())
    strong_on = float(like_on[:, 136:141].mean())
    old_on = float(like_on[:, 62:66].mean())  # 旧位置应已衰减
    print(
        f"online VB: 5 帧跟踪 {1000 * (t1 - t0) / 5:.0f}ms/帧 | "
        f"weak@{fw}: online={weak_on:.2f} 冷启动={weak_c5:.2f} | "
        f"strong@138: online={strong_on:.2f} 冷启动={strong_c5:.2f} | "
        f"旧位置@64 衰减到 {old_on:.2f}"
    )
    assert abs(weak_on - weak_c5) < 0.2, "online 与冷启动的弱边缘似然偏差过大"
    assert abs(strong_on - strong_c5) < 0.2, "强边缘偏差过大"
    assert old_on < weak_on, "旧边缘位置应衰减低于新位置"
