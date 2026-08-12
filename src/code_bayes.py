"""逐码对角高斯贝叶斯 (全分辨率生成模型) —— 码簿可枚举 + 训练时码观测
场景下的精确解 (experiment_fullres 实测: seen 0.965 vs 池化 SPN 0.485,
3s vs 181s)。

模型: P(x|c) = Π_p N(x_p; μ_cp, σ_cp)   (逐像素独立高斯, 对角协方差)
      P(c)   = (n_c + 1) / (N + K)       (Laplace 平滑码先验)

为什么不需要 SVI/EM: 码训练时观测 → 分量归属已知 → 监督充分统计量即
精确解 (在线 EM 的软分配永远不如白给的监督分配纯, 实测 flat K=288
只有 0.26)。SVI 解的是「分量归属未知」, 这里归属是白给的。

增量: (n, Σx′, Σx′²) 可加 → absorb ≡ fit (自检 1 验证)。移位基准
逐码记录 (该码首行) —— 全局基准会让远类 (x−g)² 在 float32 下
灾难性抵消 (spn.py 教训), 且后验对 σ 的敏感度被 x² 放大:
1% 的 σ 误差 × 大幅值特征 = 几十 nats 的后验漂移 (自检实测)。

边界 (experiment_fullres 未见码探针实测):
  * 未见码码级无泛化 (Laplace 地板罚 > 相邻码特征差) —— 变量级泛化
    (kind/gx/gy) 是 SPN 组合结构的阵地;
  * 依赖特征条件近确定 (逐码方差小); 噪声/光照大变时 SPN 的池化 +
    跨码共享结构更鲁棒。

契约: posterior(feats, codes, log_prior) 与 spn.SPN.posterior 一致
(demo_inverse 的先验注入/运动序列滤波/遮挡先验零改动复用)。

本文件自检: `python src/code_bayes.py`。
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import mlx.core as mx

from utils import Utils


class CodeBayes:
    """逐码对角高斯生成模型: 码 ↔ 分量一一对应。"""

    def __init__(self, cards: tuple[int, ...], dim: int, floor: float = 0.05):
        self.cards = cards  # 各码列基数 (kind, gx, gy, size, z)
        self.dim = dim
        self.floor = floor  # σ 下限 (确定性渲染 → 逐码方差≈0, 必须钳)
        self.n_codes = math.prod(cards)
        self.g: mx.array | None = None  # 全局移位基准 (未见码统计用)
        self.c: mx.array | None = None  # (K,D) 逐码移位基准 (该码首行)
        self.n = mx.zeros(self.n_codes)  # (K,) 逐码计数
        self.s1: mx.array | None = None  # (K,D) Σ(x−c)
        self.s2: mx.array | None = None  # (K,D) Σ(x−c)²
        self.n_g = 0.0  # 全局计数
        self.g1: mx.array | None = None  # (D,) 全局 Σ(x−g)
        self.g2: mx.array | None = None  # (D,) 全局 Σ(x−g)²

    def absorb(self, X: mx.array, code_idx: mx.array) -> None:
        """吸收一批: X (N,D) 原始特征, code_idx (N,) int32 码下标。"""
        if self.g is None:
            self.g = mx.mean(X, axis=0)
            self.c = mx.zeros((self.n_codes, self.dim))
            self.s1 = mx.zeros((self.n_codes, self.dim))
            self.s2 = mx.zeros((self.n_codes, self.dim))
            self.g1 = mx.zeros(self.dim)
            self.g2 = mx.zeros(self.dim)
        assert self.c is not None and self.s1 is not None and self.s2 is not None
        assert self.g is not None
        # 新码: 以该码在本批的首行设逐码移位基准 (局部性防方差抵消)
        for u in set(code_idx.tolist()):
            if self.n[u] == 0:
                first = int(mx.argmax(code_idx == u))
                self.c[u] = X[first]
        d = X - self.c[code_idx]
        oh = mx.equal(
            code_idx[:, None], mx.arange(self.n_total)[None, :]
        ).astype(mx.float32)  # (N,K) one-hot (含临时分量下标)
        assert self.g1 is not None and self.g2 is not None
        self.n = self.n + mx.sum(oh, axis=0)
        self.s1 = self.s1 + oh.T @ d
        self.s2 = self.s2 + oh.T @ (d * d)
        self.n_g += X.shape[0]
        dg = X - self.g[None, :]
        self.g1 = self.g1 + mx.sum(dg, axis=0)
        self.g2 = self.g2 + mx.sum(dg * dg, axis=0)
        mx.eval(self.n, self.s1, self.s2, self.g1, self.g2, self.c)

    @classmethod
    def fit(
        cls,
        X: mx.array,
        code_idx: mx.array,
        cards: tuple[int, ...],
        floor: float = 0.05,
    ) -> CodeBayes:
        """一次性拟合 (= 单次 absorb; 增量与批量逐位近似, 自检 1)。"""
        m = cls(cards, X.shape[1], floor)
        m.absorb(X, code_idx)
        return m

    def params(self) -> tuple[mx.array, mx.array, mx.array]:
        """→ (mu (K,D), sg (K,D), log_prior (K,)); 未见码用全局统计。"""
        assert self.g is not None and self.c is not None
        assert self.s1 is not None and self.s2 is not None
        assert self.g1 is not None and self.g2 is not None
        ns = mx.maximum(self.n, 1.0)[:, None]
        mu = self.c + self.s1 / ns
        var = self.s2 / ns - (self.s1 / ns) ** 2
        sg = mx.sqrt(mx.maximum(var, self.floor**2))
        gmu = self.g + self.g1 / self.n_g
        gvar = self.g2 / self.n_g - (self.g1 / self.n_g) ** 2
        gsg = mx.sqrt(mx.maximum(gvar, self.floor**2))
        seen = (self.n > 0)[:, None]
        mu = mx.where(seen, mu, gmu[None, :])
        sg = mx.where(seen, sg, gsg[None, :])
        log_prior = mx.log((self.n + 1.0) / (self.n_g + self.n_total))
        return mu, sg, log_prior

    def posterior(
        self,
        feats: mx.array,
        codes: mx.array,
        log_prior: mx.array | None = None,
    ) -> mx.array:
        """feats (M,D) 原始特征, codes (K,C) 全枚举 → (M,K) log 后验, 行归一。

        对角高斯点积展开 (防 (M,K,D) 实体化爆显存):
        logp(x|c) = −½(x²·a_c) + x·b_c + c_c,
        a = 1/σ², b = μ/σ², c_c = Σ_p(−½μ²/σ² − log σ)。
        log_prior (K,) 或 (M,K): 外部码先验 (与 SPN.posterior 同契约)。
        """
        mu, sg, lp = self.params()
        idx = self.code_index(codes)
        logp = self.ll(feats, mu[idx], sg[idx]) + lp[idx][None, :]
        if log_prior is not None:
            logp = logp + (log_prior[None, :] if log_prior.ndim == 1 else log_prior)
        return logp - mx.logsumexp(logp, axis=1, keepdims=True)

    @property
    def n_total(self) -> int:
        """总分量数 (码簿 + 已提升临时分量)。"""
        return int(self.n.shape[0])

    @staticmethod
    def ll(feats: mx.array, mu: mx.array, sg: mx.array) -> mx.array:
        """(M,D) × (K,D) → (M,K) 对角高斯 log 密度 (点积展开, 防实体化)。"""
        a = 1.0 / (sg * sg)
        b = mu * a
        c = -0.5 * mx.sum(mu * b, axis=1) - mx.sum(mx.log(sg), axis=1)
        return -0.5 * ((feats * feats) @ a.T) + feats @ b.T + c[None, :]

    def posterior_all(self, feats: mx.array) -> mx.array:
        """(M,D) → (M, K_total) 全分量 log 后验 (码簿 + 临时), 行归一。"""
        mu, sg, lp = self.params()
        logp = self.ll(feats, mu, sg) + lp[None, :]
        return logp - mx.logsumexp(logp, axis=1, keepdims=True)

    def gate(self, feats: mx.array) -> tuple[mx.array, mx.array]:
        """新颖度门控: → (score (M,), is_novel (M,))。

        score = log P(x|全局分量) − max_{已知分量} log P(x|c), 等先验
        似然比 (>0 判新)。已知 = 计数>0 的码簿分量 + 已提升临时分量。
        全分辨率近确定特征下已知/未知似然差数千 nats, 几乎无 ambiguous
        区; 类别不均衡场合可平移阈值 (本任务不需要)。"""
        mu, sg, _ = self.params()
        ll = self.ll(feats, mu, sg)
        known = (self.n > 0)[None, :]
        best = mx.max(mx.where(known, ll, float("-inf")), axis=1)
        assert self.g is not None and self.g1 is not None and self.g2 is not None
        gmu = self.g + self.g1 / self.n_g
        gvar = self.g2 / self.n_g - (self.g1 / self.n_g) ** 2
        gsg = mx.sqrt(mx.maximum(gvar, self.floor**2))
        lg = self.ll(feats, gmu[None, :], gsg[None, :])[:, 0]
        score = lg - best
        return score, score > 0

    def grow(self) -> int:
        """添加临时分量 (开放集提升: 新内容的归属槽), 返回下标。
        零计数初值 → params() 的未见兜底 (全局统计) 自动覆盖。"""
        assert self.s1 is not None and self.s2 is not None
        assert self.c is not None
        idx = self.n_total
        self.n = mx.concatenate([self.n, mx.zeros(1)])
        self.s1 = mx.concatenate([self.s1, mx.zeros((1, self.dim))])
        self.s2 = mx.concatenate([self.s2, mx.zeros((1, self.dim))])
        self.c = mx.concatenate([self.c, mx.zeros((1, self.dim))])
        return idx

    def absorb_stats(
        self, idx: int, n: float, s1: mx.array, s2: mx.array, c: mx.array
    ) -> None:
        """统计量直吸 (提升交接格式): (n, Σ(x−c), Σ(x−c)², 基准 c)。
        基准不同则精确换基: Σ(x−c₀) = s1 + n·Δc,
        Σ(x−c₀)² = s2 + 2Δc·s1 + n·Δc², Δc = c − c₀。
        全局统计 (未见兜底用) 同步换基更新。"""
        assert self.s1 is not None and self.s2 is not None and self.c is not None
        assert self.g is not None and self.g1 is not None and self.g2 is not None
        if self.n[idx] == 0:
            self.c[idx] = c
            self.n[idx] = n
            self.s1[idx] = s1
            self.s2[idx] = s2
        else:
            dc = c - self.c[idx]
            self.s2[idx] = self.s2[idx] + s2 + 2.0 * dc * s1 + n * dc * dc
            self.s1[idx] = self.s1[idx] + s1 + n * dc
            self.n[idx] = self.n[idx] + n
        dg = c - self.g
        self.g2 = self.g2 + s2 + 2.0 * dg * s1 + n * dg * dg
        self.g1 = self.g1 + s1 + n * dg
        self.n_g += n
        mx.eval(self.n, self.s1, self.s2, self.c, self.g1, self.g2)

    def code_index(self, codes: mx.array) -> mx.array:
        """码元组 (K,C) → 下标 (字典序, 与 cards 顺序一致)。"""
        idx = mx.zeros(codes.shape[0], dtype=mx.int32)
        for j in range(len(self.cards)):
            idx = idx * self.cards[j] + codes[:, j].astype(mx.int32)
        return idx

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """safetensors 存盘: 张量二进制体 + JSON 明文头 (config 可读,
        Utils.st_metadata 查看)。.pkl 后缀 → 旧 pickle 格式 (向后兼容)。
        extra: mx 数组以 extra.* 键入文件, 标量 JSON 化进头。"""
        if str(path).endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump({"model": self, "extra": extra or {}}, f)
            return
        assert self.g is not None, "absorb 前无存档内容"
        assert self.s1 is not None and self.s2 is not None
        assert self.c is not None and self.g1 is not None and self.g2 is not None
        arrs = {
            "n": self.n, "s1": self.s1, "s2": self.s2, "c": self.c,
            "g": self.g, "g1": self.g1, "g2": self.g2,
        }
        meta = {
            "config": json.dumps({
                "cards": list(self.cards),
                "dim": self.dim,
                "floor": self.floor,
                "n_g": self.n_g,
            })
        }
        for k, v in (extra or {}).items():
            if isinstance(v, mx.array):
                arrs[f"extra.{k}"] = v
            else:
                meta[f"extra.{k}"] = json.dumps(v)
        mx.save_safetensors(str(path), arrs, meta)

    @staticmethod
    def load(path: str | Path) -> tuple[CodeBayes, dict[str, Any]]:
        """save 的逆操作 (按扩展名识别 safetensors/pickle)。"""
        if str(path).endswith(".pkl"):
            with open(path, "rb") as f:
                d = pickle.load(f)
            return d["model"], d["extra"]
        d = mx.load(str(path))
        hd = Utils.st_metadata(path).get("__metadata__", {})
        cfg = json.loads(hd["config"])
        m = CodeBayes(tuple(cfg["cards"]), int(cfg["dim"]), float(cfg["floor"]))
        m.n_g = float(cfg["n_g"])
        m.n, m.s1, m.s2, m.c = d["n"], d["s1"], d["s2"], d["c"]
        m.g, m.g1, m.g2 = d["g"], d["g1"], d["g2"]
        extra: dict[str, Any] = {
            k[6:]: v for k, v in d.items() if k.startswith("extra.")
        }
        extra.update(
            {k[6:]: json.loads(v) for k, v in hd.items() if k.startswith("extra.")}
        )
        return m, extra


if __name__ == "__main__":
    key = mx.random.key(7)
    n, d = 400, 64
    lab = mx.concatenate([mx.zeros(n // 2), mx.ones(n // 2)]).astype(mx.int32)
    mu1 = mx.concatenate([mx.full((d // 2,), 4.0), mx.zeros(d // 2)])
    X = mx.where(lab[:, None] == 1, mu1[None, :], 0.0) + 0.5 * mx.random.normal(
        shape=(n, d), key=key
    )
    codes = mx.array([[0.0], [1.0]])

    # 1) 精确增量: 分批 absorb ≡ 一次 fit (浮点累加序差异内)
    m1 = CodeBayes.fit(X, lab, cards=(2,))
    m2 = CodeBayes(cards=(2,), dim=d)
    m2.absorb(X[:200], lab[:200])
    m2.absorb(X[200:], lab[200:])
    diff = float(
        mx.max(mx.abs(m1.posterior(X[:8], codes) - m2.posterior(X[:8], codes)))
    )
    assert diff < 1e-3, f"分批 absorb ≠ 一次 fit: {diff}"
    print(f"  ok  精确增量: 分批 ≡ 一次 (|Δpost| = {diff:.2e})")

    # 2) 判别: 后验 argmax 恢复类
    pred = mx.argmax(m1.posterior(X, codes), axis=1)
    acc = float(mx.mean((pred == lab).astype(mx.float32)))
    assert acc > 0.95, f"判别准确率过低: {acc}"
    print(f"  ok  判别: 后验恢复类 (acc = {acc:.3f})")

    # 3) 未见码: 只用类 0 训练 → code 1 后验有限且不占优 (无码级泛化机制)
    m3 = CodeBayes.fit(X[:200], lab[:200], cards=(2,))
    p3 = m3.posterior(X[200:206], codes)
    assert mx.all(mx.isfinite(p3)), "未见码后验含 NaN/inf"
    assert float(mx.max(mx.exp(p3[:, 1]))) < 0.5, "未见码不应占优"
    print("  ok  未见码: 全局先验兜底, 后验有限不占优")

    # 4) 先验注入契约 (同 SPN): 改后验且行归一
    p4 = m1.posterior(
        X[:4], codes, log_prior=mx.array([math.log(0.9), math.log(0.1)])
    )
    assert abs(float(mx.sum(mx.exp(p4))) - 4.0) < 1e-4, "先验注入后未归一"
    print("  ok  先验注入: 行归一成立 (SPN 契约)")

    # 7) 序列化 roundtrip (safetensors): save → load → 逐位一致
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        m1.save(tmp, {"mu": mx.array([0.5])})
        m6, extra6 = CodeBayes.load(tmp)
        d6 = float(
            mx.max(mx.abs(m1.posterior_all(X[:6]) - m6.posterior_all(X[:6])))
        )
        assert d6 < 1e-6, f"roundtrip 后 posterior 不一致: {d6}"
        assert float(extra6["mu"][0]) == 0.5, "extra 未随存"
    finally:
        os.unlink(tmp)
    print("  ok  序列化: safetensors roundtrip 逐位一致, extra 随存")

    # 5) 提升交接: grow + absorb_stats ≡ 直接 absorb 行 (换基恒等式)
    m4 = CodeBayes.fit(X[:200], lab[:200], cards=(2,))
    idx = m4.grow()
    c1 = X[200]  # 类 1 首行作基准
    s1 = mx.sum(X[200:] - c1[None, :], axis=0)
    s2 = mx.sum((X[200:] - c1[None, :]) ** 2, axis=0)
    m4.absorb_stats(idx, 200.0, s1, s2, c1)
    m5 = CodeBayes.fit(X[:200], lab[:200], cards=(2,))
    idx5 = m5.grow()
    m5.absorb(X[200:], mx.full((200,), idx5, dtype=mx.int32))
    d5 = float(
        mx.max(mx.abs(m4.posterior_all(X[:6]) - m5.posterior_all(X[:6])))
    )
    # 容差 0.1 nats: float32 求和序差异 × 后验 x² 敏感度 (见模块头注释)
    assert d5 < 0.1, f"统计直吸 ≠ 行吸收: {d5}"
    print(f"  ok  提升交接: absorb_stats ≡ absorb (|Δpost| = {d5:.2e})")

    # 6) 门控: 已知行判旧, 全新类 (平移 +8) 判新
    _, nov_known = m1.gate(X[:50])
    _, nov_new = m1.gate(X[200:250] + 8.0)
    assert not bool(mx.any(nov_known)), "已知行误判新"
    assert bool(mx.all(nov_new)), "新类未判新"
    print("  ok  门控: 已知判旧, 新类判新")

    print("code_bayes.py: 7 组自检 ✓")
