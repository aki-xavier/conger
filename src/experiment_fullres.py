"""全分辨率 + SVI 探索: 不池化 (144×144×3 = 62208 维) 的三种模型对照。

动机: demo 的块池化 (8×6 → 144 维) 是为 SPN 结构学习服务的
(G 检验两两列 O(c²), 62K 维不可行)。若放弃结构学习, 用固定结构扁平
混合 + 在线 EM (共轭指数族上在线 EM ≡ SVI 自然梯度更新, Hoffman 2013;
Cappé & Moulines 2009), 全分辨率能否带来收益?

三臂:
  nb    每码朴素贝叶斯 (码↔分量一一对应, CodeBayes) —— 模板法上限,
        平凡可增量 (逐码 n/μ/σ 可加);
  flat  K 分量扁平混合在线 EM: Sum(K) × Product(像素独立高斯) 固定结构
        + 分量级码联合计数 (P(码|分量)) —— SVI 对应物;
  spn   池化 SPN 全量重训 (demo 管线, 结构学习对照)。

探针 (生成结构 vs 模板记忆的核心判别): 训练只用 90% 码族 (10% 码整族
保留不训), 测试分 seen / unseen 两组。预期: nb 对未见码无机制 (≈0);
flat 靠分量跨码共享部分泛化; spn 靠组合结构 (kind×位置×尺寸×深度
独立线索) 泛化。

数值: 全分辨率臂用原始特征 + σ_floor=0.05 (确定性渲染 → 逐码方差≈0,
z-score 在近平常量像素上会爆炸, 故不标准化); 充分统计量按全局均值
移位累积 (x−g), 防 float32 方差抵消 (spn.py 教训)。

运行: python experiment_fullres.py [--k 288]
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from code_bayes import CodeBayes
from codebook import Codebook
from evaluator import Evaluator
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from riesz import RieszWavelet
from spn_learner import SPNLearner


class FullresExperiment:
    """全分辨率三臂 + 未见码泛化探针 (k = flat 臂分量数)。"""

    N_TRAIN, N_TEST = 3600, 200
    FLOOR2 = 0.05**2  # σ² 下限 (确定性渲染 → 逐码像素方差≈0)
    HOLDOUT_KEY = 123  # 码族划分种子 (与 experiment_joint 一致)

    def __init__(self, k: int = 64):
        self.k = k
        self.codebook = Codebook(InverseConfig())  # 默认配置 (仅用常量与管线)
        # 码族保留: 10% 码整族不训 (泛化探针)
        cb = Codebook
        perm = mx.random.permutation(
            cb.N_CODES, key=mx.random.key(self.HOLDOUT_KEY)
        ).tolist()
        self.unseen = set(perm[: cb.N_CODES // 10])
        self.train_codes = [i for i in range(cb.N_CODES) if i not in self.unseen]
        self.unseen_codes = perm[: cb.N_CODES // 10]

    @staticmethod
    def sample_codes(pool: list[int], n: int, key: int) -> list[int]:
        idx = mx.random.randint(0, len(pool), shape=(n,), key=mx.random.key(key))
        return [pool[int(i)] for i in idx.tolist()]

    def feats_of(
        self, idxs: list[int], renderer, cam, rw: RieszWavelet | None
    ) -> tuple[mx.array, mx.array]:
        """帧序列 → (全分辨率 (n, 9×20736), 池化 (n, 9×48))。"""
        ex = FeatureExtractor(InverseConfig())
        full, pooled = [], []
        for i in idxs:
            frame = renderer.render(
                self.codebook.to_scene(Codebook.idx_to_code(i)), cam
            )
            v, p, rw = ex.of_frame_pair(frame, rw)
            mx.eval(v, p)  # 逐帧求值, 防惰性图累积爆显存
            full.append(v)
            pooled.append(p)
        return mx.stack(full), mx.stack(pooled)

    def build(self) -> tuple[mx.array, ...]:
        """渲染 + 特征 (含缓存) → (全res/池化 × 训/测seen/测unseen, 码×3)。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        tag = f"fullres_chr_{self.N_TRAIN}_{self.N_TEST}.safetensors"
        path = cache / tag
        keys = (
            "xf_tr", "xf_ts", "xf_tu", "xp_tr", "xp_ts", "xp_tu",
            "c_tr", "c_ts", "c_tu",
        )
        if path.exists():
            d = mx.load(str(path))
            return tuple(d[k] for k in keys)  # type: ignore
        tr = self.sample_codes(self.train_codes, self.N_TRAIN, 42)
        ts = self.sample_codes(self.train_codes, self.N_TEST, 99)
        tu = self.sample_codes(self.unseen_codes, self.N_TEST, 77)
        renderer, cam = Codebook.make_renderer()
        rw = RieszWavelet(mx.zeros((Codebook.H, Codebook.W)))
        t0 = time.monotonic()
        xf_tr, xp_tr = self.feats_of(tr, renderer, cam, rw)
        xf_ts, xp_ts = self.feats_of(ts, renderer, cam, rw)
        xf_tu, xp_tu = self.feats_of(tu, renderer, cam, rw)
        print(f"渲染+特征 {time.monotonic()-t0:.0f}s → 缓存 {tag}")
        mx.save_safetensors(
            str(path),
            {
                "c_tr": mx.array(tr, dtype=mx.float32),
                "c_ts": mx.array(ts, dtype=mx.float32),
                "c_tu": mx.array(tu, dtype=mx.float32),
                "xf_tr": xf_tr, "xf_ts": xf_ts, "xf_tu": xf_tu,
                "xp_tr": xp_tr, "xp_ts": xp_ts, "xp_tu": xp_tu,
            },
        )
        d = mx.load(str(path))
        return tuple(d[k] for k in keys)  # type: ignore

    # ── 三臂 ────────────────────────────────────────────────────────

    def run_nb(
        self, xf_tr: mx.array, c_tr: list[int], xf_te: mx.array
    ) -> list[int]:
        """每码朴素贝叶斯 = CodeBayes (code_bayes.py, 机制单家)。"""
        m = CodeBayes.fit(
            xf_tr, mx.array(c_tr, dtype=mx.int32), cards=Codebook.CARDS
        )
        codes = Codebook.all_codes()
        pred = []
        for i in range(0, xf_te.shape[0], 32):
            p = m.posterior(xf_te[i : i + 32], codes)
            mx.eval(p)
            pred.extend(mx.argmax(p, axis=1).tolist())
        return pred

    def run_flat(
        self, xf_tr: mx.array, c_tr: list[int], xf_te: mx.array, n_batch: int = 5
    ) -> list[int]:
        """K 分量扁平混合在线 EM (SVI 对应物): Sum(K)×Product(像素) 固定
        结构 + 分量级码联合计数。单遍在线: 5 小批, 充分统计量累加。"""
        cb = Codebook
        g = mx.mean(xf_tr, axis=0)
        key = mx.random.key(3)
        init = mx.random.randint(0, xf_tr.shape[0], shape=(self.k,), key=key)
        # 伪计数初始化: n=1, μ=随机训练行, σ²=全局方差 —— 打破对称,
        # 且与统计量重建公式一致 (直接初始化 mu 会在重建时被冲掉)
        var0 = mx.maximum(mx.mean((xf_tr - g) ** 2, axis=0), self.FLOOR2)
        n_k = mx.ones(self.k)
        s1 = xf_tr[init] - g[None, :]  # Σ r·(x−g) 移位累积
        s2 = mx.tile(var0[None, :], (self.k, 1))
        jt = mx.zeros((self.k, cb.N_CODES))  # 分量 × 码 联合计数
        m = xf_tr.shape[0] // n_batch
        mu = sg = None
        for b in range(n_batch):
            xb = xf_tr[b * m : (b + 1) * m]
            cb_codes = c_tr[b * m : (b + 1) * m]
            # 参数 (从累积统计量重建)
            n_safe = mx.maximum(n_k, 1e-6)
            mu = g + s1 / n_safe[:, None]
            var = s2 / n_safe[:, None] - (s1 / n_safe[:, None]) ** 2
            sg = mx.sqrt(mx.maximum(var, self.FLOOR2))
            log_w = mx.log((n_k + 1.0) / (float(mx.sum(n_k)) + self.k))
            # E: 责任度
            lp = CodeBayes.ll(xb, mu, sg) + log_w[None, :]
            r = mx.exp(lp - mx.logsumexp(lp, axis=1, keepdims=True))
            mx.eval(r)
            # M: 充分统计量累加 (移位)
            d = xb - g[None, :]
            n_k = n_k + mx.sum(r, axis=0)
            s1 = s1 + r.T @ d
            s2 = s2 + r.T @ (d * d)
            oh = mx.equal(
                mx.array(cb_codes, dtype=mx.int32)[:, None],
                mx.arange(cb.N_CODES)[None, :],
            ).astype(mx.float32)
            jt = jt + r.T @ oh
            mx.eval(n_k, s1, s2, jt)
        # 最终参数
        assert mu is not None and sg is not None
        log_w = mx.log((n_k + 1.0) / (float(mx.sum(n_k)) + self.k))
        log_pc = mx.log((jt + 1.0) / (mx.sum(jt, axis=1, keepdims=True) + cb.N_CODES))
        pred = []
        for i in range(0, xf_te.shape[0], 32):
            lp = CodeBayes.ll(xf_te[i : i + 32], mu, sg) + log_w[None, :]  # (B,K)
            # log q(码) = logsumexp_k(log w_k + logp_k + log P(码|k))
            q = lp[:, :, None] + log_pc[None, :, :]  # (B,K,码)
            lq = mx.logsumexp(q, axis=1)
            mx.eval(lq)
            pred.extend(mx.argmax(lq, axis=1).tolist())
        return pred

    def run_spn(
        self, xp_tr: mx.array, c_tr: list[int], xp_te: mx.array
    ) -> list[int]:
        """池化 SPN 全量重训 (对照, demo 管线)。"""
        cb = Codebook
        code_arr = mx.array([list(cb.idx_to_code(c)) for c in c_tr], dtype=mx.float32)
        mu = xp_tr.mean(axis=0, keepdims=True)
        sd = mx.maximum(xp_tr.std(axis=0, keepdims=True), 1e-6)
        xz = (xp_tr - mu) / sd
        xj = mx.concatenate([xz, code_arr], axis=1)
        n_feat = xp_tr.shape[1]
        card = dict(zip(range(n_feat, n_feat + 5), cb.CARDS))
        tree = SPNLearner(
            disc_cols=set(card), card=card, min_n=3, max_depth=14
        ).learn(xj)
        codes = cb.all_codes()
        xz_te = (xp_te - mu) / sd
        pred = []
        for i in range(0, xz_te.shape[0], 8):
            p = tree.posterior(xz_te[i : i + 8], codes)
            mx.eval(p)
            pred.extend(mx.argmax(p, axis=1).tolist())
        return pred

    # ── 主流程 ──────────────────────────────────────────────────────

    def run(self) -> None:
        xf_tr, xf_ts, xf_tu, xp_tr, xp_ts, xp_tu, c_tr, c_ts, c_tu = self.build()
        gt_tr = [int(v) for v in c_tr.tolist()]
        gt_ts = [int(v) for v in c_ts.tolist()]
        gt_tu = [int(v) for v in c_tu.tolist()]
        print(
            f"训练 {self.N_TRAIN} (码族 {len(self.train_codes)}/{Codebook.N_CODES})"
            f" | 测 seen/unseen 各 {self.N_TEST}"
        )

        results: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for name, fn, full in [
            ("nb (每码贝叶斯)", self.run_nb, True),
            (f"flat (K={self.k} 在线EM)", self.run_flat, True),
            ("spn (池化全量)", self.run_spn, False),
        ]:
            t0 = time.monotonic()
            if full:
                pred_s = fn(xf_tr, gt_tr, xf_ts)  # type: ignore
                pred_u = fn(xf_tr, gt_tr, xf_tu)  # type: ignore
            else:
                pred_s = fn(xp_tr, gt_tr, xp_ts)  # type: ignore
                pred_u = fn(xp_tr, gt_tr, xp_tu)  # type: ignore
            acc_s = Evaluator.evaluate(pred_s, gt_ts)
            acc_u = Evaluator.evaluate(pred_u, gt_tu)
            results[name.split(" ")[0]] = (acc_s, acc_u)
            print(f"{name} ({time.monotonic()-t0:.0f}s)")
            print(f"  seen  : {acc_s}")
            print(f"  unseen: {acc_u}")

        # ── 标定断言 (2026-08-12 实测标定, 留余量) ───────────────────
        nb_s, nb_u = results["nb"]
        spn_s, spn_u = results["spn"]
        assert nb_s["code"] > 0.90, f"nb seen 应≈模板法上限: {nb_s['code']:.3f}"
        assert nb_u["code"] < 0.02, "nb 未见码码级应全灭 (无机制)"
        assert spn_u["kind"] > 0.40, (
            f"spn 未见码 kind 泛化 (组合结构): {spn_u['kind']:.3f}"
        )
        assert nb_u["gx"] < 0.25, (
            f"nb 未见码位置精度应崩 (模板无法插值): {nb_u['gx']:.3f}"
        )
        print("experiment_fullres: 完成 ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64, help="flat 臂分量数")
    args = ap.parse_args()
    FullresExperiment(k=args.k).run()
