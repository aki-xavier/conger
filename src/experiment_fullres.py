"""全分辨率 + SVI 探索: 不池化 (144×144×3 = 62208 维) 的三种模型对照。

动机: demo 的块池化 (8×6 → 144 维) 是为 learnSPN 结构学习服务的
(G 检验两两列 O(c²), 62K 维不可行)。若放弃结构学习, 用固定结构扁平
混合 + 在线 EM (共轭指数族上在线 EM ≡ SVI 自然梯度更新, Hoffman 2013;
Cappé & Moulines 2009), 全分辨率能否带来收益?

三臂:
  nb    每码朴素贝叶斯 (码↔分量一一对应, 监督充分统计量) —— 模板法上限,
        平凡可增量 (逐码 n/μ/σ 可加);
  flat  K=64 扁平混合在线 EM: Sum(K) × Product(像素独立高斯) 固定结构
        + 分量级码联合计数 (P(码|分量)) —— SVI 对应物;
  spn   池化 SPN 全量重训 (demo 管线, 结构学习对照)。

探针 (生成结构 vs 模板记忆的核心判别): 训练只用 90% 码族 (10% 码整族
保留不训), 测试分 seen / unseen 两组。预期: nb 对未见码无机制 (≈0);
flat 靠分量跨码共享部分泛化; spn 靠组合结构 (kind×位置×尺寸×深度
独立线索) 泛化。

数值: 全分辨率臂用原始特征 + σ_floor=0.05 (确定性渲染 → 逐码方差≈0,
z-score 在近平常量像素上会爆炸, 故不标准化); 充分统计量按全局均值
移位累积 (x−g), 防 float32 方差抵消 (spn.py 教训)。

运行: python experiment_fullres.py
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from code_bayes import CodeBayes
from demo_inverse import (
    N_CODES,
    all_codes,
    block_pool,
    code_to_scene,
    evaluate,
    frame_lum,
    idx_to_code,
    make_renderer,
)
from riesz import RieszWavelet
from spn import learn_spn

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=64, help="flat 臂分量数")
args = ap.parse_args()

H = W = 144
D = H * W * 3  # log_mag / phase_coh / ori_R 三通道全分辨率
FLOOR2 = 0.05**2  # σ² 下限 (确定性渲染 → 逐码像素方差≈0)
K_FLAT = args.k

# ── 码族保留: 10% 码整族不训 (泛化探针) ─────────────────────────
perm = mx.random.permutation(N_CODES, key=mx.random.key(123)).tolist()
unseen = set(perm[: N_CODES // 10])
train_codes = [i for i in range(N_CODES) if i not in unseen]
unseen_codes = perm[: N_CODES // 10]

n_train, n_test = 3600, 200


def sample_codes(pool: list[int], n: int, key: int) -> list[int]:
    idx = mx.random.randint(0, len(pool), shape=(n,), key=mx.random.key(key))
    return [pool[int(i)] for i in idx.tolist()]


def feats_of(
    idxs: list[int], renderer, cam, rw: RieszWavelet
) -> tuple[mx.array, mx.array]:
    """帧序列 → (全分辨率特征 (n,62208), 池化特征 (n,144))。"""
    full, pooled = [], []
    for i in idxs:
        frame = renderer.render(code_to_scene(idx_to_code(i)), cam)
        rw.update(frame_lum(frame))
        f = rw.features()
        v = mx.concatenate(
            [f.log_mag.reshape(-1), f.phase_coh.reshape(-1), f.ori_R.reshape(-1)]
        )
        p = mx.concatenate(
            [
                block_pool(f.log_mag).reshape(-1),
                block_pool(f.phase_coh).reshape(-1),
                block_pool(f.ori_R).reshape(-1),
            ]
        )
        mx.eval(v, p)  # 逐帧求值, 防惰性图累积爆显存
        full.append(v)
        pooled.append(p)
    return mx.stack(full), mx.stack(pooled)


def build() -> tuple[mx.array, ...]:
    """渲染 + 特征 (含缓存) → (全res/池化 × 训/测seen/测unseen, 码下标×3)。"""
    cache = Path(__file__).resolve().parent.parent / "artifacts"
    cache.mkdir(exist_ok=True)
    tag = f"fullres_{n_train}_{n_test}.npz"
    path = cache / tag
    if path.exists():
        d = mx.load(str(path))
        return tuple(d[k] for k in (  # type: ignore
            "xf_tr", "xf_ts", "xf_tu", "xp_tr", "xp_ts", "xp_tu",
            "c_tr", "c_ts", "c_tu",
        ))
    tr = sample_codes(train_codes, n_train, 42)
    ts = sample_codes(train_codes, n_test, 99)
    tu = sample_codes(unseen_codes, n_test, 77)
    renderer, cam = make_renderer()
    rw = RieszWavelet(mx.zeros((H, W)))
    t0 = time.monotonic()
    xf_tr, xp_tr = feats_of(tr, renderer, cam, rw)
    xf_ts, xp_ts = feats_of(ts, renderer, cam, rw)
    xf_tu, xp_tu = feats_of(tu, renderer, cam, rw)
    print(f"渲染+特征 {time.monotonic()-t0:.0f}s → 缓存 {tag}")
    c_tr = mx.array(tr, dtype=mx.float32)
    c_ts = mx.array(ts, dtype=mx.float32)
    c_tu = mx.array(tu, dtype=mx.float32)
    mx.savez(
        str(path),
        c_tr=c_tr, c_ts=c_ts, c_tu=c_tu,
        xf_tr=xf_tr, xf_ts=xf_ts, xf_tu=xf_tu,
        xp_tr=xp_tr, xp_ts=xp_ts, xp_tu=xp_tu,
    )
    return xf_tr, xf_ts, xf_tu, xp_tr, xp_ts, xp_tu, c_tr, c_ts, c_tu


# ── 扁平模型的共享推理: 逐像素独立高斯, 点积展开防显存 ────────────
# logp(x|c) = Σ_p −½(x−μ)²/σ² − log σ = −½(x²·a_c) + x·b_c + c_c,
# a_c = 1/σ², b_c = μ/σ², c_c = Σ_p(−½μ²/σ² − log σ)


def gauss_logp(x: mx.array, mu: mx.array, sg: mx.array) -> mx.array:
    """x (B,D), mu/sg (K,D) → (B,K) log 密度 (对角高斯)。"""
    a = 1.0 / (sg * sg)
    b = mu * a
    c = -0.5 * mx.sum(mu * b, axis=1) - mx.sum(mx.log(sg), axis=1)  # (K,)
    return -0.5 * ((x * x) @ a.T) + x @ b.T + c[None, :]


def acc_of(pred: list[int], gt: list[int]) -> dict[str, float]:
    return evaluate(pred, gt)


# ── nb 臂: 每码朴素贝叶斯 (监督充分统计量) ─────────────────────────


def run_nb(
    xf_tr: mx.array, c_tr: list[int], xf_te: mx.array
) -> list[int]:
    """每码朴素贝叶斯 = CodeBayes (code_bayes.py, 机制单家)。"""
    m = CodeBayes.fit(
        xf_tr, mx.array(c_tr, dtype=mx.int32), cards=(3, 8, 6, 2, 4)
    )
    codes = all_codes()
    pred = []
    for i in range(0, xf_te.shape[0], 32):
        p = m.posterior(xf_te[i : i + 32], codes)
        mx.eval(p)
        pred.extend(mx.argmax(p, axis=1).tolist())
    return pred


# ── flat 臂: K=64 扁平混合在线 EM (SVI 对应物) ─────────────────────


def run_flat(
    xf_tr: mx.array, c_tr: list[int], xf_te: mx.array, n_batch: int = 5
) -> list[int]:
    g = mx.mean(xf_tr, axis=0)
    key = mx.random.key(3)
    init = mx.random.randint(0, xf_tr.shape[0], shape=(K_FLAT,), key=key)
    # 伪计数初始化: n=1, μ=随机训练行, σ²=全局方差 —— 打破对称,
    # 且与统计量重建公式一致 (直接初始化 mu 会在重建时被冲掉)
    var0 = mx.maximum(mx.mean((xf_tr - g) ** 2, axis=0), FLOOR2)
    n_k = mx.ones(K_FLAT)
    s1 = xf_tr[init] - g[None, :]  # Σ r·(x−g) 移位累积
    s2 = mx.tile(var0[None, :], (K_FLAT, 1))
    jt = mx.zeros((K_FLAT, N_CODES))  # 分量 × 码 联合计数
    m = xf_tr.shape[0] // n_batch
    for b in range(n_batch):
        xb = xf_tr[b * m : (b + 1) * m]
        cb = c_tr[b * m : (b + 1) * m]
        # 参数 (从累积统计量重建)
        n_safe = mx.maximum(n_k, 1e-6)
        mu = g + s1 / n_safe[:, None]
        var = s2 / n_safe[:, None] - (s1 / n_safe[:, None]) ** 2
        sg = mx.sqrt(mx.maximum(var, FLOOR2))
        log_w = mx.log((n_k + 1.0) / (float(mx.sum(n_k)) + K_FLAT))
        # E: 责任度 (冷启动批 n=0 → 全均匀 → 由 mu 初始化打破对称)
        lp = gauss_logp(xb, mu, sg) + log_w[None, :]
        r = mx.exp(lp - mx.logsumexp(lp, axis=1, keepdims=True))
        mx.eval(r)
        # M: 充分统计量累加 (移位)
        d = xb - g[None, :]
        n_k = n_k + mx.sum(r, axis=0)
        s1 = s1 + r.T @ d
        s2 = s2 + r.T @ (d * d)
        oh = mx.equal(
            mx.array(cb, dtype=mx.int32)[:, None], mx.arange(N_CODES)[None, :]
        ).astype(mx.float32)
        jt = jt + r.T @ oh
        mx.eval(n_k, s1, s2, jt)
    # 最终参数
    n_safe = mx.maximum(n_k, 1e-6)
    mu = g + s1 / n_safe[:, None]
    var = s2 / n_safe[:, None] - (s1 / n_safe[:, None]) ** 2
    sg = mx.sqrt(mx.maximum(var, FLOOR2))
    log_w = mx.log((n_k + 1.0) / (float(mx.sum(n_k)) + K_FLAT))
    log_pc = mx.log((jt + 1.0) / (mx.sum(jt, axis=1, keepdims=True) + N_CODES))
    pred = []
    for i in range(0, xf_te.shape[0], 32):
        lp = gauss_logp(xf_te[i : i + 32], mu, sg) + log_w[None, :]  # (B,K)
        # log q(码) = logsumexp_k(log w_k + logp_k + log P(码|k))
        q = lp[:, :, None] + log_pc[None, :, :]  # (B,K,码)
        lq = mx.logsumexp(q, axis=1)
        mx.eval(lq)
        pred.extend(mx.argmax(lq, axis=1).tolist())
    return pred


# ── spn 臂: 池化 SPN 全量重训 (对照) ────────────────────────────────


def run_spn(
    xp_tr: mx.array, c_tr: list[int], xp_te: mx.array
) -> list[int]:
    code_arr = mx.array([list(idx_to_code(c)) for c in c_tr], dtype=mx.float32)
    mu = xp_tr.mean(axis=0, keepdims=True)
    sd = mx.maximum(xp_tr.std(axis=0, keepdims=True), 1e-6)
    xz = (xp_tr - mu) / sd
    xj = mx.concatenate([xz, code_arr], axis=1)
    n_feat = xp_tr.shape[1]
    card = dict(zip(range(n_feat, n_feat + 5), (3, 8, 6, 2, 4)))
    tree = learn_spn(xj, disc_cols=set(card), card=card, min_n=3, max_depth=14)
    codes = mx.array([list(idx_to_code(i)) for i in range(N_CODES)], dtype=mx.float32)
    xz_te = (xp_te - mu) / sd
    pred = []
    for i in range(0, xz_te.shape[0], 8):
        p = tree.posterior(xz_te[i : i + 8], codes)
        mx.eval(p)
        pred.extend(mx.argmax(p, axis=1).tolist())
    return pred


# ── 主流程 ────────────────────────────────────────────────────────

xf_tr, xf_ts, xf_tu, xp_tr, xp_ts, xp_tu, c_tr, c_ts, c_tu = build()
gt_tr = [int(v) for v in c_tr.tolist()]
gt_ts = [int(v) for v in c_ts.tolist()]
gt_tu = [int(v) for v in c_tu.tolist()]
print(f"训练 {n_train} (码族 {len(train_codes)}/{N_CODES}) | "
      f"测 seen/unseen 各 {n_test}")

results: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
for name, fn, full in [
    ("nb (每码贝叶斯)", run_nb, True),
    (f"flat (K={K_FLAT} 在线EM)", run_flat, True),
    ("spn (池化全量)", run_spn, False),
]:
    t0 = time.monotonic()
    if full:
        pred_s = fn(xf_tr, gt_tr, xf_ts)  # type: ignore
        pred_u = fn(xf_tr, gt_tr, xf_tu)  # type: ignore
    else:
        pred_s = fn(xp_tr, gt_tr, xp_ts)  # type: ignore
        pred_u = fn(xp_tr, gt_tr, xp_tu)  # type: ignore
    acc_s, acc_u = acc_of(pred_s, gt_ts), acc_of(pred_u, gt_tu)
    results[name.split(" ")[0]] = (acc_s, acc_u)
    print(f"{name} ({time.monotonic()-t0:.0f}s)")
    print(f"  seen  : {acc_s}")
    print(f"  unseen: {acc_u}")

# ── 标定断言 (2026-08-12 实测标定, 留余量) ───────────────────────
nb_s, nb_u = results["nb"]
spn_s, spn_u = results["spn"]
assert nb_s["code"] > 0.90, f"nb seen 应≈模板法上限: {nb_s['code']:.3f}"
assert nb_u["code"] < 0.02, "nb 未见码码级应全灭 (无机制)"
assert spn_u["kind"] > 0.40, f"spn 未见码 kind 泛化 (组合结构): {spn_u['kind']:.3f}"
assert nb_u["gx"] < 0.25, f"nb 未见码位置精度应崩 (模板无法插值): {nb_u['gx']:.3f}"
print("experiment_fullres: 完成 ✓")
