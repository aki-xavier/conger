"""结构增量研究实验: 叶增长式增量学习 vs 全量重训。

增量算法 (叶增长):
  批 1 learn_spn 建树 → 后续批样本按码路由到叶 (CatLeaf count>0 支撑)
  → 叶缓冲累积 → 缓冲 ≥ 2·min_n 且码混杂 → 叶内 learnSPN 子过程
  → 叶替换为子树 (局部生长, 根结构冻结)。

研究问题: 结构增量能否逼近全量重训 (准确率差距 + 时间收益)?
实测 (N=4000, 5 批): 时间省 ~30%, 但准确率仅全量 ~65% ——
根结构冻结是主因 (批 1 码空间划分固定, 后续只能叶内细化)。

运行: python experiment_incremental.py
"""

import time
from typing import NamedTuple

import mlx.core as mx

from demo_inverse import (
    all_codes,
    build_data,
    code_cols,
    code_to_idx,
    evaluate,
    standardize,
)
from spn import CatLeaf, GaussLeaf, Node, leaf_blocks, learn_spn, replace_leaf

n_train, n_test = 4000, 200
n_batch = 5
x_tr, c_tr, x_te, c_te = build_data(n_train, n_test, use_cache=True)
x_tr, x_te, mu, sd = standardize(x_tr, x_te)
codes = all_codes()
gt_i = [code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()]
card = dict(zip(code_cols, (3, 8, 6, 2, 4)))
N_F = 144  # 特征维 (n_feat 默认 l 模式)


def eval_tree(tree) -> float:
    """测试集码准确率 (分块推理防显存)。"""
    parts = []
    for i in range(0, n_test, 8):
        p = tree.posterior(x_te[i : i + 8], codes)
        mx.eval(p)
        parts.append(p)
    post = mx.concatenate(parts)
    pred = mx.argmax(post, axis=1).tolist()
    return evaluate(pred, gt_i)["code"]


def leaf_support(leaf):
    """叶块码支撑: 各码列的 count>0 值集合 (路由用)。"""
    nodes = [leaf] if isinstance(leaf, (GaussLeaf, CatLeaf)) else list(leaf.children)
    cats = {}
    for n in nodes:
        if isinstance(n, CatLeaf):
            # counts 必须存在: 旧 pickle 缺 counts 时 logp>-5 的 fallback
            # 会因 Laplace 平滑把所有值判为支撑 → 路由退化为总是第一叶
            assert n.counts is not None, "CatLeaf 缺 counts (旧格式 pickle?)"
            cats[n.var] = {i for i, c in enumerate(n.counts.tolist()) if c > 0}
    return cats


def code_matches(support, code):
    for j, v in enumerate(code):
        if j in support and v not in support[j]:
            return False
    return True


def code_mixed(rows):
    seen = set()
    for r in rows:
        seen.add(tuple(int(v) for v in r[N_F : N_F + 5]))
        if len(seen) > 1:
            return True
    return False


min_n = 3
batch = n_train // n_batch
acc_inc, acc_full, t_inc, t_full = [], [], [], []

# 批 1: 建初始树 + 全量基线
x1 = mx.concatenate([x_tr[:batch], c_tr[:batch]], axis=1)
t0 = time.monotonic()
tree = learn_spn(x1, disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14)
acc_inc.append(eval_tree(tree))
t_inc.append(time.monotonic() - t0)
tree_f = learn_spn(x1, disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14)
acc_full.append(eval_tree(tree_f))
t_full.append(time.monotonic() - t0)
print(
    f"批1: 增量 {acc_inc[-1]:.3f} ({t_inc[-1]:.1f}s) "
    f"叶{len(leaf_blocks(tree.root))} | 全量 {acc_full[-1]:.3f} ({t_full[-1]:.1f}s)"
)

class Reg(NamedTuple):
    path: tuple[int, ...]
    leaf: Node
    support: dict[int, set[int]]
    buffer: list[list[float]]


registry = [
    Reg(path, leaf, leaf_support(leaf), [])
    for leaf, path in leaf_blocks(tree.root)
]

for b in range(1, n_batch):
    sl = slice(b * batch, (b + 1) * batch)
    xb = mx.concatenate([x_tr[sl], c_tr[sl]], axis=1)
    t0 = time.monotonic()
    for r in xb.tolist():
        code = [int(v) for v in r[N_F : N_F + 5]]
        hit = None
        for reg in registry:
            if code_matches(reg.support, code):
                hit = reg
                break
        if hit is not None:
            hit.buffer.append(r)
        else:
            pass  # 新码样本: 原型先丢弃 (注册表支撑未覆盖的码)
    for reg in list(registry):
        if len(reg.buffer) >= 2 * min_n and code_mixed(reg.buffer):
            sub = learn_spn(
                mx.array(reg.buffer),
                disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14,
            )
            tree.root = replace_leaf(tree.root, reg.path, sub.root)
            registry.remove(reg)
            for leaf2, path2 in leaf_blocks(sub.root):
                registry.append(Reg(reg.path + path2, leaf2, leaf_support(leaf2), []))
    acc_inc.append(eval_tree(tree))
    t_inc.append(t_inc[-1] + (time.monotonic() - t0))
    t0 = time.monotonic()
    tree_f = learn_spn(
        mx.concatenate([x_tr[: (b + 1) * batch], c_tr[: (b + 1) * batch]], axis=1),
        disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14,
    )
    acc_full.append(eval_tree(tree_f))
    t_full.append(t_full[-1] + (time.monotonic() - t0))
    print(
        f"批{b+1}: 增量 {acc_inc[-1]:.3f} ({t_inc[-1]:.1f}s) "
        f"叶{len(leaf_blocks(tree.root))} | 全量 {acc_full[-1]:.3f} ({t_full[-1]:.1f}s)"
    )

# ── 结果 ───────────────────────────────────────────────────────
print("\n对比:")
for i in range(len(acc_inc)):
    print(
        f"  批{i+1}: 增量 {acc_inc[i]:.3f} ({t_inc[i]:.1f}s) | "
        f"全量 {acc_full[i]:.3f} ({t_full[i]:.1f}s)"
    )
# 断言: 时间收益 (增量累计 < 全量累计); 准确率差距如实报告
assert t_inc[-1] < t_full[-1], "增量应节省累计时间"
print(
    f"\n时间: 增量 {t_inc[-1]:.0f}s vs 全量 {t_full[-1]:.0f}s "
    f"(省 {(1 - t_inc[-1]/t_full[-1])*100:.0f}%)"
)
print(
    f"准确率: 增量 {acc_inc[-1]:.3f} vs 全量 {acc_full[-1]:.3f} "
    f"(增量 = 全量 × {acc_inc[-1]/acc_full[-1]:.2f})"
)
print("experiment_incremental: 完成 ✓")
