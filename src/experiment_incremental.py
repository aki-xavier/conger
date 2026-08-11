"""结构增量研究实验 v2: 在线 EM (软路由 + 统计常驻 + 延迟生长) vs 全量重训。

增量算法 (OnlineSPN, 见 spn.py):
  批 1 learn_spn 建树 → 每批样本按叶后验 (路径先验 × 叶似然) 软分配
  → 叶统计 (n/Σx/Σx², 码联合计数) 与 Sum 计数永久累加
  (同结构同数据下 = 批量 MLE, spn.py 自检 6 验证)
  → 叶码计数达显著性下限且过翻倍检查点 → 码空间加权 k-means 分裂
  (子叶继承父叶高斯先验, 分组错误由软路由自校正)。

v1 (叶增长 + 硬路由 + 缓冲丢弃) 实测: 时间省 30%, 准确率 = 全量 × 0.65
—— 三根因: 根结构冻结 / 证据随缓冲销毁 / min_n=3 早承诺。
v2 架构修复: 软路由拆反馈环, 统计常驻保参数精确, 延迟承诺抗早分裂。
④根生长在实现中取消: 软路由下未覆盖码被分布式吸收, 根生长冗余。

研究问题: 架构修复后增量能否追平全量 (gap 随批次收敛 vs 发散)?

运行: python experiment_incremental.py
"""

import time

import mlx.core as mx

from demo_inverse import (
    N_GX,
    N_GY,
    N_KIND,
    N_SIZE,
    N_Z,
    all_codes,
    build_data,
    code_cols,
    code_to_idx,
    evaluate,
    standardize,
)
from spn import OnlineSPN, leaf_blocks, learn_spn

n_train, n_test = 4000, 200
n_batch = 5
x_tr, c_tr, x_te, c_te = build_data(n_train, n_test, use_cache=True)
x_tr, x_te, mu, sd = standardize(x_tr, x_te)
codes = all_codes()
gt_i = [code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()]
cards = (N_KIND, N_GX, N_GY, N_SIZE, N_Z)
card = dict(zip(code_cols, cards))


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


min_n = 3
batch = n_train // n_batch
acc_inc, acc_full, t_inc, t_full = [], [], [], []

# 批 1: learn_spn 建初始树 → OnlineSPN 吸收同批数据 (统计从零累加,
# 刷新后参数与 learn_spn MLE 等价)
x1 = mx.concatenate([x_tr[:batch], c_tr[:batch]], axis=1)
t0 = time.monotonic()
base = learn_spn(x1, disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14)
learner = OnlineSPN(
    base.root, n_vars=x1.shape[1], code_cols=tuple(code_cols), cards=cards
)
learner.absorb(x1)
acc_inc.append(eval_tree(learner.to_spn()))
t_inc.append(time.monotonic() - t0)
tree_f = learn_spn(x1, disc_cols=set(code_cols), card=card, min_n=min_n, max_depth=14)
acc_full.append(eval_tree(tree_f))
t_full.append(time.monotonic() - t0)
print(
    f"批1: 增量 {acc_inc[-1]:.3f} ({t_inc[-1]:.1f}s) "
    f"叶{len(leaf_blocks(learner.root))} | 全量 {acc_full[-1]:.3f} ({t_full[-1]:.1f}s)"
)

for b in range(1, n_batch):
    sl = slice(b * batch, (b + 1) * batch)
    xb = mx.concatenate([x_tr[sl], c_tr[sl]], axis=1)
    t0 = time.monotonic()
    learner.absorb(xb)
    acc_inc.append(eval_tree(learner.to_spn()))
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
        f"叶{len(leaf_blocks(learner.root))} | 全量 {acc_full[-1]:.3f} "
        f"({t_full[-1]:.1f}s)"
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
# v2 标定 (2026-08-11): 0.385 vs v1 的 0.305; 阈值留余量
assert acc_inc[-1] > 0.33, f"v2 应显著超 v1 (0.305), 实测 {acc_inc[-1]:.3f}"
print(
    f"\n时间: 增量 {t_inc[-1]:.0f}s vs 全量 {t_full[-1]:.0f}s "
    f"(省 {(1 - t_inc[-1]/t_full[-1])*100:.0f}%)"
)
print(
    f"准确率: 增量 {acc_inc[-1]:.3f} vs 全量 {acc_full[-1]:.3f} "
    f"(增量 = 全量 × {acc_inc[-1]/acc_full[-1]:.2f})"
)
print("experiment_incremental: 完成 ✓")
