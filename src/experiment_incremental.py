"""结构增量研究实验 v2: 在线 EM (软路由 + 统计常驻 + 延迟生长) vs 全量重训。

增量算法 (OnlineSPN, 见 spn.py):
  批 1 SPNLearner 建树 → 每批样本按叶后验 (路径先验 × 叶似然) 软分配
  → 叶统计 (n/μ/M2 中心矩, 码联合计数) 与 Sum 计数永久累加
  (同结构同数据下 = 批量 MLE, spn_selftest 自检 6 验证)
  → 叶码计数过显著性下限 → 码空间加权 k-means 分裂
  (子叶继承父叶高斯先验 + 当批行按码分组播种, 软路由自校正分组错误)。

v1 (叶增长 + 硬路由 + 缓冲丢弃) 实测: 时间省 30%, 准确率 = 全量 × 0.65
—— 三根因: 根结构冻结 / 证据随缓冲销毁 / min_n=3 早承诺。
v2 架构修复: 软路由拆反馈环, 统计常驻保参数精确, 延迟承诺抗早分裂。
④根生长在实现中取消: 软路由下未覆盖码被分布式吸收, 根生长冗余。

v2 实测 (N=4000, 5 批): 准确率 = 全量 × 0.82 (0.385 vs 0.470),
且批 5 仍 +0.035/批 爬升 (全量仅 +0.015) —— 残差是收敛滞后
(分裂播种只用当批行 + 先验折扣继承) 而非结构墙。

⑤ 周期修订实测 (--rev-cap N [--rev-at 批号]):
  每批修订 (cap=2048): ×0.61 —— 反复付证据丢失代价 (重构后参数只剩
    reservoir 样本), 反而不如纯在线;
  单次中途修订 (批3, cap=2048): ×0.98 (0.460 vs 0.470) —— 结构@样本
    (2048 行够估结构) + 参数@全流 (reservoir + 后续批在线累积);
  单次修订 cap=512: ×0.28 —— 结构估计样本不足 (结构是瓶颈)。
  结论: ⑤ 的正确形态 = 稀疏调度 + reservoir 够估结构, 不是高频重构。

运行: python experiment_incremental.py [--rev-cap 2048 [--rev-at 3]]
"""

import argparse
import random
import time

import mlx.core as mx

from codebook import Codebook
from data_builder import DataBuilder
from evaluator import Evaluator
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from online_spn import OnlineSPN
from spn_learner import SPNLearner


class IncrementalExperiment:
    """增量 vs 全量对照实验 (配置入构造器, run() 驱动)。"""

    N_TRAIN, N_TEST, N_BATCH = 4000, 200, 5
    MIN_N = 3

    def __init__(self, rev_cap: int = 0, rev_at: frozenset[int] = frozenset()):
        self.rev_cap = rev_cap  # reservoir 容量 (0 = 不跑修订臂)
        self.rev_at = rev_at  # 在哪些批后修订 (空 = 每批)
        self.cfg = InverseConfig(model="spn")  # 池化特征 (结构学习需要低维)
        self.codebook = Codebook(self.cfg)
        self.data = DataBuilder(
            self.cfg, self.codebook, FeatureExtractor(self.cfg)
        )

    def run(self) -> None:
        cb = self.codebook
        x_tr, c_tr, x_te, c_te = self.data.build(
            self.N_TRAIN, self.N_TEST, use_cache=True
        )
        x_tr, x_te, mu, sd = self.data.standardize(x_tr, x_te)
        self.x_te = x_te
        self.codes = cb.all_codes()
        self.gt_i = [
            cb.code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()
        ]
        self.learner_kw = dict(
            disc_cols=set(self.cfg.code_cols), card=self.cfg.card,
            min_n=self.MIN_N, max_depth=14,
        )

        batch = self.N_TRAIN // self.N_BATCH
        acc_inc, acc_full, t_inc, t_full = [], [], [], []

        # 批 1: 建初始树 + OnlineSPN 吸收同批数据 (统计从零累加,
        # 刷新后参数与批量 MLE 等价) + 全量基线
        x1 = mx.concatenate([x_tr[:batch], c_tr[:batch]], axis=1)
        t0 = time.monotonic()
        base = SPNLearner(**self.learner_kw).learn(x1)
        learner = OnlineSPN(
            base.root,
            n_vars=x1.shape[1],
            code_cols=self.cfg.code_cols,
            cards=cb.CARDS,
        )
        learner.absorb(x1)
        acc_inc.append(self.eval_tree(learner.to_spn()))
        t_inc.append(time.monotonic() - t0)
        tree_f = SPNLearner(**self.learner_kw).learn(x1)
        acc_full.append(self.eval_tree(tree_f))
        t_full.append(time.monotonic() - t0)
        print(
            f"批1: 增量 {acc_inc[-1]:.3f} ({t_inc[-1]:.1f}s) "
            f"叶{len(learner.root.leaf_blocks())} | 全量 {acc_full[-1]:.3f} "
            f"({t_full[-1]:.1f}s)"
        )

        for b in range(1, self.N_BATCH):
            sl = slice(b * batch, (b + 1) * batch)
            xb = mx.concatenate([x_tr[sl], c_tr[sl]], axis=1)
            t0 = time.monotonic()
            learner.absorb(xb)
            acc_inc.append(self.eval_tree(learner.to_spn()))
            t_inc.append(t_inc[-1] + (time.monotonic() - t0))
            t0 = time.monotonic()
            xf = mx.concatenate(
                [x_tr[: (b + 1) * batch], c_tr[: (b + 1) * batch]], axis=1
            )
            tree_f = SPNLearner(**self.learner_kw).learn(xf)
            acc_full.append(self.eval_tree(tree_f))
            t_full.append(t_full[-1] + (time.monotonic() - t0))
            print(
                f"批{b+1}: 增量 {acc_inc[-1]:.3f} ({t_inc[-1]:.1f}s) "
                f"叶{len(learner.root.leaf_blocks())} | 全量 {acc_full[-1]:.3f} "
                f"({t_full[-1]:.1f}s)"
            )

        # ── 结果 ───────────────────────────────────────────────────
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

        if self.rev_cap > 0:
            self.run_revision(x_tr, c_tr, acc_inc[-1], acc_full[-1])
        print("experiment_incremental: 完成 ✓")

    def eval_tree(self, tree) -> float:
        """测试集码准确率 (分块推理防显存)。"""
        parts = []
        for i in range(0, self.N_TEST, 8):
            p = tree.posterior(self.x_te[i : i + 8], self.codes)
            mx.eval(p)
            parts.append(p)
        post = mx.concatenate(parts)
        pred = mx.argmax(post, axis=1).tolist()
        return Evaluator.evaluate(pred, self.gt_i)["code"]

    # ── ⑤ 周期全局修订臂 (reservoir + SPNLearner 重构 + 重吸收) ──────

    def run_revision(
        self, x_tr: mx.array, c_tr: mx.array, acc_inc: float, acc_full: float
    ) -> None:
        """全局 reservoir (Vitter R, 容量封顶) + 稀疏 learn 重构 + 重吸收。
        ⑤ 的有界形式: 精确统计与结构绑定, 修订结构必然丢失精确累积,
        除非保留原始数据 → reservoir 容量即内存-准确率权衡的横轴;
        全局重构是同容量局部子树修订的上界。"""
        cb = self.codebook
        rng = random.Random(7)
        res: list[list[float]] = []
        seen = 0

        def res_add(xb: mx.array) -> None:
            nonlocal seen
            for row in xb.tolist():
                if len(res) < self.rev_cap:
                    res.append(row)
                else:
                    j = rng.randrange(seen + 1)
                    if j < self.rev_cap:
                        res[j] = row
                seen += 1

        def revise() -> OnlineSPN:
            """从 reservoir 重构结构 + 重吸收 (统计与结构一致)。"""
            xr = mx.array(res, dtype=mx.float32)
            sub = SPNLearner(**self.learner_kw).learn(xr)
            nl = OnlineSPN(
                sub.root,
                n_vars=xr.shape[1],
                code_cols=self.cfg.code_cols,
                cards=cb.CARDS,
            )
            nl.absorb(xr)
            return nl

        batch = self.N_TRAIN // self.N_BATCH
        print(f"\n── 修订臂 (reservoir cap={self.rev_cap}) ──")
        acc_rev, t_rev = [], []
        lrn: OnlineSPN | None = None
        for b in range(self.N_BATCH):
            sl = slice(b * batch, (b + 1) * batch)
            xb = mx.concatenate([x_tr[sl], c_tr[sl]], axis=1)
            t0 = time.monotonic()
            if lrn is None:
                base = SPNLearner(**self.learner_kw).learn(xb)
                lrn = OnlineSPN(
                    base.root,
                    n_vars=xb.shape[1],
                    code_cols=self.cfg.code_cols,
                    cards=cb.CARDS,
                )
            lrn.absorb(xb)
            res_add(xb)
            if not self.rev_at or (b + 1) in self.rev_at:
                lrn = revise()
            acc_rev.append(self.eval_tree(lrn.to_spn()))
            t_rev.append((t_rev[-1] if t_rev else 0.0) + (time.monotonic() - t0))
            print(f"批{b+1}: 修订 {acc_rev[-1]:.3f} ({t_rev[-1]:.1f}s)")
        print(
            f"\n修订臂(cap={self.rev_cap}): 最终 {acc_rev[-1]:.3f} "
            f"vs 纯在线 {acc_inc:.3f} vs 全量 {acc_full:.3f} "
            f"(修订 = 全量 × {acc_rev[-1]/acc_full:.2f})"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rev-cap", type=int, default=0,
        help="周期全局修订臂的 reservoir 容量 (0=不跑该臂)",
    )
    ap.add_argument(
        "--rev-at", default="",
        help="在哪些批后修订 (逗号分隔, 如 '3'; 空=每批都修订)。实测: 每批修订"
        "反复付证据丢失代价; 中途修订一次 = 结构@样本 + 参数@全流",
    )
    args = ap.parse_args()
    rev_at = frozenset(int(s) for s in args.rev_at.split(",") if s.strip())
    IncrementalExperiment(rev_cap=args.rev_cap, rev_at=rev_at).run()
