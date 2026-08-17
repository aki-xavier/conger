"""CausalInvarianceProbe: 分析-合成精炼在 held-out 光照下的反照率不变性。

路线 ① 的端到端测量: 用 Codebook 采场景, 按光照 holdout 分组, 对每个
场景固定几何做全光照候选 re-render 得到联合后验, 边缘化出反照率 (hue)
估计, 统计池内 vs 池外准确率与不变性分数。期望: 分析-合成按构造对
held-out 光照不变 (gap≈0, 不变性≈1); 这对照 §9.1 的相关密度 (SPN) 会
随光照干预退化 —— 本探针给出那条差距的下界参照。
"""

from __future__ import annotations

import argparse
from typing import cast

from causal_invariance import InvarianceProbe, LightingHoldout
from codebook import Codebook
from inverse_config import InverseConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--factor", default="hue", choices=("hue", "lcol", "ldir"))
    ap.add_argument("--holdout-color", type=int, default=2)
    ap.add_argument("--holdout-dir", type=int, default=2)
    args = ap.parse_args()

    holdout = LightingHoldout.split(
        n_colors=len(Codebook.LIGHT_COLORS),
        n_dirs=len(Codebook.LIGHT_DIRS),
        holdout_color=args.holdout_color,
        holdout_dir=args.holdout_dir,
    )
    codebook = Codebook(InverseConfig(scene_family="single"))
    rows = cast(list, Codebook.sample(2, args.seed).tolist())
    in_rows = [
        r for r in rows if holdout.in_support(int(r[6]), int(r[7]))
    ][: args.n_scenes]
    out_rows = [
        r for r in rows if holdout.holdout(int(r[6]), int(r[7]))
    ][: args.n_scenes]
    report = InvarianceProbe.run(
        codebook, in_rows + out_rows, holdout, factor=args.factor
    )
    print(
        f"holdout lcol={holdout.holdout_colors} ldir={holdout.holdout_dirs} "
        f"| 池内 {len(in_rows)} 池外 {len(out_rows)} 场景"
    )
    print(
        f"{args.factor}: 池内 {report.in_support_accuracy:.3f} / "
        f"池外 {report.holdout_accuracy:.3f} / 不变性 "
        f"{report.invariance_score:.3f} / gap {report.gap:+.3f}"
    )
    print("per-group:", report.per_group_accuracy)


if __name__ == "__main__":
    main()
