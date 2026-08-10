"""NYUv2 (Eigen 测试切分 654) 评估: 单目深度线索 vs 米制 GT。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_nyu_eval.py [样本数]
数据: /tmp/datasets/nyu_v2_eigen/extract/val/official/*.h5
指标: 逐图尺度对齐 (ẑ = a·z_rel + b, 有效 ∩ 纹理区) 后
      δ1/δ2/δ3/RMSE/log-RMSE + Spearman (免对齐单调性)。
输出: 逐图报告 + artifacts/nyu_eval_cases.png
"""

import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from demo_eval_common import (
    aggregate,
    depth_metrics,
    print_summary,
    run_ours,
    save_cases,
)

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    """h5 → (rgb HWC uint8, depth 米制 float32)。"""
    with h5py.File(path, "r") as f:
        rgb = np.asarray(f["rgb"]).transpose(1, 2, 0)  # CHW → HWC
        depth = np.asarray(f["depth"]).astype(np.float64)
    return rgb, depth


def main(n_images: int = 10) -> None:
    import pathlib

    hs = sorted(pathlib.Path(
        "/tmp/datasets/nyu_v2_eigen/extract/val/official").glob("*.h5"))
    step = max(1, len(hs) // n_images)
    hs = hs[::step]
    print(f"NYUv2 Eigen 子集 {len(hs)} 张")

    rows = []
    t0 = time.perf_counter()
    for i, hp in enumerate(hs):
        name = hp.stem
        rgb, depth = load_h5(str(hp))
        _, enh, cue = run_ours(rgb)
        z_rel = cue.mean
        p_mask = cue.precision > 0.01
        valid = depth > 0.01
        sc = depth_metrics(z_rel, depth, valid & p_mask)
        rows.append((name, sc))
        print(f"[{i + 1}/{len(hs)}] {name}: δ1={sc['delta1']:.3f} "
              f"δ2={sc['delta2']:.3f} RMSE={sc['rmse']:.2f}m "
              f"sp={sc['spearman']:.3f} "
              f"({(time.perf_counter() - t0) / 60:.1f}min)")

    print_summary("NYUv2 Eigen", aggregate(rows))
    ranked = sorted(rows, key=lambda r: -r[1]["spearman"])
    imgs = {n: load_h5(str(next(h for h in hs if h.stem == n)))[0]
            for n, _ in rows}
    save_cases("nyu_eval_cases",
               [(n, imgs[n], sc["spearman"]) for n, sc in ranked],
               "Spearman")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
