"""iBims-1 评估: conger 单目管线 vs 高质量室内 GT (RGB-D, 无传感器噪声)。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_ibims_eval.py [样本数]
数据: /tmp/datasets/ibims1 (scripts/fetch_datasets.sh 重下载)
三通道 (对照 data/README.md 协议):
  a. 绝对深度: 单目深度线索 vs GT 深度 (有效 ∩ 纹理区), 逐图尺度对齐
     后 δ1/δ2/δ3/RMSE/log-RMSE + Spearman (免对齐单调性);
  b. 边界对齐: 增强边缘图 vs GT 边缘, 逐 τ 边界 F1;
  c. 平面图元: 我们的区域 vs GT 平面实例掩码 (wall/floor/table),
     每实例最佳区域 IoU, 检出率 (IoU>0.5)。
输出: 逐图报告 + artifacts/ibims_eval_cases.png
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from demo_eval_common import (
    aggregate,
    depth_metrics,
    edge_f1,
    print_summary,
    run_ours,
)
from utils import Utils

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_scene(mat_path: str) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray | None, list[np.ndarray], np.ndarray]:
    """.mat → (rgb, depth_m, edges|None, 平面实例掩码列, 有效掩码)。"""
    m = scipy.io.loadmat(mat_path)["data"][0, 0]
    rgb = m["rgb"].astype(np.uint8)
    depth = m["depth"].astype(np.float64)
    edges = m["edges"].astype(bool)
    if not edges.any():
        edges = None  # 部分场景无边缘 GT (corridor_01/04/06), 该通道跳过
    # 掩码是正向选择器 (1 = 保留, 名字 mask_transp/invalid 有误导性,
    # 已对照 raw 深度覆盖率验证: kitchen 98% 深度覆盖 ↔ 掩码 99.4% 1)
    valid = (m["mask_transp"].astype(bool)) & (m["mask_invalid"].astype(bool)) \
        & (depth > 0)
    planes = []
    for k in ("mask_wall", "mask_floor", "mask_table"):
        mk = m[k].astype(np.int32)
        for inst in range(1, int(mk.max()) + 1):
            if (mk == inst).sum() > 200:  # 忽略过小实例
                planes.append(mk == inst)
    return rgb, depth, edges, planes, valid


def plane_metrics(regions: np.ndarray, planes: list[np.ndarray]) -> tuple[float, float]:
    """每 GT 平面实例 → 最佳区域 IoU; 返回 (检出率, 平均最佳 IoU)。"""
    if not planes:
        return float("nan"), float("nan")
    best_ious = []
    for pm in planes:
        inter = (regions > 0) & pm
        if not inter.any():
            best_ious.append(0.0)
            continue
        # 与平面重叠最大的区域 = argmax over region ids
        rids, counts = np.unique(regions[inter], return_counts=True)
        best_r = rids[np.argmax(counts)]
        inter_n = int(counts.max())
        union_n = int((pm).sum()) + int((regions == best_r).sum()) - inter_n
        best_ious.append(inter_n / max(union_n, 1))
    det = float(np.mean([iou > 0.5 for iou in best_ious]))
    return det, float(np.mean(best_ious))


def main(n_images: int = 10) -> None:
    root = Utils.project_root()
    import pathlib

    mats = sorted(pathlib.Path(
        "/tmp/datasets/ibims1/ibims1_core_mat").glob("*.mat"))
    step = max(1, len(mats) // n_images)
    mats = mats[::step]
    print(f"iBims-1 子集 {len(mats)} 场景")

    rows = []
    t0 = time.perf_counter()
    for i, mp in enumerate(mats):
        name = mp.stem
        rgb, depth, edges, planes, valid = load_scene(str(mp))
        hier, enh, cue = run_ours(rgb)
        # 区域 (τ 固定 0.5) + 单目深度
        regions = np.array(hier.cut(0.5))
        z_rel = np.asarray(cue.mean)
        p_mask = np.asarray(cue.precision) > 0.01  # 线索有精度的像素
        sc = {}
        sc.update(depth_metrics(z_rel, depth, valid & p_mask))
        if edges is not None:
            ef = edge_f1(enh, edges)
            sc.update({f"edge_t{t}": v for t, v in ef.items()})
            sc["edge_best"] = max(ef.values())
        else:
            sc["edge_best"] = float("nan")  # 无 GT, 不计入汇总
        det, mIoU = plane_metrics(regions, planes)
        sc["plane_det"], sc["plane_iou"] = det, mIoU
        rows.append((name, sc))
        print(f"[{i + 1}/{len(mats)}] {name}: δ1={sc['delta1']:.2f} "
              f"sp={sc['spearman']:.2f} edge={sc['edge_best']:.2f} "
              f"plane_det={sc['plane_det']:.2f} "
              f"({(time.perf_counter() - t0) / 60:.1f}min)")

    print_summary("iBims-1", aggregate(rows))
    # 三通道排序三例
    ranked_d = sorted(rows, key=lambda r: -r[1]["spearman"])
    ranked_e = sorted(rows, key=lambda r: -r[1]["edge_best"])
    ranked_p = sorted(rows, key=lambda r: -r[1]["plane_det"])
    imgs = {n: load_scene(str(next(m for m in mats if m.stem == n)))[0]
            for n, _ in rows}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for c_i, (label, sel) in enumerate(
        [("深度 Spearman", ranked_d), ("边界 F1", ranked_e),
         ("平面检出", ranked_p)]
    ):
        pick = [sel[0], sel[len(sel) // 2], sel[-1]]
        for r_i, (n, sc) in enumerate(pick):
            ax = axes[r_i, c_i]
            ax.imshow(imgs[n])
            ax.set_title(f"{n} ({label})")
            ax.axis("off")
    fig.tight_layout()
    out = root / "artifacts/ibims_eval_cases.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
