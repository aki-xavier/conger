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

import math
import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import scipy.io

from demo_eval_common import (
    aggregate,
    depth_metrics,
    edge_f1,
    occlusion_boundary_recall,
    print_summary,
    run_ours,
)
from utils import Utils

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_scene(mat_path: str) -> tuple:
    """.mat → (rgb, depth_m, edges|None, 平面实例掩码列, 法向列, calib,
    有效掩码)。"""
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
    planes, normals = [], []
    for k in ("mask_wall", "mask_floor", "mask_table"):
        mk = m[k].astype(np.int32)
        paras = np.asarray(m[f"{k}_paras"]).reshape(-1, 7)
        for inst in range(1, int(mk.max()) + 1):
            if (mk == inst).sum() > 200:  # 忽略过小实例
                planes.append(mk == inst)
                n = paras[inst - 1, 4:7] if paras.shape[0] >= inst else None
                normals.append(n)
    calib = np.asarray(m["calib"]).reshape(3, 3)
    return rgb, depth, edges, planes, normals, calib, valid


def plane_metrics(
    regions: np.ndarray | mx.array, planes: list[np.ndarray]
) -> tuple[float, float]:
    """每 GT 平面实例 → 最佳区域 IoU; 返回 (检出率, 平均最佳 IoU)。MLX。"""
    if not planes:
        return float("nan"), float("nan")
    r = mx.array(regions)
    best_ious = []
    for pm in planes:
        pm_mx = mx.array(pm)
        inter = (r > 0) & pm_mx
        if int(mx.sum(inter)) == 0:
            best_ious.append(0.0)
            continue
        # 与平面重叠最大的区域: 排序分组计数 (MLX 无 unique/bincount)
        key = mx.where(inter.reshape(-1), mx.arange(inter.size), inter.size)
        idx = mx.argsort(key)[: int(mx.sum(inter))]
        vals = mx.sort(r.reshape(-1)[idx])
        nv = vals.shape[0]
        chg = mx.concatenate([mx.array([True]), vals[1:] != vals[:-1]])
        starts = Utils.nonzero(chg)
        ends = mx.concatenate([starts[1:], mx.array([nv])])
        cnt = ends - starts
        # 最长运行对应区域 = vals[starts[argmax]] (cnt 索引 ≠ vals 索引)
        best_r = int(vals[int(starts[int(mx.argmax(cnt))])])
        inter_n = int(mx.max(cnt))
        union_n = int(mx.sum(pm_mx)) + int(mx.sum(r == best_r)) - inter_n
        best_ious.append(inter_n / max(union_n, 1))
    det = float(sum(1.0 for iou in best_ious if iou > 0.5) / len(best_ious))
    return det, float(sum(best_ious) / len(best_ious))


def plane_normal_err(
    regions: np.ndarray, planes: list[np.ndarray],
    normals: list[np.ndarray | None], depth: np.ndarray,
    calib: np.ndarray,
) -> tuple[float, int]:
    """检出平面 (IoU>0.5) 的法向角误差 vs GT (米制重建后拟合)。
    平面参数通道 (README 协议"拟合残差 vs 真值参数")。"""
    fx, fy = calib[0, 0], calib[1, 1]
    cx, cy = calib[0, 2], calib[1, 2]
    errs: list[float] = []
    h, w = depth.shape
    r = mx.array(regions)
    d_mx = mx.array(depth, dtype=mx.float32)
    yy, xx = mx.meshgrid(
        mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
        indexing="ij",
    )
    for pm, n_gt in zip(planes, normals):
        if n_gt is None:
            continue
        pm_mx = mx.array(pm)
        inter = (r > 0) & pm_mx
        if int(mx.sum(inter)) == 0:
            continue
        key = mx.where(inter.reshape(-1), mx.arange(inter.size), inter.size)
        idx = mx.argsort(key)[: int(mx.sum(inter))]
        vals = mx.sort(r.reshape(-1)[idx])
        nv = vals.shape[0]
        chg = mx.concatenate([mx.array([True]), vals[1:] != vals[:-1]])
        starts = Utils.nonzero(chg)
        ends = mx.concatenate([starts[1:], mx.array([nv])])
        cnt = ends - starts
        # 最长运行对应区域 = vals[starts[argmax]] (cnt 索引 ≠ vals 索引)
        best_r = int(vals[int(starts[int(mx.argmax(cnt))])])
        inter_n = int(mx.max(cnt))
        union_n = int(mx.sum(pm_mx)) + int(mx.sum(r == best_r)) - inter_n
        if inter_n / max(union_n, 1) <= 0.5:
            continue  # 未检出
        mask = (r == best_r) & (d_mx > 0)
        if int(mx.sum(mask)) < 100:
            continue
        key = mx.where(mask.reshape(-1), mx.arange(mask.size), mask.size)
        idx = mx.argsort(key)[: int(mx.sum(mask))]
        z = d_mx.reshape(-1)[idx]
        x = (xx.reshape(-1)[idx] - cx) / fx * z  # 米制重建
        y = (yy.reshape(-1)[idx] - cy) / fy * z
        # 正规方程 lstsq (MLX 无 lstsq): 解 AᵀA·c = Aᵀz
        one = mx.ones_like(z)
        a11 = float(mx.sum(x * x))
        a12 = float(mx.sum(x * y))
        a13 = float(mx.sum(x))
        a22 = float(mx.sum(y * y))
        a23 = float(mx.sum(y))
        a33 = float(mx.sum(one))
        b1 = float(mx.sum(x * z))
        b2 = float(mx.sum(y * z))
        b3 = float(mx.sum(z))
        A = mx.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])
        bv = mx.array([b1, b2, b3])
        coef = mx.linalg.solve(A, bv, stream=mx.cpu)
        n_fit = mx.array([coef[0], coef[1], -1.0])
        n_fit = n_fit / mx.linalg.norm(n_fit)
        n_gt_u = mx.array(n_gt, dtype=mx.float32)
        n_gt_u = n_gt_u / mx.linalg.norm(n_gt_u)
        ang = float(mx.arccos(
            mx.clip(mx.abs(mx.sum(n_fit * n_gt_u)), 0.0, 1.0)
        ))
        errs.append(math.degrees(ang))
    if not errs:
        return float("nan"), 0
    return float(mx.median(mx.array(errs))), len(errs)


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
        rgb, depth, edges, planes, normals, calib, valid = load_scene(str(mp))
        hier, enh, cue = run_ours(rgb)
        z_rel = np.asarray(cue.mean)
        p_mask = np.asarray(cue.precision) > 0.01  # 线索有精度的像素
        sc = {}
        sc.update(depth_metrics(z_rel, depth, valid & p_mask))
        # 分割的下游准绳 (2026-08-10 架构检讨, 替代 BSDS-F1 口径):
        # 遮挡边界 recall (深度跳变边界 vs 我们的区域边界, confetti
        # 不惩罚) + 平面检出。两者都随 τ 变 (τ=0.5 全并 → recall 0,
        # τ=0.2 碎裂 → recall 高但多余边界 13×) —— 按 τ 扫描报告。
        for tau in (0.2, 0.3, 0.5):
            regions = mx.array(hier.cut(tau))
            occl_rec, n_occ = occlusion_boundary_recall(regions, depth, valid)
            sc[f"occl_rec_{tau}"] = occl_rec
            det, mIoU = plane_metrics(regions, planes)
            sc[f"plane_det_{tau}"] = det
            sc[f"plane_iou_{tau}"] = mIoU
            # 平面法向精度 (检出平面的米制法向角误差, τ=0.3)
            if tau == 0.3:
                nerr, nerr_n = plane_normal_err(
                    regions, planes, normals, depth, calib
                )
                sc["plane_norm_err"] = nerr
                sc["plane_norm_n"] = float(nerr_n)
        sc["n_occl"] = float(n_occ)
        if edges is not None:
            ef = edge_f1(enh, edges)
            sc.update({f"edge_t{t}": v for t, v in ef.items()})
            sc["edge_best"] = max(ef.values())
        else:
            sc["edge_best"] = float("nan")  # 无 GT, 不计入汇总
        rows.append((name, sc))
        print(f"[{i + 1}/{len(mats)}] {name}: δ1={sc['delta1']:.2f} "
              f"sp={sc['spearman']:.2f} occl_rec(0.3)={sc['occl_rec_0.3']:.2f} "
              f"plane_det(0.3)={sc['plane_det_0.3']:.2f} "
              f"({(time.perf_counter() - t0) / 60:.1f}min)")

    print_summary("iBims-1", aggregate(rows))
    # 三通道排序三例
    ranked_d = sorted(rows, key=lambda r: -r[1]["spearman"])
    ranked_o = sorted(rows, key=lambda r: -r[1]["occl_rec_0.3"])
    ranked_p = sorted(rows, key=lambda r: -r[1]["plane_det_0.3"])
    imgs = {n: load_scene(str(next(m for m in mats if m.stem == n)))[0]
            for n, _ in rows}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for c_i, (label, sel) in enumerate(
        [("深度 Spearman", ranked_d), ("遮挡边界 recall", ranked_o),
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
