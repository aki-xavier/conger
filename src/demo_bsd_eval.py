"""BSDS500 标准口径评测: conger 分割 vs 人工标注。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_bsd_eval.py [样本数]
输出: 逐图边界 F1 报告 (τ 扫描) + artifacts/bsd_eval_cases.png

协议 (BSDS 惯例近似):
  边界 F1, 容差 = 0.0075 × 图像对角线 (~4px);
  每张图多个标注者 → 与各标注者分别算 F1 取最大 (per-image max);
  分割 τ ∈ {0.2, 0.3, 0.5} 扫描 → 固定 τ 均值 (≈ODS) 与逐图最佳
  (≈OIS)。标注即参照非绝对真值: 标注者彼此也有分歧。
提速: 全链每图一遍 (τ 共享层级 cut) + VBGMM.fast_fit 级联冷启动
(coreset 重要性采样) —— 20 图从 45min 压到 <1min (2026-08-09)。
"""

import glob
import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import scipy.io
from PIL import Image

from color import Color
from demo_sam_compare import boundary_f1, colorize
from utils import Utils

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TAUS = (0.2, 0.3, 0.5)


def gt_boundaries(mat_path: str) -> list[np.ndarray]:
    """.mat 人工标注 → 各标注者的二值边界图列。"""
    m = scipy.io.loadmat(mat_path)
    gt = m["groundTruth"]
    out = []
    for j in range(gt.shape[1]):
        seg = gt[0, j]["Segmentation"][0, 0]
        b = np.zeros_like(seg, dtype=bool)
        b[:-1] |= seg[:-1] != seg[1:]
        b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
        out.append(b)
    return out


def our_boundary(regions: np.ndarray) -> np.ndarray:
    """我们的区域标签图 → 二值边界图。"""
    b = np.zeros_like(regions, dtype=bool)
    b[:-1] |= regions[:-1] != regions[1:]
    b[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    return b


def run_ours_hierarchy(rgb):
    """conger 外观链 → RegionHierarchy (τ 无关部分, 每图一次)。
    级联冷启动 + coreset 重要性采样 (稀有类防漏检)。"""
    from edgemap import EdgePrior
    from riesz import RieszWavelet
    from segment import RegionLayer
    from vbgmm import VBGMM

    h, w = rgb.shape[:2]
    lum, hs = Color.split_dual_path(rgb)
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM.fast_fit(
        VBGMM.feature_matrix(feat), (h, w), k_max=48, coreset=8192
    )
    like_l = gm_l.edge_likelihood((h, w))
    gm_h = VBGMM.fast_fit(
        VBGMM.hs_feature_matrix(hs).reshape(-1, 7), (h, w),
        k_max=32, coreset=8192,
    )
    like_h = gm_h.edge_likelihood((h, w)) * mx.abs(hs)  # 饱和度门控
    like = 1 - (1 - like_l) * (1 - like_h)
    enh = EdgePrior().enhance(like, feat, rw)
    return RegionLayer().run(enh).hierarchy


def eval_image(img_path: str, gt_path: str, tol: int) -> dict[float, float]:
    """单图: 逐 τ 的 per-image-max 边界 F1。"""
    rgb = Color.image_to_mlx(Image.open(img_path).convert("RGB"))
    hier = run_ours_hierarchy(rgb)
    gt_b = gt_boundaries(gt_path)
    scores = {}
    for tau in TAUS:
        ob = our_boundary(np.array(hier.cut(tau)))
        scores[tau] = max(boundary_f1(ob, gb, tol=tol) for gb in gt_b)
    return scores


def main(n_images: int = 20) -> None:
    root = Utils.project_root()
    imgs = sorted(glob.glob("/tmp/BSDS500/BSDS500/data/images/test/*.jpg"))
    step = max(1, len(imgs) // n_images)
    imgs = imgs[::step]
    tol = int(0.0075 * (321.0**2 + 481.0**2) ** 0.5)
    print(f"BSDS test 子集 {len(imgs)} 张, 容差 {tol}px, τ∈{TAUS}")

    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(imgs):
        name = p.split("/")[-1]
        gtp = p.replace("/images/", "/groundTruth/").replace(".jpg", ".mat")
        sc = eval_image(p, gtp, tol)
        rows.append((name, sc))
        print(f"[{i + 1}/{len(imgs)}] {name}: "
              + " ".join(f"τ{t}={v:.3f}" for t, v in sc.items())
              + f" | best={max(sc.values()):.3f} "
              f"({(time.perf_counter() - t0) / 60:.0f}min)")

    ods = {t: float(np.mean([sc[t] for _, sc in rows])) for t in TAUS}
    ois = float(np.mean([max(sc.values()) for _, sc in rows]))
    print("\n== 汇总 ==")
    for t, v in ods.items():
        print(f"  固定 τ={t}: 平均 F1 = {v:.3f} (≈ODS)")
    print(f"  逐图最佳 τ: 平均 F1 = {ois:.3f} (≈OIS)")

    # 最佳/中位/最差三例可视化
    ranked = sorted(rows, key=lambda r: -max(r[1].values()))
    pick = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for r_i, (name, sc) in enumerate(pick):
        img = Image.open(
            f"/tmp/BSDS500/BSDS500/data/images/test/{name}"
        ).convert("RGB")
        gt_b = gt_boundaries(
            "/tmp/BSDS500/BSDS500/data/groundTruth/test/"
            + name.replace(".jpg", ".mat")
        )[0]
        rgb = Color.image_to_mlx(img)
        best_tau = max(sc, key=sc.get)
        seg_regions = run_ours_hierarchy(rgb).cut(best_tau)
        axes[r_i, 0].imshow(img)
        axes[r_i, 0].set_title(f"{name} (F1={sc[best_tau]:.3f} @τ={best_tau})")
        ys, xs = np.where(gt_b)
        axes[r_i, 1].imshow(img)
        axes[r_i, 1].scatter(xs[::2], ys[::2], s=0.3, c="r")
        axes[r_i, 1].set_title("人工标注边界 (标注者0)")
        axes[r_i, 2].imshow(colorize(np.array(seg_regions)))
        axes[r_i, 2].set_title(f"conger 区域 ×{int(seg_regions.max())}")
        for ax in axes[r_i]:
            ax.axis("off")
    fig.tight_layout()
    out = root / "artifacts/bsd_eval_cases.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
