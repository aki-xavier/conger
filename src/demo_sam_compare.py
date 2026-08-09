"""SAM 参照检验: Segment Anything (vit_b) 的分割结果 vs conger 管线分割。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_sam_compare.py [图片名]
输出: artifacts/sam_cmp_<图片名> (4 面板) + 逐 mask IoU 报告。

定位: SAM 是深度学习派参照系, 不是真值 —— 报告口径是"一致度"
(IoU/覆盖/边界 F1), 分歧处各自合理与否要目检, 不分高下。
注: SAM MPS 后端不支持 float64, 须 CPU (device 选择见代码)。
有 BSDS 人工标注后本脚本退居次要参照 (demo_bsd_eval 为主)。
"""

import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from color import Color
from edgemap import EdgePrior
from grouping import PerceptualGrouping
from riesz import RieszWavelet
from segment import SceneSegmenter, grouping_contours
from utils import Utils
from vbgmm import VBGMM

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TOP_K = 8  # 取 SAM 面积前 K 的 mask 做逐一对照


def run_ours(rgb: mx.array):
    """conger 外观链 → segment.regions (与 demo_full 相同路径)。"""
    h, w = rgb.shape[:2]
    lum, hs = Color.split_dual_path(rgb)
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    like_l = gm_l.edge_likelihood((h, w))
    tex_l = gm_l.class_likelihood("texture").reshape(h, w)
    gm_h = VBGMM(VBGMM.hs_feature_matrix(hs).reshape(-1, 7), k_max=32)
    like_h = gm_h.edge_likelihood((h, w)) * mx.abs(hs)
    tex_h = gm_h.class_likelihood("texture").reshape(h, w)
    like = 1 - (1 - like_l) * (1 - like_h)
    tex = 1 - (1 - tex_l) * (1 - tex_h)
    enh = EdgePrior().enhance(like, feat, rw)
    res = PerceptualGrouping().run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    return seg


def colorize(labels: np.ndarray) -> np.ndarray:
    """标签图 → 随机但确定性的彩色。"""
    rng = np.random.default_rng(0)
    pal = rng.random((int(labels.max()) + 2, 3)) * 0.8 + 0.2
    out = pal[labels % (labels.max() + 1)]
    return out.astype(np.float32)


def boundary_f1(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    """边界 F1 (tol 像素容差): a/b 为二值边界图。"""
    from scipy import ndimage

    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() or not b.any():
        return 0.0
    hit_a = ndimage.binary_dilation(a, iterations=tol) & b
    hit_b = ndimage.binary_dilation(b, iterations=tol) & a
    prec = hit_b.sum() / max(b.sum(), 1)
    rec = hit_a.sum() / max(a.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """二值 mask 的边界 (自身与 1px 腐蚀之差)。"""
    from scipy import ndimage

    return mask & np.logical_not(ndimage.binary_erosion(mask))


def main(img_name: str = "12.png") -> None:
    root = Utils.project_root()
    im = Image.open(root / f"images/{img_name}").convert("RGB")
    rgb = Color.image_to_mlx(im)
    H, W = rgb.shape[:2]
    print(f"{img_name}: {H}×{W}")

    # ── SAM (vit_b; MPS 不支持 float64, 用 CPU) ─────────────────────
    t0 = time.perf_counter()
    sam = sam_model_registry["vit_b"](
        checkpoint=str(root / "checkpoints/sam_vit_b_01ec64.pth")
    )
    sam.to("cpu")
    gen = SamAutomaticMaskGenerator(sam)
    sam_out = gen.generate(np.array(im))
    sam_out = sorted(sam_out, key=lambda m: -m["area"])[:TOP_K]
    print(f"SAM: {len(sam_out)} 顶层 mask ({time.perf_counter() - t0:.0f}s)")

    # ── conger 分割 ──────────────────────────────────────────────────
    t1 = time.perf_counter()
    seg = run_ours(rgb)
    regions = np.array(seg.regions)
    print(f"conger: {int(seg.regions.max())} 区域 ({time.perf_counter() - t1:.0f}s)")

    # ── 逐 SAM mask 对照 ─────────────────────────────────────────────
    ours_labels = np.unique(regions[regions > 0])
    print(f"{'SAM mask':>8} {'面积%':>6} {'最佳IoU':>8}"
          f" {'双区并IoU':>9} {'边界F1':>6}")
    ious, bf1s = [], []
    iou_map = np.zeros((H, W), dtype=np.float32)
    for k, m in enumerate(sam_out):
        mask = m["segmentation"]
        area_pct = 100 * mask.sum() / (H * W)
        best_iou, best_two = 0.0, 0.0
        per_region_iou = []
        for r in ours_labels:
            rm = regions == r
            inter = (mask & rm).sum()
            if inter == 0:
                continue
            iou = inter / (mask | rm).sum()
            per_region_iou.append((iou, r))
            best_iou = max(best_iou, iou)
        per_region_iou.sort(reverse=True)
        if len(per_region_iou) >= 2:
            r1, r2 = per_region_iou[0][1], per_region_iou[1][1]
            uni = (regions == r1) | (regions == r2)
            best_two = (mask & uni).sum() / (mask | uni).sum()
        elif per_region_iou:
            best_two = best_iou
        mb = mask_boundary(mask)
        ob = np.zeros_like(mask, dtype=bool)
        if per_region_iou:
            ob = mask_boundary(regions == per_region_iou[0][1])
        f1 = boundary_f1(mb, ob)
        ious.append(best_iou)
        bf1s.append(f1)
        iou_map[mask] = best_iou
        print(f"{k:>8} {area_pct:>6.1f} {best_iou:>8.2f}"
              f" {best_two:>9.2f} {f1:>6.2f}")

    # ── 4 面板 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    ax = axes[0]
    ax.imshow(im)
    for m in sam_out:
        ys, xs = np.where(mask_boundary(m["segmentation"]))
        ax.scatter(xs[::4], ys[::4], s=0.3)
    ax.set_title(f"SAM 顶层 {len(sam_out)} mask 边界")
    ax.axis("off")
    sam_lbl = np.zeros((H, W), dtype=np.int32)
    for k, m in enumerate(sam_out):
        sam_lbl[m["segmentation"]] = k + 1
    axes[1].imshow(colorize(sam_lbl))
    axes[1].set_title(f"SAM mask (面积降序前 {len(sam_out)})")
    axes[1].axis("off")
    axes[2].imshow(colorize(regions))
    axes[2].set_title(f"conger 区域 ×{int(seg.regions.max())}")
    axes[2].axis("off")
    im3 = axes[3].imshow(iou_map, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(
        f"逐 mask 最佳 IoU (均值 {np.mean(ious):.2f}, F1 {np.mean(bf1s):.2f})"
    )
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.03)
    fig.suptitle(f"SAM × conger: {img_name}", fontsize=13)
    fig.tight_layout()
    out = root / f"artifacts/sam_cmp_{img_name.rsplit('.', 1)[0]}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "12.png")
