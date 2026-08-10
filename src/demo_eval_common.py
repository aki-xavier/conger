"""数据集评估共享基础设施 (iBims-1 / NYUv2 等)。

管线调用 + 深度度量 + 可视化, 供 demo_ibims_eval / demo_nyu_eval 复用。
度量口径:
  深度线索是"弱单调约束"级 (倍频程分辨率, 物理上限 ~3-5 档) ——
  绝对度量 (δ/RMSE) 前必须先做逐图尺度对齐 (最小二乘 ẑ = a·z+b),
  对齐后指标反映线性精度上限; Spearman 秩相关是免对齐的诚实口径
  (直接测单调性, 即线索的设计承诺)。
"""


import math

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from color import Color
from demo_sam_compare import boundary_f1
from utils import Utils

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TAUS = (0.2, 0.3, 0.5)


def run_ours(rgb: np.ndarray | mx.array) -> tuple:
    """conger 外观链: 双通路特征 → VBGMM → 边缘似然 → 增强 → 区域层级
    + 单目深度线索。返回 (hierarchy, enh, cue)。
    与 demo_bsd_eval.run_ours_hierarchy 同款冷启动 (级联 + coreset)。"""
    from edgemap import EdgePrior
    from monocular import MonocularCues
    from riesz import RieszWavelet
    from segment import RegionLayer
    from vbgmm import VBGMM

    if isinstance(rgb, np.ndarray):
        rgb = mx.array(rgb.astype(np.float32) / 255.0)  # numpy → [0,1] MLX
    h, w = rgb.shape[:2]
    lum, hs = Color.split_dual_path(rgb)
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM.fast_fit(
        VBGMM.feature_matrix(feat), (h, w), k_max=48, coreset=8192
    )
    like_l = gm_l.edge_likelihood((h, w))
    tex_l = gm_l.class_likelihood("texture").reshape(h, w)
    gm_h = VBGMM.fast_fit(
        VBGMM.hs_feature_matrix(hs).reshape(-1, 7), (h, w),
        k_max=32, coreset=8192,
    )
    like_h = gm_h.edge_likelihood((h, w)) * mx.abs(hs)
    like = 1 - (1 - like_l) * (1 - like_h)
    tex_h = gm_h.class_likelihood("texture").reshape(h, w)
    tex = 1 - (1 - tex_l) * (1 - tex_h)
    enh = EdgePrior().enhance(like, feat, rw)
    hier = RegionLayer().run(enh).hierarchy
    cue = MonocularCues().texture_scale(rw, tex)
    return hier, enh, cue


def _spearman(a: mx.array, b: mx.array) -> float:
    """Spearman 秩相关 (全 MLX): 秩 = 双重 argsort, 相关 = 手写。"""
    ra = mx.argsort(mx.argsort(a)).astype(mx.float32)
    rb = mx.argsort(mx.argsort(b)).astype(mx.float32)
    ma = mx.mean(ra)
    mb = mx.mean(rb)
    cov = mx.sum((ra - ma) * (rb - mb))
    va = mx.sum((ra - ma) ** 2)
    vb = mx.sum((rb - mb) ** 2)
    return float(cov / mx.sqrt(va * vb + 1e-9))


def depth_metrics(
    z_pred: np.ndarray | mx.array,
    z_gt: np.ndarray | mx.array,
    valid: np.ndarray | mx.array,
) -> dict:
    """单目深度线索 vs GT: 逐图尺度对齐后 δ1/δ2/δ3/RMSE/log-RMSE + Spearman。
    z_pred 为相对线索 (任意尺度); 对齐 = 有效像素上最小二乘 ẑ=a·z+b。
    全 MLX (铁规则: 矩阵计算一律 MLX; 入参 numpy 在入口转 mx)。"""
    zp = mx.array(z_pred, dtype=mx.float32)
    zg = mx.array(z_gt, dtype=mx.float32)
    vd = mx.array(valid, dtype=mx.bool_)
    v = vd & mx.isfinite(zp) & mx.isfinite(zg) & (zg > 0)
    n = int(mx.sum(v))
    if n < 100:
        return {"n": n, "delta1": float("nan"), "delta2": float("nan"),
                "delta3": float("nan"), "rmse": float("nan"),
                "log_rmse": float("nan"), "spearman": float("nan"),
                "sp_region": float("nan")}
    key = mx.where(v.reshape(-1), mx.arange(v.size), v.size)
    idx = mx.argsort(key)[:n]
    p, g = zp.reshape(-1)[idx], zg.reshape(-1)[idx]
    # 尺度对齐: 正规方程 lstsq (ẑ = a·p + b) —— MLX 无 lstsq, 解 AᵀA
    nf = float(n)
    sp_ = float(mx.sum(p))
    sg = float(mx.sum(g))
    spp = float(mx.sum(p * p))
    spg = float(mx.sum(p * g))
    denom = nf * spp - sp_ * sp_
    a = (nf * spg - sp_ * sg) / denom
    b = (spp * sg - sp_ * spg) / denom
    pred = a * p + b
    err = pred - g
    rmse = float(mx.sqrt(mx.mean(err**2)))
    log_rmse = float(mx.sqrt(mx.mean((mx.log(pred) - mx.log(g)) ** 2)))
    ratio = mx.maximum(pred / g, g / pred)
    delta1 = float(mx.mean(ratio < 1.25))
    delta2 = float(mx.mean(ratio < 1.25**2))
    delta3 = float(mx.mean(ratio < 1.25**3))
    sp = _spearman(p, g)
    # 区域级 Spearman (线索承诺口径: 16×16 块中位数聚合后测秩)
    sp_region = float("nan")
    bs = 16
    Hh, Ww = zg.shape
    bp, bg = [], []
    for r0 in range(0, Hh, bs):
        for c0 in range(0, Ww, bs):
            mv = vd[r0 : r0 + bs, c0 : c0 + bs]
            if int(mx.sum(mv)) < 8:
                continue
            key2 = mx.where(mv.reshape(-1), mx.arange(mv.size), mv.size)
            idx2 = mx.argsort(key2)[: int(mx.sum(mv))]
            bp.append(mx.median(zp[r0 : r0 + bs, c0 : c0 + bs].reshape(-1)[idx2]))
            bg.append(mx.median(zg[r0 : r0 + bs, c0 : c0 + bs].reshape(-1)[idx2]))
    if len(bp) >= 8:
        sp_region = _spearman(mx.stack(bp), mx.stack(bg))
    return {"n": n, "delta1": delta1, "delta2": delta2, "delta3": delta3,
            "rmse": rmse, "log_rmse": log_rmse, "spearman": sp,
            "sp_region": sp_region}


def edge_f1(enh: mx.array, gt_edges: np.ndarray, tol: int = 3) -> dict[float, float]:
    """增强边缘图 vs GT 二值边缘: 逐 τ 边界 F1。"""
    e = np.asarray(enh)
    return {
        t: boundary_f1(e > t, gt_edges.astype(bool), tol=tol) for t in TAUS
    }


def region_boundary(regions: np.ndarray | mx.array) -> mx.array:
    """区域标签图 → 二值边界图 (MLX)。"""
    r = mx.array(regions)
    h, w = r.shape
    b = mx.pad(r[:-1] != r[1:], [(0, 1), (0, 0)])
    b = b | mx.pad(r[:, :-1] != r[:, 1:], [(0, 0), (0, 1)])
    return b


def occlusion_boundary_recall(
    regions: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    tol: int = 3,
    rel_jump: float = 0.15,
) -> tuple[float, int]:
    """遮挡边界 recall (分割的下游准绳之一, 2026-08-10 架构检讨):
    GT = 深度跳变边界 (log 深度相对跳变 > rel_jump, 免尺度);
    我们的边界 = 区域标签变化。只测 recall —— confetti (多碎边界)
    不惩罚, 正是 BSDS-F1 与该口径的关键差异 (BSDS 惩罚碎裂, 项目
    下游 (场景图合并) 对碎裂自愈; 漏遮挡界 = 漏 T 结 = 深度序错,
    才是致命)。返回 (recall, GT 边界像素数)。"""
    d = mx.log(mx.maximum(mx.array(depth, dtype=mx.float32), 1e-3))
    thr = math.log(1.0 + rel_jump)
    gx = mx.abs(mx.diff(d, axis=1)) > thr  # (H,W-1)
    gy = mx.abs(mx.diff(d, axis=0)) > thr  # (H-1,W)
    # 双向传播 (像素两侧任一侧跳变即边界); pad 而非 ArrayAt (无 __ior__)
    gtb = mx.pad(gx, [(0, 0), (0, 1)]) | mx.pad(gx, [(0, 0), (1, 0)])
    gtb = gtb | mx.pad(gy, [(0, 1), (0, 0)]) | mx.pad(gy, [(1, 0), (0, 0)])
    gtb = gtb & mx.array(valid, dtype=mx.bool_)
    n_gt = int(mx.sum(gtb))
    if n_gt < 50:
        return float("nan"), n_gt
    ob = region_boundary(regions)
    # scipy.ndimage 膨胀: MLX 无形态学算子, 属"不能使用 mlx"例外
    from scipy import ndimage

    ob_np = np.asarray(ob)
    hit = ndimage.binary_dilation(ob_np, iterations=tol) & np.asarray(gtb)
    return float(hit.sum() / n_gt), n_gt


def aggregate(rows: list[tuple[str, dict]]) -> dict:
    """行列表 → 汇总 (逐指标均值, nan 忽略)。纯 Python 标量统计。"""
    import statistics

    keys = list(rows[0][1].keys())
    out = {}
    for k in keys:
        vals = [sc[k] for _, sc in rows if math.isfinite(sc[k])]
        out[k] = statistics.fmean(vals) if vals else float("nan")
    return out


def print_summary(title: str, agg: dict) -> None:
    print(f"\n== {title} 汇总 (n={int(agg['n'])} 像素/图均值) ==")
    if "delta1" in agg:
        print(
            f"  尺度对齐后: δ1={agg['delta1']:.3f} δ2={agg['delta2']:.3f} "
            f"δ3={agg['delta3']:.3f} RMSE={agg['rmse']:.3f}m "
            f"logRMSE={agg['log_rmse']:.3f}"
        )
    if "spearman" in agg:
        print(f"  Spearman 秩相关 (免对齐, 单调性): {agg['spearman']:.3f} "
              f"(区域级: {agg.get('sp_region', float('nan')):.3f})")
    for k, v in agg.items():
        if k not in ("n", "delta1", "delta2", "delta3", "rmse",
                     "log_rmse", "spearman", "sp_region") and np.isfinite(v):
            print(f"  {k}: {v:.3f}")


def save_cases(name: str, cases: list[tuple[str, np.ndarray, float]],
               key: str, extra: list[str] | None = None) -> None:
    """三例 (最佳/中位/最差) 可视化 → artifacts/<name>.png。
    cases: [(标题, RGB 图, 指标值)] 已按指标排序。extra: 额外副标题。"""
    pick = [cases[0], cases[len(cases) // 2], cases[-1]]
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    for ax, (title, img, val) in zip(axes, pick):
        ax.imshow(img)
        ax.set_title(f"{title} ({key}={val:.3f})")
        ax.axis("off")
    fig.tight_layout()
    out = Utils.project_root() / f"artifacts/{name}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)
