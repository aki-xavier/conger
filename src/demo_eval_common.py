"""数据集评估共享基础设施 (iBims-1 / NYUv2 等)。

管线调用 + 深度度量 + 可视化, 供 demo_ibims_eval / demo_nyu_eval 复用。
度量口径:
  深度线索是"弱单调约束"级 (倍频程分辨率, 物理上限 ~3-5 档) ——
  绝对度量 (δ/RMSE) 前必须先做逐图尺度对齐 (最小二乘 ẑ = a·z+b),
  对齐后指标反映线性精度上限; Spearman 秩相关是免对齐的诚实口径
  (直接测单调性, 即线索的设计承诺)。
"""


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


def depth_metrics(z_pred: np.ndarray, z_gt: np.ndarray, valid: np.ndarray) -> dict:
    """单目深度线索 vs GT: 逐图尺度对齐后 δ1/δ2/δ3/RMSE/log-RMSE + Spearman。
    z_pred 为相对线索 (任意尺度); 对齐 = 有效像素上最小二乘 ẑ=a·z+b。
    valid 为参与评估的像素掩码 (GT 有效 ∩ 线索有精度)。"""
    v = valid & np.isfinite(z_pred) & np.isfinite(z_gt) & (z_gt > 0)
    n = int(v.sum())
    if n < 100:
        return {"n": n, "delta1": float("nan"), "delta2": float("nan"),
                "delta3": float("nan"), "rmse": float("nan"),
                "log_rmse": float("nan"), "spearman": float("nan"),
                "sp_region": float("nan")}
    p, g = z_pred[v], z_gt[v]
    # 尺度对齐 (最小二乘, 含截距吸收直流)
    a, b = np.polyfit(p, g, 1)
    pred = a * p + b
    err = pred - g
    rmse = float(np.sqrt(np.mean(err**2)))
    log_rmse = float(np.sqrt(np.mean((np.log(pred) - np.log(g)) ** 2)))
    ratio = np.maximum(pred / g, g / pred)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25**2))
    delta3 = float(np.mean(ratio < 1.25**3))
    # Spearman 秩相关 (免对齐, 测单调性; 手写秩相关, 免 scipy 类型依赖)
    rp = np.argsort(np.argsort(p)).astype(np.float64)
    rg = np.argsort(np.argsort(g)).astype(np.float64)
    sp = float(np.corrcoef(rp, rg)[0, 1])
    # 区域级 Spearman (线索承诺口径: 区域级序数骨架, 16×16 块中位
    # 数聚合后测秩; 像素级测的是逐像素承诺, 比模块承诺更严)
    sp_region = float("nan")
    bs = 16
    Hh, Ww = z_pred.shape
    bp, bg = [], []
    for r0 in range(0, Hh, bs):
        for c0 in range(0, Ww, bs):
            m = valid[r0 : r0 + bs, c0 : c0 + bs]
            if int(m.sum()) < 8:
                continue
            bp.append(np.median(z_pred[r0 : r0 + bs, c0 : c0 + bs][m]))
            bg.append(np.median(z_gt[r0 : r0 + bs, c0 : c0 + bs][m]))
    if len(bp) >= 8:
        rp2 = np.argsort(np.argsort(np.array(bp))).astype(np.float64)
        rg2 = np.argsort(np.argsort(np.array(bg))).astype(np.float64)
        sp_region = float(np.corrcoef(rp2, rg2)[0, 1])
    return {"n": n, "delta1": delta1, "delta2": delta2, "delta3": delta3,
            "rmse": rmse, "log_rmse": log_rmse, "spearman": sp,
            "sp_region": sp_region}


def edge_f1(enh: mx.array, gt_edges: np.ndarray, tol: int = 3) -> dict[float, float]:
    """增强边缘图 vs GT 二值边缘: 逐 τ 边界 F1。"""
    e = np.asarray(enh)
    return {
        t: boundary_f1(e > t, gt_edges.astype(bool), tol=tol) for t in TAUS
    }


def aggregate(rows: list[tuple[str, dict]]) -> dict:
    """行列表 → 汇总 (逐指标均值, nan 忽略)。"""
    keys = list(rows[0][1].keys())
    out = {}
    for k in keys:
        vals = [sc[k] for _, sc in rows if np.isfinite(sc[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
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
                     "log_rmse", "spearman") and np.isfinite(v):
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
