"""特征相关性诊断 (2026-08-10, 压缩启示的实质可做项)。

7 个谱特征 (log_mag/slope/resid/bump/spread/ori_R/phase_coh) 同源自
Riesz 功率谱 —— 检查真实图像上是否近冗余 (|ρ|>0.95), 冗余则降维:
更快更稳的 GMM + 更好的条件数。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_feature_diag.py [每集图像数]
数据: iBims (室内) + BSDS (自然) 混合, 域稳定性对照。
输出: 相关矩阵 + 近冗余对 + 特征值/主成分能量保留 + 建议。
"""

import glob
import sys

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image

from color import Color
from riesz import RieszWavelet
from utils import Utils
from vbgmm import VBGMM

FEATS = VBGMM.FEAT_NAMES


def sample_features(rgb: np.ndarray, max_pix: int = 20000) -> np.ndarray:
    """RGB 图 → 采样像素的 z 标准化特征 (N,7) (GMM 实际看到的空间)。"""
    lum, _ = Color.split_dual_path(
        mx.array(rgb.astype(np.float32) / 255.0)
    )
    rw = RieszWavelet(lum)
    x = np.asarray(VBGMM.feature_matrix(rw.features()), dtype=np.float64)
    if x.shape[0] > max_pix:
        idx = mx.linspace(0, x.shape[0] - 1, max_pix).astype(mx.int32)
        x = np.asarray(x)[np.asarray(idx)]
    return x


def diag(features: np.ndarray, label: str) -> dict:
    """z 标准化 → 相关矩阵/特征值 → 冗余报告 (全 MLX)。"""
    zf = mx.array(features, dtype=mx.float32)
    mu = mx.mean(zf, axis=0)
    sd = mx.std(zf, axis=0) + 1e-9
    z = (zf - mu) / sd
    zc = z - mx.mean(z, axis=0)
    corr = (zc.T @ zc) / zc.shape[0]  # 相关矩阵 (特征已单位方差)
    eig = mx.linalg.eigh(corr, stream=mx.cpu)[0][::-1]  # 升序 → 降序
    cum = mx.cumsum(eig) / mx.sum(eig)
    cond = float(eig[0]) / max(float(eig[-1]), 1e-12)
    corr = np.asarray(corr)  # 仅打印/可视化用 (互操作桥)
    print(f"\n== {label} (n={features.shape[0]} 像素) ==")
    print(f"  相关矩阵 (行/列 = {FEATS}):")
    print("      " + " ".join(f"{n[:4]:>5}" for n in FEATS))
    for i, row in enumerate(corr):
        print(f"  {FEATS[i][:4]:>5} " + " ".join(f"{v:5.2f}" for v in row))
    pairs = []
    for i in range(7):
        for j in range(i + 1, 7):
            r = abs(corr[i, j])
            if r > 0.9:
                pairs.append((FEATS[i], FEATS[j], r))
    print(f"  近冗余对 (|ρ|>0.9): {[(a, b, f'{r:.2f}') for a, b, r in pairs] or '无'}")
    print(f"  特征值: {np.round(eig, 3).tolist()}")
    print(f"  主成分方差保留: 1维={cum[0]:.3f} 2维={cum[1]:.3f} "
          f"3维={cum[2]:.3f} 5维={cum[4]:.3f}")
    print(f"  条件数: {cond:.1f}")
    return {"label": label, "corr": corr, "eig": eig, "pairs": pairs, "cond": cond}


def main(n_images: int = 6) -> None:
    root = Utils.project_root()
    # 混合域样本: iBims (室内) + BSDS (自然)
    mats = sorted(glob.glob("/tmp/datasets/ibims1/ibims1_core_mat/*.mat"))
    bsds = sorted(glob.glob("/tmp/datasets/BSDS500/BSDS500/data/images/test/*.jpg"))
    picks = {
        "iBims": [mats[i] for i in np.linspace(0, len(mats) - 1, n_images).astype(int)],
        "BSDS": [bsds[i] for i in np.linspace(0, len(bsds) - 1, n_images).astype(int)],
    }
    results = {}
    for domain, paths in picks.items():
        feats = []
        for p in paths:
            if p.endswith(".mat"):
                import scipy.io

                rgb = scipy.io.loadmat(p)["data"][0, 0]["rgb"].astype(np.uint8)
            else:
                rgb = np.asarray(Image.open(p).convert("RGB"))
            feats.append(sample_features(rgb))
        X = np.concatenate(feats)
        results[domain] = diag(X, f"{domain} 特征相关诊断")

    # 跨域汇总: 两个域最强相关对是否一致 (域稳定性)
    print("\n== 跨域对比 (|ρ|>0.9 的冗余对) ==")
    all_pairs = set(
        (a, b) for res in results.values() for a, b, _ in res["pairs"]
    )
    if not all_pairs:
        print("  两域均无 |ρ|>0.9 冗余对 —— 特征空间 7 维独立, 无需降维")
    for a, b in sorted(all_pairs):
        ri = next((r for x, y, r in results["iBims"]["pairs"]
                   if (x, y) == (a, b)), None)
        rb = next((r for x, y, r in results["BSDS"]["pairs"]
                   if (x, y) == (a, b)), None)
        print(f"  {a}-{b}: iBims |ρ|={ri:.2f}, BSDS |ρ|={rb:.2f}")

    # 可视化: 两域相关矩阵热图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = None
    for ax, (domain, res) in zip(axes, results.items()):
        im = ax.imshow(res["corr"], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(7), [f[:4] for f in FEATS], rotation=45)
        ax.set_yticks(range(7), [f[:4] for f in FEATS])
        ax.set_title(f"{domain} 相关矩阵 (cond={res['cond']:.0f})")
    fig.colorbar(im, ax=axes)
    fig.tight_layout()
    out = root / "artifacts/feature_corr_diag.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
