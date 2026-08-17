"""BasisSweep: 逐族白化基内在维截断的精度-内存扫描 (post-hoc, 不重训)。

对已有的全维模型做 `model_memory.truncate_basis` 后置截断, 在子集测试集
上评估, 找各族的甜点 D。单物体已验 D≈48; 多物体 (layered/composite)
全维 D≈1360, 甜点未必 48, 需实测。默认只扫截断维 (全维 predict 在多物体
2916 帧上 ~12min, 跳过), 全维基线见 docs §4。
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from composite_reconstructor import CompositeReconstructor
from data_builder import DataBuilder
from evaluator import LAYERED_FACTORS, LAYERED_TARGET_COLS
from inverse_app import InverseApp
from inverse_config import InverseConfig
from layered_reconstructor import LayeredReconstructor
from mixture_spn import MixtureSPN
from model_memory import model_size_mb, truncate_basis

MODELS = {
    "layered": "artifacts/spn_layered_anchor_mix_144x144_luloluphluorchlochphchorchlochphchorchrachra_k3h6c3d3o2sv3rp2_sl8.safetensors",
    "composite": "artifacts/spn_composite_mix_144x144_luloluphluorchlochphchorchlochphchorchrachra_k3h6c3d3o2sv1rp2_cp2.safetensors",
}
TARGET_NAMES = ("u0", "v0", "s0", "z0", "u1", "v1", "s1", "z1")


def _eval(p_gt: mx.array, t_pred: mx.array, scene_pred, p_train: mx.array):
    """紧凑评估: 离散因子准确率 + 8 连续目标 R² (对齐 Evaluator.report)。"""
    gt = p_gt[:, list(LAYERED_TARGET_COLS)]
    base = mx.mean(p_train[:, list(LAYERED_TARGET_COLS)], axis=0, keepdims=True)
    ss_base = mx.sum((gt - base) ** 2, axis=0)
    ss_res = mx.sum((gt - t_pred) ** 2, axis=0)
    r2 = 1.0 - ss_res / mx.maximum(ss_base, 1e-12)
    pred = mx.array(scene_pred, dtype=mx.float32)
    acc = {nm: float(mx.mean((pred[:, j] == p_gt[:, j]).astype(mx.float32))) for nm, j in LAYERED_FACTORS}
    r2s = {f"{nm}_r2": float(r2[i]) for i, nm in enumerate(TARGET_NAMES)}
    return acc, r2s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="layered", choices=("layered", "composite"))
    ap.add_argument("--subset", type=int, default=256)
    ap.add_argument("--dims", default="512,256,128,64,48,32")
    args = ap.parse_args()

    cfg = InverseConfig(scene_family=args.family, replicates=1)
    app = InverseApp(cfg)
    f_tr, p_tr, f_ti, p_ti, f_te, p_te, s_tr, s_ti, s_te = app.data.build(1)
    c_tr = DataBuilder.scene_classes(p_tr)
    t_tr = DataBuilder.targets(p_tr)
    recon = CompositeReconstructor if args.family == "composite" else LayeredReconstructor
    recon.residual_targets(t_tr, c_tr, s_tr)  # 仅验证契约, 训练目标不参与扫描

    full = MixtureSPN.load(MODELS[args.family])
    n = min(args.subset, f_ti.shape[0])
    print(f"{args.family}: full D={full.f_mu.shape[1]} K={full.f_mu.shape[0]} subset={n}")
    for D in (int(x) for x in args.dims.split(",")):
        m = full if D >= full.f_mu.shape[1] else truncate_basis(full, D)
        ti_raw, ci_p, _ = m.predict(f_ti[:n])
        te_raw, ce_p, _ = m.predict(f_te[:n])
        ci_pred = recon.params(ti_raw, ci_p, s_ti[:n])
        ce_pred = recon.params(te_raw, ce_p, s_te[:n])
        ti_pred = recon.targets_from_params(ci_pred)
        te_pred = recon.targets_from_params(ce_pred)
        ai, ri = _eval(p_ti[:n], ti_pred, ci_pred, p_tr)
        ae, re = _eval(p_te[:n], te_pred, ce_pred, p_tr)
        print(
            f"D={m.f_mu.shape[1]:4d} {model_size_mb(m):6.0f}MB | "
            f"I kind0={ai['kind0']:.3f} kind1={ai['kind1']:.3f} hue0={ai['hue0']:.3f} "
            f"hue1={ai['hue1']:.3f} lcol={ai['lcol']:.3f} ldir={ai['ldir']:.3f} | "
            f"u0={ri['u0_r2']:.2f} v0={ri['v0_r2']:.2f} z0={ri['z0_r2']:.2f} "
            f"u1={ri['u1_r2']:.2f} v1={ri['v1_r2']:.2f} z1={ri['z1_r2']:.2f} | "
            f"E kind0={ae['kind0']:.3f} lcol={ae['lcol']:.3f} u0={re['u0_r2']:.2f} "
            f"v0={re['v0_r2']:.2f} z1={re['z1_r2']:.2f}"
        )


if __name__ == "__main__":
    main()
