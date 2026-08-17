"""ModelMemoryBenchmark: 按需加载 + 动态遗忘的量化权衡。

① split 序列化 + 分级加载: 全量 / transform(白化基) / components(分量表)
   的大小与加载耗时。② basis 截断 (D↓) 与分量驱逐 (K↓) 扫描: 模型大小、
   SPN 精度 (kind/hue/lcol/ldir + u/v/s/z RMSE)、推理速度。
"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx

from mixture_spn import MixtureSPN
from model_memory import (
    forget_components,
    load_components,
    load_transform,
    model_size_mb,
    split_save,
    truncate_basis,
)
from scene_reconstructor import SceneReconstructor


def _evaluate(
    model: MixtureSPN, F: mx.array, P: mx.array, S: mx.array
) -> tuple[dict, dict]:
    """SPN-only 精度: 离散因子准确率 + u/v/s/z RMSE (物理单位)。"""
    t_pred, cat_p, _ = model.predict(F)
    kind = mx.argmax(cat_p[:, :3], axis=1)
    hue = mx.argmax(cat_p[:, 3:9], axis=1)
    lcol = mx.argmax(cat_p[:, 9:12], axis=1)
    ldir = mx.argmax(cat_p[:, 12:15], axis=1)
    acc = {
        "kind": float(mx.mean(mx.equal(kind, P[:, 0]).astype(mx.float32))),
        "hue": float(mx.mean(mx.equal(hue, P[:, 5]).astype(mx.float32))),
        "lcol": float(mx.mean(mx.equal(lcol, P[:, 6]).astype(mx.float32))),
        "ldir": float(mx.mean(mx.equal(ldir, P[:, 7]).astype(mx.float32))),
    }
    phys = SceneReconstructor.physical_targets(t_pred, S, kind)
    gt = P[:, 1:5]
    rmse = mx.sqrt(mx.mean((phys - gt) ** 2, axis=0))
    return acc, {n: float(rmse[i]) for i, n in enumerate(("u", "v", "s", "z"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="artifacts/spn_kindgeo_mix_144x144_luloluphluorchlochphchorchlochphchorchrachra_k3h6c3d3o1sv3rp2_st4.safetensors",
    )
    ap.add_argument(
        "--data",
        default="artifacts/mix_144x144_luloluphluorchlochphchorchlochphchorchrachra_k3h6c3d3o1sv3rp2_st4_ti0.safetensors",
    )
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--split-path", default="artifacts/_split_kindgeo")
    args = ap.parse_args()

    model = MixtureSPN.load(args.model)
    d = mx.load(args.data)
    F, P, S = d["F"][: args.n_frames], d["P"][: args.n_frames], d["S"][: args.n_frames]

    print("=== ① split 序列化 + 分级加载 ===")
    t0 = time.time()
    tp, cp = split_save(model, args.split_path)
    print(f"split_save: {time.time() - t0:.2f}s")
    for p in (tp, cp):
        print(f"  {os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")
    t0 = time.time()
    load_components(args.split_path)
    print(f"load_components (分量表): {time.time() - t0:.4f}s")
    t0 = time.time()
    load_transform(args.split_path)
    print(f"load_transform (白化基): {time.time() - t0:.3f}s")

    print("\n=== ② basis 截断 (D↓, 内存+速度) ===")
    for dm in (497, 256, 128, 64, 32):
        m = truncate_basis(model, dm)
        t0 = time.time()
        acc, rmse = _evaluate(m, F, P, S)
        mx.eval()
        dt = time.time() - t0
        print(
            f"D={m.f_mu.shape[1]:4d} size={model_size_mb(m):6.1f}MB "
            f"kind={acc['kind']:.3f} hue={acc['hue']:.3f} lcol={acc['lcol']:.3f} "
            f"ldir={acc['ldir']:.3f} | u={rmse['u']:.2f} v={rmse['v']:.2f} "
            f"s={rmse['s']:.3f} z={rmse['z']:.3f} | {1000 * dt / F.shape[0]:.1f}ms/f"
        )

    print("\n=== ③ 分量驱逐 (K↓, 速度) ===")
    for km in (1296, 648, 324, 162):
        m = forget_components(model, km, "coreset")
        t0 = time.time()
        acc, rmse = _evaluate(m, F, P, S)
        mx.eval()
        dt = time.time() - t0
        print(
            f"K={m.f_mu.shape[0]:4d} size={model_size_mb(m):6.1f}MB "
            f"kind={acc['kind']:.3f} hue={acc['hue']:.3f} lcol={acc['lcol']:.3f} "
            f"ldir={acc['ldir']:.3f} | u={rmse['u']:.2f} v={rmse['v']:.2f} "
            f"s={rmse['s']:.3f} z={rmse['z']:.3f} | {1000 * dt / F.shape[0]:.1f}ms/f"
        )


if __name__ == "__main__":
    main()
