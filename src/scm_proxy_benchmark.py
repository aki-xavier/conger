"""SCMProxyBenchmark: 用真实 renderer 校准外观机制代理 (路线 ②)。

固定几何, 渲染全因子 (hue,lcol,ldir)=54 干预样本, 取前景平均 RGB,
拟合乘法机制 `albedo[hue] ⊙ lighting[lcol,ldir]`, 报告重构误差与
反照率不变性分数。期望: MeshStandardMaterial 的反照率×光照物理使
乘法分解精确 (不变性≈1), 反照率项恢复真实图元色 (与 Codebook.obj_color
对齐); 偏离纯乘法的残差暴露机制的非模块性。
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from codebook import Codebook
from inverse_config import InverseConfig
from scm_proxy import AppearanceMechanism
from stereo import StereoDepth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.parse_args()

    cfg = InverseConfig(scene_family="single")
    codebook = Codebook(cfg)
    renderer, cam_l, cam_r = Codebook.make_renderer()
    # 固定几何 (球体, 中心偏上), 只扫外观因子
    base = (0, 72.0, 72.0, 0.45, 3.2)
    rgb = mx.zeros(
        (Codebook.N_HUE, len(Codebook.LIGHT_COLORS), len(Codebook.LIGHT_DIRS), 3)
    )
    for h in range(Codebook.N_HUE):
        for lc in range(len(Codebook.LIGHT_COLORS)):
            for ld in range(len(Codebook.LIGHT_DIRS)):
                prm = base + (float(h), float(lc), float(ld))
                scene = codebook.to_scene(prm)
                fl = renderer.render(scene, cam_l)
                w = StereoDepth.foreground_weights(fl)
                rgb[h, lc, ld] = AppearanceMechanism.foreground_mean_rgb(fl, w)

    mechanism = AppearanceMechanism().fit(rgb)
    err = mechanism.reconstruction_error(rgb)
    inv = mechanism.albedo_invariance(rgb)
    assert mechanism.albedo is not None and mechanism.lighting is not None
    print(f"重构误差 {err:.4f} / 反照率不变性 {inv:.4f}")
    print("反照率机制项 (单位光照下, 归一化到最大通道):")
    for h in range(Codebook.N_HUE):
        a = mechanism.albedo[h]
        a = a / max(float(mx.max(a)), 1e-8)
        true_hex = f"#{Codebook.obj_color(h):06x}"
        print(
            f"  hue {h} ({true_hex}): "
            f"R {float(a[0]):.3f} G {float(a[1]):.3f} B {float(a[2]):.3f}"
        )
    print("光照机制项 (单位均值, 每 (lcol,ldir) 的 RGB 增益):")
    for lc in range(len(Codebook.LIGHT_COLORS)):
        row = []
        for ld in range(len(Codebook.LIGHT_DIRS)):
            g = mechanism.lighting[lc, ld]
            row.append(
                f"({lc},{ld})=[{float(g[0]):.2f},{float(g[1]):.2f},{float(g[2]):.2f}]"
            )
        print("  " + " ".join(row))


if __name__ == "__main__":
    main()
