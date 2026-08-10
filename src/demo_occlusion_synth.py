"""E3 判别: 合成世界全链序正确率 (2026-08-10 序数修复讨论)。

目的: 隔离"映射逻辑 bug" vs "室内几何失效"。realtime 广告牌世界
(真遮挡: 3 块不同深度广告牌 z=2/5/3 vs 背景 4, 板内光栅线在板边
终止 → 真 T 结, GT 深度已知)。跑全链 (grouping → segment →
constraints_from_grouping → 对 GT 深度验证序正确率)。

- E3 通过 (≈1)  → 映射逻辑通, 问题是室内几何 → 修 T 结质量门/边界供给
- E3 失败 (≈0.5) → constraints_from_grouping 有真 bug → 改区域感知映射

用法: PYTHONPATH=src .venv/bin/python3 src/demo_occlusion_synth.py
"""

import time

import mlx.core as mx
import numpy as np

from utils import Utils

H, W = 128, 256


def make_world():
    """realtime 闭环合成世界同款: 三块光栅广告牌 + 背景, GT 深度 zmap。
    (E3 探针: 该世界能产出约束, 序正确率 n=5 不显著但非 ≈1)"""
    RECTS = [
        (20, 60, 30, 70, 2.0, 0.60),
        (20, 60, 150, 190, 5.0, 0.85),
        (70, 110, 90, 130, 3.0, 0.55),
    ]
    canvas = mx.full((H, W), 0.15)
    zmap = mx.full((H, W), 4.0)
    for r0, r1, c0, c1, z, val in RECTS:
        gr = Utils.make_grating((r1 - r0, c1 - c0), 6.0, 0.0)
        canvas[r0:r1, c0:c1] = 0.15 + gr * (val - 0.15)
        zmap[r0:r1, c0:c1] = z
    return canvas, zmap


def main() -> None:
    from edgemap import EdgePrior
    from fusion import OcclusionOrder
    from grouping import PerceptualGrouping
    from riesz import RieszWavelet
    from segment import SceneSegmenter, grouping_contours
    from vbgmm import VBGMM

    img, zmap = make_world()
    H0, W0 = img.shape
    rw = RieszWavelet(img)  # 合成世界已是 2D 亮度图
    feat = rw.features()
    gm_l = VBGMM.fast_fit(
        VBGMM.feature_matrix(feat), (H0, W0), k_max=48, coreset=8192
    )
    like = gm_l.edge_likelihood((H0, W0))
    tex = gm_l.class_likelihood("texture").reshape(H0, W0)
    enh = EdgePrior().enhance(like, feat, rw)
    t0 = time.perf_counter()
    res = PerceptualGrouping().run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    sub = seg.subregions
    cons = OcclusionOrder.constraints_from_grouping(res, sub)
    print(f"全链: {time.perf_counter() - t0:.1f}s, T 结 {len(res.t_junctions)}, "
          f"约束 {len(cons)}")

    # 序正确率: front/behind 区域在 GT 深度的中位
    ok = tot = 0
    sub_np = np.asarray(sub)
    zm_np = np.asarray(zmap)
    for cn in cons:
        m_f = (sub_np == cn.front) & (zm_np > 0)
        m_b = (sub_np == cn.behind) & (zm_np > 0)
        nf, nb = int(m_f.sum()), int(m_b.sum())
        if nf < 10 or nb < 10:
            continue
        zf = np.median(zm_np[m_f])
        zb = np.median(zm_np[m_b])
        tot += 1
        ok += 1 if zf < zb else 0
        print(f"  约束@{int(cn.pos[0])},{int(cn.pos[1])}: "
              f"front={cn.front}({nf}px,z={zf:.2f}) "
              f"behind={cn.behind}({nb}px,z={zb:.2f}) "
              f"{'✓' if zf < zb else '✗'}")
    print(f"可验证 {tot} 约束, 序正确率: {ok / max(tot, 1):.3f} "
          f"({'≈1: 映射逻辑通' if ok / max(tot, 1) > 0.7 else '≈0.5: 映射有 bug'})")

    # E4 对照: T 结落在板边 (深度跳变) 的比例
    d = np.log(np.maximum(np.asarray(zmap), 1e-3))
    thr = np.log(1.15)
    gtb_np = np.zeros(zmap.shape, dtype=bool)
    gtb_np[:, 1:] |= np.abs(np.diff(d, axis=1)) > thr
    gtb_np[1:, :] |= np.abs(np.diff(d, axis=0)) > thr
    from scipy import ndimage

    gtb_d = ndimage.binary_dilation(gtb_np, iterations=5)
    n_real = sum(
        1 for t in res.t_junctions
        if gtb_d[int(round(t.pos[0])), int(round(t.pos[1]))]
    )
    print(f"T 结在深度跳变邻域: {n_real}/{len(res.t_junctions)} "
          f"({100 * n_real / max(len(res.t_junctions), 1):.0f}%)")


if __name__ == "__main__":
    main()
