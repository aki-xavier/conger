"""性能瓶颈评估: realtime 闭环管线阶段级计时 (v2, 全直接调用)。

方法: time.perf_counter + mx.eval (cProfile 扭曲 GPU 惰性执行)。
前台 (关键路径): Riesz 特征 → VBGMM online → 似然 → 增强 → 提交,
逐阶段打点。后台 (并行线程): grouping → segment → fusion+scenegraph,
直接调用同一输入计时。合成世界 (realtime 闭环同款)。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_perf_profile.py [帧数]
"""

import statistics
import sys
import time

import mlx.core as mx

from utils import Utils

H, W = 128, 256


def make_world():
    """realtime 闭环合成世界同款: 三块广告牌 + 相机 x 平移。"""
    from edgemap import EdgePrior as _EP

    FX, DX = 100.0, 0.04
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
    yy2, xx2 = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )

    def wframe(k: int) -> mx.array:
        dx_f = k * FX * DX / zmap
        smp = _EP.precomp_gather((H, W), mx.zeros((H, W)), dx_f, yy2, xx2)
        return smp(canvas) + mx.random.normal(
            (H, W), key=mx.random.key(20 + k)
        ) * 0.01

    return wframe


def profile(n_frames: int = 4) -> None:
    from edgemap import EdgePrior
    from fusion import DepthCue, DepthFusionLayer
    from grouping import PerceptualGrouping
    from riesz import RieszWavelet
    from segment import SceneSegmenter, grouping_contours
    from vbgmm import VBGMM

    wframe = make_world()
    img0 = wframe(0)

    # ── 前台逐阶段计时 (关键路径) ────────────────────────────────
    fg = {"riesz": [], "features": [], "online": [], "like": [],
          "enhance": []}
    rw = RieszWavelet(img0)
    gm = VBGMM(VBGMM.feature_matrix(rw.features()), k_max=48)
    for k in range(n_frames):
        img = wframe(k)
        t = time.perf_counter()
        rw.update(img)
        mx.eval(rw.scales)
        fg["riesz"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        feat = rw.features()
        mx.eval(feat.log_mag)
        fg["features"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        x = VBGMM.feature_matrix(feat)
        r = gm.online_update(x, rho=0.3)
        mx.eval(r)
        fg["online"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        like = gm.class_likelihood("edge", x=x, r=r)
        mx.eval(like)
        fg["like"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        enh = EdgePrior().enhance(like.reshape(H, W), feat, rw)
        mx.eval(enh)
        fg["enhance"].append(1000 * (time.perf_counter() - t))

    print(f"\n== realtime 闭环性能 (合成世界 {H}x{W}, {n_frames} 帧) ==")
    print("  帧预算对照: 30fps=33ms / 20fps=50ms / 15fps=67ms")
    print("  ── 前台 (关键路径, 逐阶段) ──")
    total = 0.0
    for name, vals in fg.items():
        med = statistics.median(vals)
        total += med
        print(f"    {name:<12} 中位 {med:7.1f} ms")
    print(f"    {'Σ 前台':<12}       {total:7.1f} ms")

    # ── 后台逐阶段 (并行线程, 直接调用同一输入) ──────────────────
    feat = rw.features()
    x = VBGMM.feature_matrix(feat)
    like = gm.edge_likelihood((H, W))
    tex = gm.class_likelihood("texture").reshape(H, W)
    enh = EdgePrior().enhance(like, feat, rw)
    mx.eval(enh)
    pg = PerceptualGrouping()
    seg = SceneSegmenter()
    bg = {}
    t = time.perf_counter()
    res = pg.run(enh, feat.mean_ori)
    mx.eval(res.edgels.pos)
    bg["grouping"] = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    polys, circs = grouping_contours(res)
    seg_r = seg.run(enh, like, tex, polys, circs)
    mx.eval(seg_r.regions)
    bg["segment"] = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    fr = DepthFusionLayer().run(
        [DepthCue(mx.full((H, W), 3.0), mx.full((H, W), 1.0))],
        seg_r.subregions, boundary=enh,
    )
    mx.eval(fr.render)
    bg["fusion+scenegraph"] = 1000 * (time.perf_counter() - t)

    print("  ── 后台 (并行线程) ──")
    bg_total = 0.0
    for name, v in bg.items():
        bg_total += v
        print(f"    {name:<22} {v:7.1f} ms")
    print(f"    {'Σ 后台':<22} {bg_total:7.1f} ms (并行, 与前台不叠加)")
    print(f"    后台/前台 = {bg_total / total:.1f}× —— 若后台跟不上, "
          f"tracker 丢弃中间帧 (只保留最新)")


if __name__ == "__main__":
    profile(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
