"""完整管线全环节验证 (真实自然照片): 19 面板总图。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_full.py [图片名]

环节: 输入 → 双通路 → riesz → vbgmm(L/HS) → 似然融合 → edgemap →
grouping → segment → monocular 深度 → 融合层 → 重力支撑/视平线/
灭点/光源一致性 (spatial_priors) → 本征分解 → C6 分层。
stereo 需要立体对, temporal 需要序列, 均不适用单帧 (见各自自检)。
"""

import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
from PIL import Image

from color import Color
from edgemap import EdgePrior
from fusion import DepthFusionLayer, OcclusionOrder
from grouping import MetelliGate, PerceptualGrouping
from intrinsic import IntrinsicDecomposition
from layers import LayeredPosterior
from monocular import MonocularCues
from riesz import RieszWavelet
from segment import SceneSegmenter, grouping_contours
from spatial_priors import (
    GravitySupport,
    HorizonCue,
    LightFromAbove,
    VanishingPoints,
)
from utils import Utils
from vbgmm import VBGMM

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def hs_to_rgb(hs: mx.array) -> mx.array:
    """复数色度 → 色轮可视化。"""
    h = (mx.arctan2(mx.imag(hs), mx.real(hs)) / (2 * mx.pi)) % 1.0
    s = mx.clip(mx.abs(hs), 0.0, 1.0)
    return Color.hsl_to_rgb(mx.stack([h, s, mx.full(hs.shape, 0.5)], -1))


def main(img_name: str = "nat1018.jpg") -> None:
    root = Utils.project_root()
    im = Image.open(root / f"images/{img_name}")
    if im.mode != "RGB":
        im = im.convert("RGB")
    rgb = Color.image_to_mlx(im)
    H, W = rgb.shape[:2]
    print(f"{img_name}: {H}×{W}")
    t0 = time.perf_counter()

    # ── 双通路 → riesz → vbgmm ──────────────────────────────────────
    lum, hs = Color.split_dual_path(rgb)
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    like_l = gm_l.edge_likelihood((H, W))
    tex_l = gm_l.class_likelihood("texture").reshape(H, W)
    gm_h = VBGMM(VBGMM.hs_feature_matrix(hs).reshape(-1, 7), k_max=32)
    like_h = gm_h.edge_likelihood((H, W)) * mx.abs(hs)  # 饱和度门控
    tex_h = gm_h.class_likelihood("texture").reshape(H, W)
    like = 1 - (1 - like_l) * (1 - like_h)
    tex = 1 - (1 - tex_l) * (1 - tex_h)
    t1 = time.perf_counter()
    print(f"外观前端: {t1 - t0:.0f}s")

    # ── edgemap → grouping → segment ────────────────────────────────
    prior = EdgePrior()
    enh = prior.enhance(like, feat, rw)
    res = PerceptualGrouping().run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    sub = seg.subregions
    t2 = time.perf_counter()
    print(f"组织/分割: {t2 - t1:.0f}s (链 {len(res.chains)}, "
          f"T {len(res.t_junctions)}, X {len(res.x_junctions)})")

    # ── 单目深度 → 融合层 ────────────────────────────────────────────
    cue = MonocularCues().texture_scale(rw, tex)
    occ = OcclusionOrder.constraints_from_grouping(res, sub)
    fr = DepthFusionLayer().run([cue], sub, occlusion=occ, boundary=enh)
    t3 = time.perf_counter()
    print(f"单目深度+融合: {t3 - t2:.0f}s")

    # ── 空间先验 ────────────────────────────────────────────────────
    verdicts = GravitySupport().analyze(fr, sub)
    gi = GravitySupport().ground_index(fr, sub)
    horizon = None
    if gi >= 0:
        g = fr.fits[gi].params
        gidx = Utils.nonzero((sub == gi + 1).reshape(-1))
        v_c = float(mx.mean((gidx // W).astype(mx.float32))) / H - 0.5
        horizon = HorizonCue().estimate(g, (0.0, v_c))
    vps = VanishingPoints().detect(res, (H, W))
    lfa_c = LightFromAbove().consistency(fr.render, lum)
    print(f"空间先验: 地面{'有' if gi >= 0 else '无'}, "
          f"支撑判定 {len(verdicts)}, 灭点 {len(vps)}, LFA={lfa_c:.2f}")

    # ── 本征分解 ────────────────────────────────────────────────────
    dec = IntrinsicDecomposition().estimate(fr.render, lum, sub)
    t4 = time.perf_counter()
    print(f"本征分解: {t4 - t3:.0f}s")

    # ── C6 分层 ─────────────────────────────────────────────────────
    mxs = MetelliGate().validate(res.x_junctions, lum)
    field = LayeredPosterior().from_metelli(mxs, res.x_junctions, lum, sub)
    t5 = time.perf_counter()
    print(f"分层: {t5 - t4:.0f}s (Metelli 锚点 {len(mxs)})")

    # ── 19 面板总图 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 4, figsize=(20, 22))

    def show(ax, arr, title, cmap="gray"):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    show(axes[0, 0], rgb, "① 输入 RGB")
    show(axes[0, 1], lum, "② L 亮度通路")
    show(axes[0, 2], hs_to_rgb(hs), "③ HS 色度通路 (复数)")
    show(axes[0, 3], feat.phase_coh, "④ riesz 相位一致性")

    show(axes[1, 0], like_l, "⑤ L 边缘似然")
    show(axes[1, 1], like_h, "⑥ HS 边缘似然 (饱和门控)")
    show(axes[1, 2], like, "⑦ 融合边缘似然")
    show(axes[1, 3], enh, "⑧ edgemap 增强边界")

    ax = axes[2, 0]
    ax.imshow(enh, cmap="gray", alpha=0.35)
    for i, ch in enumerate(res.chains):
        p = res.edgels.pos[ch]
        ax.plot(p[:, 1], p[:, 0], "-", linewidth=0.6,
                color=plt.cm.tab20(i % 20))
    for t in res.t_junctions:
        ax.plot(t.pos[1], t.pos[0], "rx", markersize=6)
    for x in res.x_junctions:
        ax.plot(x.pos[1], x.pos[0], "bo", markersize=6, fillstyle="none")
    ax.set_title(f"⑨ grouping 链×{len(res.chains)} "
                 f"T×{len(res.t_junctions)} X×{len(res.x_junctions)}",
                 fontsize=10)
    ax.axis("off")
    show(axes[2, 1], seg.regions.astype(mx.float32), "⑩ segment 区域",
         cmap="tab20")
    show(axes[2, 2], fr.render, "⑪ 单目相对深度 (融合渲染)", cmap="viridis")
    show(axes[2, 3], fr.prior_map, "⑫ 深度不连续反馈 prior_map")

    # ⑬ 重力支撑 overlay: 接触=绿, 悬空=红
    ax = axes[3, 0]
    ov = rgb * 0.4
    for vd in verdicts:
        if vd.is_ground:
            tint = mx.array([0.0, 0.5, 0.0])
        elif vd.contact:
            tint = mx.array([0.0, 0.8, 0.0])
        else:
            tint = mx.array([0.8, 0.0, 0.0])
        m = (sub == vd.region)[..., None]
        ov = mx.where(m, ov + tint * 0.6, ov)
    ax.imshow(mx.clip(ov, 0, 1))
    ax.set_title(f"⑬ 重力支撑: 绿=接触 红=悬空 ({len(verdicts)} 区)",
                 fontsize=10)
    ax.axis("off")
    # ⑭ 视平线 overlay
    ax = axes[3, 1]
    ax.imshow(rgb)
    if horizon is not None:
        v_h, slope = horizon
        cols = mx.arange(W)
        rows = H / 2 + max(H, W) * (v_h + slope * (cols - W / 2) / max(H, W))
        ax.plot(cols.tolist(), rows.tolist(), "r-", linewidth=2,
                label="视平线")
        ax.set_ylim(H, 0)
        ax.legend(fontsize=9)
        ax.set_title("⑭ 视平线 (地面平面解析)", fontsize=10)
    else:
        ax.set_title("⑭ 视平线 (未检出地面)", fontsize=10)
    ax.axis("off")
    # ⑮ 灭点 overlay
    ax = axes[3, 2]
    ax.imshow(rgb)
    for vr, vc, wgt in vps[:3]:
        ax.plot(vc, vr, "y*", markersize=18, markeredgewidth=2)
    ax.set_title(f"⑮ 灭点 ×{len(vps)} (线性透视)", fontsize=10)
    ax.axis("off")
    # ⑯ 光源一致性数值 + shading 图
    show(axes[3, 3], dec.shading, f"⑯ shading 图 (LFA 一致性 {lfa_c:.2f})")

    show(axes[4, 0], dec.albedo, "⑰ 反照率图 (本征分解)", cmap="viridis")
    show(axes[4, 1], field.opacity, f"⑱ 层覆盖度 α (锚点 {len(mxs)})",
         cmap="viridis")
    show(axes[4, 2], field.base, "⑲ 分层底层 base (去遮)")
    sup = LayeredPosterior().suppress(enh, field)
    show(axes[4, 3], sup, "⑳ 遮层抑制后 enh")

    fig.suptitle(f"conger 完整管线: {img_name}", fontsize=14)
    fig.tight_layout()
    out = root / f"artifacts/full_{img_name.rsplit('.', 1)[0]}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
    print(f"总耗时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nat1018.jpg")
