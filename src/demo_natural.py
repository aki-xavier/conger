"""自然图像全链路可视化 demo: 双通路 (L + HS 复数) 完整管线。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_natural.py [图片名]
输出: artifacts/pipeline_<图片名> (12 面板总图)

覆盖: 输入 → 双通路分解 → riesz → vbgmm(L/HS) → 似然融合 →
edgemap → grouping → segment。fusion/scenegraph/temporal 是
深度/运动层, 单帧静态图无真实深度源, 不在本图 (机制见各自自检)。
"""

import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
from PIL import Image

from color import Color
from edgemap import EdgePrior
from grouping import PerceptualGrouping
from riesz import RieszWavelet
from segment import SceneSegmenter, grouping_contours
from utils import Utils
from vbgmm import VBGMM

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def hs_to_rgb(hs: mx.array) -> mx.array:
    """复数色度 → 色轮可视化 (H=相位, S=幅值, L=0.5)。"""
    h = (mx.arctan2(mx.imag(hs), mx.real(hs)) / (2 * mx.pi)) % 1.0
    s = mx.clip(mx.abs(hs), 0.0, 1.0)
    hsl = mx.stack([h, s, mx.full(hs.shape, 0.5)], axis=-1)
    return Color.hsl_to_rgb(hsl)


def main(img_name: str = "12.png") -> None:
    root = Utils.project_root()
    im = Image.open(root / f"images/{img_name}")
    if im.mode != "RGB":
        im = im.convert("RGB")
    rgb = Color.image_to_mlx(im)
    H, W = rgb.shape[:2]
    print(f"{img_name}: {H}×{W}")

    # ── 双通路分解 ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    lum, hs = Color.split_dual_path(rgb)

    # ── L 支路: riesz → vbgmm ──────────────────────────────────────
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    like_l = gm_l.edge_likelihood((H, W))
    tex_l = gm_l.class_likelihood("texture").reshape(H, W)
    t1 = time.perf_counter()
    print(f"L 支路 (riesz+vbgmm): {t1 - t0:.1f}s")

    # ── HS 支路: 复数 riesz (Re/Im) → vbgmm ─────────────────────────
    x_hs = VBGMM.hs_feature_matrix(hs).reshape(-1, 7)
    gm_h = VBGMM(x_hs, k_max=32)
    like_h = gm_h.edge_likelihood((H, W))
    like_h = like_h * mx.abs(hs)  # 饱和度门控 (灰区色度噪声抑制)
    tex_h = gm_h.class_likelihood("texture").reshape(H, W)
    t2 = time.perf_counter()
    print(f"HS 支路 (双 riesz+vbgmm): {t2 - t1:.1f}s")

    # ── 似然级融合 (概率 OR) ─────────────────────────────────────────
    like = 1 - (1 - like_l) * (1 - like_h)
    tex = 1 - (1 - tex_l) * (1 - tex_h)

    # ── edgemap → grouping → segment ────────────────────────────────
    prior = EdgePrior()
    enh = prior.enhance(like, feat, rw)
    t3 = time.perf_counter()
    print(f"edgemap: {t3 - t2:.1f}s")
    pg = PerceptualGrouping()
    res = pg.run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    t4 = time.perf_counter()
    print(f"grouping: {t4 - t3:.1f}s (链 {len(res.chains)}, "
          f"T {len(res.t_junctions)}, X {len(res.x_junctions)})")
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    t5 = time.perf_counter()
    print(f"segment: {t5 - t4:.1f}s (区域 {int(mx.max(seg.regions))})")

    # ── 12 面板总图 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))

    def show(ax, arr, title, cmap="gray"):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    show(axes[0, 0], rgb, "① 输入 RGB")
    show(axes[0, 1], lum, "② L 亮度通路 (实数)")
    show(axes[0, 2], hs_to_rgb(hs), "③ HS 色度通路 (复数, 色轮编码)")
    show(axes[0, 3], feat.phase_coh, "④ riesz 相位一致性 (L)")

    show(axes[1, 0], like_l, "⑤ L 边缘似然 (vbgmm)")
    show(axes[1, 1], like_h, "⑥ HS 边缘似然 (等亮度边可见)")
    show(axes[1, 2], like, "⑦ 融合边缘似然 (概率 OR)")
    show(axes[1, 3], tex, "⑧ 融合纹理似然")

    show(axes[2, 0], enh, "⑨ edgemap 增强边界 (Pb)")
    # ⑩ grouping: 链 + T/X 结
    ax = axes[2, 1]
    ax.imshow(enh, cmap="gray", alpha=0.35)
    for i, ch in enumerate(res.chains):
        p = res.edgels.pos[ch]
        ax.plot(p[:, 1], p[:, 0], "-", linewidth=0.8,
                color=plt.cm.tab20(i % 20))
    for t in res.t_junctions:
        ax.plot(t.pos[1], t.pos[0], "rx", markersize=8, markeredgewidth=2)
    for x in res.x_junctions:
        ax.plot(x.pos[1], x.pos[0], "bo", markersize=8, fillstyle="none",
                markeredgewidth=2)
    ax.set_title(f"⑩ grouping 链 ×{len(res.chains)} "
                 f"T结 ×{len(res.t_junctions)}(红) X结 ×{len(res.x_junctions)}(蓝)",
                 fontsize=11)
    ax.axis("off")
    show(axes[2, 2], seg.regions.astype(mx.float32), "⑪ segment 区域 (τ=0.3)",
         cmap="tab20")
    show(axes[2, 3], seg.ucm, "⑫ UCM 超度量等高线")

    fig.suptitle(
        f"conger 全链路: {img_name} | fusion/scenegraph/temporal 为深度/运动层, "
        f"单帧静态图无深度源, 见各模块自检",
        fontsize=13,
    )
    fig.tight_layout()
    out = root / f"artifacts/pipeline_{img_name.rsplit('.', 1)[0]}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
    print(f"总耗时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "12.png")
