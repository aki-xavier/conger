"""序数约束的真实场景消费验证 (OcclusionOrder 此前只有合成验证)。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_occlusion.py [图片名]
输出: 违序统计 + artifacts/occl_<图片名> (有/无约束深度对照)

口径: 无真值 → 测 (a) 融合前违序条数 (单目线索与 T 结偏序的
分歧量), (b) 投影后违序 = 0 (机制正确性), (c) 修正幅度分布
(约束实际做了多少功), (d) 目检深度图序数骨架是否改善。
"""

import sys

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image

from color import Color
from edgemap import EdgePrior
from fusion import DepthFusionLayer, OcclusionOrder
from grouping import PerceptualGrouping
from monocular import MonocularCues
from riesz import RieszWavelet
from segment import SceneSegmenter, grouping_contours
from utils import Utils
from vbgmm import VBGMM

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main(img_name: str = "12.png") -> None:
    root = Utils.project_root()
    im = Image.open(root / f"images/{img_name}")
    if im.mode != "RGB":
        im = im.convert("RGB")
    rgb = Color.image_to_mlx(im)
    H, W = rgb.shape[:2]

    # 外观链 (与 demo_full 同)
    lum, hs = Color.split_dual_path(rgb)
    rw = RieszWavelet(lum)
    feat = rw.features()
    gm_l = VBGMM.fast_fit(VBGMM.feature_matrix(feat), (H, W), k_max=48,
                          coreset=8192)
    like_l = gm_l.edge_likelihood((H, W))
    tex_l = gm_l.class_likelihood("texture").reshape(H, W)
    gm_h = VBGMM.fast_fit(VBGMM.hs_feature_matrix(hs).reshape(-1, 7),
                          (H, W), k_max=32, coreset=8192)
    like_h = gm_h.edge_likelihood((H, W)) * mx.abs(hs)
    tex_h = gm_h.class_likelihood("texture").reshape(H, W)
    like = 1 - (1 - like_l) * (1 - like_h)
    tex = 1 - (1 - tex_l) * (1 - tex_h)
    enh = EdgePrior().enhance(like, feat, rw)
    res = PerceptualGrouping().run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    sub = seg.subregions
    print(f"{img_name}: 链 {len(res.chains)}, T 结 {len(res.t_junctions)}")

    # 序数约束生成
    occ = OcclusionOrder.constraints_from_grouping(res, sub)
    print(f"有效序数约束: {len(occ)} 条 (映射成功率 "
          f"{100 * len(occ) / max(len(res.t_junctions), 1):.0f}%)")

    # 单目线索 → 有/无约束各融合一遍
    cue = MonocularCues().texture_scale(rw, tex)
    layer = DepthFusionLayer()
    fr_no = layer.run([cue], sub, boundary=enh)
    fr_oc = layer.run([cue], sub, occlusion=occ, boundary=enh)

    # 违序统计: 约束处 前区深度 > 后区深度 = 违序
    def violations(fr) -> list[float]:
        out = []
        s = float(max(H, W))
        for cn in occ:
            u = (cn.pos[1] - W / 2) / s
            v = (cn.pos[0] - H / 2) / s
            ff = fr.fits[cn.front - 1]
            fb = fr.fits[cn.behind - 1]
            if ff.kind != "plane" or fb.kind != "plane":
                continue
            zf = ff.params[0] * u + ff.params[1] * v + ff.params[2]
            zb = fb.params[0] * u + fb.params[1] * v + fb.params[2]
            out.append(zf - zb)  # > 0 = 违序
        return out

    v_no = violations(fr_no)
    v_oc = violations(fr_oc)
    n_viol = sum(1 for x in v_no if x > 0)
    print(f"融合前违序: {n_viol}/{len(v_no)} 条 "
          f"(单目线索与 T 结的分歧率 {100 * n_viol / max(len(v_no), 1):.0f}%)")
    print(f"投影后违序: {sum(1 for x in v_oc if x > 1e-6)}/{len(v_oc)} 条")
    if v_no:
        mag = [abs(x) for x in v_no if x > 0]
        print(f"修正幅度: 中位 {sorted(mag)[len(mag) // 2]:.3f}, "
              f"最大 {max(mag):.3f}" if mag else "无违序无需修正")

    # 面板: 原图 / 无约束深度 / 有约束深度 / 差值 + T 结点位
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(im)
    for t in res.t_junctions:
        axes[0].plot(t.pos[1], t.pos[0], "rx", markersize=5)
    axes[0].set_title(f"输入 + T 结 ×{len(res.t_junctions)}")
    axes[1].imshow(np.array(fr_no.render), cmap="viridis")
    axes[1].set_title("单目深度 (无序数约束)")
    axes[2].imshow(np.array(fr_oc.render), cmap="viridis")
    axes[2].set_title("单目深度 + 序数约束")
    diff = np.array(fr_oc.render - fr_no.render)
    im3 = axes[3].imshow(diff, cmap="RdBu", vmin=-np.abs(diff).max(),
                         vmax=np.abs(diff).max())
    axes[3].set_title("约束修正场 (红=前移 蓝=后移)")
    fig.colorbar(im3, ax=axes[3], fraction=0.03)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = root / f"artifacts/occl_{img_name.rsplit('.', 1)[0]}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "12.png")
