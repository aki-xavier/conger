"""单目深度线索层 (prior.md 可计算子集: 纹理梯度 + T 结序数)。

交付物定位: 粗粒度相对深度 / 区域级序数骨架 (类 2.5D 草图)。
物理上限 (顾问定): riesz 倍频程尺度 → λ̂ 分辨率 ~八度 → 全图只有
3-5 个可分辨深度档; 纹理纵长畸变 (foreshortening, 沿梯度 ∝1/d²)
使 λ̂ 只有单调性没有线性精度。所以本层只生产"弱单调约束"级
线索, 下游零新代码 —— 全部走 fusion.DepthFusionLayer
(EdgeAwareSmooth/PrimitiveFit/OcclusionOrder 复用)。

红线: 相对深度不接 temporal/MotorEKF 度量闭环 —— 尺度模糊会
污染运动估计; 度量深度属于运动视差通道。HS 支路纹理尺度留作
第二步 (L 验证先行)。

模块流程:

  riesz scales (energy × lams) → λ̂ 逐像素 (能量加权)
       │  区域中位数聚合 (robust 于区域内遮挡边)
       ▼  仅纹理区给精度 (tex_like 门控), 平坦区弃权
  DepthCue(z_rel, precision) → DepthFusionLayer.run(...) → 相对深度场
"""

import mlx.core as mx

from fusion import DepthCue
from riesz import RieszWavelet
from utils import Utils


class MonocularCues:
    """单目线索生产者: 纹理梯度 (主) + 视平线高度 (默认关)。"""

    tex_precision: float = 0.5  # 纹理区线索精度 (弱单调约束级,
    # 远低于运动视差/传感器线索 —— 倍频程分辨率上限决定)

    def texture_scale(
        self, rw: RieszWavelet, tex_like: mx.array
    ) -> DepthCue:
        """纹理尺度线索: λ̂ = Σ e_s·λ_s / Σ e_s (逐像素能量加权),
        16×16 块中位数聚合 → z_rel ∝ 1/λ̂ (同一表面世界尺度恒定的
        统计假设)。tex_like 低的区域弃权 (精度≈0, 靠弱先验兜底 ——
        宁可高不确定, 不要自信垃圾)。无循环依赖: 纹理类是谱分类,
        不依赖深度。"""
        h, w = tex_like.shape
        e = mx.stack([s.energy for s in rw.scales], axis=-1)  # (H,W,S)
        lams = mx.array(rw.lams, dtype=mx.float32)
        lam_hat = (e * lams).sum(axis=-1) / mx.maximum(e.sum(axis=-1), 1e-12)
        # 块中位数聚合 (16×16): 比逐像素鲁棒, 又不会像区域中位数
        # 那样被大区域拍平梯度 (实测: 地面合成一个区时区域中位数
        # 把深度梯度全抹掉, Spearman 0.9→−1.0)
        bs = 16
        z_blk = mx.zeros((h, w))
        p_blk = mx.zeros((h, w))
        for r0 in range(0, h, bs):
            for c0 in range(0, w, bs):
                blk_l = lam_hat[r0 : r0 + bs, c0 : c0 + bs]
                blk_t = tex_like[r0 : r0 + bs, c0 : c0 + bs]
                z_blk = z_blk.at[r0 : r0 + bs, c0 : c0 + bs].add(
                    mx.median(blk_l)
                )
                p_blk = p_blk.at[r0 : r0 + bs, c0 : c0 + bs].add(
                    mx.median(blk_t)
                )
        med_global = float(mx.median(lam_hat))
        # z ∝ 1/λ̂ → min-max 归一到 [1,5] (clip 会饱和:
        # med_global/med >2 的区域全被钳到 5, 实测毁掉排序)
        z_rel = med_global / mx.maximum(z_blk, 1e-6)
        vmin = mx.min(z_rel)
        span = mx.maximum(mx.max(z_rel) - vmin, 1e-6)
        z_rel = 1.0 + 4.0 * (z_rel - vmin) / span
        return DepthCue(z_rel, p_blk * self.tex_precision)


if __name__ == "__main__":
    # ── 合成验证: 地面纹理梯度 + 遮挡广告牌 + 平世界弃权 ────────────
    from color import Color
    from edgemap import EdgePrior
    from fusion import DepthFusionLayer, OcclusionOrder
    from grouping import PerceptualGrouping
    from segment import SceneSegmenter, grouping_contours
    from vbgmm import VBGMM

    H, W = 96, 128
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    # 地面: 波长向底部递增 (顶部细纹理 = 远) → z_true ∝ 1/λ(y)
    lam_y = 4.0 + 12.0 * yy / H
    phase = mx.cumsum(2 * mx.pi / lam_y, axis=0)  # 变频正弦的相位积分
    ground = 0.5 + 0.3 * mx.sin(phase)
    # 广告牌: 高亮近平坦矩形 (微光栅), 遮在地面中段 —— 边界强对比,
    # 地面横纹链在其竖边处中断 → 真 T 结
    bb = (yy > 30) & (yy < 70) & (xx > 40) & (xx < 88)
    board_tex = 0.85 + 0.05 * mx.sin(2 * mx.pi * xx / 6.0)
    img = mx.where(bb, board_tex, ground)
    img = img + mx.random.normal((H, W), key=mx.random.key(21)) * 0.01

    # 真实前端全链: riesz → vbgmm → edgemap → grouping → segment
    rw = RieszWavelet(img)
    feat = rw.features()
    gm = VBGMM(VBGMM.feature_matrix(feat), k_max=24)
    like = gm.edge_likelihood((H, W))
    tex = gm.class_likelihood("texture").reshape(H, W)
    enh = EdgePrior().enhance(like, feat, rw)
    res = PerceptualGrouping().run(enh, feat.mean_ori)
    polys, circs = grouping_contours(res)
    seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
    sub = seg.subregions

    # 线索 → 现有融合层 (零新下游代码)。序数部分用真值区域图:
    # 测的是 线索→融合→序数投影 的接线, 不是分割 (分割对近平坦
    # 亮板的归属是另一个问题, 不在这里纠缠)
    cue = MonocularCues().texture_scale(rw, tex)
    sub2 = mx.where(bb, 1, 2).astype(mx.int32)
    from fusion import OrdinalConstraint

    occ = [OrdinalConstraint((50.0, 40.0), front=1, behind=2)]
    fr = DepthFusionLayer().run([cue], sub, boundary=enh)
    dep = fr.render
    fr2 = DepthFusionLayer().run([cue], sub2, occlusion=occ, boundary=enh)
    dep2 = fr2.render

    # 1. 序数骨架: 区域中位深度与真值序 (顶远底近) 的秩相关
    def spearman(a: list[float], b: list[float]) -> float:
        """Spearman 秩相关 (纯 Python, 小样本)。"""
        def rank(v: list[float]) -> list[float]:
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for i, o in enumerate(order):
                rk[o] = float(i)
            return rk
        ra, rb = rank(a), rank(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        va = sum((x - ma) ** 2 for x in ra)
        vb = sum((y - mb) ** 2 for y in rb)
        return cov / max((va * vb) ** 0.5, 1e-12)

    rows = [10, 25, 45, 80, 90]  # 由远及近采样行 (避开广告牌)
    z_est = [float(dep[r, 10]) for r in rows]
    z_true = [float(1.0 / (4.0 + 12.0 * r / H)) for r in rows]
    rho = spearman(z_est, z_true)
    assert rho > 0.8, f"深度序秩相关 {rho:.2f} < 0.8: {z_est}"
    print(f"1. 纹理梯度: 深度序 Spearman={rho:.2f} (估计 "
          f"{[f'{z:.2f}' for z in z_est]}) ✓")

    # 2. 遮挡序数: 广告牌 ≤ 同列背景 (序数投影生效)
    z_bb = float(dep2[50, 64])
    z_bg_top = float(dep2[50, 20])
    assert z_bb <= z_bg_top + 1e-6, (
        f"序数约束后广告牌应在前: {z_bb:.2f} vs {z_bg_top:.2f}"
    )
    print(f"2. 序数约束: 广告牌 {z_bb:.2f} ≤ 地面 {z_bg_top:.2f} ✓")

    # 3. 平世界弃权: 无纹理 → 输出弱先验场而非自信垃圾
    flat = mx.full((H, W), 0.5) + mx.random.normal(
        (H, W), key=mx.random.key(22)) * 0.01
    rw_f = RieszWavelet(flat)
    feat_f = rw_f.features()
    gm_f = VBGMM(VBGMM.feature_matrix(feat_f), k_max=8)
    tex_f = gm_f.class_likelihood("texture").reshape(H, W)
    sub_f = mx.ones((H, W), dtype=mx.int32)
    cue_f = MonocularCues().texture_scale(rw_f, tex_f)
    p_max = float(mx.max(cue_f.precision))
    assert p_max < 0.2, f"平世界应弃权: max precision {p_max:.2f}"
    print(f"3. 平世界弃权: 线索精度上限 {p_max:.3f} ≈ 0 ✓")

    # ── 自然图目检: 12.png 相对深度场 ─────────────────────────────
    from PIL import Image

    im = Image.open(Utils.project_root() / "images/12.png").convert("L")
    arr = Color.image_to_mlx(im)
    Hn, Wn = arr.shape
    rw_n = RieszWavelet(arr)
    feat_n = rw_n.features()
    gm_n = VBGMM(VBGMM.feature_matrix(feat_n), k_max=48)
    like_n = gm_n.edge_likelihood((Hn, Wn))
    tex_n = gm_n.class_likelihood("texture").reshape(Hn, Wn)
    enh_n = EdgePrior().enhance(like_n, feat_n, rw_n)
    res_n = PerceptualGrouping().run(enh_n, feat_n.mean_ori)
    polys_n, circs_n = grouping_contours(res_n)
    seg_n = SceneSegmenter(tau=0.3).run(enh_n, like_n, tex_n, polys_n, circs_n)
    cue_n = MonocularCues().texture_scale(rw_n, tex_n)
    occ_n = OcclusionOrder.constraints_from_grouping(res_n, seg_n.subregions)
    fr_n = DepthFusionLayer().run(
        [cue_n], seg_n.subregions, occlusion=occ_n, boundary=enh_n
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("输入")
    axes[1].imshow(tex_n, cmap="gray")
    axes[1].set_title("纹理似然 (线索精度源)")
    im2 = axes[2].imshow(fr_n.render, cmap="viridis")
    axes[2].set_title("单目相对深度 (纹理梯度+序数)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.axis("off")
    out = Utils.project_root() / "artifacts/monocular_12.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
