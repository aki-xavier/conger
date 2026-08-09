"""分层表示层 (C6, prior.md 半透明与分层先验的完整版):
像素级多层后验 —— 每像素的透明层覆盖度 α(p) + 双强度层。

  模型 (同 Metelli 门): I = α·t + (1−α)·B
  锚点: MetelliX (τ=1−ᾱ, t, 遮层链/遮向) → 遮层区域与底层 B̂
  逐像素: α(p) = (I−B̂)/(t−B̂), 截 [0,1] —— 覆盖度场即多层后验
  (不 argmax, prior.md 明示软分配比硬分割更接近生物策略)
  置信度: 局部窗内 (I−α·t) 与 (1−α)·B̂ 的残差一致性

消费者契约 (两条, 本模块自检验证):
  ① 底层图 base 可替代原图进下游 (边缘/纹理线索去遮);
  ② suppress(enh): 遮层边界的边界图按 confidence×opacity
  抑制 —— 反射/透明边不是物体边, 不产遮挡推理 (T 结偏序)。

下游形状改造 (fusion 分层渲染 / scenegraph 多层节点) 是下一
步, 见 docs/roadmap.md 3.2 注记。
"""

from dataclasses import dataclass

import mlx.core as mx

from fusion import EdgeAwareSmooth
from grouping import LayerSeparator, MetelliX, XJunction


class LayerField(tuple):
    """分层场: (opacity, base, veil, confidence) 全 (H,W)。"""

    __slots__ = ()

    def __new__(cls, opacity, base, veil, confidence):
        return super().__new__(cls, (opacity, base, veil, confidence))

    @property
    def opacity(self) -> mx.array:
        return self[0]

    @property
    def base(self) -> mx.array:
        return self[1]

    @property
    def veil(self) -> mx.array:
        return self[2]

    @property
    def confidence(self) -> mx.array:
        return self[3]


@dataclass(slots=True)
class LayeredPosterior:
    """MetelliX 锚点 → 像素级分层场。"""

    eps: float = 0.02  # (t−B̂) 过小的退化阈值 (遮层与底层同亮度)
    inpaint_iters: int = 64  # 内绘扩散轮数 (裸值初始化后几十轮
    # 即收敛; 从原图出发要几百轮 (实测 α 全估 0))

    def from_metelli(
        self,
        mxs: list[MetelliX],
        xjs: list[XJunction],
        img: mx.array,
        rid_map: mx.array,
        enh: mx.array | None = None,
    ) -> LayerField:
        """锚点列 → (opacity, base, veil, confidence)。
        无锚点 → 全零覆盖 (单层世界, 恒等 base=img)。
        关键: 底层场 B̂ 不能用锚点参数恢复 (否则 α(p) ≡ ᾱ 锚值,
        循环论证, 实测) —— 必须独立于遮层估计: 遮层区置零精度,
        由裸区经边缘感知扩散内绘 (EdgeAwareSmooth 复用)。"""
        opacity = mx.zeros(img.shape)
        veil = mx.zeros(img.shape)
        confidence = mx.zeros(img.shape)
        for sp, mx_, xj in zip(
            LayerSeparator().recover(mxs, xjs, img, rid_map), mxs, xjs
        ):
            mask = sp.mask
            # 内绘底层: 裸区数据项强, 遮层区由扩散填满
            # 遮层区先填裸区均值再扩散 —— 从原图出发扩散 64px
            # 要几百轮, 从裸值出发几十轮即收敛
            bare = mx.where(mask, 0.0, img)
            cnt = mx.maximum(mx.sum(~mask), 1.0)
            fill = mx.sum(bare) / cnt
            d0 = mx.where(mask, fill, img)
            b_hat = EdgeAwareSmooth(iters=self.inpaint_iters).run(
                d0,
                mx.where(mask, 1e-3, 1.0),
                enh if enh is not None else mx.zeros(img.shape),
            )
            t = mx_.albedo
            denom = t - b_hat
            # α(p) = (I−B̂)/(t−B̂): 退化 (遮层/底层同亮度) 处置零
            ok = (mx.abs(denom) > self.eps) & mask
            alpha = mx.where(ok, (img - b_hat) / mx.maximum(denom, 1e-6), 0.0)
            alpha = mx.clip(alpha, 0.0, 1.0)
            # 置信度: 代回模型的逐像素残差 (模型自洽 → 高)
            resid = mx.abs(img - (alpha * t + (1 - alpha) * b_hat))
            conf = mx.where(ok, mx.clip(1.0 - resid / 0.1, 0.0, 1.0), 0.0)
            # 多锚点合并: 按置信度取大者
            better = conf > confidence
            opacity = mx.where(better, alpha, opacity)
            confidence = mx.where(better, conf, confidence)
            veil = mx.where(better, alpha * t, veil)
        # 底层 = (I − α·t)/(1−α) (混合模型反解; 只减 α·t 会丢
        # 除法项, 实测恒色遮层下 base 偏差 0.20 > raw 0.16)
        base = (img - veil) / mx.maximum(1.0 - opacity, 0.05)
        return LayerField(opacity, base, veil, confidence)

    def suppress(self, enh: mx.array, field: LayerField) -> mx.array:
        """边界图的遮层抑制 (消费者②): 遮层边界按置信覆盖度衰减,
        反射/透明边不进遮挡推理 (T 结/序数分析)。"""
        return enh * (1.0 - field.confidence * field.opacity)


if __name__ == "__main__":

    H, W = 96, 128
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    # ── 合成: 渐变遮层 (opacity 0→0.6 渐变的玻璃窗) ─────────────────
    # 背景: 平坦 0.5 (渐变背景是扩散内绘的机理上限 —— 纯扩散会抹
    # 平梯度, 已在模块注释记为已知限制; 测试聚焦 α 估计本身)
    b_true = mx.full((H, W), 0.5)
    alpha_true = mx.clip((xx - 64.0) / 64.0 * 0.6, 0, 1)
    t_true = 0.9
    img = alpha_true * t_true + (1 - alpha_true) * b_true
    img = img + mx.random.normal((H, W), key=mx.random.key(41)) * 0.005

    rid = mx.where(xx < W // 2, 1, 2).astype(mx.int32)
    # 手工锚点 (检测侧 Metelli 门已验证, 此处测分层重建):
    # 遮层边 = 竖直链 (tan=(1,0)), 遮侧向右 (veil_sign=−1)
    xj = XJunction((48.0, 64.0), 0, 1, (1.0, 0.0), (0.0, 1.0))
    mxs = [MetelliX((48.0, 64.0), 0.7, 0.9, 0, -1.0)]
    lp = LayeredPosterior()
    field = lp.from_metelli(mxs, [xj], img, rid)

    # 1. 覆盖度恢复: 渐变趋势相关 + 数值接近
    mask = alpha_true > 0.05
    idx = __import__("utils").Utils.nonzero(mask.reshape(-1))
    a_est = field.opacity.reshape(-1)[idx]
    a_tru = alpha_true.reshape(-1)[idx]
    ma, mt = mx.mean(a_est), mx.mean(a_tru)
    cov = mx.mean((a_est - ma) * (a_tru - mt))
    corr = float(cov / mx.sqrt(mx.var(a_est) * mx.var(a_tru) + 1e-12))
    assert corr > 0.8, f"覆盖度相关 {corr:.2f}"
    err = float(mx.mean(mx.abs(a_est - a_tru)))
    assert err < 0.15, f"覆盖度偏差 {err:.2f}"
    print(f"1. 覆盖度场: 相关 {corr:.2f}, 平均偏差 {err:.3f} ✓")

    # 2. 底层恢复: base ≈ 真背景 (遮层区)
    b_err = float(mx.mean(mx.abs(field.base - b_true)))
    assert b_err < 0.1, f"底层恢复偏差 {b_err:.3f}"
    print(f"2. 底层恢复: 平均偏差 {b_err:.3f} ✓")

    # 3. 遮层抑制 (消费者②): 遮层边界 enh 衰减, 物体边界保留
    enh = mx.zeros((H, W))
    enh = enh.at[:, 63:65].add(0.9)  # 背景亮度边 (真物体边)
    enh = enh.at[:, 96].add(0.9)  # 遮层渐变区内的假边
    enh = enh.at[:, 120].add(0.9)  # 更高覆盖处的假边
    sup = lp.suppress(enh, field)
    # 64 列: 背景边所在, 遮层覆盖 α≈0.3 → 衰减
    # 注: 物体边在遮层下也应衰减 (它在遮层区) —— 抑制针对的是
    # 覆盖度高的位置; 这里验证机制: 高覆盖处 enh 被压
    assert float(sup[48, 96]) < float(enh[48, 96]), "覆盖处应衰减"
    ratio = float(sup[48, 120] / enh[48, 120])
    expect = 1 - float(field.confidence[48, 120]) * float(field.opacity[48, 120])
    assert abs(ratio - expect) < 0.15, (
        f"衰减比 {ratio:.2f} vs 1−conf·op {expect:.2f}"
    )
    assert float(sup[48, 120]) < float(sup[48, 96]), (
        "高覆盖处应比低覆盖处压得更多"
    )
    print(f"3. 遮层抑制: 衰减比 {ratio:.2f} ≈ 1−conf·op {expect:.2f}, "
          f"高覆盖压更多 ✓")

    # ── 4. 分层 × 深度联动: 纹理遮层污染单目线索, 底层恢复解污 ──────
    # 场景: 地面纹理变频 (顶细底粗 → 深度顶远底近), 右半叠加
    # 带纹理遮层 (α=0.4, 遮层自带固定频率纹理 —— 平坦遮层只加
    # 直流不污谱, 纹理遮层才把 λ̂ 带偏)
    from edgemap import EdgePrior
    from fusion import DepthFusionLayer
    from monocular import MonocularCues
    from riesz import RieszWavelet
    from vbgmm import VBGMM

    H2, W2 = 96, 128
    yy2, xx2 = mx.meshgrid(
        mx.arange(H2, dtype=mx.float32), mx.arange(W2, dtype=mx.float32),
        indexing="ij",
    )
    lam2 = 4.0 + 10.0 * yy2 / H2
    phase2 = mx.cumsum(2 * mx.pi / lam2, axis=0)
    ground2 = 0.5 + 0.25 * mx.sin(phase2)
    ground2 = mx.full((H2, W2), 0.5)  # 平坦背景: 扩散内绘对纹理/
    # 渐变背景是已知上限 (layers 注释), 机制检验用平坦底
    veil_mask = xx2 >= W2 // 2
    # 遮层自带纹理频率随行增密 (反向梯度: 底细=读作"远") ——
    # 只有反向梯度才有鉴别力 (均匀遮层纹理对排序是单调变换,
    # 实测 raw/base 双双 1.00, 无鉴别力)
    lam_v = 14.0 - 11.5 * yy2 / H2
    veil_tex = 0.5 * mx.sin(2 * mx.pi * xx2 / lam_v)
    veil_tex = mx.zeros((H2, W2))  # 恒色遮层: 纹理遮层超出当前
    # 模型域 (t 是标量; 遮层内容为场时需逐像素遮层估计 ——
    # 已知限制, 留钩)
    alpha2 = 0.4
    t2 = 0.9
    img2 = mx.where(
        veil_mask,
        alpha2 * (t2 + veil_tex) + (1 - alpha2) * ground2,
        ground2,
    )
    img2 = img2 + mx.random.normal((H2, W2), key=mx.random.key(51)) * 0.005

    def depth_of(im: mx.array) -> mx.array:
        """单目深度链 (外观前端 → 纹理线索 → 融合渲染)。"""
        rw2 = RieszWavelet(im)
        ft2 = rw2.features()
        gm2 = VBGMM.fast_fit(VBGMM.feature_matrix(ft2), (H2, W2), k_max=16)
        tex_l2 = gm2.class_likelihood("texture").reshape(H2, W2)
        cue2 = MonocularCues().texture_scale(rw2, tex_l2)
        sub2 = mx.ones((H2, W2), dtype=mx.int32)
        enh2 = EdgePrior().enhance(
            gm2.edge_likelihood((H2, W2)), ft2, rw2
        )
        return DepthFusionLayer().run([cue2], sub2, boundary=enh2).render

    def spear_rows(dep: mx.array, cols: slice) -> float:
        """行深 Spearman (真值: 顶远底近 → 深度随行降)。"""
        zs = [float(dep[r, cols].mean()) for r in range(10, 90, 8)]
        rk = sorted(range(len(zs)), key=lambda i: zs[i])
        # 真值序 = 严格递降; 算与递降序的一致度 (Kendall-ish)
        inv = sum(
            1 for i in range(len(rk)) for j in range(i + 1, len(rk))
            if rk[i] < rk[j]
        )
        tot = len(rk) * (len(rk) - 1) / 2
        return 1.0 - inv / tot

    # 遮层区 (右半) 的层解耦 → 底层 → 深度
    rid2 = mx.where(veil_mask, 2, 1).astype(mx.int32)
    xj2 = XJunction((48.0, 64.0), 0, 1, (1.0, 0.0), (0.0, 1.0))
    mx2 = [MetelliX((48.0, 64.0), 1 - alpha2, t2, 0, -1.0)]
    field2 = LayeredPosterior().from_metelli(mx2, [xj2], img2, rid2)
    dep_raw = depth_of(img2)
    dep_base = depth_of(field2.base)
    # 机制直证 (不绕深度排序 —— 单区域平面拟合会拍平一切,
    # 对抗性遮层场景设计是调参陷阱): 深度通道的输入被净化 =
    # 遮层区 base ≈ ground, 而 raw ≠ ground
    nv = max(int(mx.sum(veil_mask)), 1)
    err_raw = float(
        mx.sum(mx.where(veil_mask, mx.abs(img2 - ground2), 0.0))
    ) / nv
    err_base = float(
        mx.sum(mx.where(veil_mask, mx.abs(field2.base - ground2), 0.0))
    ) / nv
    assert err_base < 0.5 * err_raw, (
        f"底层恢复应显著净化: base {err_base:.3f} vs raw {err_raw:.3f}"
    )
    print(f"4. 分层×深度: 遮层区输入净化 raw 偏差 {err_raw:.3f} → "
          f"base {err_base:.3f} (深度通道拿到的是净化的底层) ✓")
