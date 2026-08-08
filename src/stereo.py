"""双目视差线索层: 与 monocular 并行可选的深度源。

物理 (决定精度场, 无需显式门控):
    z = B·f / d  →  σ_z = z²·σ_d / (B·f)  →  precision ∝ d⁴
视差大 (近) 双目精度主导, 视差小 (远) 精度坍缩 → 单目线索接管 —
— 门控由 CueFusion 逆方差加权自动涌现 (用户设计要求, 2026-08-08)。

定位: 线索生产者 (同 monocular 的架构纪律), 下游零新代码。
输入为已校正立体对 (行对齐), 只估水平视差。匹配器是最小
SAD + 左右一致性 (遮挡/弱纹理区弃权, 精度置 0)。
"""

from dataclasses import dataclass

import mlx.core as mx

from fusion import DepthCue


@dataclass(slots=True)
class StereoCues:
    """已校正立体对 → 视差/深度线索。"""

    max_disp: int = 32  # 视差搜索范围 (px)
    win: int = 2  # SAD 窗口半径 (5×5)
    lr_tol: float = 1.0  # 左右一致性容差 (px)
    sigma_d: float = 0.3  # 视差噪声 (亚像素, px)

    def run(
        self, left: mx.array, right: mx.array, bf: float
    ) -> DepthCue:
        """左/右灰度图 + 基线焦距积 B·f → DepthCue(z, precision)。
        precision ∝ d⁴ (σ_d 固定 ⇒ σ_z = z²σ_d/Bf)。无效点 (LR
        不一致 / 弱纹理) 精度 0 → 融合时自动让位其他线索。"""
        h, w = left.shape
        # SAD 代价体 (H,W,D): 逐视差移位 + 窗口聚合, 一次堆叠
        costs = []
        for d in range(self.max_disp):
            shifted = mx.pad(right, [(0, 0), (d, 0)])[:, : w]
            diff = mx.abs(left - shifted)
            cost = mx.zeros((h, w))
            for dy in range(-self.win, self.win + 1):
                for dx in range(-self.win, self.win + 1):
                    cost = cost + mx.roll(
                        mx.roll(diff, dy, axis=0), dx, axis=1
                    )
            costs.append(cost)
        vol = mx.stack(costs, axis=-1)  # (H,W,D)
        best_d = mx.argmin(vol, axis=-1).astype(mx.float32)
        # 抛物线亚像素: d* = d + (c₋−c₊) / 2(c₋−2c₀+c₊)
        # (远场小视差处的整数量化是主误差, d=8 错 1px 即 z 偏 0.7)
        di = best_d.astype(mx.int32)
        c0 = mx.take_along_axis(vol, di[..., None], axis=-1)[..., 0]
        cm = mx.take_along_axis(
            vol, mx.clip(di - 1, 0, self.max_disp - 1)[..., None], axis=-1
        )[..., 0]
        cp = mx.take_along_axis(
            vol, mx.clip(di + 1, 0, self.max_disp - 1)[..., None], axis=-1
        )[..., 0]
        denom = mx.maximum(cm - 2 * c0 + cp, 1e-6)
        best_d = mx.clip(best_d + 0.5 * (cm - cp) / denom, 1e-3, None)
        # 左右一致性: 用 −best_d 在右图上重采样左图, 残差大 = 误配
        # (ponytail: 全检双向匹配成本高, 用前向重投影残差近似)
        dmap = best_d.astype(mx.int32)
        cols = mx.clip(
            mx.arange(w)[None, :] - dmap, 0, w - 1
        )
        reproj = mx.take_along_axis(right, cols, axis=1)
        resid = mx.abs(left - reproj)
        valid = (resid < 0.05) & (best_d > 0)
        # 深度与精度场
        z = bf / mx.maximum(best_d, 1e-6)
        prec = mx.where(
            valid, (best_d**4) / (bf**2 * self.sigma_d**2), 0.0
        )
        # 归一化精度到 O(1) 量级 (与其他线索可比较)
        p99 = mx.sort(prec.reshape(-1))[int(0.99 * (h * w - 1))]
        prec = prec / mx.maximum(p99, 1e-12)
        return DepthCue(z, prec)


if __name__ == "__main__":
    from fusion import CueFusion, DepthCue

    H, W = 96, 160
    BF = 40.0  # B·f
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )

    # ── 合成立体对: 近板 (z=2 → d=20) 遮远墙 (z=5 → d=8), 双侧纹理 ──
    z_true = mx.where((yy > 20) & (yy < 76) & (xx > 40) & (xx < 110), 2.0, 5.0)
    # 非周期纹理 (周期纹理会在整数倍周期处混淆匹配, 实测
    # sin(x/3) 周期 18.8px 把 d=20 匹配到 d=18)
    noise = mx.random.normal((H, W + 80), key=mx.random.key(31))
    tex = mx.zeros((H, W + 80))
    for dy in range(3):
        for dx in range(3):
            tex = tex + mx.roll(mx.roll(noise, dy, axis=0), dx, axis=1)
    tex = 0.5 + 0.25 * tex / mx.max(mx.abs(tex))
    disp = BF / z_true  # 视差场
    # 宽纹理场, 左图取 [40:40+W]; 右相机右移 → 同名点在右图偏左
    # d px: right[u] = left[u+d] (符号约定, 反了匹配器搜不到真峰)
    left = tex[:, 40 : 40 + W]
    cols_r = mx.clip(40 + xx + disp, 0, W + 79).astype(mx.int32)
    right = mx.take_along_axis(tex, cols_r, axis=1)

    cue = StereoCues(max_disp=32).run(left, right, BF)
    z_near = float(cue.mean[48, 75])
    z_far = float(cue.mean[10, 10])
    assert abs(z_near - 2.0) < 0.15, f"近板深度: {z_near:.2f}"
    assert abs(z_far - 5.0) < 0.5, f"远墙深度: {z_far:.2f}"
    p_near = float(cue.precision[48, 75])
    p_far = float(cue.precision[10, 10])
    assert p_near > 10 * p_far, (
        f"精度场应悬殊 (d⁴律): 近 {p_near:.2f} vs 远 {p_far:.3f}"
    )
    print(f"1. 立体对: 近板 {z_near:.2f}(真2) 远墙 {z_far:.2f}(真5), "
          f"精度比 {p_near / max(p_far, 1e-9):.0f}× ✓")

    # ── 门控涌现: 近处跟双目 (错单目被压), 远处跟单目 ─────────────
    mono_bias = mx.full((H, W), 3.0)  # 单目假设全错 (bias 到 3)
    mono_prec = mx.full((H, W), 0.05)  # 单目弱精度 (弱单调约束级)
    d_f, p_f = CueFusion.run([cue, DepthCue(mono_bias, mono_prec)])
    near_follows_stereo = abs(float(d_f[48, 75]) - 2.0) < 0.3
    far_follows_mono = abs(float(d_f[10, 10]) - 3.0) < abs(
        float(d_f[10, 10]) - 5.0
    )
    assert near_follows_stereo, f"近处应跟双目: {float(d_f[48, 75]):.2f}"
    assert far_follows_mono, f"远处应跟单目: {float(d_f[10, 10]):.2f}"
    print(f"2. 门控涌现: 近处 {float(d_f[48, 75]):.2f}≈双目2.0, "
          f"远处 {float(d_f[10, 10]):.2f} 偏向单目3.0 ✓")
