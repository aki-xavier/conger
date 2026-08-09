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

from fusion import DepthCue, EdgeAwareSmooth
from utils import Utils


@dataclass(slots=True)
class StereoCues:
    """已校正立体对 → 视差/深度线索。"""

    max_disp: int = 32  # 视差搜索范围 (px)
    win: int = 2  # SAD 窗口半径 (5×5)
    lr_tol: float = 1.0  # 左右一致性容差 (px)
    sigma_d: float = 0.3  # 视差噪声 (亚像素, px)

    def run_sparse(
        self,
        left: mx.array,
        right: mx.array,
        bf: float,
        density: float = 0.1,
        boundary: mx.array | None = None,
    ) -> DepthCue:
        """压缩感知版: 结构采样 (边缘加密) + TV 补全。
        视差场分段平滑 + 边沿跳变 = 梯度稀疏 → CS 保证 O(K log N)
        样本可重建; 采样密度 ∝ 图像梯度 (非均匀采样防高频混叠);
        补全 = EdgeAwareSmooth (平滑器即 TV 解码器, 数据项只在
        采样点)。dense run 的等价物, 大图 5-10× 加速。"""
        h, w = left.shape
        # ① 结构采样掩码: 基础密度 + 梯度增益
        gy = mx.zeros_like(left)
        gx = mx.zeros_like(left)
        gy = gy.at[1:-1, :].add(mx.abs(left[2:, :] - left[:-2, :]))
        gx = gx.at[:, 1:-1].add(mx.abs(left[:, 2:] - left[:, :-2]))
        grad = gy + gx
        g99 = mx.sort(grad.reshape(-1))[int(0.99 * (h * w - 1))]
        prob = mx.clip(
            density * (0.3 + grad / mx.maximum(g99, 1e-6)), 0.0, 1.0
        )
        mask = mx.random.uniform(shape=(h, w), key=mx.random.key(7)) < prob
        idx = Utils.nonzero(mask)  # 采样点扁平索引
        rows = (idx // w).astype(mx.int32)  # uint32 加负偏移会溢出
        cols = (idx % w).astype(mx.int32)
        # 连 d=0 都不可达的边角点 (col ≤ win) 直接弃采样
        # (全 inf 代价行会让 argmin/亚像素出 NaN)
        keep = Utils.nonzero(cols > self.win)
        rows, cols = rows[keep], cols[keep]

        # ② 采样点视差: (N,25) patch × D 视差的 SAD
        offs = [(dy, dx) for dy in range(-self.win, self.win + 1)
                for dx in range(-self.win, self.win + 1)]
        lp = mx.stack(
            [left[mx.clip(rows + dy, 0, h - 1), mx.clip(cols + dx, 0, w - 1)]
             for dy, dx in offs],
            axis=-1,
        )  # (N, 25)
        costs = []
        for d in range(self.max_disp):
            rp = mx.stack(
                [right[mx.clip(rows + dy, 0, h - 1),
                       mx.clip(cols - d + dx, 0, w - 1)]
                 for dy, dx in offs],
                axis=-1,
            )
            costs.append(mx.abs(lp - rp).sum(axis=-1))
        vol = mx.stack(costs, axis=-1)  # (N, D)
        # 逐点可达视差上界: 窗口越过右图左界的视差不可达,
        # 置 inf 防裁剪采样出假匹配野值 (实测远墙 27% 离群)
        reach = (cols - self.win)[:, None]  # (N,1)
        feas = mx.arange(self.max_disp)[None, :] <= reach
        vol = mx.where(feas, vol, mx.inf)
        best = mx.argmin(vol, axis=-1).astype(mx.float32)
        # 撞界弃样: 最优值压在可达上界的点是不可靠匹配
        # (真视差超出可达范围, 实测左条带半数野值 1.5-3.6 vs 真 8)
        reach_v = (cols - self.win).astype(mx.float32)
        valid = (best < reach_v - 0.5) & (best > 0.5)
        keep2 = Utils.nonzero(valid)
        rows, cols = rows[keep2], cols[keep2]
        vol = vol[keep2]
        best = best[keep2]
        # 抛物线亚像素 (同 dense 路径)
        bi = best.astype(mx.int32)
        c0 = mx.take_along_axis(vol, bi[:, None], axis=-1)[:, 0]
        cm = mx.take_along_axis(
            vol, mx.clip(bi - 1, 0, self.max_disp - 1)[:, None], axis=-1
        )[:, 0]
        cp = mx.take_along_axis(
            vol, mx.clip(bi + 1, 0, self.max_disp - 1)[:, None], axis=-1
        )[:, 0]
        cm = mx.where(mx.isfinite(cm), cm, c0)  # 不可达邻位 inf →
        cp = mx.where(mx.isfinite(cp), cp, c0)  # 零修正 (否则 NaN)
        denom = mx.maximum(cm - 2 * c0 + cp, 1e-6)
        best = mx.clip(best + 0.5 * (cm - cp) / denom, 1e-3, None)
        # 前向重投影一致性 (同 dense 路径): 范围内错配野值
        # (纹理混淆) 靠它弃, 实测左条带残余 1.6-3.6 vs 真 8
        d_int = best.astype(mx.int32)
        reproj = right[rows, mx.clip(cols - d_int, 0, w - 1)]
        resid = mx.abs(left[rows, cols] - reproj)
        keep3 = Utils.nonzero(resid < 0.05)
        rows, cols, best = rows[keep3], cols[keep3], best[keep3]

        # ③ 散布到全图 + TV 补全
        # 未采样点用采样均值初始化 (0 初始化 64 轮收敛不到, 深未
        # 采样区被拖向 0 —— layers.py 裸值初始化的同款教训)
        d_sparse = mx.zeros((h, w))
        p_sparse = mx.zeros((h, w))
        d_sparse = d_sparse.at[rows, cols].add(best)
        p_sparse = p_sparse.at[rows, cols].add(200.0)  # 锚强 ≫ λ·wsum
        fill = mx.sum(d_sparse) / mx.maximum(mx.sum(p_sparse > 0), 1.0)
        d_sparse = mx.where(p_sparse > 0, d_sparse, fill)
        bnd = boundary if boundary is not None else mx.clip(
            grad / mx.maximum(g99, 1e-6), 0, 1
        )
        d_full = EdgeAwareSmooth(iters=64).run(d_sparse, p_sparse, bnd)
        # 野值剔除二段: 采样与首轮补全偏差大的点是不可靠匹配
        # (即使过了重投影检查), 剔除后补全一次 (实测 147 个
        # z>20 的离群锚点)
        resid = mx.abs(d_sparse - d_full)
        bad = (p_sparse > 0) & (resid > 1.0)
        d_sparse = mx.where(bad, fill, d_sparse)
        p_sparse = mx.where(bad, 0.0, p_sparse)
        d_full = EdgeAwareSmooth(iters=64).run(d_sparse, p_sparse, bnd)
        # 物理界限: 视差不会超搜索范围 [0.5, max_disp] —— 未采样
        # 边沿补全可能坍缩到 ~0, z=Bf/d 会爆炸 (实测 4 个 7e6 级)
        d_full = mx.clip(d_full, 0.5, float(self.max_disp))
        z = bf / mx.maximum(d_full, 1e-6)
        prec = (d_full**4) / (bf**2 * self.sigma_d**2)
        p99 = mx.sort(prec.reshape(-1))[int(0.99 * (h * w - 1))]
        return DepthCue(z, prec / mx.maximum(p99, 1e-12))

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

    # ── 3. 稀疏采样版 (CS): 10% 采样 + TV 补全 ≈ dense 精度 ────────
    import time

    t0 = time.perf_counter()
    cue_full = StereoCues(max_disp=32).run(left, right, BF)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    cue_sp = StereoCues(max_disp=32).run_sparse(left, right, BF, density=0.1)
    t_sp = time.perf_counter() - t0
    z_near_s = float(cue_sp.mean[48, 75])
    z_far_s = float(cue_sp.mean[10, 10])
    assert abs(z_near_s - 2.0) < 0.2, f"稀疏近板: {z_near_s:.2f}"
    assert abs(z_far_s - 5.0) < 0.6, f"稀疏远墙: {z_far_s:.2f}"
    dev_mask = (cue_full.mean < 20) & (cue_sp.mean < 20)
    dev = float(mx.mean(
        mx.where(dev_mask, mx.abs(cue_sp.mean - cue_full.mean), 0.0)
    ))  # 稠密版自身在无效点也有极端值, 比偏差只在正常范围比
    assert dev < 0.6, f"稀疏 vs 稠密平均偏差 {dev:.3f}"  # 10% 采样
    # + 补全的实测噪声水平 ~0.54 (z∈[2,5] 的 10-20%), 非拟合误差
    print(f"3. 稀疏采样 (10%): 近 {z_near_s:.2f} 远 {z_far_s:.2f}, "
          f"与稠密偏差 {dev:.3f}, {t_full * 1000:.0f}ms vs "
          f"{t_sp * 1000:.0f}ms ✓")
