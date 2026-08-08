"""本征图像分解 (intrinsic decomposition): I = ρ·(n·L + a)。

prior.md 光学先验的载体: 单一光源 (全局共享 L) + 反射率恒常
(区域反照率 ρ) + 光源上方 (初值)。有了深度 → 法向, 不做盲
Retinex: 几何把光照估计变成交替最小二乘问题。

  模型: I_p = ρ_region(p) · (n_p·L + a)
  交替: 固定 ρ 解 L (全局 3×3 LSQ); 固定 L 解 ρ (逐区域 1D LSQ)
  初值: L = 光源上方先验 (0,−0.7,0.7) 归一
  阴影处理: n·L ≤ 0 的像素退出光照估计 (背光面不参与)

判别应用 (Adelson 棋盘阴影的逻辑): 分解后逐像素反照率
ρ_p = I/(n·L+a) 在区域内应近常数; 区域内 ρ 双峰 = 换漆,
平坦 = 阴影/形状。
"""

from dataclasses import dataclass

import mlx.core as mx


class DecompResult(tuple):
    """分解产物 (L, 反照率图, shading 图) —— 具名元组。"""

    __slots__ = ()

    def __new__(cls, light: mx.array, albedo: mx.array, shading: mx.array):
        return super().__new__(cls, (light, albedo, shading))

    @property
    def light(self) -> mx.array:
        return self[0]

    @property
    def albedo(self) -> mx.array:
        return self[1]

    @property
    def shading(self) -> mx.array:
        return self[2]


@dataclass(slots=True)
class IntrinsicDecomposition:
    """本征分解器: 深度 + 亮度 + 区域 → (光照 L, 反照率, shading)。"""

    iters: int = 8  # 交替轮数
    ambient_q: float = 0.1  # 环境光 = 亮度分布的该分位
    min_lit: float = 0.05  # n·L 低于此视为背光, 退出估计

    def normals(self, depth: mx.array) -> mx.array:
        """深度图 → 单位法向场 (H,W,3): n ∝ (−∂z/∂u, −∂z/∂v, 1),
        像素差分 × s 换算归一化坐标导数 (同 LFA 的量纲教训)。"""
        s = float(max(depth.shape))
        dz_u = mx.zeros_like(depth)
        dz_v = mx.zeros_like(depth)
        dz_u = dz_u.at[:, 1:-1].add((depth[:, 2:] - depth[:, :-2]) * 0.5 * s)
        dz_v = dz_v.at[1:-1, :].add((depth[2:, :] - depth[:-2, :]) * 0.5 * s)
        n = mx.stack([-dz_u, -dz_v, mx.ones_like(depth)], axis=-1)
        return n / mx.maximum(mx.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

    def estimate(
        self, depth: mx.array, img: mx.array, regions: mx.array
    ) -> DecompResult:
        """交替 LSQ 估计全局光照与区域反照率。"""
        n = self.normals(depth)
        flat_i = img.reshape(-1)
        srt = mx.sort(flat_i)
        amb = float(srt[int(self.ambient_q * (flat_i.shape[0] - 1))])
        light = mx.array([0.0, -0.7, 0.7])  # 光源上方先验初值
        light = light / mx.linalg.norm(light)
        lab = regions.reshape(-1)
        n_reg = int(mx.max(regions))
        nf = n.reshape(-1, 3)
        rho = mx.full((n_reg + 1,), 0.5)
        for _ in range(self.iters):
            ndotl = nf @ light  # (N,)
            lit = ndotl > self.min_lit
            # 固定 L → 逐区域 ρ (1D LSQ: ρ = ΣI(n·L) / Σ(n·L)²)
            num = mx.zeros((n_reg + 1,)).at[lab].add(
                flat_i * ndotl * lit
            )
            den = mx.zeros((n_reg + 1,)).at[lab].add(ndotl * ndotl * lit)
            rho = num / mx.maximum(den, 1e-9)
            # 固定 ρ → 全局 L (增广 [n|1] 4×4 LSQ: I/ρ = n·L + c;
            # 截距吸收环境光失配 —— 不显式建模会把常数项逼进 L
            # 的方向, 交替几轮就漂走 (实测 cos 0.95→−0.6))
            rho_pix = mx.maximum(rho[lab], 1e-3)
            y = (flat_i / rho_pix) * lit
            wn = mx.concatenate(
                [nf * lit[:, None], lit[:, None]], axis=-1
            )  # (N,4)
            g = wn.T @ wn
            # 脊随量级: 近平面场景法向趋同 → g 近奇异, 固定 1e-6
            # 在 float32 下保不住正定性 (Eigen LLT 直接 terminate)
            g = g + (1e-6 * float(mx.trace(g)) / 4.0 + 1e-9) * mx.eye(4)
            b = wn.T @ y
            sol = mx.linalg.solve(g, b, stream=mx.cpu)
            light = sol[:3]
            # 前半球约束: 可见面必被前向照明 —— 交替 LSQ 符号自由,
            # 不锁会漂到背面 → lit 全灭 → 下一轮换出零向量 (实测)
            light = mx.where(light[2] < 0, -light, light)
            light = light / mx.maximum(mx.linalg.norm(light), 1e-9)
        ndotl = (nf @ light).reshape(img.shape)
        shading = mx.maximum(ndotl, 0.0) + amb
        albedo = img / mx.maximum(shading, 1e-3)
        return DecompResult(light, albedo, shading)


if __name__ == "__main__":
    H, W = 96, 128
    s = float(W)
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    u = (xx - W / 2) / s
    v = (yy - H / 2) / s

    dec = IntrinsicDecomposition()

    # ── 1. 光照恢复: 光滑曲面 (法向连续张成) + 已知 L → 恢复方向 ──
    # (少数离散倾角在原理上不可辨识: 每区域一个法向, 方程数不够;
    # 曲面法向连续分布才可解)
    l_true = mx.array([0.3, -0.6, 0.74])
    l_true = l_true / mx.linalg.norm(l_true)
    rr1 = mx.maximum(1.2 - u**2 - v**2, 0.05)
    z = 3.0 - 0.8 * mx.sqrt(rr1)  # 朝向相机的球凸包
    n = dec.normals(z)
    ndl = mx.maximum((n.reshape(-1, 3) @ l_true).reshape(H, W), 0.0)
    img1 = 0.6 * (ndl + 0.05)
    sub1 = mx.ones((H, W), dtype=mx.int32)
    r1 = dec.estimate(z, img1, sub1)
    cos_err = float(mx.sum(r1.light * l_true))
    assert cos_err > 0.98, f"L 恢复余弦 {cos_err:.3f}"
    print(f"1. 光照估计: L 方向余弦 {cos_err:.3f} (真值对比) ✓")

    # ── 2. 反射率恒常: 同一场景内 同漆异倾角 → ρ 相等; 异漆 → 比值 ──
    # (必须同一估计: L 归一化使 ρ 绝对值与 |L| 尺度共轭, 跨场景
    # 比绝对值无意义, 同场景内比值才是恒常性检验)
    z2 = mx.where(xx < W // 2, mx.full((H, W), 3.0), 3.0 + 0.8 * v)
    n2 = dec.normals(z2)
    ndl2 = mx.maximum((n2.reshape(-1, 3) @ l_true).reshape(H, W), 0.0)
    albedo_true = mx.where(xx < W // 2, 0.7, 0.7)  # 同漆
    img2 = albedo_true * (ndl2 + 0.05)
    sub2 = mx.where(xx < W // 2, 1, 2).astype(mx.int32)
    r2 = dec.estimate(z2, img2, sub2)
    ra = float(r2.albedo[48, 32])
    rb = float(r2.albedo[48, 96])
    assert abs(ra - rb) / max(ra, 1e-6) < 0.15, (
        f"同漆异倾角反照率应一致: {ra:.2f} vs {rb:.2f}"
    )
    print(f"2. 反射率恒常: 同漆两倾角 ρ={ra:.2f}/{rb:.2f} ✓")

    # ── 3. 换漆 vs 阴影判别: 平坦区内的亮度阶跃 = 换漆 ────────────
    z_flat = mx.full((H, W), 3.0)
    img_paint = mx.where(xx < W // 2, 0.25, 0.75)  # 平坦平面双色漆
    r3 = dec.estimate(z_flat, img_paint, sub1)
    # 平坦法向下 shading 应近常数 → ρ 图保留阶跃 = 判为换漆
    alb = r3.albedo
    step_kept = abs(float(alb[48, 20]) - float(alb[48, 100])) > 0.3
    shade_var = float(mx.std(r3.shading[44:52, 8:120]))
    assert step_kept and shade_var < 0.05, (
        f"换漆阶跃应留在反照率图 (shading 方差 {shade_var:.3f})"
    )
    print(f"3. 换漆/阴影: 阶跃留在 ρ 图, shading 平坦 "
          f"(std={shade_var:.3f}) ✓")
