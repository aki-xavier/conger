"""深度融合层 (Depth Fusion, flow.md §5): 线索融合 → 区域提升 → 深度反馈。

模块流程:

  线索 (DepthCue: mean + precision 稠密场, 契约见下)
       │  CueFusion: 高斯乘积 (逆方差加权) —— 共轭闭式
       ▼  稠密深度后验 (mean, precision)
  PrimitiveFit: 以分割子区域为拟合单元, 加权 LSQ 提升为
       平面/球图元 (Kasa), 残差选模, 协方差 = (AᵀWA)⁻¹
       正规方程充分统计量按区域 scatter-add, (R,3,3)/(R,4,4) 批量
       求逆 —— 无逐区域 Python/MLX 同步 (grouping 的教训)
       ▼  → 图元 blade (cga Plane/Sphere) + Σ + 残差稠密场
  反馈: 渲染图元深度 → 不连续图 D → SceneSegmenter(prior_map=D)
       (§2 迭代协议的虚线边, 本层是 prior_map 的生产者)

  [留钩] ManhattanCoupling: 法向直方图聚类 + 正交约束精化, 未实现
  [留钩] 分层表示 (§5.3): 多层软分配/透明分裂, 判据依赖缺口 C6, 未实现

**架构声明**: 管线当前无真实深度来源 (无双目/传感器/运动视差)。
本层交付的是机制与契约 —— 第一个真实线索按设计是 §6 的运动视差
(未实现)。自检全部用合成地面真值 + 合成高斯线索。

坐标约定: 图像 (row, col) → 3D (x=col, y=row, z=深度), 与 grouping
的 point(c, r, 0) 一致; 拟合在各向同性归一化坐标 (减中心÷边长) 下
进行保条件数, 输出 blade 前换回像素单位。离线层, 不做逐帧承诺。
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from cga import Multivector, Plane, Sphere


class DepthCue(NamedTuple):
    """深度线索契约: 逐像素高斯读数 (均值场 + 精度场)。"""

    mean: mx.array  # (H,W) 深度均值
    precision: mx.array  # (H,W) 精度 (逆方差), 越大越可信


class CueFusion:
    """贝叶斯线索整合: p(d|c₁..c_M) ∝ p(d)∏p(c_i|d)^w_i 的高斯闭式。"""

    @staticmethod
    def run(
        cues: list[DepthCue], prior_precision: float = 1e-3
    ) -> tuple[mx.array, mx.array]:
        """精度加权融合 → (均值, 精度)。prior 为弱全局精度 (分段平滑
        的职责由图元化接管, 见 flow.md §5.2), 遮挡偏序不进表决。"""
        p = mx.full(cues[0].mean.shape, prior_precision)
        d = mx.zeros(cues[0].mean.shape)
        for c in cues:
            p = p + c.precision
            d = d + c.precision * c.mean
        return d / p, p


# ── 区域提升: 加权 LSQ 图元化 ─────────────────────────────────────


class PrimFit(NamedTuple):
    """单区域拟合产物: blade + 类型 + 参数协方差 + 渲染参数 + 残差。"""

    blade: Multivector | None  # Plane/Sphere blade; None = 退化, 留稠密场
    kind: str  # "plane" / "sphere" / "dense"
    cov: mx.array | None  # (P,P) 参数协方差
    params: tuple[float, ...]  # 平面 (a,b,c) / 球 (cu,cv,cz,ρ) (归一化坐标)
    sign: float  # 球渲染的半球符号 (平面恒 1)
    rms: float  # 加权 RMS 残差


class PrimitiveFit:
    """稠密深度后验 → 逐区域图元 (加权最小二乘 + 模型选择)。

    权重 = 融合精度 (区域软分配由子区域硬标签承担)。批量方式:
    正规方程的充分统计量 (Σw·f_if_j, Σw·f_i·z) 按区域 scatter-add,
    再 (R,3,3)/(R,4,4) 一次性求逆 —— 无逐区域 MLX 同步。
    已知近似: 球拟合用 Kasa 代数残差 (非几何正交距离)。
    """

    min_points: int = 10  # 区域最少像素数 (退化守卫)

    def run(
        self, d: mx.array, prec: mx.array, sub: mx.array
    ) -> tuple[list[PrimFit], mx.array, mx.array]:
        """深度均值 + 精度 + 子区域标签 → (图元列, 渲染深度, 残差场)。"""
        h, w = d.shape
        s = float(max(h, w))
        yy, xx = mx.meshgrid(
            mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )
        u = ((xx - w / 2) / s).reshape(-1)  # 各向同性归一化坐标
        v = ((yy - h / 2) / s).reshape(-1)
        z = d.reshape(-1)
        wt = prec.reshape(-1)
        lab = sub.reshape(-1)
        n_reg = int(mx.max(sub))
        n = n_reg + 1  # 0 号槽 = 掩码/无标签像素

        def scatter(val: mx.array) -> mx.array:
            """按区域标签求和 (scatter-add, (n_reg+1,))。"""
            return mx.zeros((n,)).at[lab].add(val)

        cnt = scatter(mx.ones_like(z))
        wsum = scatter(wt)  # 权重和 (归一化分母, 别用像素数)

        # ── 平面: z = a·u + b·v + c, feats = [u, v, 1] ─────────────
        feats3 = [u, v, mx.ones_like(u)]
        g3 = []
        for i in range(3):
            for j in range(3):
                g3.append(scatter(wt * feats3[i] * feats3[j]))
        b3 = [scatter(wt * feats3[i] * z) for i in range(3)]
        G3 = mx.stack(g3, axis=-1).reshape(n, 3, 3)
        B3 = mx.stack(b3, axis=-1)[..., None]
        # 微小脊正则: 0 号槽/退化区域的 G 奇异会让 LU 直接 terminate
        # (MLX C++ 层异常无法 try) —— 相对量级 ~1e-10, 不扰动真拟合
        cov3 = mx.linalg.inv(G3 + 1e-6 * mx.eye(3), stream=mx.cpu)  # (n,3,3)
        th3 = (cov3 @ B3)[:, :, 0]  # (n,3)

        # ── 球 (Kasa): u²+v²+z_c² + D·u + E·v + F·z_c + G = 0 ──────
        # z 逐区域中心化 (z_c = z − z̄): 浅球盖上 z ≈ 常数 → [z,1]
        # 近共线, G 病态, Kasa 直接失效 (实测 cz 偏 0.6)
        mz = scatter(wt * z) / mx.maximum(wsum, 1.0)
        z_c = z - mz[lab]
        feats4 = [u, v, z_c, mx.ones_like(u)]
        tgt = -(u**2 + v**2 + z_c**2)
        g4 = []
        for i in range(4):
            for j in range(4):
                g4.append(scatter(wt * feats4[i] * feats4[j]))
        b4 = [scatter(wt * feats4[i] * tgt) for i in range(4)]
        G4 = mx.stack(g4, axis=-1).reshape(n, 4, 4)
        B4 = mx.stack(b4, axis=-1)[..., None]
        cov4 = mx.linalg.inv(G4 + 1e-6 * mx.eye(4), stream=mx.cpu)
        th4 = (cov4 @ B4)[:, :, 0]  # (n,4): D,E,F,G

        # 球参数还原 (中心平移回原坐标) + 半球符号
        cu, cv = -th4[:, 0] / 2, -th4[:, 1] / 2
        cz = -th4[:, 2] / 2 + mz
        rho2 = (th4[:, 0] ** 2 + th4[:, 1] ** 2 + th4[:, 2] ** 2) / 4 - th4[:, 3]
        rho = mx.sqrt(mx.maximum(rho2, 0.0))
        sign = mx.where(mz >= cz, 1.0, -1.0)  # 朝向区域质量侧的半球

        # ── 逐像素两种模型的预测与残差 (参数按 lab gather) ──────────
        pred_pl = th3[lab, 0] * u + th3[lab, 1] * v + th3[lab, 2]
        rr = mx.maximum(
            rho2[lab] - (u - cu[lab]) ** 2 - (v - cv[lab]) ** 2, 0.0
        )  # 钳底: 负值开方得 NaN 会沿 minimum 传染整个选模
        pred_sp = cz[lab] + sign[lab] * mx.sqrt(rr)
        ok = (cnt > self.min_points) & mx.all(mx.isfinite(th3), axis=1)
        ok = ok & mx.all(mx.isfinite(th4), axis=1) & (rho2 > 0)
        res_pl = (z - pred_pl) ** 2 * wt
        res_sp = (z - pred_sp) ** 2 * wt
        rms_pl = mx.sqrt(scatter(res_pl) / mx.maximum(wsum, 1.0))
        rms_sp = mx.sqrt(scatter(res_sp) / mx.maximum(wsum, 1.0))
        rms_sp = mx.where(ok, rms_sp, math.inf)
        rms_pl = mx.where(
            (cnt > self.min_points) & mx.all(mx.isfinite(th3), axis=1),
            rms_pl, math.inf,
        )
        use_sphere = rms_sp < rms_pl  # (n,) 模型选择

        # 渲染深度与残差场 (未图元化区域留稠密深度)
        fitted = mx.isfinite(mx.minimum(rms_pl, rms_sp))
        pred = mx.where(use_sphere[lab], pred_sp, pred_pl)
        is_fit = fitted[lab] & (lab > 0)
        render = mx.where(is_fit, pred, z).reshape(h, w)
        resid = mx.where(is_fit, z - pred, 0.0).reshape(h, w)

        # ── 逐区域 PrimFit (blade 换回像素单位) ─────────────────────
        fits: list[PrimFit] = []
        finite_l = fitted.tolist()
        sph_l = use_sphere.tolist()
        for r in range(1, n_reg + 1):
            if not finite_l[r]:
                fits.append(PrimFit(None, "dense", None, (), 1.0, math.inf))
                continue
            if not sph_l[r]:
                a, b, c = (float(t) for t in th3[r])
                na, nb = a / s, b / s  # 换回像素单位
                nc = c - a * (w / 2) / s - b * (h / 2) / s
                nl = math.sqrt(na * na + nb * nb + 1.0)
                blade = Plane((na / nl, nb / nl, -1.0 / nl), nc / nl)
                fits.append(
                    PrimFit(blade, "plane", cov3[r], (a, b, c), 1.0,
                            float(rms_pl[r]))
                )
            else:
                prm = (float(cu[r]), float(cv[r]), float(cz[r]), float(rho[r]))
                blade = Sphere(
                    (prm[0] * s + w / 2, prm[1] * s + h / 2, prm[2]), prm[3] * s
                )
                fits.append(
                    PrimFit(blade, "sphere", cov4[r], prm, float(sign[r]),
                            float(rms_sp[r]))
                )
        return fits, render, resid


# ── 曼哈顿耦合 (留钩) ──────────────────────────────────────────────


class ManhattanCoupling:
    """法向直方图聚类 + 正交约束精化 (flow.md §5.4)。未实现 ——
    接口预留, 默认不参与管线。"""

    def refine(self, fits: list[PrimFit]) -> list[PrimFit]:
        """对平面组施加 ip(n_i, n_j)=0 约束并重投影精化 (未实现)。"""
        return fits


# ── 总装门面 ──────────────────────────────────────────────────────


class FusionResult(NamedTuple):
    """融合层输出: 稠密后验 + 图元 + 渲染 + 反馈图。"""

    depth: mx.array  # (H,W) 融合深度均值
    precision: mx.array  # (H,W) 融合精度
    fits: list[PrimFit]  # 逐子区域图元 (blade=None = 残差留稠密场)
    render: mx.array  # (H,W) 图元渲染深度 (未图元化处为稠密深度)
    prior_map: mx.array  # (H,W) ∈[0,1] 深度不连续反馈 D (→ 分割层)
    residual: mx.array  # (H,W) 拟合残差稠密场


@dataclass(slots=True)
class DepthFusionLayer:
    """融合层门面: 线索 + 分割结果 → FusionResult (含 prior_map 反馈)。"""

    prior_precision: float = 1e-3  # 弱全局先验精度
    feedback_quantile: float = 0.99  # D 归一化分位数

    def run(self, cues: list[DepthCue], subregions: mx.array) -> FusionResult:
        """线索列 + 分割层子区域 → 融合/提升/反馈全量产物。"""
        d, p = CueFusion.run(cues, self.prior_precision)
        fits, render, resid = PrimitiveFit().run(d, p, subregions)
        # 深度不连续 → 归一化反馈图 (4 邻域差分的最大值)
        dy = mx.abs(render[1:, :] - render[:-1, :])
        dx = mx.abs(render[:, 1:] - render[:, :-1])
        dmap = mx.zeros(render.shape)
        dmap = dmap.at[1:, :].add(dy).at[:, 1:].add(dx)
        flat = mx.sort(dmap.reshape(-1))
        q = flat[int(self.feedback_quantile * (flat.shape[0] - 1))]
        prior_map = mx.clip(dmap / mx.maximum(q, 1e-12), 0.0, 1.0)
        return FusionResult(d, p, fits, render, prior_map, resid)


if __name__ == "__main__":
    H, W = 96, 128

    # ── 1. 线索融合: 逆方差加权 ─────────────────────────────────────
    c_weak = DepthCue(mx.full((8, 8), 1.0), mx.full((8, 8), 1.0))
    c_strong = DepthCue(mx.full((8, 8), 2.0), mx.full((8, 8), 3.0))
    dm, dp = CueFusion.run([c_weak, c_strong])
    # 解析: (1·1+3·2)/(1+3+1e-3) ≈ 1.75, 偏向强线索
    assert abs(float(dm[0, 0]) - 1.75) < 1e-2, float(dm[0, 0])
    assert abs(float(dp[0, 0]) - 4.0) < 1e-2
    print(f"1. 融合: mean={float(dm[0, 0]):.3f} (弱1.0/强2.0, 偏向强) ✓")

    # ── 2/4. 平面提升: 平面世界应选平面, 参数复原 ───────────────────
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    s = float(W)
    u_n = (xx - W / 2) / s
    v_n = (yy - H / 2) / s
    z_plane = 0.4 * u_n + 0.2 * v_n + 3.0
    sub = mx.ones((H, W), dtype=mx.int32)
    cues = [DepthCue(z_plane, mx.full((H, W), 10.0))]
    fr = DepthFusionLayer().run(cues, sub)
    f = fr.fits[0]
    assert f.kind == "plane", f"平面世界应选平面, 得 {f.kind}"
    a, b, c = f.params
    assert abs(a - 0.4) < 1e-2 and abs(b - 0.2) < 1e-2 and abs(c - 3.0) < 1e-2, (
        f"平面参数 {f.params}"
    )
    assert f.rms < 1e-3, f"平面残差 {f.rms}"
    print(f"2. 平面提升: (a,b,c)=({a:.3f},{b:.3f},{c:.3f}) rms={f.rms:.2e} ✓")

    # ── 3/4. 球提升: 球面世界应选球, 参数复原 ───────────────────────
    cu0, cv0, cz0, rho0 = 0.0, 0.0, 3.0, 1.5
    rr0 = rho0**2 - u_n**2 - v_n**2
    z_sph = cz0 - mx.sqrt(mx.maximum(rr0, 0.0))  # 上半球朝向相机
    valid = rr0 > 0.05
    sub2 = mx.where(valid, 1, 0).astype(mx.int32)
    cues2 = [DepthCue(z_sph, mx.full((H, W), 10.0))]
    fr2 = DepthFusionLayer().run(cues2, sub2)
    f2 = fr2.fits[0]
    assert f2.kind == "sphere", f"球面世界应选球, 得 {f2.kind}"
    cu, cv, cz, rho = f2.params
    assert abs(cu) < 2e-2 and abs(cv) < 2e-2 and abs(cz - cz0) < 2e-2
    assert abs(rho - rho0) < 2e-2, f"球参数 {f2.params}"
    print(f"3. 球提升: c=({cu:.3f},{cv:.3f},{cz:.3f}) ρ={rho:.3f} ✓")

    # ── 5. 反馈闭环: 双平面世界, 缺口边界由真实反馈保持 ─────────────
    # 两个深度平面在 col 64 相接 (深度跳变); enh 在边界上有 16px 缺口
    z_two = mx.where(xx < W // 2, 2.0, 5.0)
    E = mx.random.uniform(shape=(H, W), key=mx.random.key(1)) * 0.04
    E[:, 64] = 0.7  # 边缘图: 边界强但...
    E[40:56, 64] = 0.02  # ...中间 16px 缺口
    from segment import SceneSegmenter

    zero_like = mx.zeros((H, W))
    seg0 = SceneSegmenter(tau=0.5).run(E, zero_like, zero_like)
    fr3 = DepthFusionLayer().run([DepthCue(z_two, mx.full((H, W), 10.0))],
                                 seg0.subregions)
    seg1 = SceneSegmenter(tau=0.5).run(
        E, zero_like, zero_like, prior_map=fr3.prior_map, w_prior=0.8
    )
    pt_l, pt_r = (48, 32), (48, 96)
    same0 = int(seg0.regions[pt_l]) == int(seg0.regions[pt_r])
    same1 = int(seg1.regions[pt_l]) == int(seg1.regions[pt_r])
    assert same0, "无反馈: 缺口弧被稀释, 两区应在 τ=0.5 合并"
    assert not same1, "真实深度反馈: 缺口被 D 填补, 两区应保持分离"
    print("5. 反馈闭环: 融合产物 D 保持缺口边界分离 (§2 协议首轮) ✓")

    # ── 6. 退化区域: 残差留稠密场 ───────────────────────────────────
    sub3 = mx.ones((H, W), dtype=mx.int32)
    sub3[:5, :5] = 2  # 25px 小区域 (>= min_points=10 不退化), 用 2x2
    sub3[:2, :2] = 3  # 4px 区域 → 退化
    fr4 = DepthFusionLayer().run(cues, sub3)
    assert fr4.fits[2].kind == "dense", "4px 区域应退化留稠密场"
    print("6. 退化守卫: 小区域未图元化, 残差留稠密场 ✓")
