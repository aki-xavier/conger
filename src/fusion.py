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
  OcclusionOrder: T 结遮挡偏序 → 序数约束 z_front ≤ z_behind
       (prior.md 物理先验, 高权重不可下调), 图元提升后半空间投影
  反馈: 渲染图元深度 → 不连续图 D → SceneSegmenter(prior_map=D)
       (§2 迭代协议的虚线边, 本层是 prior_map 的生产者)

  ManhattanCoupling: 平行/正交吸附 (prior.md 平直与正交先验),
       近满足才吸附, 无证据逐位不动 (run(manhattan=True) 开启)
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
    """贝叶斯线索整合: p(d|c₁..c_M) ∝ p(d)∏p(c_i|d) 的高斯闭式
    (线索权重即其 precision 本身, 无额外指数权重)。"""

    @staticmethod
    def run(
        cues: list[DepthCue], prior_precision: float = 1e-3
    ) -> tuple[mx.array, mx.array]:
        """精度加权融合 → (均值, 精度)。prior 为弱全局精度 (分段平滑
        的职责由图元化接管, 见 flow.md §5.2), 遮挡偏序不进表决。
        cues 须非空 (契约)。"""
        if not cues:
            raise ValueError("CueFusion.run: cues 不能为空")
        p = mx.full(cues[0].mean.shape, prior_precision)
        d = mx.zeros(cues[0].mean.shape)
        for c in cues:
            p = p + c.precision
            d = d + c.precision * c.mean
        return d / p, p


@dataclass(slots=True)
class EdgeAwareSmooth:
    """稠密深度场的边缘感知平滑 (prior.md 紧凑性与平滑性先验:
    E = E_data + λ·E_smooth, 表面大多平滑、深度变化稀疏)。

    二次能量的加权 Jacobi 不动点迭代:
        z ← (p·d + λ·Σ_q w_pq·z_q) / (p + λ·Σ_q w_pq)
    邻接权 w = 1 − E_边界 (取两像素 max), 边界处平滑断开 →
    区域内扩散填洞/降噪, 跨边界不渗漏。数据项权重 = 融合精度
    (弱线索区被平滑主导 ≈ 内绘, 强线索区保持)。
    近似: precision 不随平滑更新 (docstring 注明, 不重估)。"""

    iters: int = 8
    lam: float = 4.0

    def run(self, d: mx.array, p: mx.array, boundary: mx.array) -> mx.array:
        """融合均值/精度 + 边界强度图 (∈[0,1], 如 enh) → 平滑深度。"""
        e = mx.clip(boundary, 0.0, 1.0)
        wr = 1.0 - mx.maximum(e[:, 1:], e[:, :-1])  # 水平邻接权 (H, W-1)
        wd = 1.0 - mx.maximum(e[1:, :], e[:-1, :])  # 垂直邻接权 (H-1, W)
        data = p * d
        z = d
        for _ in range(self.iters):
            num = mx.zeros_like(z)
            wsum = mx.zeros_like(z)
            num = num.at[:, 1:].add(wr * z[:, :-1])
            num = num.at[:, :-1].add(wr * z[:, 1:])
            wsum = wsum.at[:, 1:].add(wr)
            wsum = wsum.at[:, :-1].add(wr)
            num = num.at[1:, :].add(wd * z[:-1, :])
            num = num.at[:-1, :].add(wd * z[1:, :])
            wsum = wsum.at[1:, :].add(wd)
            wsum = wsum.at[:-1, :].add(wd)
            z = (data + self.lam * num) / (p + self.lam * wsum)
        return z


# ── 区域提升: 加权 LSQ 图元化 ─────────────────────────────────────


class OrdinalConstraint(NamedTuple):
    """遮挡序数约束: front 区域深度 ≤ behind 区域深度 (T 结语义)。
    prior.md 物理先验: "A 在 B 前" 是严格序数约束, 在任何距离成立,
    以极高权重编码, 不被其他线索的冲突下调。"""

    pos: tuple[float, float]  # (row, col) 结点位置
    front: int  # 前表面区域 id (rid)
    behind: int  # 后表面区域 id (rid)


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

    @staticmethod
    def plane_blade(a: float, b: float, c: float, h: int, w: int) -> Multivector:
        """归一化平面参数 (a,b,c) → 像素单位 blade。
        z = na·x + nb·y + nc ⇔ (−na)x + (−nb)y + z = nc:
        法向取 (−na,−nb,1), 否则 dist 符号整体反 (闭环期踩过)。"""
        s = float(max(h, w))
        na, nb = a / s, b / s  # 换回像素单位
        nc = c - a * (w / 2) / s - b * (h / 2) / s
        nl = math.sqrt(na * na + nb * nb + 1.0)
        return Plane((-na / nl, -nb / nl, 1.0 / nl), nc / nl)

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
        # 条件数守卫: 线状子区域的 G 近奇异, inv 会吐出 |a|~1e5 的
        # 陡坡平面 (实测污染场景渲染到 3e5) —— 这类区域留稠密场
        ev3 = mx.linalg.eigh(G3, stream=mx.cpu)[0]
        cond3 = ev3[:, 0] / mx.maximum(ev3[:, -1], 1e-12) > 1e-8

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
        ev4 = mx.linalg.eigh(G4, stream=mx.cpu)[0]
        cond4 = ev4[:, 0] / mx.maximum(ev4[:, -1], 1e-12) > 1e-8

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
        ok = (cnt > self.min_points) & cond4
        ok = ok & mx.all(mx.isfinite(th4), axis=1) & (rho2 > 0)
        # 注: 球门不要求 cond3 (平面 Gram 条件数) —— 线状区域平面
        # 退化时球拟合再好也不该被拒 (两模型的门各自独立)
        res_pl = (z - pred_pl) ** 2 * wt
        res_sp = (z - pred_sp) ** 2 * wt
        rms_pl = mx.sqrt(scatter(res_pl) / mx.maximum(wsum, 1.0))
        rms_sp = mx.sqrt(scatter(res_sp) / mx.maximum(wsum, 1.0))
        rms_sp = mx.where(ok, rms_sp, math.inf)
        rms_pl = mx.where(
            (cnt > self.min_points) & cond3 & mx.all(mx.isfinite(th3), axis=1),
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
                fits.append(
                    PrimFit(self.plane_blade(a, b, c, h, w), "plane",
                            cov3[r], (a, b, c), 1.0, float(rms_pl[r]))
                )
            else:
                prm = (float(cu[r]), float(cv[r]), float(cz[r]), float(rho[r]))
                # 注意: x,y 换像素而 z 保持深度单位 —— blade 在
                # (px,px,depth) 混合量纲空间, 非米制 (已知近似;
                # motor 对齐对球只做中心点变换, 半径不动)
                blade = Sphere(
                    (prm[0] * s + w / 2, prm[1] * s + h / 2, prm[2]), prm[3] * s
                )
                # 协方差 Δ 变换: cov4 是 Kasa (D,E,F,G) 空间的, params
                # 是 (cu,cv,cz,ρ) 空间 —— 不变换下游 (scenegraph 信息
                # 滤波/Bhattacharyya) 拿到的就是语义错位的协方差
                dd, ee, ff = (float(t) for t in th4[r, :3])
                rho_v = max(prm[3], 1e-6)
                jm = mx.array(
                    [
                        [-0.5, 0.0, 0.0, 0.0],
                        [0.0, -0.5, 0.0, 0.0],
                        [0.0, 0.0, -0.5, 0.0],
                        [dd / (4 * rho_v), ee / (4 * rho_v),
                         ff / (4 * rho_v), -1.0 / (2 * rho_v)],
                    ]
                )
                cov_s = jm @ cov4[r] @ jm.T
                fits.append(
                    PrimFit(blade, "sphere", cov_s, prm, float(sign[r]),
                            float(rms_sp[r]))
                )
        return fits, render, resid


# ── T 结序数深度约束 (prior.md 物理先验: 遮挡逻辑) ─────────────────


@dataclass(slots=True)
class OcclusionOrder:
    """T 结遮挡偏序的消费者: 序数约束 z_front ≤ z_behind 的半空间投影。

    约束不进 CueFusion 的高斯表决 (半边约束无共轭闭式), 在图元提升
    之后投影: 违序时两区域深度向约束边界做 KKT 修正 (按结点处预测
    方差分摊, 序本身绝对; 满序时不动 → 无自确认)。修正全摊到截距
    c (保坡度), 渲染随区域整体平移。协方差不收缩 (保守, 序数投影
    只移位不缩不确定度)。近似: 只接管平面-平面对, 球/稠密侧跳过。"""

    margin: float = 0.0  # 严格序间隔 (深度单位)

    @staticmethod
    def constraints_from_grouping(
        res, rid_map: mx.array, offset: float = 2.0
    ) -> list[OrdinalConstraint]:
        """T 结偏序 → 序数约束 (链 → 区域几何映射)。
        behind 链近端点位于背景侧: 背景方向 = 端点 − 结点; front 链
        最近 edgel 的单位法向按此定向, 两侧偏移 offset 采样区域 id。
        res 为鸭子类型 (GroupingResult), 避免 fusion→grouping 依赖环。"""
        h, w = rid_map.shape
        pos, normal = res.edgels.pos, res.edgels.normal
        out: list[OrdinalConstraint] = []

        def _rid(r: float, c: float) -> int:
            rr = min(max(int(round(r)), 0), h - 1)
            cc = min(max(int(round(c)), 0), w - 1)
            return int(rid_map[rr, cc])

        for t in res.t_junctions:
            jr, jc = t.pos
            ch_b = res.chains[t.behind]
            e0, e1 = pos[int(ch_b[0])], pos[int(ch_b[-1])]
            d0 = float((e0[0] - jr) ** 2 + (e0[1] - jc) ** 2)
            d1 = float((e1[0] - jr) ** 2 + (e1[1] - jc) ** 2)
            en = e0 if d0 <= d1 else e1
            br, bc = float(en[0]) - jr, float(en[1]) - jc  # 背景方向
            bn = math.hypot(br, bc)
            if bn < 1e-6:
                continue
            br, bc = br / bn, bc / bn
            ch_f = res.chains[t.front]
            pts = pos[ch_f]
            i = int(mx.argmin(
                (pts[:, 0] - jr) ** 2 + (pts[:, 1] - jc) ** 2
            ))
            idx = int(ch_f[i])
            nr, nc_ = float(normal[idx, 0]), float(normal[idx, 1])
            if nr * br + nc_ * bc < 0:
                nr, nc_ = -nr, -nc_  # 法向定向到背景侧
            pr, pc = float(pts[i, 0]), float(pts[i, 1])
            rid_f = _rid(pr - offset * nr, pc - offset * nc_)
            rid_b = _rid(pr + offset * nr, pc + offset * nc_)
            if rid_f <= 0 or rid_b <= 0 or rid_f == rid_b:
                continue  # 采样落到同区/无标签 → 约束不可信, 跳过
            out.append(OrdinalConstraint(t.pos, rid_f, rid_b))
        return out

    def enforce(
        self,
        fits: list[PrimFit],
        render: mx.array,
        sub: mx.array,
        cons: list[OrdinalConstraint],
    ) -> tuple[list[PrimFit], mx.array]:
        """序数约束的半空间投影: 违序平面对 → 方差分摊修正。
        fits[r-1] ↔ 区域 r (PrimitiveFit 的既有约定)。"""
        h, w = render.shape
        s = float(max(h, w))
        fits = list(fits)
        for cn in cons:
            if cn.front > len(fits) or cn.behind > len(fits):
                continue
            ff, fb = fits[cn.front - 1], fits[cn.behind - 1]
            if ff.kind != "plane" or fb.kind != "plane":
                continue  # 只接管平面-平面对
            u = (cn.pos[1] - w / 2) / s
            v = (cn.pos[0] - h / 2) / s
            fvec = mx.array([u, v, 1.0])
            zf = ff.params[0] * u + ff.params[1] * v + ff.params[2]
            zb = fb.params[0] * u + fb.params[1] * v + fb.params[2]
            gap = zb - zf
            if gap >= self.margin:
                continue  # 已满足序 → 不动 (无自确认)
            # KKT: min (Δf²/var_f + Δb²/var_b) s.t. z_b'−z_f' ≥ margin
            vf = float(fvec @ ff.cov @ fvec)
            vb = float(fvec @ fb.cov @ fvec)
            need = self.margin - gap
            tot = vf + vb + 1e-12
            dzf, dzb = -need * vf / tot, need * vb / tot
            nf = ff._replace(
                params=(ff.params[0], ff.params[1], ff.params[2] + dzf),
                blade=PrimitiveFit.plane_blade(
                    ff.params[0], ff.params[1], ff.params[2] + dzf, h, w),
            )
            nb = fb._replace(
                params=(fb.params[0], fb.params[1], fb.params[2] + dzb),
                blade=PrimitiveFit.plane_blade(
                    fb.params[0], fb.params[1], fb.params[2] + dzb, h, w),
            )
            fits[cn.front - 1], fits[cn.behind - 1] = nf, nb
            # 平面截距修正 = 区域渲染整体平移
            render = mx.where(sub == cn.front, render + dzf, render)
            render = mx.where(sub == cn.behind, render + dzb, render)
        return fits, render


# ── 曼哈顿耦合 (平直与正交先验) ──────────────────────────────────────


@dataclass(slots=True)
class ManhattanCoupling:
    """平直与正交先验 (prior.md 几何与结构: 室内/城市场景默认平面
    与直角)。深度图只能表达非垂直面, 完整曼哈顿系退化为两条
    保守规则, 都只在近满足时吸附 (无正交证据不强加):

    ① 平行吸附: 法向互夹 < cluster_deg 的平面组 (≥2) 吸附到
       加权均值法向 (共享方向 = 平行地板/墙面);
    ② 正交吸附: |n_i·n_j| < sin(ortho_deg) 的近正交对, 各反向
       旋转一半残差到严格正交 (小角一步 + 一次抛光)。

    吸附保区域质心处深度不变 (锚点), 修正全进截距/坡度;
    |m_z| < 0.3 的近垂直面不可表达为 z(x,y), 跳过 (docstring 级
    已知限制)。协方差/rms 不变 (保守, 吸附不缩不确定度)。
    顺序语义: 先平行后正交, 正交旋转可能微扰刚吸附的平行对齐
    (corner 概率低, 不回查)。"""

    cluster_deg: float = 10.0  # 平行聚类角
    ortho_deg: float = 10.0  # 正交吸附的余差门

    @staticmethod
    def _normal(fit: PrimFit) -> list[float] | None:
        """平面 blade 的米制单位法向 (vals[1:4]); 非平面返回 None。"""
        if fit.kind != "plane" or fit.blade is None:
            return None
        v = fit.blade.values
        return [float(v[1]), float(v[2]), float(v[3])]

    def refine(
        self, fits: list[PrimFit], sub: mx.array
    ) -> list[PrimFit]:
        """平面组施加平行/正交吸附; 返回新 fits (无证据时逐位不变)。
        fits[r-1] ↔ 区域 r (PrimitiveFit 约定)。"""
        h, w = sub.shape
        s = float(max(h, w))
        ns: dict[int, list[float]] = {
            i: n for i in range(len(fits)) if (n := self._normal(fits[i]))
        }
        wt = {i: 1.0 / max(fits[i].rms, 1e-3) for i in ns}
        cos_c = math.cos(math.radians(self.cluster_deg))
        sin_o = math.sin(math.radians(self.ortho_deg))

        def dot(a: list[float], b: list[float]) -> float:
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def norm(a: list[float]) -> list[float]:
            nl = math.sqrt(dot(a, a)) + 1e-12
            return [v / nl for v in a]

        # ── ① 平行聚类: 按权重贪心, 组内吸附到加权均值法向 ──────
        order = sorted(ns, key=lambda i: -wt[i])
        clustered: set[int] = set()
        snapped: dict[int, list[float]] = {}
        for i in order:
            if i in clustered:
                continue
            grp = [
                j for j in order
                if j not in clustered and abs(dot(ns[i], ns[j])) > cos_c
            ]
            if len(grp) < 2:
                continue
            clustered.update(grp)
            ref = ns[i]
            mean = [0.0, 0.0, 0.0]
            for j in grp:
                sgn = 1.0 if dot(ns[j], ref) >= 0 else -1.0
                for k in range(3):
                    mean[k] += wt[j] * sgn * ns[j][k]
            mean = norm(mean)
            for j in grp:
                snapped[j] = mean if dot(ns[j], mean) >= 0 else [-v for v in mean]

        # ── ② 近正交对: 各反向旋一半残差到严格正交 ──────────────
        for i in order:
            for j in order:
                if j <= i:
                    continue
                ni = snapped.get(i, ns[i])
                nj = snapped.get(j, ns[j])
                d = dot(ni, nj)
                if abs(d) >= sin_o or abs(d) < 1e-12:
                    continue  # 无正交证据 / 已严格正交
                for _ in range(2):  # 小角一步 + 抛光
                    ni = norm([ni[k] - d / 2 * nj[k] for k in range(3)])
                    nj = norm([nj[k] - d / 2 * ni[k] for k in range(3)])
                    d = dot(ni, nj)
                snapped[i], snapped[j] = ni, nj

        if not snapped:
            return fits

        # ── 应用: 新法向 + 质心锚点 → 新截距/坡度, 重建 blade ─────
        lab = sub.reshape(-1)
        yy, xx = mx.meshgrid(
            mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )

        def scatter(v: mx.array) -> mx.array:
            return mx.zeros((len(fits) + 1,)).at[lab].add(v)

        cnt = scatter(mx.ones((int(lab.shape[0]),)))
        uc = scatter(xx.reshape(-1)) / mx.maximum(cnt, 1.0)
        vc = scatter(yy.reshape(-1)) / mx.maximum(cnt, 1.0)
        out = list(fits)
        for i, m in snapped.items():
            f = fits[i]
            if abs(m[2]) < 0.3:
                continue  # 近垂直面不可表达为 z(x,y), 跳过
            r = i + 1
            cu = (float(uc[r]) - w / 2) / s
            cv = (float(vc[r]) - h / 2) / s
            a, b, c = f.params
            z_ref = a * cu + b * cv + c  # 旧平面在质心处的深度
            na, nb = -m[0] / m[2], -m[1] / m[2]  # 像素单位坡度
            a2, b2 = na * s, nb * s  # 归一化坐标参数
            c2 = z_ref - a2 * cu - b2 * cv
            out[i] = f._replace(
                params=(a2, b2, c2),
                blade=PrimitiveFit.plane_blade(a2, b2, c2, h, w),
            )
        return out


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

    def run(
        self,
        cues: list[DepthCue],
        subregions: mx.array,
        occlusion: list[OrdinalConstraint] | None = None,
        boundary: mx.array | None = None,
        manhattan: bool = False,
    ) -> FusionResult:
        """线索列 + 分割层子区域 → 融合/提升/反馈全量产物。
        occlusion: T 结序数约束 (OcclusionOrder.constraints_from_grouping
        的产物), 图元提升后做半空间投影, 不进高斯表决。
        boundary: 边界强度图 (如 enh) —— 给则稠密场先过边缘感知
        平滑 (E_data+λE_smooth, 紧凑性先验), 再图元化。
        manhattan: True 则图元提升后施加平行/正交吸附
        (ManhattanCoupling; 默认关 —— 非正交场景不应付代价)。
        注意 manhattan 路径重渲 render 但不重算 residual
        (残差场保持吸附前口径)。"""
        d, p = CueFusion.run(cues, self.prior_precision)
        if boundary is not None:
            d = EdgeAwareSmooth().run(d, p, boundary)
        fits, render, resid = PrimitiveFit().run(d, p, subregions)
        if manhattan:
            fits = ManhattanCoupling().refine(fits, subregions)
            # 吸附改了拟合参数 → 渲染重建 (平面平移/坡度变, 逐区重渲)
            h, w = d.shape
            s = float(max(h, w))
            yy, xx = mx.meshgrid(
                mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
                indexing="ij",
            )
            u = (xx - w / 2) / s
            v = (yy - h / 2) / s
            # 从原渲染出发只重渲平面区 (球/稠密区保持; 未变平面
            # 重渲出相同值, 同一进程内逐位确定)
            for r, f in enumerate(fits, start=1):
                if f.kind != "plane":
                    continue
                pred = f.params[0] * u + f.params[1] * v + f.params[2]
                render = mx.where(subregions == r, pred, render)
        if occlusion:
            fits, render = OcclusionOrder().enforce(
                fits, render, subregions, occlusion
            )
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
    # 协方差已 Δ 变换到 (cu,cv,cz,ρ) 空间: 对角元应为正且量级合理
    assert f2.cov is not None
    dg = [float(f2.cov[i, i]) for i in range(4)]
    assert all(x > 0 for x in dg), f"球 cov 对角应正: {dg}"
    assert dg[2] < 1e-2, f"cz 方差量级: {dg[2]:.2e} (Kasa cz 天然欠优)"
    print(f"3. 球提升: c=({cu:.3f},{cv:.3f},{cz:.3f}) ρ={rho:.3f} cov✓ ✓")

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
    sub3[:5, :5] = 2  # 21px 区域 (≥ min_points=10 不退化)
    sub3[:2, :2] = 3  # 4px 区域 → 退化
    fr4 = DepthFusionLayer().run(cues, sub3)
    assert fr4.fits[2].kind == "dense", "4px 区域应退化留稠密场"
    print("6. 退化守卫: 小区域未图元化, 残差留稠密场 ✓")

    # ── 7. T 结序数约束: 违序投影 + 满序不动 (prior.md 物理先验) ────
    sub_oc = mx.where(xx < W // 2, 1, 2).astype(mx.int32)
    z_wrong = mx.where(xx < W // 2, 5.0, 2.0)  # 深度序颠倒的世界
    cn = OrdinalConstraint((48.0, 64.0), front=1, behind=2)
    layer = DepthFusionLayer()
    cue_oc = [DepthCue(z_wrong, mx.full((H, W), 10.0))]
    fr_bad = layer.run(cue_oc, sub_oc)
    assert float(fr_bad.render[48, 32]) > float(fr_bad.render[48, 96]), (
        "违序世界: 前区渲染应比后区远"
    )
    fr_fix = layer.run(cue_oc, sub_oc, occlusion=[cn])
    # 约束在结点处求值 (u=0,v=0 → z=c): 序应精确成立且落在原值之间
    c_f, c_b = fr_fix.fits[0].params[2], fr_fix.fits[1].params[2]
    assert c_f <= c_b + 1e-6, f"序数投影后结点处仍违序: {c_f}/{c_b}"  # float32 底
    assert 2.0 < c_f < c_b + 1e-6 and c_b < 5.0, (
        f"方差分摊应把两者折中原区间内: {c_f:.2f}/{c_b:.2f}"
    )
    # 满序不动: 正确序世界加同一约束, 渲染逐位不变 (无自确认)
    z_right = mx.where(xx < W // 2, 2.0, 5.0)
    cue_ok = [DepthCue(z_right, mx.full((H, W), 10.0))]
    fr_a = layer.run(cue_ok, sub_oc)
    fr_b = layer.run(cue_ok, sub_oc, occlusion=[cn])
    # 满序不动: 正确序世界加同一约束, 渲染不变 (无自确认)。
    # 容差 1e-3: scatter-add 的 GPU atomic 归约乱序, 同输入两跑
    # 本身有 ~1e-4 抖动 (实测, 逐跑有波动), 非约束引入 ——
    # 真被约束改动时位移量级 ~1.5, 1e-3 足以区分
    assert bool(mx.all(mx.abs(fr_a.render - fr_b.render) < 1e-3)), (
        "已满足序时约束应不动"
    )
    print("7. 序数约束: 违序投影成立, 满序逐位不动 (高权重不可下调) ✓")

    # ── 8. 链→区域映射: T 结几何 → 序数约束 ──────────────────────────
    from types import SimpleNamespace as _NS

    # front 链: row 48 水平 (遮挡轮廓; 上=前物体 rid1, 下=背景 rid2)
    # behind 链: col 60 垂直 (背景边缘, 止于结点 (48,60))
    fcols = list(range(20, 101, 4))
    brows = list(range(49, 91, 4))
    pts = [(48.0, float(c)) for c in fcols]
    pts += [(float(r), 60.0) for r in brows]
    nrm = [(1.0, 0.0)] * len(fcols) + [(0.0, 1.0)] * len(brows)
    ed = _NS(
        pos=mx.array(pts, dtype=mx.float32),
        normal=mx.array(nrm, dtype=mx.float32),
    )
    tj = _NS(pos=(48.0, 60.0), front=0, behind=1, support=(3, 0))
    res = _NS(
        edgels=ed,
        chains=[mx.array(list(range(len(fcols)))),
                mx.array(list(range(len(fcols), len(pts))))],
        t_junctions=[tj],
    )
    rid = mx.where(yy < 48, 1, 2).astype(mx.int32)
    cons = OcclusionOrder.constraints_from_grouping(res, rid)
    assert len(cons) == 1, f"应得 1 条约束: {cons}"
    assert cons[0].front == 1 and cons[0].behind == 2, cons[0]
    print("8. 链→区域映射: T 结 → (前=1, 后=2) 序数约束 ✓")

    # ── 9. 边缘感知平滑: 区域内降噪 + 边界不渗漏 (紧凑性先验) ──────
    sm = EdgeAwareSmooth()
    # 9a. 平面 + 噪声, 无边界 → 平滑后更靠真值
    z_true = 0.4 * u_n + 3.0
    z_noisy = z_true + mx.random.normal((H, W), key=mx.random.key(7)) * 0.3
    p_flat = mx.full((H, W), 1.0)
    z_sm = sm.run(z_noisy, p_flat, mx.zeros((H, W)))
    err_raw = float(mx.mean(mx.abs(z_noisy - z_true)))
    err_sm = float(mx.mean(mx.abs(z_sm - z_true)))
    assert err_sm < 0.6 * err_raw, f"降噪: {err_sm:.3f} vs {err_raw:.3f}"
    # 9b. 双深度世界 + 强边界 → 阶梯保持 (不跨边界渗漏)
    z_two9 = mx.where(xx < W // 2, 2.0, 5.0)
    b9 = mx.zeros((H, W))
    b9 = b9.at[:, 64].add(1.0)
    p9 = mx.full((H, W), 0.5)  # 弱线索 → 平滑主导, 考验渗漏
    z_sm9 = sm.run(z_two9, p9, b9)
    assert abs(float(z_sm9[48, 62]) - 2.0) < 0.3, (
        f"边界旁 2px 应保 2: {float(z_sm9[48, 62]):.2f}"
    )
    assert abs(float(z_sm9[48, 68]) - 5.0) < 0.3, (
        f"右区应保 5: {float(z_sm9[48, 68]):.2f}"
    )
    # 9c. 对照: 同输入无边界 → 阶梯被抹 (验证边界图是阻断源)
    z_nb = sm.run(z_two9, p9, mx.zeros((H, W)))
    bleed = float(mx.abs(z_nb[48, 62] - 2.0))
    assert bleed > 0.3, f"无边界对照应渗漏: {bleed:.2f}"
    print("9. 边缘平滑: 区域内降噪, 边界阶梯保持 (对照组渗漏) ✓")

    # ── 10. 曼哈顿耦合: 平行/正交吸附, 无证据不动 (平直正交先验) ────
    mc = ManhattanCoupling()

    def _n(f) -> list[float]:
        """拟合参数的米制单位法向 (−na,−nb,1)。"""
        na, nb = f.params[0] / s, f.params[1] / s
        nl = math.sqrt(na * na + nb * nb + 1.0)
        return [-na / nl, -nb / nl, 1.0 / nl]

    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    sub10 = mx.where(xx < W // 2, 1, 2).astype(mx.int32)
    # 10a. 近平行对 (米制坡度 0.02 / −0.03 ≈ 2.9°) → 法向吸附一致
    z10a = mx.where(xx < W // 2, 0.02 * s * u_n + 3.0, -0.03 * s * u_n + 2.5)
    fr10 = layer.run([DepthCue(z10a, mx.full((H, W), 10.0))], sub10)
    f_ref = mc.refine(fr10.fits, sub10)
    assert _dot(_n(f_ref[0]), _n(f_ref[1])) > 1.0 - 1e-9, (
        f"平行吸附后法向应一致: {_dot(_n(f_ref[0]), _n(f_ref[1]))}"
    )
    # 区域 1 (左半) 质心: col 31.5 → u_c, row 47.5 → v_c
    u_c1, v_c1 = (31.5 - W / 2) / s, (47.5 - H / 2) / s
    z_c_before = (
        fr10.fits[0].params[0] * u_c1
        + fr10.fits[0].params[1] * v_c1
        + fr10.fits[0].params[2]
    )
    z_c_after = (
        f_ref[0].params[0] * u_c1
        + f_ref[0].params[1] * v_c1
        + f_ref[0].params[2]
    )
    assert abs(z_c_before - z_c_after) < 1e-6, "质心锚点深度应不变"
    # 10b. 屋顶对 (米制坡度 ±~1 = ±45°, 近正交) → 吸附后严格正交
    z10b = mx.where(xx < W // 2, 0.95 * s * u_n + 3.0, -1.08 * s * u_n + 3.05)
    fr10b = layer.run([DepthCue(z10b, mx.full((H, W), 10.0))], sub10)
    d_before = _dot(_n(fr10b.fits[0]), _n(fr10b.fits[1]))
    f_ref2 = mc.refine(fr10b.fits, sub10)
    d_after = _dot(_n(f_ref2[0]), _n(f_ref2[1]))
    assert abs(d_after) < 1e-4 < abs(d_before), (
        f"正交吸附: {d_before:.4f} → {d_after:.2e}"
    )
    # 10c. 非曼哈顿对 (米制 0.2 / −0.35, dot≈0.86) → 逐位不动
    z10c = mx.where(xx < W // 2, 0.2 * s * u_n + 3.0, -0.35 * s * u_n + 3.0)
    fr10c = layer.run([DepthCue(z10c, mx.full((H, W), 10.0))], sub10)
    f_ref3 = mc.refine(fr10c.fits, sub10)
    same = all(
        a.params == b.params for a, b in zip(fr10c.fits, f_ref3)
    )
    assert same, "无平行/正交证据时应逐位不动"
    # 10d. flag 通路冒烟: manhattan=True 端到端可跑, 非曼哈顿世界
    # 渲染与默认一致 (GPU atomic 抖动 1e-3 内, 同测试 7)
    fr10d = layer.run([DepthCue(z10c, mx.full((H, W), 10.0))], sub10,
                      manhattan=True)
    assert bool(mx.all(mx.abs(fr10d.render - fr10c.render) < 1e-3))
    print("10. 曼哈顿: 平行/正交吸附成立, 无证据逐位不动 ✓")
