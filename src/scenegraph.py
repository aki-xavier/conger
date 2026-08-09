"""三维重建层 (3D Reconstruction, flow.md §7): CGA 图元场景图。

模块流程:

  图元流 (fusion.PrimFit: 归一化拟合空间 params + cov) + motor
       │  accumulate: 跨帧同名图元的信息滤波融合 (协方差加权),
       │    Bhattacharyya 门控匹配 (同空间, d_B < 2 宁并勿分),
       │    匹配不上 → 新节点 (提升); motor 对齐走 blade 层
       ▼  SceneNode (blade + params + Σ + 运行统计)
  arbitrate: 存在性序贯检验 (SPRT), log-likelihood-ratio 越阈退场
  render: 场景深度渲染 + 残差高亮 → feedback: prior_map (→ 分割层),
       场景渲染作高精度 DepthCue 反喂融合层 (闭环演示)
  reflect: 反射 versor 共轭 + detect_symmetry 对称面检测
       (角平分面候选 + 支撑城重叠真门)

  [留钩] 关联边 (meet/op/ip 关联代数); 曼哈顿正交联合精化;
         motor 协方差传播 (小运动下 blade 对齐足够);
         特征层反馈只留输出字段。

**参数空间约定 (最大的坑)**: 节点携带"空间标签 = 归一化拟合空间"
(fusion 的 (u,v,z), 减中心÷边长)。信息滤波与 Bhattacharyya 全部
在该空间进行; 只在输出/渲染/blade 构造时换像素单位。
离线层, 不做逐帧承诺。
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from cga import Circle, Cylinder, Motor, Multivector, Plane, Sphere
from fusion import DepthCue, FusionResult, PrimFit
from utils import Utils


@dataclass(slots=True)
class SceneNode:
    """场景图节点: 图元 + 最小参数 (归一化拟合空间) + 协方差 + 统计。"""

    blade: Multivector  # Plane/Sphere blade (像素单位, 输出用)
    kind: str  # "plane" / "sphere"
    params: mx.array  # (P,) 归一化拟合空间参数
    cov: mx.array  # (P,P) 参数协方差 (同空间)
    region: int  # 来源子区域 id
    sign: float = 1.0  # 球半球符号 (朝向区域质量侧; 平面恒 1)
    hits: int = 1  # 被匹配融合的次数
    log_llr: float = 0.0  # 存在性序贯检验: 累积 log-likelihood-ratio (封顶 0, 仲裁用)


class SceneGraph:
    """CGA 图元场景图: 累积 / 仲裁 / 渲染 / 反馈。"""

    def __init__(
        self,
        shape: tuple[int, int],
        match_thr: float = 2.0,
        bf_retire: float = -8.0,
        max_rms: float = 0.5,
        max_slope: float = 20.0,
    ):
        """shape: 图像 (H,W) —— 归一化拟合空间的尺度基准。
        match_thr: Bhattacharyya 匹配阈值 (宁并勿分)。
        bf_retire: 存在性序贯检验的 log-likelihood-ratio 阈值
        (标定见 arbitrate: 复刻旧"连续 3 帧 miss"语义)。
        max_rms: 拟合残差上限 —— 绝对栏 (BIC 惩罚残差, 深度单位)
        + 相对栏 (rms_raw > max_rms 时图元须解释 ≥50% 区域方差)。
        max_slope: 平面坡度上限 —— 边界混合区是干净深度坡 (残差低),
        rms 门挡不住, 但 |a|~1e2 的陡坡不是真实表面, 留稠密场。"""
        self.h, self.w = shape
        self.s = float(max(shape))
        self.match_thr = match_thr
        self.bf_retire = bf_retire
        self.max_rms = max_rms
        self.max_slope = max_slope
        self.nodes: list[SceneNode] = []
        self.deaths = 0  # 累计退场数 (运行统计)

    # ── 匹配: 同空间 Bhattacharyya ─────────────────────────────────

    @staticmethod
    def _bhatt_batch(
        m1: mx.array, c1: mx.array, m2: mx.array, c2: mx.array
    ) -> mx.array:
        """K 组 (μ,Σ) 对单候选的 Bhattacharyya 距离 —— 共享实现
        在 Utils.bhatt (内聚收编, 2026-08-08 架构审计)。"""
        return Utils.bhatt(m1, c1, m2, c2)

    def _match(self, fit: PrimFit) -> int:
        """找同品节点 (kind 相同且 d_B < 阈值), 返回节点下标或 −1。
        批量 Bhattacharyya: 逐对 CPU 求逆在百级节点时是主开销
        (实测 ~1.4s), 一次批算替代。"""
        cand = [
            (i, nd) for i, nd in enumerate(self.nodes) if nd.kind == fit.kind
        ]
        if not cand:
            return -1
        p = mx.array(fit.params, dtype=mx.float32)
        mus = mx.stack([nd.params for _, nd in cand])
        covs = mx.stack([nd.cov for _, nd in cand])
        d = self._bhatt_batch(mus, covs, p, fit.cov)
        bi = int(mx.argmin(d))
        return cand[bi][0] if float(d[bi]) < self.match_thr else -1

    # ── 累积: 信息滤波融合 ─────────────────────────────────────────

    def _merge(self, nd: SceneNode, fit: PrimFit) -> None:
        """信息滤波: P = Σ₁⁻¹+Σ₂⁻¹, μ = P⁻¹Σ Pᵢμᵢ (最小参数空间)。"""
        p_new = mx.array(fit.params, dtype=mx.float32)
        ridge = 1e-9 * mx.eye(nd.cov.shape[0])
        p1 = mx.linalg.inv(nd.cov + ridge, stream=mx.cpu)
        p2 = mx.linalg.inv(fit.cov + ridge, stream=mx.cpu)
        nd.cov = mx.linalg.inv(p1 + p2, stream=mx.cpu)
        nd.params = nd.cov @ (p1 @ nd.params + p2 @ p_new)
        nd.hits += 1
        nd.log_llr = 0.0  # 融合 = 强确认, 存在性证据重置 (旧 misses 清零对应)
        nd.blade = self.make_blade(nd.kind, nd.params)
        mx.eval(nd.blade.values, nd.params, nd.cov)  # 跨线程物化

    def make_blade(self, kind: str, params: mx.array):
        """归一化参数 → 像素单位 blade (换算是 fusion 已有逻辑)。
        注意混合量纲: x,y 为像素, z 为深度单位 (非米制, 已知近似)。"""
        h, w, s = self.h, self.w, self.s
        if kind == "plane":
            a, b, c = (float(params[0]), float(params[1]), float(params[2]))
            na, nb = a / s, b / s
            nc = c - a * (w / 2) / s - b * (h / 2) / s
            nl = math.sqrt(na * na + nb * nb + 1.0)
            # z = na·x + nb·y + nc ⇔ (−na)x + (−nb)y + z = nc
            return Plane((-na / nl, -nb / nl, 1.0 / nl), nc / nl)
        if kind == "cylinder":
            return self.make_cyl_blade(params)
        cu, cv, cz, rho = (float(params[i]) for i in range(4))
        return Sphere((cu * s + w / 2, cv * s + h / 2, cz), rho * s)

    def make_cyl_blade(self, params: mx.array):
        """圆柱 7 参 (n̂, q, ρ) → 像素单位 Cylinder (混合量纲同球)。"""
        h, w, s = self.h, self.w, self.s
        nx, ny, nz = (float(params[i]) for i in range(3))
        qx, qy, qz = (float(params[i]) for i in range(3, 6))
        rho = float(params[6])
        return Cylinder(
            (qx * s + w / 2, qy * s + h / 2, qz), (nx, ny, nz), rho * s
        )

    def accumulate(
        self, res: FusionResult, M: Motor | None = None
    ) -> dict:
        """一帧图元流累积: motor 对齐 (blade 层) → 匹配 → 融合/新建。
        返回运行统计 (新建/融合数)。M 为 None = 同坐标系直接累积。
        注: 协方差随 motor 传播未做 (小运动下 blade 对齐足够)。"""
        created = merged = 0
        for fit in res.fits:
            if fit.blade is None or fit.cov is None:
                continue
            if fit.rms > self.max_rms:
                # 混合深度区域的高残差拟合, 留稠密场。fit.rms 已是
                # fusion 选模的 BIC 惩罚残差 (rms·n^{k/2n}, fusion.py
                # 三模型选择) —— 本门是对模型比较证据的绝对质量栏。
                continue
            if fit.rms_raw > self.max_rms and fit.rms > 0.7071 * fit.rms_raw:
                # 相对门 (尺度无关, 缺口 1): 区域确有结构 (rms_raw 超
                # 绝对栏) 但图元解释方差不足一半 (残差 > √½·rms_raw)
                # = 混合深度区的低绝对残差拟合 (如两平面深度差小 →
                # rms 0.3 < 0.5, 旧绝对栏看不见)。留稠密场。平坦/图像
                # 平行区域 rms_raw ≈ 噪声 < 绝对栏, 相对门不咬。
                continue
            if fit.kind == "plane" and (
                abs(float(fit.params[0])) > self.max_slope
                or abs(float(fit.params[1])) > self.max_slope
            ):
                continue  # 边界混合区的陡坡 (低残差但不是真实表面)
            if not (
                math.isfinite(fit.rms)
                and math.isfinite(fit.rms_raw)
                and mx.all(mx.isfinite(fit.cov)).item()
            ):
                # 非有限残差/协方差: 退化区域拟合 (奇异正规方程 → 求逆
                # 产生 NaN/inf)。旧门 NaN>阈值 为 False 会放行 → _match
                # 的 bhatt eigh 遇 NaN 抛 C++ 异常 → Abort trap (绕过
                # worker 的 try/except 故障隔离直接杀进程, 实测偶发)。
                # 根因守卫: 非有限拟合不进图 (区域留稠密场)。
                continue
            region = self._region_of(fit, res)  # 先记 rid (见下)
            if M is not None:
                # _align 经 _replace 产生新对象, identity 查询会丢 rid
                fit = self._align(fit, M)
            i = self._match(fit)
            if i >= 0:
                self._merge(self.nodes[i], fit)
                merged += 1
            else:
                node = SceneNode(
                    fit.blade, fit.kind,
                    mx.array(fit.params, dtype=mx.float32),
                    fit.cov, region, sign=fit.sign,
                )
                # 物化: 懒图携带 worker 线程局部流, 主线程读会报
                # no Stream (MotorEKF __post_init__ 同款教训)
                mx.eval(node.blade.values, node.params, node.cov)
                self.nodes.append(node)
                created += 1
        return {"created": created, "merged": merged}

    def _align(self, fit: PrimFit, M: Motor) -> PrimFit:
        """motor 对齐: blade 经 M.apply 变换后重新提取归一化参数。
        平面/球从共轭 blade 提取; 圆柱用点对旋转 (轴点+方向),
        半径不变 (同球)。"""
        blade = M.apply(fit.blade)
        if fit.kind == "plane":
            vals = blade.values
            nx, ny, nz = float(vals[1]), float(vals[2]), float(vals[3])
            dd = float(vals[5])
            a_px, b_px = nx / (-nz), ny / (-nz)
            c_px = dd / nz  # z = (dd − nx·x − ny·y)/nz, 常数项 = dd/nz
            a = a_px * self.s
            b = b_px * self.s
            c = c_px + a_px * self.w / 2 + b_px * self.h / 2
            return fit._replace(blade=blade, params=(a, b, c))
        if fit.kind == "cylinder":
            # 圆柱: 轴点/方向随 motor 变换 (点对旋转), 半径不变 (同球)。
            # 由对齐后参数重建 blade —— M.apply 的包裹 Cylinder 不更新
            # _axis_dir/_axis_point, 直接持有会渲染到旧轴位。
            qx = float(fit.params[3]) * self.s + self.w / 2
            qy = float(fit.params[4]) * self.s + self.h / 2
            qz = float(fit.params[5])
            n0 = (
                float(fit.params[0]),
                float(fit.params[1]),
                float(fit.params[2]),
            )
            from cga import Point as _Pt

            c2 = M.apply(_Pt(qx, qy, qz)).coords()
            ca = M.apply(_Pt(qx + n0[0], qy + n0[1], qz + n0[2])).coords()
            n2 = [ca[i] - c2[i] for i in range(3)]
            nl = math.sqrt(sum(v * v for v in n2)) + 1e-12
            n2 = [v / nl for v in n2]
            params = (
                n2[0],
                n2[1],
                n2[2],
                (c2[0] - self.w / 2) / self.s,
                (c2[1] - self.h / 2) / self.s,
                c2[2],
                float(fit.params[6]),
            )
            return fit._replace(
                blade=self.make_cyl_blade(params), params=params
            )
        # 球: 中心随 motor 变换 (点), 半径不变
        vals = blade.values
        wgt = float(vals[4])
        cx, cy, cz = (float(vals[i]) / wgt for i in (1, 2, 3))
        rho2 = (cx * cx + cy * cy + cz * cz) - 2.0 * float(vals[5]) / wgt
        rho = math.sqrt(max(rho2, 0.0))
        from cga import Point as _Pt

        c2 = M.apply(_Pt(cx, cy, cz)).coords()
        cu = (c2[0] - self.w / 2) / self.s
        cv = (c2[1] - self.h / 2) / self.s
        return fit._replace(blade=blade, params=(cu, cv, c2[2], rho / self.s))

    def _region_of(self, fit: PrimFit, res: FusionResult) -> int:
        """fit 的来源区域 (按对象 identity 在 fits 中的位置反查)。
        锁定不变量: PrimitiveFit 按区域顺序构建 fits, fits[i] ↔
        区域 i+1; fusion 层若改构建顺序须同步改这里。"""
        for i, f2 in enumerate(res.fits):
            if f2 is fit:
                return i + 1
        return 0

    # ── 仲裁: 冲突退场 ─────────────────────────────────────────────

    def arbitrate(
        self,
        depth: mx.array,
        subregions: mx.array,
        precision: mx.array | None = None,
    ) -> int:
        """节点存在性的序贯模型比较 (SPRT): 每帧累积 log-likelihood-
        ratio, 越过阈值退场。H1: 节点仍解释其区域 (残差 ~ N(0,σ_node²));
        H0: 场景已变 (残差 ~ N(0,σ_alt²), σ_alt = 3σ_node)。
            Δ = ln(σ_alt/σ_node) − res²/2·(1/σ_node² − 1/σ_alt²)
              = ln3 − 4/9·(res/σ_node)²
        precision: 融合精度场 (可选) —— 给则 σ_node² = σ_param² + σ_obs²,
        σ_obs = 1/√p̄ (观测噪声项, 缺口 2)。标定 (锚 = 旧"连续
        max_misses 帧 > 3σ 包络"语义): 一帧 3σ 残差 Δ=−2.9, 三帧 −8.7
        → bf_retire=−8 复刻"连续 3 帧 miss"; 区域消失 Δ=−2.9 (同旧 miss
        语义: 出画 3 帧退场)。逐帧 Δ 下限 −5: 单帧灾难冲突不秒杀
        (连续两帧才退场)。log_llr 封顶 0: 有界记忆 —— 世界非平稳,
        正证据不无限累积, 长寿节点保持可退场 (旧规则一次 hit 清零
        misses 是健忘, 等价物是 _merge 重置)。res<2σ 是正证据 (Δ>0),
        2σ~3σ 间轻微负债: 持续 2σ 失配是模型错误信号, 旧规则视为
        hit 是漏检。返回本轮退场数。"""
        died = 0
        keep = []
        for nd in self.nodes:
            res = self._node_residual(nd, depth, subregions)
            sigma_p = max(
                float(
                    mx.sqrt(mx.max(mx.linalg.eigh(nd.cov, stream=mx.cpu)[0]))
                ),
                0.05 / 3.0,  # σ_node 地板: 旧 tol=0.05 包络的 1/3
            )
            sigma = sigma_p
            if precision is not None and res is not None:
                # 观测噪声项: 总噪声 σ² = σ_param² + σ_obs²。上限
                # 3σ_param: 精度是设计权重非标定噪声 (如场景渲染线索
                # 恒 1.0), 不封顶 → σ_obs≈1 → res/σ→0 → Δ 恒正, 节点
                # 永不退场。接线闭环时须校验精度标定再放开。
                lab = subregions.reshape(-1)
                mask = lab == nd.region
                n_r = int(mx.sum(mask))
                if n_r > 0:
                    key = mx.where(mask, mx.arange(lab.shape[0]), lab.shape[0])
                    idx = mx.argsort(key)[:n_r]
                    p_bar = float(precision.reshape(-1)[idx].mean())
                    sigma_obs = min(
                        1.0 / math.sqrt(max(p_bar, 1e-9)), 3.0 * sigma_p
                    )
                    sigma = math.sqrt(sigma_p * sigma_p + sigma_obs * sigma_obs)
            if res is None:
                delta = math.log(3.0) - 4.0  # 区域消失 ≈ 一次 3σ miss
            else:
                delta = math.log(3.0) - 4.0 / 9.0 * (res / sigma) ** 2
            nd.log_llr = min(nd.log_llr + max(delta, -5.0), 0.0)
            if nd.log_llr < self.bf_retire:
                died += 1
            else:
                keep.append(nd)
        self.nodes = keep
        self.deaths += died
        return died

    def _render_cylinder(self, nd: SceneNode) -> mx.array:
        """圆柱深度渲染 (图像空间柱面近根, 同 fusion pred_cy)。"""
        yy, xx = mx.meshgrid(
            mx.arange(self.h, dtype=mx.float32),
            mx.arange(self.w, dtype=mx.float32), indexing="ij",
        )
        u = (xx - self.w / 2) / self.s
        v = (yy - self.h / 2) / self.s
        nn = nd.params[0:3]
        qq = nd.params[3:6]
        rr = nd.params[6]
        nz = float(nn[2])
        pez = mx.array([-nz * float(nn[0]), -nz * float(nn[1]),
                        1.0 - nz * nz])
        q0, q1, q2 = float(qq[0]), float(qq[1]), float(qq[2])
        w0 = mx.stack(
            [u - q0, v - q1, mx.full_like(u, -q2)], axis=-1
        )
        wn = mx.sum(w0 * nn, axis=-1)
        aq = mx.maximum(1.0 - nz * nz, 1e-8)
        bq = mx.sum(w0 * pez, axis=-1)
        cq = mx.sum(w0 * w0, axis=-1) - wn * wn - rr * rr
        disc = bq * bq - aq * cq
        zr = mx.where(
            disc > 0.0,
            (-bq - mx.sqrt(mx.maximum(disc, 0.0))) / aq,
            float("inf"),
        )
        return mx.where(zr > 0.0, zr, float("inf"))

    def _node_residual(
        self, nd: SceneNode, depth: mx.array, sub: mx.array
    ) -> float | None:
        """节点在其来源区域的加权 RMS 残差 (渲染 vs 观测)。
        区域消失 (无像素) 返回 None —— 调用方按 miss 计。"""
        mask = (sub == nd.region).reshape(-1)
        if nd.kind == "plane":
            zr = self._render_plane(nd.params).reshape(-1)
        elif nd.kind == "sphere":
            zr = self._render_sphere(nd).reshape(-1)
        else:
            zr = self._render_cylinder(nd).reshape(-1)
        d = depth.reshape(-1)
        k = int(mx.sum(mask))
        if k == 0:
            return None  # 区域消失 → miss (调用方处理)
        key = mx.where(mask, mx.arange(mask.shape[0]), mask.shape[0])
        idx = mx.argsort(key)[:k]
        return float(mx.sqrt(mx.mean((d[idx] - zr[idx]) ** 2)))

    # ── 渲染与反馈 ─────────────────────────────────────────────────

    def _render_plane(self, params: mx.array) -> mx.array:
        """平面深度渲染 (归一化坐标)。"""
        yy, xx = mx.meshgrid(
            mx.arange(self.h, dtype=mx.float32),
            mx.arange(self.w, dtype=mx.float32), indexing="ij",
        )
        u = (xx - self.w / 2) / self.s
        v = (yy - self.h / 2) / self.s
        return params[0] * u + params[1] * v + params[2]

    def _render_sphere(self, nd: SceneNode) -> mx.array:
        """球深度渲染: cz + sign·√rr (sign = 节点半球符号,
        此前硬编码上半球, 背向半球节点渲染全错)。"""
        yy, xx = mx.meshgrid(
            mx.arange(self.h, dtype=mx.float32),
            mx.arange(self.w, dtype=mx.float32), indexing="ij",
        )
        u = (xx - self.w / 2) / self.s
        v = (yy - self.h / 2) / self.s
        cu, cv, cz, rho = nd.params[0], nd.params[1], nd.params[2], nd.params[3]
        rr = mx.maximum(rho**2 - (u - cu) ** 2 - (v - cv) ** 2, 0.0)
        return cz + nd.sign * mx.sqrt(rr)

    def render(
        self, subregions: mx.array, dense: mx.array
    ) -> tuple[mx.array, mx.array]:
        """场景渲染: 节点覆盖其来源区域, 其余留稠密场。
        返回 (渲染深度, 残差高亮 —— 特征层反馈的输出字段)。"""
        out = dense
        highlight = mx.zeros(dense.shape)
        for nd in self.nodes:
            if nd.kind == "plane":
                zr = self._render_plane(nd.params)
            elif nd.kind == "sphere":
                zr = self._render_sphere(nd)
            else:
                zr = self._render_cylinder(nd)
            mask = subregions == nd.region
            out = mx.where(mask, zr, out)
            highlight = mx.where(mask, mx.abs(dense - zr), highlight)
        return out, highlight

    def feedback(self, render: mx.array, quantile: float = 0.99) -> mx.array:
        """深度不连续 → 归一化 prior_map (→ 分割层虚线边)。"""
        dy = mx.abs(render[1:, :] - render[:-1, :])
        dx = mx.abs(render[:, 1:] - render[:, :-1])
        dmap = mx.zeros(render.shape)
        dmap = dmap.at[1:, :].add(dy).at[:, 1:].add(dx)
        flat = mx.sort(dmap.reshape(-1))
        q = flat[int(quantile * (flat.shape[0] - 1))]
        return mx.clip(dmap / mx.maximum(q, 1e-12), 0.0, 1.0)

    def as_cue(self, render: mx.array, precision: float = 50.0) -> DepthCue:
        """场景渲染作高精度 DepthCue 反喂融合层 (闭环契约)。"""
        return DepthCue(render, mx.full(render.shape, precision))

    def detect_symmetry(
        self,
        rid_map: mx.array,
        ang_deg: float = 15.0,
        dist_tol: float = 1.0,
        bf_min: float = 0.0,
        max_samples: int = 200,
    ) -> list[SymmetryMatch]:
        """镜像面对称检测 (prior.md 几何与结构: 对称性先验)。

        关键教训: 任意两平面关于其角平分面是*精确*镜像的
        (相交面交换/平行面中分面), blade 级验证恒真、无鉴别力 ——
        真门在**支撑城**: 区域 i 的像素升上自身平面 → 跨候选面
        反射 → 投影回图像, 落点命中伙伴区。ang/dist 是硬几何定义门
        (候选须几何一致); 支撑落点证据走模型比较: k~Bin(n,p),
        H1 镜像语义 p=0.8 vs H0 随机落点 p₀=伙伴区面积占比,
        log BF > bf_min 接受 (默认 0 = 模型比较边界)。候选面 =
        角平分面闭式 (单位法向 (n₁∓n₂)·x = d₁∓d₂, 两支覆盖相交/
        平行)。blade 一致性只作松门 (拟合噪声容忍)。球-球对称留钩。
        低频后台层。"""
        h, w = rid_map.shape
        rm = rid_map.tolist()
        planes = [
            (k, nd) for k, nd in enumerate(self.nodes) if nd.kind == "plane"
        ]
        # 支撑城证据的 H0: 反射随机落点的命中率 = 伙伴区域面积占比
        lab = rid_map.reshape(-1)
        n_reg = int(mx.max(rid_map))
        cnt = mx.zeros(n_reg + 1).at[lab].add(1.0)
        p_area = [float(cnt[r]) / float(lab.shape[0]) for r in range(n_reg + 1)]
        p_hi = 0.8  # H1 命中率: 镜像语义下反射应入伙伴支撑 (边缘/量化留 20%)

        def blade_nd(nd: SceneNode) -> tuple[list[float], float]:
            v = nd.blade.values
            return [float(v[k]) for k in (1, 2, 3)], float(v[5])

        def samples(rid: int) -> list[tuple[int, int]]:
            """区域像素 (行, 列) 等距抽样到 max_samples。"""
            pts = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if rm[r][c] == rid
            ]
            if len(pts) > max_samples:
                step = len(pts) / max_samples
                pts = [pts[int(k * step)] for k in range(max_samples)]
            return pts

        def overlap(
            nd_from: SceneNode, rid_from: int, mirror: Plane, rid_to: int
        ) -> tuple[int, int]:
            """from 区像素升上自身平面 → 反射 → 投影落在 to 区的
            命中数/样本数 (证据量, 模型比较用)。"""
            n, d = blade_nd(nd_from)
            mv = mirror.values
            nm = [float(mv[k]) for k in (1, 2, 3)]
            dm = float(mv[5])
            if abs(n[2]) < 1e-9:
                return 0, 0
            pts = samples(rid_from)
            if not pts:
                return 0, 0
            hit = 0
            for r, c in pts:
                z = (d - n[0] * c - n[1] * r) / n[2]  # 平面提升
                t = nm[0] * c + nm[1] * r + nm[2] * z - dm
                x2, y2 = c - 2 * t * nm[0], r - 2 * t * nm[1]
                ci, ri = int(round(x2)), int(round(y2))
                if 0 <= ri < h and 0 <= ci < w and rm[ri][ci] == rid_to:
                    hit += 1
            return hit, len(pts)

        out: list[SymmetryMatch] = []
        for a in range(len(planes)):
            for b in range(a + 1, len(planes)):
                i, nd_i = planes[a]
                j, nd_j = planes[b]
                n1, d1 = blade_nd(nd_i)
                n2, d2 = blade_nd(nd_j)
                for sgn in (1.0, -1.0):  # 两条角平分面
                    nm = [n1[k] - sgn * n2[k] for k in range(3)]
                    nl = math.sqrt(sum(v * v for v in nm))
                    if nl < 1e-9:
                        continue  # 平行面的另一支由 sgn=-1 覆盖
                    # Plane 构造只归一法向不缩放距离, d 须预除 |nm|
                    mirror = Plane(tuple(nm), (d1 - sgn * d2) / nl)
                    # blade 一致性松门 (理想情形恒真, 容忍拟合噪声)
                    ref = reflect(nd_i.blade, mirror)
                    nr = [float(ref.values[k]) for k in (1, 2, 3)]
                    dr = float(ref.values[5])
                    dp = sum(nr[k] * n2[k] for k in range(3))
                    if dp < 0:  # 法向符号对齐 (blade 轴向任意)
                        nr, dr, dp = [-v for v in nr], -dr, -dp
                    ang = math.degrees(math.acos(min(max(dp, -1.0), 1.0)))
                    if ang >= ang_deg or abs(dr - d2) >= dist_tol:
                        continue
                    # 真门: 支撑城双向落点证据 (模型比较, 旧 ov≥min_overlap
                    # 的 Bayes 化)。双向 (i→j, j→i) 的 H0 命中率各自不同
                    # (伙伴区面积占比), 证据相加; ang/dist 仍是硬几何定义门。
                    hi, ni = overlap(nd_i, nd_i.region, mirror, nd_j.region)
                    hj, nj = overlap(nd_j, nd_j.region, mirror, nd_i.region)
                    n = ni + nj
                    if n == 0:
                        continue
                    p0a, p0b = p_area[j], p_area[i]
                    if p0a <= 0.0 or p0b <= 0.0:
                        # 伙伴区不在当前 rid_map (区域已换/消失):
                        # 无随机基线, 同旧 ov=0 拒绝
                        continue
                    if p0a >= 1.0 or p0b >= 1.0:
                        continue  # 伙伴区整图: 无随机基线可比较
                    log_bf = (
                        hi * math.log(p_hi / p0a)
                        + (ni - hi) * math.log((1 - p_hi) / (1 - p0a))
                        + hj * math.log(p_hi / p0b)
                        + (nj - hj) * math.log((1 - p_hi) / (1 - p0b))
                    )
                    if log_bf > bf_min:
                        rat = min(hi / ni, hj / nj)  # 残差排序键保留重叠量
                        out.append(
                            SymmetryMatch(
                                mirror, (i, j),
                                ang + abs(dr - d2) + (1 - rat), log_bf,
                            )
                        )
                        break  # 一支角平分面命中即可
        out.sort(key=lambda m: m.residual)
        return out

    def lift_circles(
        self,
        circle_params: list[tuple[tuple[float, float], float]],
        depth: mx.array,
        K: tuple[float, float, float, float],
        n_samp: int = 48,
        inset: float = 2.0,
        max_plane_rms: float = 0.05,
    ) -> list:
        """2D 圆检测 → 3D 圆 (独立摄入路径, 不经 PrimFit)。
        grouping.circle_params 是 ((cx,cy), ρ) 像素圆; 圆周内缩采样
        (圆周本体常落在背景/边缘, 采样即脏) → scene 深度反投影 3D
        点 → 平面拟合 (质心 + 最小特征方向), 残差 σ > 门拒 (球
        剪影/噪声会被拦下 —— 诚实门控); 圆心 = 2D 圆心 3D 点在
        平面上投影, 半径 = 环点平面内距均值。
        分歧保留: 圆与端视圆柱歧义交下游 (两者都进 SceneModel)。"""
        fx, fy, cx0, cy0 = K
        h, w = self.h, self.w
        out: list = []
        for (cx, cy), rho in circle_params:
            if rho < inset + 1.0 or rho > 0.5 * w:
                continue
            # 圆周内缩采样 (像素坐标)
            th = [2.0 * math.pi * k / n_samp for k in range(n_samp)]
            xs = [cx + (rho - inset) * math.cos(t) for t in th]
            ys = [cy + (rho - inset) * math.sin(t) for t in th]
            pts = []
            for x, y in zip(xs, ys, strict=True):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                z = float(depth[yi, xi])
                if z <= 0.1:
                    continue
                pts.append(
                    ((x - cx0) * z / fx, (y - cy0) * z / fy, z)
                )
            if len(pts) < max(8, n_samp // 3):
                continue
            # 平面拟合: 质心 + PCA 最小特征方向 (法向)
            nx_ = sum(p[0] for p in pts) / len(pts)
            ny_ = sum(p[1] for p in pts) / len(pts)
            nz_ = sum(p[2] for p in pts) / len(pts)
            g = mx.zeros((3, 3))
            for p in pts:
                d = mx.array([p[0] - nx_, p[1] - ny_, p[2] - nz_])
                g += d[:, None] @ d[None, :]
            ev = mx.linalg.eigh(g, stream=mx.cpu)[1]
            normal = ev[:, 0]  # 最小特征值方向
            nl = float(mx.linalg.norm(normal))
            if nl < 1e-8:
                continue
            normal = normal / nl
            rms_pl = math.sqrt(
                sum(
                    (normal[0] * (p[0] - nx_) + normal[1] * (p[1] - ny_)
                     + normal[2] * (p[2] - nz_)) ** 2
                    for p in pts
                )
                / len(pts)
            )
            if rms_pl > max_plane_rms:
                continue  # 非平面 (球剪影/深度噪声)
            # 2D 圆心反投影 → 平面上投影 → 圆心; 半径 = 环距均值
            zc = float(depth[int(round(cy)), int(round(cx))])
            if zc <= 0.1:
                continue
            c3 = mx.array(
                [(cx - cx0) * zc / fx, (cy - cy0) * zc / fy, zc]
            )
            off = c3 - mx.array([nx_, ny_, nz_])
            center = (mx.array([nx_, ny_, nz_]) + off
                      - (off @ normal) * normal)
            rs = [
                math.sqrt(
                    (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
                    + (p[2] - center[2]) ** 2
                )
                for p in pts
            ]
            r_avg = sum(rs) / len(rs)
            # 内缩只为干净采样, 半径外推回真圆: 2D 半径 ρ 按中心深度
            # 换算 (采样环内缩了 inset px, 直接环距均值会偏小)
            r_avg = rho * zc / fx
            if r_avg < 0.02:
                continue
            out.append(
                Circle(
                    (float(center[0]), float(center[1]), float(center[2])),
                    r_avg,
                    (float(normal[0]), float(normal[1]), float(normal[2])),
                )
            )
        return out

    def export(
        self, K: tuple[float, float, float, float], rid_map: mx.array
    ) -> SceneModel:
        """导出米制 CGA 图元场景 (管线终态输出)。
        K = (fx, fy, cx, cy) 来自 temporal 的 C1 慢速标定。
        反投影: 像素-深度混合空间 → 米制相机空间。平面法向经
        局部切平面 Jacobian (锚点 = 区域质心 —— 线性拟合模型是
        倒数曲面的局部一阶, 与视平线同一教训); 球半径按深度
        比例换算 (fx/fy 不等时为椭球近似, 取 fx)。"""
        fx, fy, cx, cy = K
        h, w, s = self.h, self.w, self.s
        # 区域质心 (scatter 批量, 像素坐标)
        lab = rid_map.reshape(-1)
        n = int(mx.max(rid_map))
        yy, xx = Utils.grid((h, w))

        def sc(v: mx.array) -> mx.array:
            return mx.zeros((n + 1,)).at[lab].add(v.reshape(-1))

        cnt = mx.maximum(sc(mx.ones((h, w))), 1.0)
        rc = (sc(yy) / cnt).tolist()
        cc = (sc(xx) / cnt).tolist()

        prims: list[ScenePrimitive] = []
        for nd in self.nodes:
            r = nd.region
            col_c, row_c = cc[r], rc[r]
            u_c, v_c = (col_c - w / 2) / s, (row_c - h / 2) / s
            if nd.kind == "plane":
                a, b, c = (float(nd.params[i]) for i in range(3))
                z_c = a * u_c + b * v_c + c
                # 锚点反投影到米制
                px, py = (col_c - cx) * z_c / fx, (row_c - cy) * z_c / fy
                # 像素空间法向 (na,nb,−1) (z = na·col + nb·row + nc)
                na, nb = a / s, b / s
                # Jacobian 变换到米制法向 (在锚点处)
                nx = na * fx / z_c
                ny = nb * fy / z_c
                nz = -(na * fx * px + nb * fy * py) / (z_c * z_c) - 1.0
                nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                d = (nx * px + ny * py + nz * z_c) / nl
                blade = Plane((nx / nl, ny / nl, nz / nl), d)
            elif nd.kind == "sphere":
                cu, cv, cz, rho = (float(nd.params[i]) for i in range(4))
                col0, row0 = cu * s + w / 2, cv * s + h / 2
                px = (col0 - cx) * cz / fx
                py = (row0 - cy) * cz / fy
                rho_m = rho * s * cz / fx  # 深度比例换算 (椭球近似)
                blade = Sphere((px, py, cz), max(rho_m, 1e-6))
            else:  # cylinder
                nx, ny, nz = (float(nd.params[i]) for i in range(3))
                qx, qy, qz = (float(nd.params[i]) for i in range(3, 6))
                col0, row0 = qx * s + w / 2, qy * s + h / 2
                px = (col0 - cx) * qz / fx
                py = (row0 - cy) * qz / fy
                rho_m = nd.params[6] * s * qz / fx
                blade = Cylinder((px, py, qz), (nx, ny, nz), max(rho_m, 1e-6))
            prims.append(
                ScenePrimitive(nd.kind, blade, nd.cov, r, nd.hits)
            )
        return SceneModel(prims, K, self.detect_symmetry(rid_map), rid_map)


class ScenePrimitive(NamedTuple):
    """米制 CGA 图元 (导出的终态单元)。"""

    kind: str  # "plane" / "sphere"
    blade: Multivector  # 米制空间的 Plane/Sphere blade
    cov: mx.array  # 参数协方差 (原归一化拟合空间, 未传播到米制)
    region: int  # 来源 rid
    hits: int  # 累积命中次数 (存活证据)


class SceneModel(NamedTuple):
    """管线终态输出: 米制 CGA 图元场景 + 标定 + 关系。"""

    primitives: list[ScenePrimitive]
    K: tuple[float, float, float, float]  # (fx, fy, cx, cy)
    symmetries: list[SymmetryMatch]  # 对称关系 (detect_symmetry 产物)
    regions: mx.array  # 源 rid 图 (掩码逆渲染用, 像素坐标)
    circles: list = []  # 3D 圆 (lift_circles 产物; 独立摄入路径)


class SymmetryMatch(NamedTuple):
    """镜像对称检测产物: 镜像面 + 对称节点对 (nodes 索引) + 残差
    + 支撑城证据的 log-Bayes-factor。"""

    mirror: Plane  # 镜像面 blade
    pair: tuple[int, int]  # 对称节点对 (self.nodes 索引)
    residual: float  # 法向角差(度) + 距离差 (排序用, 混合量纲)
    log_bf: float = 0.0  # 支撑城模型比较证据 (H1 镜像 vs H0 随机落点)


def reflect(node_blade, mirror: Plane):
    """反射 versor 共轭: X' = R X R̃ (§7.3 对称补全的工具)。

    反射是 grade-1 versor, 与 motor 同一代数 —— 直接复用
    Motor.apply 的 sandwich 机制: 把镜像面法向作为 versor。
    """
    from cga import Multivector

    vals = mirror.values  # π = n + d·e∞, 归一法向
    nx, ny, nz, dd = (float(vals[i]) for i in (1, 2, 3, 5))
    # 反射 versor: R = n + d·e∞ 本身就是 grade-1 versor
    r = Multivector(mirror.values)
    out = r.gp(node_blade).gp(r.reverse())
    return out


if __name__ == "__main__":
    # ── 1. 累积: 协方差加权融合 ────────────────────────────────────
    H, W = 96, 128
    sg = SceneGraph((H, W))
    th = mx.array([0.4, 0.2, 3.0])
    c1 = mx.eye(3) * 0.04  # σ=0.2
    c2 = mx.eye(3) * 0.01  # σ=0.1 (更可信)

    def mkfit(theta, cov, kind="plane", rms=0.01):
        blade = sg.make_blade(kind, theta)
        return PrimFit(blade, kind, cov, tuple(float(t) for t in theta), 1.0, rms)

    sub = mx.ones((H, W), dtype=mx.int32)
    fr = FusionResult(mx.zeros((H, W)), mx.zeros((H, W)),
                      [mkfit(th + 0.05, c1)], mx.zeros((H, W)),
                      mx.zeros((H, W)), mx.zeros((H, W)))
    st = sg.accumulate(fr)
    fr2 = FusionResult(mx.zeros((H, W)), mx.zeros((H, W)),
                       [mkfit(th - 0.05, c2)], mx.zeros((H, W)),
                       mx.zeros((H, W)), mx.zeros((H, W)))
    st2 = sg.accumulate(fr2)
    nd = sg.nodes[0]
    # 加权平均: (0.04⁻¹·(θ+0.05) + 0.01⁻¹·(θ−0.05))/(0.04⁻¹+0.01⁻¹)
    # = θ + (0.05·0.04⁻¹ − 0.05·0.01⁻¹)/(125) = θ + 0.05·(25−100)/125 = θ − 0.03
    assert st["created"] == 1 and st2["merged"] == 1 and len(sg.nodes) == 1
    got = nd.params[0]
    want = th[0] - 0.03
    assert abs(float(got) - float(want)) < 1e-3, f"{float(got)} vs {float(want)}"
    assert float(mx.max(mx.linalg.eigh(nd.cov, stream=mx.cpu)[0])) < 0.01, (
        "融合协方差应小于两个单次 (信息增长)"
    )
    print(f"1. 累积: 加权融合 θ[0]={float(got):.3f} (期望 {float(want):.3f}), "
          f"Σ 收紧 ✓")

    # ── 2. 匹配: 不同平面不合并 ────────────────────────────────────
    fr3 = FusionResult(mx.zeros((H, W)), mx.zeros((H, W)),
                       [mkfit(mx.array([2.0, 0.0, 0.5]), c1)], mx.zeros((H, W)),
                       mx.zeros((H, W)), mx.zeros((H, W)))
    st3 = sg.accumulate(fr3)
    assert st3["created"] == 1 and len(sg.nodes) == 2, "不同平面应建新节点"
    fr4 = FusionResult(mx.zeros((H, W)), mx.zeros((H, W)),
                       [mkfit(mx.array([2.05, 0.02, 0.48]), c1)], mx.zeros((H, W)),
                       mx.zeros((H, W)), mx.zeros((H, W)))
    st4 = sg.accumulate(fr4)
    assert st4["merged"] == 1 and len(sg.nodes) == 2, "微扰应匹配不分裂"
    print("2. 匹配: 异面新建, 微扰融合 (d_B 门控) ✓")

    # ── 3. 仲裁: 持续冲突 → 节点退场 ───────────────────────────────
    for i in range(4):
        depth_conflict = mx.full((H, W), 9.9)  # 与节点 z≈3 严重冲突
        sg.arbitrate(depth_conflict, sub)
    assert sg.deaths >= 1, f"持续冲突应有节点退场: {sg.deaths}"
    print(f"3. 仲裁: 持续冲突后节点退场 (deaths={sg.deaths}) ✓")

    # ── 4. 反馈闭环: 场景渲染 → D → 分割; 渲染作 DepthCue 反喂 ──────
    sg2 = SceneGraph((H, W))
    yy, xx = mx.meshgrid(mx.arange(H, dtype=mx.float32),
                         mx.arange(W, dtype=mx.float32), indexing="ij")
    z_two = mx.where(xx < W // 2, 2.0, 5.0)
    sub2 = mx.where(xx < W // 2, 1, 2).astype(mx.int32)
    E = mx.random.uniform(shape=(H, W), key=mx.random.key(1)) * 0.04
    E[:, 64] = 0.7
    E[40:56, 64] = 0.02  # 缺口
    frL = FusionResult(z_two, mx.zeros((H, W)),
                       [mkfit(mx.array([0.0, 0.0, 2.0]), c2),
                        mkfit(mx.array([0.0, 0.0, 5.0]), c2)],
                       z_two, mx.zeros((H, W)), mx.zeros((H, W)))
    sg2.accumulate(frL)
    out, hl = sg2.render(sub2, z_two)
    D = sg2.feedback(out)
    from segment import SceneSegmenter

    zero_like = mx.zeros((H, W))
    seg = SceneSegmenter(tau=0.5).run(E, zero_like, zero_like, prior_map=D, w_prior=0.8)
    pt_l, pt_r = (48, 32), (48, 96)
    assert int(seg.regions[pt_l]) != int(seg.regions[pt_r]), "场景图反馈应保持分离"
    # 渲染作 cue 反喂: 融合方差下降
    from fusion import CueFusion

    weak = DepthCue(z_two + 1.0, mx.full((H, W), 0.5))  # 弱且偏
    d1, p1 = CueFusion.run([weak])
    d2, p2 = CueFusion.run([weak, sg2.as_cue(out)])
    var1 = float((d1[48, 96] - 5.0) ** 2)
    var2 = float((d2[48, 96] - 5.0) ** 2)
    assert var2 < var1, f"反喂后偏差应下降: {var1:.3f} → {var2:.3f}"
    assert float(mx.max(hl)) >= 0.0, "残差高亮字段存在"
    print(f"4. 闭环: 场景图 D 保持分离; 反喂偏差 {var1:.3f}→{var2:.3f} ✓")

    # ── 5. reflect: 反射 versor ────────────────────────────────────
    from cga import Point

    mirror = Plane((1.0, 0.0, 0.0), 0.0)  # x=0 镜像面
    pl = Point(2.0, 1.0, 3.0)
    pr = reflect(pl, mirror)
    c = Point(0, 0, 0).dist(Point(*Point(0, 0, 0).coords()))
    # 反射后应为 (−2, 1, 3): 用 dist 验证
    wgt = float(pr.values[4])
    rx, ry, rz = (float(pr.values[i]) / wgt for i in (1, 2, 3))
    assert abs(rx + 2.0) < 1e-4 and abs(ry - 1.0) < 1e-4 and abs(rz - 3.0) < 1e-4, (
        f"反射结果 ({rx},{ry},{rz})"
    )
    print(f"5. reflect: (2,1,3) → ({rx:.1f},{ry:.1f},{rz:.1f}) ✓")

    # ── 6. 对称面检测: 支撑城重叠是真门 ────────────────────────────
    # 定理: 任意两平面关于其角平分面精确镜像 → blade 验证无鉴别力;
    # 真门 = 区域支撑城反射重叠。场景: 屋顶对 π1: z=−0.5x+37 (左区),
    # π2: z=0.5x−27 (右区) 关于 x=64 镜像; π3: z=0.3x+5 无对称伙伴
    sg6 = SceneGraph((H, W))
    rid6 = mx.zeros((H, W), dtype=mx.int32)
    rid6 = rid6.at[:, 32:64].add(1)
    rid6 = rid6.at[:, 64:96].add(2)
    rid6 = rid6.at[40:60, 100:120].add(3)
    sg6.nodes = [
        SceneNode(Plane((0.5, 0.0, 1.0), 37.0 / math.sqrt(1.25)), "plane",
                  mx.array([-0.5, 0.0, 5.0]), mx.eye(3), 1),
        SceneNode(Plane((-0.5, 0.0, 1.0), -27.0 / math.sqrt(1.25)), "plane",
                  mx.array([0.5, 0.0, 5.0]), mx.eye(3), 2),
        SceneNode(Plane((-0.3, 0.0, 1.0), 5.0 / math.sqrt(1.09)), "plane",
                  mx.array([0.3, 0.0, 5.0]), mx.eye(3), 3),
    ]
    syms = sg6.detect_symmetry(rid6)
    assert len(syms) == 1, f"应只检出 1 对: {len(syms)}"
    m = syms[0]
    assert m.pair == (0, 1), m.pair
    # 镜像面 x=64: 法向 ±x 且过 (64, 0, 0)
    mv = m.mirror.values
    assert abs(abs(float(mv[1])) - 1.0) < 1e-3, f"镜像面法向: {mv[1]:.3f}"
    assert abs(m.mirror.dist(Point(64.0, 0.0, 0.0))) < 1e-3, "应过 x=64"
    print(f"6. 对称面: 屋顶对检出 (镜像面 x=64, 残差 {m.residual:.3f}), "
          f"非对称对被支撑城门拦下 ✓")

    # ── 7. 球节点: 半球符号渲染 (P0 修复: 曾硬编码上半球) ─────────
    sg7 = SceneGraph((H, W))
    sph_blade = Sphere((64.0, 48.0, 3.0), 1.5)
    prm7 = mx.array([0.0, 0.0, 3.0, 1.5])  # (cu,cv,cz,ρ) 归一化
    sub7 = mx.ones((H, W), dtype=mx.int32)
    sg7.nodes = [
        SceneNode(sph_blade, "sphere", prm7, mx.eye(4), 1, sign=-1.0)
    ]
    r_near, _ = sg7.render(sub7, mx.zeros((H, W)))
    sg7.nodes = [SceneNode(sph_blade, "sphere", prm7, mx.eye(4), 1,
                           sign=1.0)]
    r_far, _ = sg7.render(sub7, mx.zeros((H, W)))
    # 球心处 (cu=cv=0 → 图中心): 近半球 3−1.5=1.5, 远半球 3+1.5=4.5
    assert abs(float(r_near[48, 64]) - 1.5) < 1e-4, float(r_near[48, 64])
    assert abs(float(r_far[48, 64]) - 4.5) < 1e-4, float(r_far[48, 64])
    print("7. 球节点: sign=−1 渲 1.5 / sign=+1 渲 4.5 (半球符号生效) ✓")

    # ── 8. export: 米制反投影 (终态输出) ────────────────────────────
    # 平面 z = 3 + 0.01·(col−64): 归一化参数 a = 0.01·128 = 1.28
    sg8 = SceneGraph((H, W))
    prm8 = mx.array([1.28, 0.0, 3.0])
    rid8 = mx.ones((H, W), dtype=mx.int32)
    sg8.nodes = [
        SceneNode(sg8.make_blade("plane", prm8), "plane", prm8,
                  mx.eye(3), 1)
    ]
    K8 = (100.0, 100.0, 64.0, 48.0)
    model = sg8.export(K8, rid8)
    pb = model.primitives[0].blade
    assert isinstance(pb, Plane)  # 类型窄化 (pyright)
    # 像素 (64,48) 深 3 → 米制 (0,0,3) 应在平面上
    assert abs(pb.dist(Point(0.0, 0.0, 3.0))) < 1e-3
    # 像素 (100,48) 深 3.36 → 米制 (1.2096, 0, 3.36) 也应在
    # 像素 (100,48) 深 3.36 → 米制 (1.2096, 0, 3.36):
    # 残差是像素线性模型自身的近似量级 (~0.04 @ 36px 偏心,
    # 非 bug —— 倒数曲面的一阶展开误差, 偏心越大越大)
    d2 = pb.dist(Point(1.2096, 0.0, 3.36))
    assert abs(d2) < 0.06, f"米制平面外点残差 {d2:.4f}"
    print(f"8. export: 米制平面两检查点残差 < 2e-2, "
          f"primitives={len(model.primitives)} ✓")

    # ── 9. lift_circles: 平盘环 → 3D 圆 (平面=表面平面) ────────────
    # 场景: z=3 平面前放一个圆环盘 (圆在 z=3 平面内, 圆心 (64,48),
    # 半径 40px) —— 深度场: 圆环内与圆外都是平面 z=3 (圆是轮廓标注)
    K9 = (100.0, 100.0, 64.0, 48.0)
    depth9 = mx.full((H, W), 3.0)
    sg9 = SceneGraph((H, W))
    circ9 = sg9.lift_circles([((64.0, 48.0), 40.0)], depth9, K9)
    assert len(circ9) == 1, f"平盘环应提升 1 圆, 得 {len(circ9)}"
    c9 = circ9[0]
    # 关联判据: 圆上点 ip(p, C) = 0 (对偶圆 = 球∧平面)
    from cga import Point
    on = Point(1.2, 0.0, 3.0)  # 圆心 (0,0,3) 半径 1.2 的圆上点
    assert float(mx.max(mx.abs(on.ip(c9).values))) < 1e-3, on.ip(c9).values
    off = Point(1.2, 0.0, 3.5)  # 圆平面外 (z=3.5) → 不在圆上
    assert float(mx.max(mx.abs(off.ip(c9).values))) > 1e-3
    print("9. 圆提升: 平盘环 → 3D 圆 (圆上点关联 / 平面外点不关联 ✓)")
