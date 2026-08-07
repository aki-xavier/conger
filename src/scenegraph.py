"""三维重建层 (3D Reconstruction, flow.md §7): CGA 图元场景图。

模块流程:

  图元流 (fusion.PrimFit: 归一化拟合空间 params + cov) + motor
       │  accumulate: 跨帧同名图元的信息滤波融合 (协方差加权),
       │    Bhattacharyya 门控匹配 (同空间, d_B < 2 宁并勿分),
       │    匹配不上 → 新节点 (提升); motor 对齐走 blade 层
       ▼  SceneNode (blade + params + Σ + 运行统计)
  arbitrate: 节点残差持续 > 3σ 包络 → 退场, 区域交还残差场
  render: 场景深度渲染 + 残差高亮 → feedback: prior_map (→ 分割层),
       场景渲染作高精度 DepthCue 反喂融合层 (闭环演示)
  reflect: 反射 versor 共轭 (§7.3 对称补全的工具; 检测器留钩)

  [留钩] 关联边 (meet/op/ip 关联代数); 曼哈顿正交联合精化;
         对称面检测; motor 协方差传播 (小运动下 blade 对齐足够);
         特征层反馈只留输出字段。

**参数空间约定 (最大的坑)**: 节点携带"空间标签 = 归一化拟合空间"
(fusion 的 (u,v,z), 减中心÷边长)。信息滤波与 Bhattacharyya 全部
在该空间进行; 只在输出/渲染/blade 构造时换像素单位。
离线层, 不做逐帧承诺。
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from cga import Motor, Plane, Sphere
from fusion import DepthCue, FusionResult, PrimFit
from vbgmm import VBGMM


@dataclass(slots=True)
class SceneNode:
    """场景图节点: 图元 + 最小参数 (归一化拟合空间) + 协方差 + 统计。"""

    blade: object  # Plane/Sphere blade (像素单位, 输出用)
    kind: str  # "plane" / "sphere"
    params: mx.array  # (P,) 归一化拟合空间参数
    cov: mx.array  # (P,P) 参数协方差 (同空间)
    region: int  # 来源子区域 id
    hits: int = 1  # 被匹配融合的次数
    misses: int = 0  # 连续冲突帧数 (仲裁用)


class SceneGraph:
    """CGA 图元场景图: 累积 / 仲裁 / 渲染 / 反馈。"""

    def __init__(
        self,
        shape: tuple[int, int],
        match_thr: float = 2.0,
        max_misses: int = 3,
    ):
        """shape: 图像 (H,W) —— 归一化拟合空间的尺度基准。
        match_thr: Bhattacharyya 匹配阈值 (宁并勿分)。
        max_misses: 连续冲突多少帧后节点退场。"""
        self.h, self.w = shape
        self.s = float(max(shape))
        self.match_thr = match_thr
        self.max_misses = max_misses
        self.nodes: list[SceneNode] = []
        self.deaths = 0  # 累计退场数 (运行统计)

    # ── 匹配: 同空间 Bhattacharyya ─────────────────────────────────

    def _bhatt(self, m1, c1, m2, c2) -> float:
        """两组 (μ,Σ) 的 Bhattacharyya 距离 (同参数空间内)。"""
        cb = (c1 + c2) * 0.5
        dm = (m1 - m2)[:, None]
        inv = mx.linalg.inv(cb + 1e-9 * mx.eye(cb.shape[0]), stream=mx.cpu)
        t1 = float(dm.T @ inv @ dm) / 8.0
        t2 = 0.5 * (
            VBGMM.logdet_spd(cb)
            - 0.5 * (VBGMM.logdet_spd(c1) + VBGMM.logdet_spd(c2))
        )
        return t1 + t2

    def _match(self, fit: PrimFit) -> int:
        """找同品节点 (kind 相同且 d_B < 阈值), 返回节点下标或 −1。"""
        best, bi = self.match_thr, -1
        p = mx.array(fit.params, dtype=mx.float32)
        for i, nd in enumerate(self.nodes):
            if nd.kind != fit.kind:
                continue
            d = self._bhatt(nd.params, nd.cov, p, fit.cov)
            if d < best:
                best, bi = d, i
        return bi

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
        nd.misses = 0
        nd.blade = self.make_blade(nd.kind, nd.params)

    def make_blade(self, kind: str, params: mx.array):
        """归一化参数 → 像素单位 blade (换算是 fusion 已有逻辑)。"""
        h, w, s = self.h, self.w, self.s
        if kind == "plane":
            a, b, c = (float(params[0]), float(params[1]), float(params[2]))
            na, nb = a / s, b / s
            nc = c - a * (w / 2) / s - b * (h / 2) / s
            nl = math.sqrt(na * na + nb * nb + 1.0)
            return Plane((na / nl, nb / nl, -1.0 / nl), nc / nl)
        cu, cv, cz, rho = (float(params[i]) for i in range(4))
        return Sphere((cu * s + w / 2, cv * s + h / 2, cz), rho * s)

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
            if M is not None:
                fit = self._align(fit, M)
            i = self._match(fit)
            if i >= 0:
                self._merge(self.nodes[i], fit)
                merged += 1
            else:
                self.nodes.append(
                    SceneNode(
                        fit.blade, fit.kind,
                        mx.array(fit.params, dtype=mx.float32),
                        fit.cov, self._region_of(fit, res),
                    )
                )
                created += 1
        return {"created": created, "merged": merged}

    def _align(self, fit: PrimFit, M: Motor) -> PrimFit:
        """motor 对齐: blade 经 M.apply 变换后重新提取归一化参数。"""
        blade = M.apply(fit.blade)
        if fit.kind == "plane":
            vals = blade.values
            nx, ny, nz = float(vals[1]), float(vals[2]), float(vals[3])
            dd = float(vals[5])
            a_px, b_px = nx / (-nz), ny / (-nz)
            c_px = -dd / nz
            a = a_px * self.s
            b = b_px * self.s
            c = c_px + a_px * self.w / 2 + b_px * self.h / 2
            return fit._replace(blade=blade, params=(a, b, c))
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
        """fit 的来源区域 (按参数在 fits 列表中的位置反查)。"""
        for i, f2 in enumerate(res.fits):
            if f2 is fit:
                return i + 1
        return 0

    # ── 仲裁: 冲突退场 ─────────────────────────────────────────────

    def arbitrate(self, depth: mx.array, subregions: mx.array) -> int:
        """节点残差 > 3σ 包络持续 max_misses 帧 → 退场。
        返回本轮退场数。σ 包络 = 节点协方差的主特征尺度。"""
        died = 0
        keep = []
        for nd in self.nodes:
            res = self._node_residual(nd, depth, subregions)
            sigma = float(mx.sqrt(mx.max(mx.linalg.eigh(nd.cov, stream=mx.cpu)[0])))
            tol = max(3.0 * sigma, 0.05)
            if res > tol:
                nd.misses += 1
            else:
                nd.misses = 0
            if nd.misses >= self.max_misses:
                died += 1
            else:
                keep.append(nd)
        self.nodes = keep
        self.deaths += died
        return died

    def _node_residual(self, nd: SceneNode, depth: mx.array, sub: mx.array) -> float:
        """节点在其来源区域的加权 RMS 残差 (渲染 vs 观测)。"""
        mask = (sub == nd.region).reshape(-1)
        if nd.kind == "plane":
            zr = self._render_plane(nd.params).reshape(-1)
        else:
            zr = self._render_sphere(nd.params).reshape(-1)
        d = depth.reshape(-1)
        k = int(mx.sum(mask))
        if k == 0:
            return 0.0
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

    def _render_sphere(self, params: mx.array) -> mx.array:
        """球深度渲染 (上半球, 朝向相机)。"""
        yy, xx = mx.meshgrid(
            mx.arange(self.h, dtype=mx.float32),
            mx.arange(self.w, dtype=mx.float32), indexing="ij",
        )
        u = (xx - self.w / 2) / self.s
        v = (yy - self.h / 2) / self.s
        cu, cv, cz, rho = params[0], params[1], params[2], params[3]
        rr = mx.maximum(rho**2 - (u - cu) ** 2 - (v - cv) ** 2, 0.0)
        return cz - mx.sqrt(rr)

    def render(
        self, subregions: mx.array, dense: mx.array
    ) -> tuple[mx.array, mx.array]:
        """场景渲染: 节点覆盖其来源区域, 其余留稠密场。
        返回 (渲染深度, 残差高亮 —— 特征层反馈的输出字段)。"""
        out = dense
        highlight = mx.zeros(dense.shape)
        for nd in self.nodes:
            zr = (
                self._render_plane(nd.params)
                if nd.kind == "plane"
                else self._render_sphere(nd.params)
            )
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
    assert sg.deaths >= 1, f"3 帧冲突应有节点退场: {sg.deaths}"
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
