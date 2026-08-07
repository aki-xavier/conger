"""知觉组织层 (Perceptual Grouping): 像素域概率边缘图 → 2D CGA 图元。

设计见 docs/flow.md §1。输入: 边缘似然 L_e (VBGMM 后验, 不硬阈值)、
法向场 mean_ori。输出: edgel 集 (亚像素), 轮廓链 (亲和图的高亲和
链), 链段 line blade, 跨缺口补全概率 (不硬连接) 与高置信补全弧段
的 circle blade, T 结与遮挡偏序。

工作空间: 2D 共形子代数 —— 图像点嵌入 5D CGA 的 z=0 平面
(x=col, y=row), 圆/线保持在子代数内。数值纪律 (flow.md §0.3):
blade 输出与残差验证走 cga 的 float64 标量通道; 向量化成对几何
只涉及局部小量差 (≤ link_radius), 无远原点抵消, 用 float32。
残差 (d²−ρ²)/(2ρ) 即归一化对偶球 ip 残差, 近圆处 ≈ 径向距离;
CGA blade 只承载输出图元 (稀疏通用语), 不进逐对循环。

模块流程 (run() 总装):

  L_e (边缘似然, 不硬阈值) + mean_ori (法向场)
       │  ① extract_edgels: 法向 NMS (复用 EdgePrior 预计算
       │     gather) + 抛物线亚像素顶点 → edgel 集 (pos/normal/
       │     tangent/strength); near_pairs 网格桶去重 (强者留)
       ▼
  ② affinity: near_pairs 候选对 (≤ link_radius) → 候选共圆
     (两法向线交点为圆心) / 平行退化切向线残差, res_max 钳顶,
     res_floor 吸收定位噪声 → w = 邻近性 × exp(−κ·共圆残差)
       ▼
  ③ group: 每 edgel 沿切向前/后各取最高亲和伙伴连边 (度≤2 →
     链必为路径; 不要求互选) → 度1端点起走出有序链 (闭环二轮
     兜底), ≥ min_chain 留链
       ▼
  ④ 每链两端点提升 CGA 点 → line blade (p1∧p2∧e∞)
       ▼
  ⑤ complete: 异链端点两两 (≤ gap_max) 复用同一套邻近×共圆
     残差 → p(连续) 排序输出 (不硬连接); ≥ complete_thr 的弧段
     额外拟合 circle blade
       ▼
  ⑥ detect_t_junctions: 折线相交 + 链端点延长线两类候选 →
     局部切向 ±臂支撑统计 (带内·死区外: 数量+延伸) → 一侧通过
     一侧中断 = T 结 (front≻behind 偏序); 双臂皆通 = X, 不出偏序;
     dedupe 3px 去重
       ▼
  GroupingResult (edgels / chains / lines / completions /
  circles / t_junctions)
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import mlx.core as mx

from cga import circle, line, point
from cga.multivector import Multivector
from edgemap import EdgePrior
from utils import Utils


class Edgels(NamedTuple):
    """edgel 集合 (不可变): 亚像素位置 + 局部几何 + 强度。
    pos/normal/tangent 的坐标序为 (row, col)。"""

    pos: mx.array  # (N,2) 亚像素 (row, col)
    normal: mx.array  # (N,2) 单位法向 (轴向, 符号任意)
    tangent: mx.array  # (N,2) 单位切向 = 法向旋转 90°
    strength: mx.array  # (N,) L_e 峰值 ∈[0,1]

    def cga_point(self, idx: int) -> Multivector:
        """提升为 2D conformal 点 (x=col, y=row, z=0)。不做反投影 —
        深度未知时图像点只对应 3D 射线 (flow.md §1.1)。"""
        r, c = float(self.pos[idx, 0]), float(self.pos[idx, 1])
        return point(c, r, 0.0)


class TJunction(NamedTuple):
    """T 结: 交叉点 + 遮挡偏序 (front 遮 behind) + 被遮侧两臂支撑数。"""

    pos: tuple[float, float]  # (row, col)
    front: int  # 遮挡者链 id (连续通过)
    behind: int  # 被遮者链 id (一侧支撑中断)
    support: tuple[int, int]  # 被遮线两臂 edgel 支撑 (+侧, −侧)


class GroupingResult(NamedTuple):
    """组织层输出 (flow.md §1): 2D CGA 直接形式图元 + 偏序约束。"""

    edgels: Edgels
    chains: list[mx.array]  # 每条链: (L,) int edgel 索引, 沿轮廓有序
    lines: list[Multivector]  # 每链一条 line blade (p1∧p2∧e∞)
    completions: list[tuple[int, int, float]]  # (端点i, 端点j, p(连续))
    circles: list[Multivector]  # 高置信补全弧段的 circle blade (对偶)
    t_junctions: list[TJunction]  # T 结集合 (遮挡偏序 front≻behind)


@dataclass(slots=True)
class PerceptualGrouping:
    """无状态知觉组织先验: 良好连续性 (共圆亲和) + 邻近性 + 闭合性
    (概率化补全) + 遮挡逻辑 (T 结, 高权重) + 一般视角 (惩罚常数)。"""

    edgel_thr: float = 0.2  # edgel 提取的 L_e 阈值
    edgel_min_dist: float = 0.5  # edgel 去重半径 (px, 抑制亚像素重复)
    sigma_d: float = 8.0  # 邻近性距离尺度 (px)
    kappa: float = 1.0  # 共圆残差衰减 (残差量纲 px)
    link_radius: float = 9.0  # 亲和近邻候选截断半径 (px)
    link_thr: float = 0.3  # 建链亲和阈值
    res_max: float = 3.0  # 残差钳顶 = 几何不一致的惩罚常数 (px)
    res_floor: float = 0.3  # 亚像素定位噪声地板, 低于此不惩罚 (px)
    det_eps: float = 0.05  # 法向平行的行列式阈值 (≈3°)
    rho_min: float = 1.5  # 候选圆最小半径 (px)
    min_chain: int = 3  # 最短链 (edgel 数)
    gap_max: float = 16.0  # 跨缺口补全的最大端点间距 (px)
    complete_thr: float = 0.5  # 高置信补全阈值 (输出 circle blade)
    t_band: float = 2.0  # T 结支撑统计的带半宽 (px)
    t_radius: float = 12.0  # 支撑统计沿线臂长 (px)
    t_arm_min: float = 1.5  # 臂内死区 (避开交叉点邻域, px)
    t_support: int = 2  # 一臂被视为"有支撑"的最少 edgel 数
    t_span: float = 5.0  # 支撑臂最小延伸 (px, 防碎链伪 T)

    # ── 1. edgel 提取与提升 ────────────────────────────────────────

    @staticmethod
    def nonzero(sel: mx.array) -> mx.array:
        """布尔掩码 → 扁平索引 (MLX 无布尔索引, argsort 技巧:
        选中位给原下标, 未选中给 N, 升序排序后前 k 个即索引)。"""
        flat = sel.reshape(-1)
        k = int(mx.sum(flat))
        key = mx.where(flat, mx.arange(flat.shape[0]), flat.shape[0])
        return mx.argsort(key)[:k]

    @staticmethod
    def near_pairs(pos: mx.array, radius: float) -> list[tuple[int, int]]:
        """网格桶找距离 ≤ radius 的候选对 (i<j), 避免 N×N 距离矩阵
        (自然图 edgel N≈1e4, N² float32 即数百 MB)。桶边长 = radius,
        检查 3×3 邻桶, 逐对精算距离。"""
        pl = pos.tolist()
        buckets: dict[tuple[int, int], list[int]] = {}
        for a, (r, c) in enumerate(pl):
            buckets.setdefault((int(r // radius), int(c // radius)), []).append(a)
        pairs: set[tuple[int, int]] = set()
        r2 = radius * radius
        for (br, bc), members in buckets.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for a in members:
                        ra, ca = pl[a]
                        for b in buckets.get((br + dr, bc + dc), ()):
                            if (
                                b > a
                                and (ra - pl[b][0]) ** 2 + (ca - pl[b][1]) ** 2 <= r2
                            ):
                                pairs.add((a, b))
        return sorted(pairs)

    def extract_edgels(self, like: mx.array, mean_ori: mx.array) -> Edgels:
        """沿法向对 L_e 做 NMS + 抛物线亚像素插值 (flow.md §1.1)。
        采样复用 EdgePrior 的预计算双线性 gather。"""
        yy, xx = EdgePrior.grid(like.shape)
        n_row, n_col = mx.sin(mean_ori), mx.cos(mean_ori)
        gp = EdgePrior.precomp_gather(like.shape, n_row, n_col, yy, xx)
        gm = EdgePrior.precomp_gather(like.shape, -n_row, -n_col, yy, xx)
        v0, vp, vm = like, gp(like), gm(like)

        mask = (v0 >= vp) & (v0 >= vm) & (v0 >= self.edgel_thr)
        # 抛物线顶点偏移 (法向 ±1px 三点), 分母近零 (=平台/鞍) 时 t=0
        denom = vm - 2.0 * v0 + vp
        t = 0.5 * (vm - vp) / mx.where(mx.abs(denom) > 1e-6, denom, -1e-6)
        t = mx.clip(t, -1.0, 1.0)
        t = mx.where(mx.abs(denom) > 1e-6, t, 0.0)

        idx = self.nonzero(mask)
        pos = mx.stack(
            [(yy + t * n_row).reshape(-1)[idx], (xx + t * n_col).reshape(-1)[idx]],
            axis=-1,
        )
        nrm = mx.stack([n_row.reshape(-1)[idx], n_col.reshape(-1)[idx]], -1)
        strg = v0.reshape(-1)[idx]

        # 去重: NMS 只沿法向比较, 切向相邻像素的亚像素顶点可能落在
        # 几乎同一位置 → 近距离重复对保留强度大者 (强度相同保留下标小者)
        sl = strg.tolist()
        drop = [False] * pos.shape[0]
        for a, b in self.near_pairs(pos, self.edgel_min_dist):
            # b > a: b 出局 ⇔ 不强于 a (等强度时大下标出局)
            drop[b if sl[b] <= sl[a] else a] = True
        keep = mx.array([a for a in range(pos.shape[0]) if not drop[a]], dtype=mx.int32)
        return Edgels(
            pos=pos[keep],
            normal=nrm[keep],
            tangent=mx.stack([-nrm[keep, 1], nrm[keep, 0]], -1),
            strength=strg[keep],
        )

    # ── 2. 亲和度: 邻近性 × 共圆关联残差 ───────────────────────────

    def affinity(self, ed: Edgels) -> tuple[mx.array, mx.array, mx.array]:
        """候选近邻对 (dist ≤ link_radius) 的亲和度 (flow.md §1.2)。

        候选共圆由两 edgel 联合确定: 圆心 = 两条法向线的交点 (几何
        一致时唯一)。残差取对称形式 |res_i|+|res_j|, res = (d²−ρ²)/(2ρ)
        (归一化对偶球 ip 残差, 长度量纲); 法向近平行时退化为切向线
        距离 (ρ→∞ 极限), 几何不一致由 res_max 钳顶给惩罚常数。

        返回 (i, j, w) 三个 (P,) 数组, i<j。"""
        pos, nrm = ed.pos, ed.normal
        pairs = self.near_pairs(pos, self.link_radius)
        i = mx.array([p[0] for p in pairs], dtype=mx.int32)
        j = mx.array([p[1] for p in pairs], dtype=mx.int32)
        d2 = mx.sum((pos[j] - pos[i]) ** 2, axis=-1)  # (P,) 逐对平方距离

        xi, xj = pos[i], pos[j]
        ni, nj = nrm[i], nrm[j]
        d = xj - xi

        def cross(a: mx.array, b: mx.array) -> mx.array:
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        det = cross(ni, nj)  # 法向线交点存在 ⇔ det≠0
        t = cross(d, nj) / mx.where(mx.abs(det) > 1e-12, det, 1e-12)
        c = xi + t[:, None] * ni
        rho_i = mx.sqrt(mx.sum((c - xi) ** 2, axis=-1))
        rho_j = mx.sqrt(mx.sum((c - xj) ** 2, axis=-1))
        safe_i, safe_j = mx.maximum(rho_i, 1e-9), mx.maximum(rho_j, 1e-9)
        # 对称残差: j 对 i 的圆 + i 对 j 的圆 (同一圆心, 半径互换)
        res_c = mx.abs((rho_j**2 - rho_i**2) / (2.0 * safe_i))
        res_c = res_c + mx.abs((rho_i**2 - rho_j**2) / (2.0 * safe_j))
        # ρ→∞ 极限: 到对方切向线的距离 (直线连续也是良好连续)
        res_l = mx.abs(mx.sum(d * ni, axis=-1)) + mx.abs(mx.sum(d * nj, axis=-1))
        is_circle = (mx.abs(det) >= self.det_eps) & (rho_i >= self.rho_min)
        res = mx.minimum(mx.where(is_circle, res_c, res_l), self.res_max)
        # 亚像素定位噪声地板: 像素采样的 edgel 位置有 ~±0.15px 量化
        # 起伏, 相邻对残差被放大而"跳过一个"反而更共圆 → 交错碎链;
        # 地板以下的残差视为定位噪声, 不惩罚
        res = mx.maximum(res - self.res_floor, 0.0)

        w = mx.exp(-d2 / (2.0 * self.sigma_d**2)) * mx.exp(-self.kappa * res)
        return i, j, w

    # ── 3. 轮廓编组: 高亲和链 ──────────────────────────────────────

    def group(self, ed: Edgels) -> list[mx.array]:
        """每个 edgel 沿自身切向前/后各取最高亲和伙伴即连边 (不要求
        互选 —— 单个不对称就会在长轮廓上制造断点); 链 = 连接图上的
        路径, 从端点起走出有序序列。"""
        i, j, w = self.affinity(ed)
        # best[side][a] = (b, w_ab): a 沿切向前(0)/后(1)的最佳伙伴
        best: list[dict[int, tuple[int, float]]] = [{}, {}]
        for a, b, wab in zip(i.tolist(), j.tolist(), w.tolist()):
            if wab < self.link_thr:
                continue
            side_a = float(mx.sum(ed.tangent[a] * (ed.pos[b] - ed.pos[a])))
            side_b = float(mx.sum(ed.tangent[b] * (ed.pos[a] - ed.pos[b])))
            for src, dst, side in ((a, b, side_a), (b, a, side_b)):
                s = 0 if side >= 0 else 1
                cur = best[s].get(src)
                if cur is None or wab > cur[1]:
                    best[s][src] = (dst, wab)

        # 每节点取前/后最佳即成边 (度 ≤2 由构造保证 → 链必为路径;
        # 不要求互选 —— 单个不对称就会在长轮廓上制造断点)
        edges: set[tuple[int, int]] = set()
        for s in range(2):
            for a, (b, _) in best[s].items():
                edges.add((min(a, b), max(a, b)))
        adj: dict[int, list[int]] = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # 从度 1 端点起走: 度 ≤2 的图上单向走法覆盖整条路径, 一条轮廓
        # 一条链 (从内部节点起走只会走一个方向, 另一臂被拆成第二条链);
        # 闭环无端点, 第二轮任意起点兜底
        chains: list[mx.array] = []
        seen: set[int] = set()
        starts = sorted(n for n in adj if len(adj[n]) == 1)
        for start in starts + sorted(adj):
            if start in seen:
                continue
            walk = [start]
            seen.add(start)
            prev, cur = -1, start
            while True:
                nxt = [k for k in adj[cur] if k != prev and k not in seen]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                walk.append(cur)
                seen.add(cur)
            if len(walk) >= self.min_chain:
                chains.append(mx.array(walk))
        return chains

    # ── 4. 跨缺口补全: 输出概率, 不硬连接 ──────────────────────────

    def complete(
        self, ed: Edgels, chains: list[mx.array]
    ) -> tuple[list[tuple[int, int, float]], list[Multivector]]:
        """不同链端点两两评估 p(连续|两端几何) —— 同一套邻近性 ×
        共圆残差 (臆造边缘会制造假深度不连续, 故只给概率)。
        p ≥ complete_thr 的弧段额外输出拟合圆 blade (编组中间产物)。"""
        owner = {}
        for cid, ch in enumerate(chains):
            for idx in ch.tolist():
                owner[idx] = cid
        ends = [int(ch[0]) for ch in chains] + [int(ch[-1]) for ch in chains]

        completions: list[tuple[int, int, float]] = []
        circles: list[Multivector] = []
        for ia in range(len(ends)):
            for ib in range(ia + 1, len(ends)):
                a, b = ends[ia], ends[ib]
                if owner[a] == owner[b]:
                    continue
                d2 = float(mx.sum((ed.pos[b] - ed.pos[a]) ** 2))
                if d2 > self.gap_max**2:
                    continue
                res, circ = self.pair_geometry(ed, a, b)
                prob = math.exp(-d2 / (2.0 * self.sigma_d**2))
                prob *= math.exp(-self.kappa * res)
                completions.append((a, b, prob))
                if prob >= self.complete_thr and circ is not None:
                    (cx, cy), rho = circ
                    circles.append(circle((cx, cy, 0.0), rho, (0, 0, 1)))
        completions.sort(key=lambda t: -t[2])
        return completions, circles

    def pair_geometry(
        self, ed: Edgels, a: int, b: int
    ) -> tuple[float, tuple[tuple[float, float], float] | None]:
        """两端点共圆残差 (长度量纲, px) 与候选圆参数 ((x,y), ρ)。
        法向近平行 (直线状) 时圆为 None。坐标转 (x=col, y=row)。"""
        xa, xb = ed.pos[a], ed.pos[b]
        na, nb = ed.normal[a], ed.normal[b]
        d = xb - xa
        det = float(na[0] * nb[1] - na[1] * nb[0])
        if abs(det) < self.det_eps:
            res = abs(float(mx.sum(d * na))) + abs(float(mx.sum(d * nb)))
            res = max(min(res, self.res_max) - self.res_floor, 0.0)
            return res, None
        t = float(d[0] * nb[1] - d[1] * nb[0]) / det
        c = xa + t * na
        rho_a = float(mx.sqrt(mx.sum((c - xa) ** 2)))
        rho_b = float(mx.sqrt(mx.sum((c - xb) ** 2)))
        if rho_a < self.rho_min:
            return self.res_max, None
        res = abs((rho_b**2 - rho_a**2) / (2.0 * rho_a))
        res += abs((rho_a**2 - rho_b**2) / (2.0 * rho_b))
        res = max(min(res, self.res_max) - self.res_floor, 0.0)
        center = (float(c[1]), float(c[0]))  # (x=col, y=row)
        return min(res, self.res_max), (center, rho_a)

    # ── 5. T 结检测: 交叉几何 + 竖杠中断证据 ───────────────────────

    def detect_t_junctions(self, ed: Edgels, chains: list[mx.array]) -> list[TJunction]:
        """线线求交给候选, 竖杠中断统计判 T (flow.md §1.4):
        候选两个来源 —— 折线直接相交; 链端点沿末端切向的延长线
        (≤ t_radius) 与其他链相交 (真实 T 的竖杠止于遮挡边,
        折线本身不相交)。交叉点处被遮线一侧有支撑一侧无支撑 →
        T 结与偏序; 两侧皆有支撑 → X 交叉, 不产生偏序。"""
        polylines = [ed.pos[ch] for ch in chains]
        cands: list[tuple[int, int, mx.array]] = []
        for a in range(len(chains)):
            for b in range(a + 1, len(chains)):
                cands += self.polyline_intersections(a, polylines[a], b, polylines[b])
        cands += self.endpoint_ray_candidates(ed, chains)

        out: list[TJunction] = []
        for a, b, q in cands:
            ta = self.local_tangent(ed, chains[a], q)
            tb = self.local_tangent(ed, chains[b], q)
            sa = self.arms(ed, q, ta, chains[a])
            sb = self.arms(ed, q, tb, chains[b])
            # 通过 = 两臂都有连续支撑 (数量够且延伸够长); 竖杠 =
            # 一侧有真实支撑 (防碎链伪 T), 一侧无支撑
            a_through = all(c >= self.t_support and s >= self.t_span for c, s in sa)
            b_through = all(c >= self.t_support and s >= self.t_span for c, s in sb)
            a_stub = any(c >= self.t_support and s >= self.t_span for c, s in sa)
            b_stub = any(c >= self.t_support and s >= self.t_span for c, s in sb)
            if a_through and not b_through and b_stub:
                out.append(
                    TJunction((float(q[0]), float(q[1])), a, b, tuple(c for c, _ in sb))
                )
            elif b_through and not a_through and a_stub:
                out.append(
                    TJunction((float(q[0]), float(q[1])), b, a, tuple(c for c, _ in sa))
                )
            # 其余 = X 交叉或伪交叉, 不产生偏序
        return self.dedupe(out)

    def endpoint_ray_candidates(
        self, ed: Edgels, chains: list[mx.array]
    ) -> list[tuple[int, int, mx.array]]:
        """链端点延长线候选: 竖杠链端点沿末端段方向延长, 命中其他
        链的最近交点 (延长 ≤ t_radius) → (遮挡链, 竖杠链, 交点)。"""
        out: list[tuple[int, int, mx.array]] = []
        for b, ch in enumerate(chains):
            ends = (
                (ed.pos[ch[0]], ed.pos[ch[0]] - ed.pos[ch[1]]),
                (ed.pos[ch[-1]], ed.pos[ch[-1]] - ed.pos[ch[-2]]),
            )
            for e, d_out in ends:
                norm = float(mx.sqrt(mx.sum(d_out**2)))
                if norm < 1e-9:
                    continue
                t_out = d_out / norm
                hit: tuple[float, int, mx.array] | None = None
                for a, other in enumerate(chains):
                    if a == b:
                        continue
                    pa = ed.pos[other]
                    for ia in range(pa.shape[0] - 1):
                        p1, p2 = pa[ia], pa[ia + 1]
                        d1 = p2 - p1
                        det = float(t_out[0] * d1[1] - t_out[1] * d1[0])
                        if abs(det) < 1e-9:
                            continue
                        de = p1 - e
                        s = float(de[0] * d1[1] - de[1] * d1[0]) / det
                        u = float(de[0] * t_out[1] - de[1] * t_out[0]) / det
                        if 0.0 <= s <= self.t_radius and 0.0 <= u <= 1.0:
                            if hit is None or s < hit[0]:
                                hit = (s, a, e + s * t_out)
                if hit is not None:
                    out.append((hit[1], b, hit[2]))
        return out

    def polyline_intersections(
        self, a: int, pa: mx.array, b: int, pb: mx.array
    ) -> list[tuple[int, int, mx.array]]:
        """两折线的线段求交 (参数式 t,u ∈ [0,1])。"""
        out = []
        for ia in range(pa.shape[0] - 1):
            p1, p2 = pa[ia], pa[ia + 1]
            d1 = p2 - p1
            for ib in range(pb.shape[0] - 1):
                p3, p4 = pb[ib], pb[ib + 1]
                d2 = p4 - p3
                det = float(d1[0] * d2[1] - d1[1] * d2[0])
                if abs(det) < 1e-9:
                    continue
                d3 = p3 - p1
                t = float(d3[0] * d2[1] - d3[1] * d2[0]) / det
                u = float(d3[0] * d1[1] - d3[1] * d1[0]) / det
                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                    out.append((a, b, p1 + t * d1))
        return out

    @staticmethod
    def local_tangent(ed: Edgels, chain: mx.array, q: mx.array) -> mx.array:
        """链上离 q 最近的 edgel 的切向。"""
        d = mx.sum((ed.pos[chain] - q) ** 2, axis=-1)
        return ed.tangent[chain[int(mx.argmin(d))]]

    def arms(
        self, ed: Edgels, q: mx.array, direction: mx.array, chain: mx.array
    ) -> tuple[tuple[int, float], tuple[int, float]]:
        """链 edgel 在 ±臂的支撑 (带内, 死区外): 每臂 (数量, 最大延伸)。"""
        v = ed.pos[chain] - q
        along = mx.sum(v * direction, axis=-1)
        perp = v - along[:, None] * direction
        perp = mx.sqrt(mx.sum(perp**2, axis=-1))
        band = perp <= self.t_band
        far = mx.abs(along) >= self.t_arm_min
        near = mx.abs(along) <= self.t_radius

        def stat(sel: mx.array) -> tuple[int, float]:
            hit = band & far & near & sel
            n = int(mx.sum(hit))
            span = float(mx.max(mx.where(hit, mx.abs(along), 0.0)))
            return n, span

        return stat(along > 0), stat(along < 0)

    @staticmethod
    def dedupe(junctions: list[TJunction], radius: float = 3.0) -> list[TJunction]:
        """3px 内的重复候选只留第一个。"""
        kept: list[TJunction] = []
        for t in junctions:
            dup = any(
                (t.pos[0] - k.pos[0]) ** 2 + (t.pos[1] - k.pos[1]) ** 2 < radius**2
                for k in kept
            )
            if not dup:
                kept.append(t)
        return kept

    # ── 总装 ───────────────────────────────────────────────────────

    def run(self, like: mx.array, mean_ori: mx.array) -> GroupingResult:
        """L_e + 法向场 → edgel/链/line/补全/circle/T 结。"""
        ed = self.extract_edgels(like, mean_ori)
        chains = self.group(ed)
        lines = []
        for ch in chains:
            p1, p2 = ed.cga_point(int(ch[0])), ed.cga_point(int(ch[-1]))
            lines.append(line(p1, p2))
        completions, circles = self.complete(ed, chains)
        t_junctions = self.detect_t_junctions(ed, chains)
        return GroupingResult(
            edgels=ed,
            chains=chains,
            lines=lines,
            completions=completions,
            circles=circles,
            t_junctions=t_junctions,
        )

    def visualize(self, like: mx.array, res: GroupingResult, out_path: str | Path):
        """链按 id 着色, edgel 带切向刻度, 补全画虚线, T 结画叉。"""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(like, cmap="gray")
        pos, tan = res.edgels.pos, res.edgels.tangent
        ax.quiver(
            pos[:, 1],
            pos[:, 0],
            tan[:, 1],
            tan[:, 0],
            scale=40,
            width=0.003,
            color="cyan",
            headwidth=0,
            headlength=0,
        )
        for cid, ch in enumerate(res.chains):
            p = pos[ch]
            ax.plot(p[:, 1], p[:, 0], ".-", lw=1.5, ms=3, label=f"chain {cid}")
        for a, b, prob in res.completions[:8]:
            pa, pb = pos[a], pos[b]
            ax.plot([pa[1], pb[1]], [pa[0], pb[0]], "y--", lw=prob * 2)
        for t in res.t_junctions:
            ax.plot(t.pos[1], t.pos[0], "rx", ms=12, mew=2.5)
        ax.legend(fontsize=7, loc="upper right")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    # ── 合成真值验证 (不经过前端管线, 直接构造 L_e / mean_ori) ─────
    # 场构造: 折线/弧段栅格化为 σ=1 的高斯脊 (强度 0.9), 法向已知。
    H, W = 96, 128

    def ridge_field(segs: list[tuple[mx.array, mx.array]]) -> tuple[mx.array, mx.array]:
        """segs: (点列 (M,2) (row,col), 法向列 (M,2)) → (L_e, mean_ori)。"""
        yy, xx = mx.meshgrid(
            mx.arange(H, dtype=mx.float32),
            mx.arange(W, dtype=mx.float32),
            indexing="ij",
        )
        like = mx.zeros((H, W), dtype=mx.float32)
        ori = mx.zeros((H, W), dtype=mx.float32)
        for pts, nrms in segs:
            for k in range(pts.shape[0]):
                p, nv = pts[k], nrms[k]
                d2 = (yy - p[0]) ** 2 + (xx - p[1]) ** 2
                g = 0.9 * mx.exp(-d2 / 2.0)
                upd = g > like
                like = mx.where(upd, g, like)
                ori = mx.where(upd, mx.arctan2(nv[0], nv[1]), ori)
        return like, ori

    def hline(r: float, c0: float, c1: float, step: float = 0.5):
        cs = mx.arange(c0, c1 + step, step)
        pts = mx.stack([mx.full_like(cs, r), cs], axis=-1)
        nrm = mx.stack([mx.ones_like(cs), mx.zeros_like(cs)], axis=-1)
        return pts, nrm

    def vline(c: float, r0: float, r1: float, step: float = 0.5):
        rs = mx.arange(r0, r1 + step, step)
        pts = mx.stack([rs, mx.full_like(rs, c)], axis=-1)
        nrm = mx.stack([mx.zeros_like(rs), mx.ones_like(rs)], axis=-1)
        return pts, nrm

    def arc(
        center: tuple[float, float],
        rho: float,
        a0: float,
        a1: float,
        step: float = 0.02,
    ):
        ang = mx.arange(math.radians(a0), math.radians(a1), step)
        pts = mx.stack(
            [center[0] + rho * mx.sin(ang), center[1] + rho * mx.cos(ang)], axis=-1
        )
        nrm = mx.stack([mx.sin(ang), mx.cos(ang)], axis=-1)
        return pts, nrm

    pg = PerceptualGrouping()

    # A. 直线 + 12px 缺口 (> link_radius): 应得 2 链 + 补全概率
    la, ma = ridge_field([hline(48, 20, 54), hline(48, 66, 104)])
    ra = pg.run(la, ma)
    print(
        f"A. gap: chains={len(ra.chains)} (期望 2), "
        f"top completions={[(a, b, f'{p:.2f}') for a, b, p in ra.completions[:3]]}"
    )

    # B. 圆弧 + 20° 缺口: 高置信补全应输出 circle
    lb, mb = ridge_field([arc((48, 64), 24, -60, 40), arc((48, 64), 24, 60, 140)])
    rb = pg.run(lb, mb)
    print(
        f"B. arc gap: chains={len(rb.chains)}, circles={len(rb.circles)} "
        f"(期望 ≥1), top completion p={rb.completions[0][2]:.2f}"
    )

    # C. T 配置: 横线通过, 竖线止于横线 → T 结, front=横链
    lc, mc = ridge_field([hline(48, 16, 112), vline(64, 50, 88)])
    rc = pg.run(lc, mc)
    print(f"C. T: chains={len(rc.chains)}, t_junctions={len(rc.t_junctions)}")
    for t in rc.t_junctions:
        print(
            f"   T @({t.pos[0]:.1f},{t.pos[1]:.1f}) "
            f"front=chain{t.front} behind=chain{t.behind} support={t.support}"
        )

    # D. X 配置: 两线均通过 → 0 个 T 结
    ld, md = ridge_field([hline(48, 16, 112), vline(64, 24, 88)])
    rd = pg.run(ld, md)
    print(f"D. X: chains={len(rd.chains)}, t_junctions={len(rd.t_junctions)} (期望 0)")

    path = Utils.project_root() / "artifacts/grouping_synth.png"
    pg.visualize(lc, rc, path)
    print(path)

    # ── 真实管线 smoke: T 图像走 riesz→vbgmm→edgemap→grouping ─────
    from riesz import RieszWavelet
    from vbgmm import VBGMM

    img = mx.full((128, 128), 0.2)
    img[40:56, 16:112] = 0.7  # 横杠
    img[56:96, 60:76] = 0.7  # 竖杠 (顶边没入横杠 → T)
    img = img + mx.random.normal((128, 128), key=mx.random.key(5)) * 0.01

    rw = RieszWavelet(img)
    feat = rw.features()
    gm = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    like = gm.edge_likelihood((128, 128))
    prior = EdgePrior()
    enh = prior.enhance(like, feat, rw)
    res = pg.run(enh, feat.mean_ori)
    print(
        f"pipeline T: edgels={res.edgels.pos.shape[0]}, "
        f"chains={len(res.chains)}, t_junctions={len(res.t_junctions)}"
    )
    for t in res.t_junctions:
        print(
            f"   T @({t.pos[0]:.1f},{t.pos[1]:.1f}) "
            f"front=chain{t.front} behind=chain{t.behind} support={t.support}"
        )
    path2 = Utils.project_root() / "artifacts/grouping_T.png"
    pg.visualize(enh, res, path2)
    print(path2)
