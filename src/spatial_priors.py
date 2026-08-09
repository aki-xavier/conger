"""空间先验分析层 (prior.md 物理/几何先验, 深度就位后解锁):
重力支撑 / 视平线 / 线性透视 / 光源上方。

定位: 全部是 fusion/scenegraph 产物的**消费者** (深度图 + 图元
拟合 + 链), 输出报告/线索, 不改下游 (高内聚低耦合纪律)。
四项共享一个家: 都是"有了深度才能算"的空间解释。

  ① GravitySupport: 地面平面 + 区域底部接触分析 → 支撑/悬空
  ② HorizonCue: 地面平面 → 视平线解析位置 (+可选弱高度线索)
  ③ VanishingPoints: 链方向族 → 平行族交点聚类 → 灭点
  ④ LightFromAbove: 法向×亮度一致性 → 凹凸歧义消解
"""

from dataclasses import dataclass

import mlx.core as mx

from utils import Utils


@dataclass(slots=True)
class SupportVerdict:
    """单区域支撑判定。"""

    region: int
    contact: bool  # 底部与地面接触
    gap: float  # 地面深度−物体深度 (≈0 接触; 显著负 = 底部深于
    # 地面线 = 悬空/嵌进地面的几何不可能态)
    is_ground: bool = False


@dataclass(slots=True)
class HorizonCue:
    """视平线 (prior.md 物理先验: 地平面上的物体越接近视平线越远)。
    地面拟合平面 z = a·u+b·v+c → 视平线闭式 v = c/b − (a/b)·u。
    推导: 精确透视下平面 z = f_n·d/(n_x u+n_y v+n_z f_n), 与线性
    模型一阶对应给出 n_x/n_z = −a·f_n/c, n_y/n_z = −b·f_n/c,
    代入灭线方程 n_x u + n_y v + n_z f_n = 0 即得 (f_n 恰好消去)。
    """

    def estimate(
        self,
        ground_params: tuple[float, ...],
        centroid: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """地面拟合参数 (a,b,c) + 拟合区域质心 (u,v, 归一化) →
        (v_h 截距, 斜率 −a/b)。展开点必须取区域质心: 线性拟合是
        倒数曲面的局部割线, 在图像中心展开会引入 ~15% 系统偏差
        (实测 2.10 vs 真 2.48); 质心展开一阶精确。"""
        a, b, c = ground_params
        if abs(b) < 1e-6:
            raise ValueError("地面无纵向倾斜, 视平线不可估")
        uc, vc = centroid
        c_c = a * uc + b * vc + c  # 展开点深度
        return vc + c_c / b, -a / b


@dataclass(slots=True)
class LightFromAbove:
    """光源上方先验 (prior.md 光学: 人类最核心的偏见, 空心面具的
    根源)。有深度就有法向: n ∝ (−∂z/∂x, −∂z/∂y, 1)。上照光下
    上仰面 (−∂z/∂y > 0) 更亮 —— 亮度与上仰分量的相关符号即
    一致性检验; 负相关 = 照明方向假设不成立, 触发凹凸翻转
    (空心面具式歧义消解)。"""

    def consistency(self, depth: mx.array, img: mx.array) -> float:
        """深度图 + 亮度图 → 光源上方一致性 (相关, ∈[−1,1])。
        上照光下 上仰面更亮: 相关(上仰分量, 亮度) 为正 —— 相关
        对象是亮度本身而非亮度梯度 (梯度符号随形状翻, 实测)。"""
        dz_dy = mx.zeros_like(depth)
        dz_dy = dz_dy.at[1:-1, :].add(
            (depth[2:, :] - depth[:-2, :]) * 0.5
        )
        # 像素差分 × s → 归一化坐标导数 (阈值 0.05 才有量纲意义)
        up = -dz_dy * float(max(depth.shape))  # 上仰分量 (−∂z/∂v)
        inner_up = up[1:-1, :]
        inner_i = img[1:-1, :]
        mask = mx.abs(inner_up) > 0.05  # 平坦区无判别力, 排除
        k = int(mx.sum(mask))
        if k < 50:
            return 0.0  # 证据不足弃权
        idx = Utils.nonzero(mask.reshape(-1))
        av = inner_up.reshape(-1)[idx]
        bv = inner_i.reshape(-1)[idx]
        ma, mb = mx.mean(av), mx.mean(bv)
        cov = mx.mean((av - ma) * (bv - mb))
        va = mx.mean((av - ma) ** 2)
        vb = mx.mean((bv - mb) ** 2)
        return float(cov / mx.maximum(mx.sqrt(va * vb), 1e-12))


@dataclass(slots=True)
class VanishingPoints:
    """线性透视 (prior.md 几何: 平行线远方汇聚, Gibson)。
    链 → PCA 直线 (方向+质心) → 线对交点按链长加权投票 →
    网格桶聚类出灭点。只取方向分布的两端 (近水平/近竖直外的
    斜线才携带汇聚信息; 完全平行于图像轴的线对交点在无穷远)。"""

    min_len: float = 15.0  # 链最短弧长 (短链方向噪声大)
    bucket: float = 8.0  # 灭点聚类桶 (px)
    min_share: float = 0.02  # 灭点最少票额占比 (绝对票数不随链数
    # 缩放 —— 自然图纹理链数千时绝对阈值全漏 (实测 8473 个))

    def detect(
        self, res, shape: tuple[int, int]
    ) -> list[tuple[float, float, float]]:
        """GroupingResult → [(row, col, 权重)] 灭点列 (按权重降序)。"""
        h, w = shape
        lines = []  # (方向单位向量, 质心, 弧长)
        for ch in res.chains:
            pts = res.edgels.pos[ch]
            n_pts = int(pts.shape[0])
            if n_pts < 3:
                continue
            mean = pts.mean(axis=0)
            cov = (pts - mean).T @ (pts - mean) / n_pts
            ev, evec = mx.linalg.eigh(cov, stream=mx.cpu)
            if float(ev[1]) < 1e-9:
                continue  # 退化 (圆/点)
            direc = evec[:, 1]  # 主方向
            length = float(4.0 * mx.sqrt(ev[1]))  # ±2σ 弦长代理
            if length < self.min_len:
                continue
            lines.append(((float(direc[0]), float(direc[1])),
                          (float(mean[0]), float(mean[1])), length))
        # 线对交点投票
        votes: dict[tuple[int, int], list[float]] = {}
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                (d1, c1, l1) = lines[i]
                (d2, c2, l2) = lines[j]
                det = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(det) < 0.05:  # 近平行 (交点在图外/无穷)
                    continue
                # 解 c1 + t·d1 = c2 + s·d2
                dc = (c2[0] - c1[0], c2[1] - c1[1])
                t = (dc[0] * d2[1] - dc[1] * d2[0]) / det
                pr, pc = c1[0] + t * d1[0], c1[1] + t * d1[1]
                if not (-w < pr < h + w and -w < pc < 2 * w):
                    continue  # 交点太远不可信
                key = (int(pr // self.bucket), int(pc // self.bucket))
                wgt = min(l1, l2)
                if key not in votes:
                    votes[key] = [0.0, 0.0, 0.0]
                votes[key][0] += wgt
                votes[key][1] += wgt * pr
                votes[key][2] += wgt * pc
        total = sum(v[0] for v in votes.values())
        out = [
            (v[1] / v[0], v[2] / v[0], v[0])
            for v in votes.values()
            if v[0] >= self.min_share * total
        ]
        out.sort(key=lambda t: -t[2])
        return out


@dataclass(slots=True)
class GravitySupport:
    """重力感 (prior.md 物理先验): 地面参考 + 支撑关系推断。
    地面签名: 深度随图像行递增 (b>0, 归一化坐标) 的最大面积平面;
    支撑判定: 区域底行处的地面深度 vs 该区域自身拟合深度 —
    近似相等 = 接触 (受支撑), 显著更小 = 悬空 (触发"挂在哪里"
    推断)。只消费平面拟合; 球/稠密区跳过。"""

    tilt_min: float = 0.1  # 地面最小纵向坡度 (排除竖直墙面)
    contact_tol: float = 0.15  # 接触容差 (相对地面深度比例)
    min_area: float = 0.01  # 最小对象面积占比 —— 支撑是对象级关系,
    # 纹理碎片拟合出的微区不是对象 (自然图实测数百噪声判定)

    def ground_index(self, res, sub: mx.array) -> int:
        """fits 中的地面索引 (无地面返回 -1)。签名: 平面 + b>tilt_min
        + 面积最大 (区域像素数由 sub 统计)。"""
        cnt: dict[int, int] = {}
        for v in sub.reshape(-1).tolist():
            cnt[v] = cnt.get(v, 0) + 1
        best, best_area = -1, 0
        for i, f in enumerate(res.fits):
            if f.kind != "plane" or f.params[1] < self.tilt_min:
                continue
            area = cnt.get(i + 1, 0)  # fits[i] ↔ 区域 i+1 (不变量)
            if area > best_area:
                best, best_area = i, area
        return best

    def analyze(self, res, sub: mx.array) -> list[SupportVerdict]:
        """FusionResult + 区域图 → 逐区域支撑判定 (无地面则全弃权)。"""
        gi = self.ground_index(res, sub)
        if gi < 0:
            return []
        ga, gb, gc = res.fits[gi].params
        h, w = sub.shape
        s = float(max(h, w))
        out: list[SupportVerdict] = []
        lab = sub.tolist()
        for i, f in enumerate(res.fits):
            rid = i + 1
            if f.kind != "plane":
                continue
            if i == gi:
                out.append(SupportVerdict(rid, True, 0.0, is_ground=True))
                continue
            rows = [r for r in range(h) if rid in lab[r]]
            if not rows:
                continue
            area = sum(lab[r].count(rid) for r in rows)
            if area < self.min_area * h * w:
                continue  # 碎片微区不判定 (非对象)
            bottom = max(rows)
            cols = [c for c in range(w) if lab[bottom][c] == rid]
            uc = (sum(cols) / len(cols) - w / 2) / s  # 底行质心 (归一化)
            vb = (bottom - h / 2) / s
            z_ground = ga * uc + gb * vb + gc
            z_obj = f.params[0] * uc + f.params[1] * vb + f.params[2]
            gap = z_ground - z_obj  # 正 = 物体在地面后方 (嵌/离)
            contact = abs(gap) <= self.contact_tol * max(abs(z_ground), 1e-6)
            out.append(SupportVerdict(rid, contact, gap))
        return out


if __name__ == "__main__":
    from fusion import DepthCue, DepthFusionLayer

    # ── 合成: 倾斜地面 + 接触盒子 + 悬空盒子 ────────────────────────
    H, W = 96, 128
    s = float(W)
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    u = (xx - W / 2) / s
    v = (yy - H / 2) / s
    # 地面: z = 2 + 3.84·v (即 0.03·row + 0.56), 底行 z≈3.4
    z_g = 2.0 + 3.84 * v
    # 盒子 A 坐在地面 (中心 row 60): 底行 row 70 的地面深度 ≈ 3.35,
    # 平面常数深度 = 接触点深度 (垂直于视线的竖板)
    z_a = float(2.0 + 3.84 * (70 - H / 2) / s)
    # 盒子 B 悬空 (底行 row 40, 同深度但地面在 row 40 只有 2.96 →
    # 它在空中 (底缘与地面同深度但底下还有地面可见))
    box_a = (yy > 50) & (yy <= 70) & (xx > 20) & (xx < 50)
    box_b = (yy > 20) & (yy <= 40) & (xx > 80) & (xx < 110)
    z = mx.where(box_a, z_a, z_g)
    z = mx.where(box_b, z_a, z)  # B 与 A 同深, 但底行地面更深
    sub = mx.zeros((H, W), dtype=mx.int32)
    sub = mx.where(box_a, 1, sub)
    sub = mx.where(box_b, 2, sub)
    sub = mx.where((sub == 0) & (yy > 48), 3, sub)  # 地面只取下半
    cue = DepthCue(z, mx.full((H, W), 10.0))
    fr = DepthFusionLayer().run([cue], sub)

    verdicts = GravitySupport().analyze(fr, sub)
    by_rid = {vd.region: vd for vd in verdicts}
    assert by_rid[3].is_ground, "地面应被识别"
    assert by_rid[1].contact, f"盒子 A 应接触地面: gap={by_rid[1].gap:.3f}"
    assert not by_rid[2].contact, f"盒子 B 应悬空: gap={by_rid[2].gap:.3f}"
    print(f"1. 重力支撑: 地面 rid3, A 接触 (gap={by_rid[1].gap:.3f}), "
          f"B 悬空 (gap={by_rid[2].gap:.3f}) ✓")

    # ── 2. 视平线: 精确透视地面 → 闭式恢复 ───────────────────────
    # 真 3D 地面: 法向 n=(0,−0.6,0.8), d=2, fx=100 → 精确深度场
    fx = 100.0
    f_n = fx / s
    nx_, ny_, nz_ = 0.0, -0.3, 0.954  # 缓倾斜 (曲率小, 线性拟合赢)
    z_exact = f_n * 2.0 / (nx_ * u + ny_ * v + nz_ * f_n)
    cue2 = DepthCue(z_exact, mx.full((H, W), 10.0))
    sub2 = mx.where(yy > 48, 1, 0).astype(mx.int32)
    fr2 = DepthFusionLayer().run([cue2], sub2)
    g2 = fr2.fits[0].params
    v_h, slope_h = HorizonCue().estimate(
        g2, (0.0, float((71.5 - H / 2) / s))  # 区域质心 (下半区)
    )
    # 解析真值: v = −n_z·f_n/n_y (u=0 处)
    v_true = -nz_ * f_n / ny_
    assert abs(v_h - v_true) < 0.05, f"视平线: {v_h:.3f} vs 真 {v_true:.3f}"
    print(f"2. 视平线: v={v_h:.3f} vs 解析真值 {v_true:.3f} "
          f"(行 {(H / 2 + s * v_h):.1f}) ✓")

    # ── 3. 光源上方: 上照凸起一致, 反转亮度判为不一致 ────────────
    rr2 = mx.maximum(1.0 - u**2 - v**2, 0.05)
    z_bump = 3.0 - 0.8 * mx.sqrt(rr2)  # 半球凸起 (朝向相机)
    # Lambert 上照光: 上仰面 (−∂z/∂v > 0) 更亮 → I = 0.5 + 0.5·up
    shaded = mx.clip(0.5 - 0.4 * v / mx.sqrt(rr2), 0, 1)
    lfa = LightFromAbove()
    c_pos = lfa.consistency(z_bump, shaded)
    c_neg = lfa.consistency(z_bump, 1.0 - shaded)  # 反转 = 上照凹坑错觉
    assert c_pos > 0.2 and c_neg < -0.2, f"一致性: {c_pos:.2f}/{c_neg:.2f}"
    print(f"3. 光源上方: 凸起一致 {c_pos:.2f}, 反转不一致 {c_neg:.2f} ✓")

    # ── 4. 线性透视: 走廊平行线族 → 灭点 ──────────────────────────
    from types import SimpleNamespace as _NS

    # 四条地面缝线汇聚于灭点 (48, 64): 直线过灭点, 从底部发起
    chains_vp = []
    pts_all = []
    for x0 in (8, 32, 96, 120):  # 底部起点
        pts = []
        for t in range(0, 40, 2):
            r = 95 - t
            c = x0 + (64 - x0) * (95 - r) / 95 * 1.0
            pts.append((float(r), float(x0 + (64 - x0) * (95 - r) / 95)))
        base = len(pts_all)
        pts_all.extend(pts)
        chains_vp.append(mx.array(list(range(base, base + len(pts)))))
    res_vp = _NS(
        edgels=_NS(pos=mx.array(pts_all, dtype=mx.float32)),
        chains=chains_vp,
    )
    vps = VanishingPoints(min_len=10.0, min_share=0.1).detect(res_vp, (H, W))
    assert vps, "应检出灭点"
    vr, vc, _w = vps[0]
    assert abs(vr - 0.0) < 12 and abs(vc - 64) < 8, (
        f"灭点: ({vr:.1f},{vc:.1f}) 期望 (≈0,64)"
    )
    print(f"4. 线性透视: 灭点 ({vr:.1f},{vc:.1f}) 期望 (≈0,64) "
          f"权重 {_w:.0f} ✓")
