"""时空累积层 (Temporal Fusion, flow.md §6): motor 流形上的状态估计。

模块流程:

  帧间三维点对应 (真实管线: GroupingTracker 稳定链质心 + 融合层深度;
                 合成验证: 直接给定)
       │  MotorEKF: 状态 = twist ξ ∈ R⁶ (切空间坐标, 经
       │    Motor.velocity_bivector 打包, 与 dM/dt = −½V·M 约定一致)
       │    预测: 常速度转移 ξ⁻ = ξ, P += Q (低速平滑先验)
       │    更新: 点-点残差 + 数值雅可比 (Motor.apply 扰动, 不手推)
       ▼  → M_t (Motor) + 协方差 P
  compensate(field, depth, M): 把上一帧场扭曲到当前帧视角
       (§6.3: VBGMM warm 必须先补偿再复用; 小运动退化原样复用)
  predict_points(pts, M): 视差预测契约 → §5 DepthCue 的生产者

  [留钩] 线-线/点-面混合观测 (§6.2 后半); 加速度限制; 最小刚性多
         目标分裂 (依赖缺口 C4); 图元对应直接输入 (现用链质心对应,
         是注明的偏离); 整帧渲染钩子留接口。

数值纪律: 雅可比数值化 (ε=1e-4 对 ξ 六分量扰动, Motor.to_matrix
变换点云), 杜绝约定漂移; 纯 MLX; 离线层, 不做逐帧承诺。
"""

from dataclasses import dataclass, field

import mlx.core as mx

from cga import Motor
from edgemap import EdgePrior


def xi_to_motor(xi: mx.array) -> Motor:
    """twist ξ ∈ R⁶ → motor: B = ω + v∧e∞ (velocity_bivector 约定
    打包), M = exp(−B)。符号约定经 cga 自检 roundtrip 校准。"""
    w = (float(xi[0]), float(xi[1]), float(xi[2]))
    v = (float(xi[3]), float(xi[4]), float(xi[5]))
    return Motor.exp(Motor.velocity_bivector(w, v), 1.0)


def motor_to_xi(M: Motor) -> mx.array:
    """motor → twist ξ ∈ R⁶ (velocity_bivector 打包约定的逆)。"""
    vals = M.log().values
    w = (float(vals[10]), -float(vals[7]), float(vals[6]))  # e23,e31,e12
    v = (float(vals[9]), float(vals[12]), float(vals[14]))  # e_i∧e∞
    return mx.array([*w, *v], dtype=mx.float32)


def transform_points(xi: mx.array, pts: mx.array) -> mx.array:
    """M(ξ) 作用于点云: pts (N,3) → (N,3) (to_matrix 与 sandwich
    一致, cga 自检已验)。"""
    hm = xi_to_motor(xi).to_matrix()
    r = mx.array([row[:3] for row in hm[:3]], dtype=mx.float32)
    t = mx.array([hm[0][3], hm[1][3], hm[2][3]], dtype=mx.float32)
    return pts @ r.T + t


@dataclass(slots=True)
class MotorEKF:
    """motor 流形上的 EKF (Bayro-Corrochano & Zhang 2000 的切空间版)。

    状态 ξ ∈ R⁶ (当前帧间 motor 的 twist); 预测 = 常速度转移
    (低速平滑先验, Q 小); 更新 = 点-点对应残差的 EKF (数值雅可比)。
    """

    q_vel: float = 1e-4  # 转移噪声 (速度一致性先验强度)
    r_obs: float = 1e-4  # 观测噪声方差 (对应点定位精度)
    r_slow: float = 0.0  # 低速先验 (Weiss slow-and-smooth): >0 则
    # 每轮更新加零速度伪观测 ξ~N(0,r_slow) —— 证据弱/含糊 (孔径)
    # 时偏向低速解释, 证据强时影响可略; 默认关 (无双义场景不付代价)
    n_iter: int = 2  # 迭代 EKF 轮数 (小运动 1–2 轮足够)
    xi: mx.array = field(default_factory=lambda: mx.zeros(6))
    cov: mx.array = field(
        default_factory=lambda: mx.eye(6) * 1.0  # 初始大不确定度
    )

    def __post_init__(self):
        """初始状态物化: 懒图携带创建线程的流, 后台线程 (tracker
        worker) 首触会报 no Stream in current thread。"""
        mx.eval(self.xi, self.cov)

    def predict(self) -> None:
        """常速度转移: ξ⁻ = ξ, P⁻ = P + Q。"""
        self.cov = self.cov + mx.eye(6) * self.q_vel

    def update(self, p_prev: mx.array, q_cur: mx.array) -> None:
        """点-点对应更新: p_prev (N,3) 上帧点, q_cur (N,3) 本帧点。
        残差 r = q − T(ξ;p), 雅可比数值化 (Motor.apply 扰动)。"""
        for _ in range(self.n_iter):
            pred = transform_points(self.xi, p_prev)
            resid = (q_cur - pred).reshape(-1)  # (3N,)
            # 数值雅可比: 对 ξ 六分量 ±ε 扰动
            eps = 1e-4
            jac = []
            for jdx in range(6):
                dp = mx.zeros(6).at[jdx].add(eps)
                jac.append(
                    (transform_points(self.xi + dp, p_prev) - pred).reshape(-1)
                    / eps
                )
            h = mx.stack(jac, axis=-1)  # (3N, 6)
            # 信息形式更新: 反演全在 6×6 切空间 —— 直接求 (3N)² 的
            # S⁻¹ 在 float32 下条件数爆炸 (实测逆误差 ~16)
            p_inv = mx.linalg.inv(self.cov + 1e-9 * mx.eye(6), stream=mx.cpu)
            a = h.T @ h / self.r_obs + p_inv  # 信息矩阵 (6,6)
            rhs = h.T @ resid / self.r_obs
            if self.r_slow > 0:
                # 零速度伪观测 (低速先验): 信息 +I/r_slow,
                # 残差 0−ξ → 步长向低速收缩
                a = a + mx.eye(6) / self.r_slow
                rhs = rhs - self.xi / self.r_slow
            step = mx.linalg.inv(a, stream=mx.cpu) @ rhs
            self.xi = self.xi + step
            self.cov = mx.linalg.inv(a, stream=mx.cpu)  # 后验协方差 = 信息逆

    def motor(self) -> Motor:
        """当前估计的帧间 motor。"""
        return xi_to_motor(self.xi)


# ── 运动补偿与视差预测 ─────────────────────────────────────────────


class MotionCompensation:
    """motor + 深度 → 场扭曲与点预测 (§6.3 / 视差契约)。"""

    @staticmethod
    def compensate(field: mx.array, depth: mx.array, M: Motor) -> mx.array:
        """把上一场按 M 扭曲到当前帧视角 (双线性采样)。

        逐像素反投影 (x,y,depth) → M⁻¹ 作用 → 取回源位置采样。
        小运动退化: 最大位移 < 0.5px 时原样返回 (§6.3 退化分支)。
        """
        h, w = field.shape
        hm = M.to_matrix()
        # M⁻¹ 的 4x4: 旋转转置 + 平移反号 (刚体)
        r = [[hm[i][j] for j in range(3)] for i in range(3)]
        t = [hm[0][3], hm[1][3], hm[2][3]]
        rt = [[r[j][i] for j in range(3)] for i in range(3)]
        tinv = [-sum(rt[i][j] * t[j] for j in range(3)) for i in range(3)]
        yy, xx = EdgePrior.grid(field.shape)
        z = depth
        # 源位置 = M⁻¹ · (x, y, depth)
        sx = rt[0][0] * xx + rt[0][1] * yy + rt[0][2] * z + tinv[0]
        sy = rt[1][0] * xx + rt[1][1] * yy + rt[1][2] * z + tinv[1]
        dy, dx = sy - yy, sx - xx
        if float(mx.max(mx.abs(dy))) + float(mx.max(mx.abs(dx))) < 0.5:
            return field  # 小运动退化
        sample = EdgePrior.precomp_gather(field.shape, dy, dx, yy, xx)
        return sample(field)

    @staticmethod
    def predict_points(pts: mx.array, M: Motor) -> mx.array:
        """视差预测契约: 三维点集按 M 预测下一帧位置
        (→ §5 DepthCue 的生产者; 整帧渲染钩子留接口)。"""
        hm = M.to_matrix()
        r = mx.array([row[:3] for row in hm[:3]], dtype=mx.float32)
        t = mx.array([hm[0][3], hm[1][3], hm[2][3]], dtype=mx.float32)
        return pts @ r.T + t


# ── 慢速在线标定环 (flow.md §8 C1) ────────────────────────────────


def project_points(pts: mx.array, k: mx.array) -> mx.array:
    """针孔投影: u = fx·X/Z + cx, v = fy·Y/Z + cy。
    pts (N,3) (相机系, Z>0), k = (fx, fy, cx, cy) → (N,2) 像素。"""
    z = mx.maximum(pts[:, 2], 1e-9)
    u = k[0] * pts[:, 0] / z + k[2]
    v = k[1] * pts[:, 1] / z + k[3]
    return mx.stack([u, v], axis=-1)


@dataclass(slots=True)
class SlowCalibration:
    """慢速在线标定环 (C1): K = (fx, fy, cx, cy) 作为慢变状态挂
    时空累积层, 与 MotorEKF 同构但时间常数大几个数量级。

    驱动信号 = 预测编码残差的系统性模式 (三维点重投影误差);
    离线棋盘格标定值仍是首选初值, 本环只负责漂移修正。
    观测对 K 是线性的 → 解析雅可比 (精确, 不需数值差分);
    "增益取小"的正确实现是过程噪声 q 控制稳态卡尔曼增益
    (≈√(q·h²/r), 时间常数 ≈ 1/g 帧) —— 用 lr 直接阻尼会让滤波
    指数冻结, 修正永远到不了 (实测)。
    [留钩] 畸变系数 (直线弯曲残差通道)、曼哈顿正交残差驱动。
    """

    k: mx.array  # (4,) 初始 K (离线初值)
    q: float = 1e-3  # 慢变过程噪声 (随机游走漂移率) —— 慢通道的
    # 真正旋钮: 稳态增益 ≈ √(q·h²/r), q 太小会让滤波冻结不修正
    lr: float = 1.0  # 额外阻尼 (一般 1.0, 慢化由 q 承担)
    r_obs: float = 1.0  # 像素观测噪声方差 (px²)
    cov: mx.array = field(
        default_factory=lambda: mx.eye(4) * 25.0  # 初值不确定度 (5² px²)
    )

    def __post_init__(self):
        """初始协方差物化 (同 MotorEKF, 后台线程安全)。"""
        mx.eval(self.k, self.cov)

    def update(self, pts: mx.array, pixels: mx.array) -> None:
        """一帧标定更新: pts (N,3) 三维点 (预测编码坐标), pixels
        (N,2) 实测投影。解析雅可比 H (2N,4): du/dfx=X/Z, du/dcx=1。"""
        z = mx.maximum(pts[:, 2], 1e-9)
        pred = project_points(pts, self.k)
        resid = (pixels - pred).reshape(-1)  # (2N,)
        # 解析雅可比 (对 K 线性, 精确)
        h = mx.zeros((2 * pts.shape[0], 4))
        h = h.at[::2, 0].add(pts[:, 0] / z)
        h = h.at[::2, 2].add(1.0)
        h = h.at[1::2, 1].add(pts[:, 1] / z)
        h = h.at[1::2, 3].add(1.0)
        # 信息形式 (与 MotorEKF 同构, 反演全在 4×4)
        self.cov = self.cov + mx.eye(4) * self.q
        p_inv = mx.linalg.inv(self.cov + 1e-9 * mx.eye(4), stream=mx.cpu)
        a = h.T @ h / self.r_obs + p_inv
        step = mx.linalg.inv(a, stream=mx.cpu) @ (h.T @ resid / self.r_obs)
        self.k = self.k + self.lr * step  # 小增益: 时间常数 ≈ 1/lr 帧
        self.cov = mx.linalg.inv(a, stream=mx.cpu)


# ── 总装门面 ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TemporalFusionLayer:
    """时空层门面: 帧间对应 → 运动估计 (motor + 协方差)。"""

    ekf: MotorEKF = field(default_factory=MotorEKF)

    def step(self, p_prev: mx.array, q_cur: mx.array) -> Motor:
        """一帧: 预测 + 更新 → 当前帧间 motor。"""
        self.ekf.predict()
        self.ekf.update(p_prev, q_cur)
        return self.ekf.motor()


if __name__ == "__main__":
    # ── 1. 核心: 已知 motor (含螺旋节距) 的恢复 ─────────────────────
    rng_key = mx.random.key(0)
    pts = mx.random.normal((60, 3), key=rng_key) * 0.5
    pts = pts + mx.array([0.0, 0.0, 3.0])  # 3m 前的点云
    m_true = Motor((0.3, 0.5, 0.8), 0.08, (0.05, -0.03, 0.02))  # 螺旋
    q_true = MotionCompensation.predict_points(pts, m_true)
    noise = mx.random.normal(q_true.shape, key=mx.random.key(1)) * 0.001
    q_obs = q_true + noise

    layer = TemporalFusionLayer()
    m_est = layer.step(pts, q_obs)
    err = mx.max(mx.abs(MotionCompensation.predict_points(pts, m_est) - q_true))
    assert float(err) < 5e-3, f"变换点偏差 {float(err):.4f}"
    print(f"1. motor 恢复: 变换点最大偏差 {float(err):.2e} (含螺旋) ✓")

    # ── 2. 转移先验: 常速度序列的平滑效应 ──────────────────────────
    m_step_true = Motor((0, 0, 1), 0.05, (0.02, 0.0, 0.0))
    layer2 = TemporalFusionLayer()
    pts_t = pts
    errs_raw, errs_ekf = [], []
    for f in range(3):
        pts_next = MotionCompensation.predict_points(pts_t, m_step_true)
        q_n = pts_next + mx.random.normal(
            pts_next.shape, key=mx.random.key(10 + f)
        ) * 0.002
        layer2.step(pts_t, q_n)  # 逐帧增量对应 (常速度 → 增量恒定)
        # 原始单帧估计: 独立 EKF (大初始协方差, 无历史)
        single = MotorEKF()
        single.predict()
        single.update(pts_t, q_n)
        errs_raw.append(float(mx.max(mx.abs(
            transform_points(single.xi, pts_t) - pts_next
        ))))
        errs_ekf.append(float(mx.max(mx.abs(
            transform_points(layer2.ekf.xi, pts_t) - pts_next
        ))))
        pts_t = pts_next
    import statistics
    assert statistics.mean(errs_ekf) <= statistics.mean(errs_raw) * 1.05, (
        f"EKF 平滑应不劣于单帧: {errs_ekf} vs {errs_raw}"
    )
    print(f"2. 转移先验: EKF 平均误差 {statistics.mean(errs_ekf):.2e} "
          f"≤ 单帧 {statistics.mean(errs_raw):.2e} ✓")

    # ── 3. 运动补偿: 平移场的对齐 ──────────────────────────────────
    H, W = 96, 128
    yy, xx = mx.meshgrid(mx.arange(H, dtype=mx.float32),
                         mx.arange(W, dtype=mx.float32), indexing="ij")
    f0 = mx.exp(-((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 100.0)
    m_shift = Motor(None, 0.0, (-3.0, -2.0, 0.0))  # 内容移动 (−3,−2)
    depth = mx.full((H, W), 3.0)
    # f1 = f0 平移后 (补偿应复原): f1[p] = f0[M⁻¹p] → compensate(f0) ≈ f1
    smp = EdgePrior.precomp_gather((H, W), mx.full((H, W), 2.0),
                                   mx.full((H, W), 3.0), yy, xx)
    f1 = smp(f0)
    f1_c = MotionCompensation.compensate(f0, depth, m_shift)
    from utils import Utils
    c_raw = Utils.corrcoef(f0.reshape(-1), f1.reshape(-1))
    c_cmp = Utils.corrcoef(f1_c.reshape(-1), f1.reshape(-1))
    assert c_cmp > 0.99 and c_cmp > c_raw, f"补偿后 {c_cmp:.4f} vs 未补偿 {c_raw:.4f}"
    print(f"3. 运动补偿: 相关 {c_raw:.3f} → {c_cmp:.4f} ✓")

    # ── 4. 图元跨帧变换 (cga blade 句柄) ───────────────────────────
    from cga import Plane, Point
    pl = Plane((0.0, 0.0, 1.0), 3.0)  # z = 3 平面
    pl_moved = m_step_true.apply(pl)
    # 平移 (0.02,0,0) 后, 点 (0.02,y,3) 应在新平面上
    p_check = Point(0.02, 1.0, 3.0)
    assert abs(pl_moved.dist(p_check)) < 1e-4, "图元变换后关联应保持"
    print("4. 图元跨帧变换: M.apply(plane) 关联保持 ✓")

    # ── 5. 退化守卫: 共线点集的轴向旋转不可观 ──────────────────────
    # 点全在 x 轴上: 绕 x 轴的旋转不改变任何点位置 → 残差恒零,
    # EKF 应停在 ξ=0 (不收敛于真值) —— 知道不收敛比悄悄错收敛好
    pts_line = mx.random.normal((40, 3), key=mx.random.key(7))
    pts_line = pts_line * mx.array([1.0, 0.0, 0.0])  # 全部在 x 轴
    m_bad = Motor((1, 0, 0), 0.3, (0.0, 0.0, 0.0))  # 绕 x 轴旋转
    q_bad = MotionCompensation.predict_points(pts_line, m_bad)
    layer3 = TemporalFusionLayer()
    layer3.step(pts_line, q_bad)
    xi_est = layer3.ekf.xi
    dev = float(mx.max(mx.abs(xi_est - motor_to_xi(m_bad))))
    assert float(mx.max(mx.abs(xi_est))) < 1e-3, "不可观方向应保持 ξ=0"
    print(f"5. 退化守卫: 轴向旋转不可观, 估计保持 0 (与真值偏差 {dev:.3f}, "
          f"已报告未收敛) ✓")

    # ── 6. C1 慢速标定: K 漂移修正 ─────────────────────────────────
    k_true = mx.array([100.0, 105.0, 64.0, 48.0])
    cal = SlowCalibration(k=mx.array([95.0, 100.0, 64.0, 48.0]))  # fx/fy 偏 5%
    single_errs = []
    for f in range(50):
        pts_c = mx.random.normal((30, 3), key=mx.random.key(100 + f))
        pts_c = pts_c * mx.array([1.0, 1.0, 0.5]) + mx.array([0.0, 0.0, 4.0])
        pix = project_points(pts_c, k_true)
        pix = pix + mx.random.normal(pix.shape, key=mx.random.key(200 + f)) * 0.5
        cal.update(pts_c, pix)
        # 单帧无历史估计 (同结构, 协方差重置 + lr=1)
        one = SlowCalibration(k=cal.k, lr=1.0)
        one.cov = mx.eye(4) * 25.0
        one.update(pts_c, pix)
        single_errs.append(float(mx.abs(one.k[0] - k_true[0]) / k_true[0]))
    rel = mx.abs(cal.k - k_true) / k_true
    assert float(mx.max(rel[:2])) < 0.01, f"K 漂移修正失败: {cal.k}"
    import statistics as _st
    assert float(rel[0]) < _st.mean(single_errs), (
        f"慢通道应优于单帧: {float(rel[0]):.4f} vs {_st.mean(single_errs):.4f}"
    )
    print(f"6. C1 标定: fx 误差 {float(rel[0]):.4f} (单帧均值 "
          f"{_st.mean(single_errs):.4f}), K={cal.k.tolist()} ✓")

    # ── 7. 低速先验 (Weiss slow-and-smooth) ──────────────────────
    # 7a. 收缩性: 同一份噪声对应, 有先验的步长模长必不大于无先验
    pts7 = mx.array(
        [[x, y, 3.0] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
    )
    xi7 = mx.array([0.0, 0.0, 0.0, 0.025, 0.0, 0.0])  # 半 twist 约定
    q7 = transform_points(xi7, pts7)  # 真值对应
    q7n = q7 + mx.random.normal(q7.shape, key=mx.random.key(3)) * 0.3
    ekf0 = MotorEKF()
    ekf0.predict()
    ekf0.update(pts7, q7n)
    ekf1 = MotorEKF(r_slow=0.01)
    ekf1.predict()
    ekf1.update(pts7, q7n)
    n0, n1 = float(mx.sum(ekf0.xi**2)), float(mx.sum(ekf1.xi**2))
    assert n1 < n0, f"先验应收缩: {n1:.4f} vs {n0:.4f}"
    # 7b. 强一致证据下先验偏置可略: 干净对应 + 先验仍复原平移
    ekf2 = MotorEKF(r_slow=0.01)
    ekf2.predict()
    ekf2.update(pts7, q7)
    dx7 = 2.0 * float(ekf2.xi[3])  # 半 twist → 物理位移
    assert abs(dx7 - 0.05) < 0.01, f"干净对应下应复原: {dx7:.4f}"
    print(f"7. 低速先验: 噪声下收缩 ({n1:.4f}<{n0:.4f}), "
          f"干净对应偏置可略 (dx={dx7:.4f}) ✓")
