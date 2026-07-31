"""CGA Motors — Rigid-body transformations in conformal space.

In CGA, rigid-body motions (rotation + translation) are represented by
even-grade multivectors called *motors*. A motor M transforms any object O as:

    O' = M O M̃

where M̃ is the reverse of M.

Key primitives:
  - Rotor:     R = exp(-θ/2 * B)    for rotation by θ around bivector plane B
  - Translator: T = 1 - (t/2)*e∞   for translation by vector t
  - Motor:     M = T * R           composite rigid-body motion

The velocity state of a rigid body is a bivector V (6 degrees of freedom),
and the equation of motion is: dM/dt = -1/2 * V * M
"""

import math

import mlx.core as mx

from cga.algebra import (
    E0,
    EINF,
    gp,
    op,
    point,
    reverse,
)
from cga.multivector import (
    NUM_COMPONENTS,
    Multivector,
    mv_scalar,
)


def rotor(axis: tuple[float, float, float], angle: float) -> Multivector:
    """Create a rotor for rotation around an axis by a given angle.

    R = cos(θ/2) - sin(θ/2) * (nx*e23 + ny*e31 + nz*e12)

    Where e31 = e3∧e1 = -e13.

    Args:
        axis: (ax, ay, az) — rotation axis (will be normalized).
        angle: rotation angle in radians.

    Returns:
        A unit rotor (grade 0 + grade 2 multivector).
    """
    ax, ay, az = axis
    norm_ax = math.sqrt(ax * ax + ay * ay + az * az)
    if norm_ax < 1e-12:
        return mv_scalar(1.0)

    ax, ay, az = ax / norm_ax, ay / norm_ax, az / norm_ax
    half_angle = angle / 2.0
    s = math.cos(half_angle)
    sf = math.sin(half_angle)

    vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    vals[0] = s  # cos(θ/2)
    vals[6] = -sf * az  # e12: -nz
    vals[7] = sf * ay  # e13: +ny  (since -ny*e31 = +ny*e13)
    vals[10] = -sf * ax  # e23: -nx

    return Multivector(vals)


def rotor_from_quaternion(q: tuple[float, float, float, float]) -> Multivector:
    """Create a rotor from a quaternion in (w, x, y, z) order (MJCF convention).

    Args:
        q: Quaternion (w, x, y, z). Will be normalized.

    Returns:
        The equivalent unit rotor.
    """
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return mv_scalar(1.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), w)
    return rotor((x, y, z), angle)


def translator(displacement: tuple[float, float, float]) -> Multivector:
    """Create a translator for translation by a displacement vector.

    T = 1 - (t ∧ e∞)/2

    Args:
        displacement: (tx, ty, tz) translation vector.

    Returns:
        A translator (grade 0 + grade 2 multivector).
    """
    tx, ty, tz = displacement
    tv = Multivector.vector(tx, ty, tz)
    return mv_scalar(1.0) - op(tv, EINF) * 0.5


def motor_from_rotor_translator(R: Multivector, T: Multivector) -> Multivector:
    """Combine a rotor and translator into a motor: M = T * R.

    Args:
        R: A rotor (rotation).
        T: A translator (translation).

    Returns:
        The combined motor M.
    """
    return gp(T, R)


def motor(
    rotation_axis: tuple[float, float, float] | None = None,
    rotation_angle: float = 0.0,
    translation: tuple[float, float, float] = (0, 0, 0),
) -> Multivector:
    """Create a motor from rotation and translation.

    Args:
        rotation_axis: (ax, ay, az) — rotation axis, None for no rotation.
        rotation_angle: rotation angle in radians.
        translation: (tx, ty, tz) translation vector.

    Returns:
        A motor M such that M = T * R.
    """
    if rotation_axis is not None and abs(rotation_angle) > 1e-12:
        ro = rotor(rotation_axis, rotation_angle)
    else:
        ro = mv_scalar(1.0)

    tx, ty, tz = translation
    if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
        tr = translator(translation)
    else:
        tr = mv_scalar(1.0)

    return motor_from_rotor_translator(ro, tr)


def identity_motor() -> Multivector:
    """Return the identity motor (no rotation, no translation)."""
    return mv_scalar(1.0)


def apply_motor(obj: Multivector, M: Multivector) -> Multivector:
    """Apply a motor transformation to a CGA object.

    O' = M O M̃  where M̃ is the reverse of M.

    参数序 (obj, M), 与 simu.cga 一致。

    Args:
        obj: Any CGA multivector (point, line, plane, sphere, etc.).
        M: A motor.

    Returns:
        The transformed object.
    """
    M_rev = reverse(M)
    return gp(gp(M, obj), M_rev)


# ── 3×3 mlx 助手 (SE(3) exp/log 用, 与 simu.cga.motors 一致) ──────────────


def _as_mat3(matrix) -> mx.array:
    """Coerce a 3x3 (or flat length-9) array-like to an mx float32 matrix."""
    m = mx.array(matrix, dtype=mx.float32)
    if m.ndim == 1 and m.size == 9:
        m = m.reshape(3, 3)
    return m


def matrix_to_quaternion(matrix) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to `(w, x, y, z)` quaternion."""
    m = _as_mat3(matrix)
    trace = float(mx.diagonal(m).sum())
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            float(m[2, 1] - m[1, 2]) / s,
            float(m[0, 2] - m[2, 0]) / s,
            float(m[1, 0] - m[0, 1]) / s,
        )
    if float(m[0, 0]) > float(m[1, 1]) and float(m[0, 0]) > float(m[2, 2]):
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        return (
            float(m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            float(m[0, 1] + m[1, 0]) / s,
            float(m[0, 2] + m[2, 0]) / s,
        )
    if float(m[1, 1]) > float(m[2, 2]):
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        return (
            float(m[0, 2] - m[2, 0]) / s,
            float(m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            float(m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
    return (
        float(m[1, 0] - m[0, 1]) / s,
        float(m[0, 2] + m[2, 0]) / s,
        float(m[1, 2] + m[2, 1]) / s,
        0.25 * s,
    )


def motor_from_matrix(R, t) -> Multivector:
    """Build a motor from a 3x3 rotation matrix and translation vector.

    M = T(t) · R, so that motor_to_matrix(motor_from_matrix(R, t)) == [R|t].
    """
    tv = tuple(float(x) for x in mx.array(t, dtype=mx.float32))
    return gp(
        translator(tv),
        rotor_from_quaternion(matrix_to_quaternion(_as_mat3(R))),
    )


def exp_bivector(B: Multivector, scale: float = 1.0) -> Multivector:
    """Exponentiate a bivector: exp(-scale · B), which is a motor.

    分解 B 的旋转/平移部分, 走 SE(3) 指数映射 (Rodrigues + SO(3) 左
    雅可比), 对一般螺旋运动 (非零节距) 精确——闭式 B² 符号分类只对
    纯旋转/纯平移成立。B 的分量约定为半 twist:
    B = ½(ω̄_bivector + v̄∧e∞), 与运动方程 dM/dt = -½·V·M 一致。

      - 纯平移 (ω̄ = 0): 1 - scale·B        (B² = 0, 级数截断, 精确)
      - 纯旋转 (v̄ = 0): rotor(ω̄/|ω̄|, |ω̄|)
      - 一般螺旋:        T(V̄·v̄) · R(rodrigues(ω̄))
    """
    Bv = B * scale
    vals = Bv.values
    wx, wy, wz = float(vals[10]), -float(vals[7]), float(vals[6])  # e23,e31,e12
    vx, vy, vz = float(vals[9]), float(vals[12]), float(vals[14])  # e_i∧e∞

    w_bar = mx.array([2.0 * wx, 2.0 * wy, 2.0 * wz], dtype=mx.float32)
    v_bar = mx.array([2.0 * vx, 2.0 * vy, 2.0 * vz], dtype=mx.float32)

    theta = float(mx.sqrt((w_bar * w_bar).sum()))
    v_norm = float(mx.sqrt((v_bar * v_bar).sum()))
    if theta < 1e-12:
        if v_norm < 1e-12:
            return mv_scalar(1.0)
        # 纯平移: Bv 幂零, 级数截断
        return mv_scalar(1.0) - Bv
    if v_norm < 1e-12:
        # 纯旋转 (过原点)
        axis = (w_bar / theta).tolist()
        return rotor((axis[0], axis[1], axis[2]), theta)

    # 一般螺旋: Rodrigues + SO(3) 左雅可比
    bx, by, bz = float(w_bar[0]), float(w_bar[1]), float(w_bar[2])
    W = mx.array(
        [
            [0.0, -bz, by],
            [bz, 0.0, -bx],
            [-by, bx, 0.0],
        ],
        dtype=mx.float32,
    )
    WW = mx.matmul(W, W, stream=mx.cpu)  # CPU stream: GPU matmul is reduced-precision
    theta2 = theta * theta
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    a_r, b_r = sin_t / theta, (1.0 - cos_t) / theta2
    a_v, b_v = (1.0 - cos_t) / theta2, (theta - sin_t) / (theta2 * theta)
    eye = mx.eye(3)
    R = eye + a_r * W + b_r * WW
    V = eye + a_v * W + b_v * WW
    t = mx.matmul(V, v_bar, stream=mx.cpu).tolist()
    return motor_from_matrix(R, t)


def log_motor(M: Multivector) -> Multivector:
    """Logarithm of a motor: the bivector Bv with exp(-Bv) = M.

    走 SE(3) 矩阵对数 (含 θ≈π 的对称部分恢复分支), 对一般螺旋运动
    (非零节距) 精确——"标量+二重向量"闭式只对纯旋转成立, 纯平移
    (幂零) 还会整体归零。结果分量约定为半 twist:
    Bv = ½(ω̄_bivector + v̄∧e∞), 与 dM/dt = -½·V·M 一致。
    """
    T = mx.array(motor_to_matrix(M), dtype=mx.float32)
    R = T[:3, :3]
    t = T[:3, 3]

    trace = float(mx.diagonal(R).sum())
    cos_theta = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    # θ 用 atan2 提取 (|antisym| = 2·sinθ): acos((trace-1)/2) 在 θ→0 时
    # 把 float32 trace 噪声 (~1e-7) 二次放大成 ~1e-4 的 θ 误差。
    antisym = mx.stack(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    sin_theta_abs = 0.5 * float(mx.sqrt((antisym * antisym).sum()))
    theta = math.atan2(sin_theta_abs, cos_theta)

    if theta < 1e-9:
        # 纯平移: v̄ = t
        w_bar = mx.zeros(3, dtype=mx.float32)
        v_bar = t
    else:
        sin_theta = math.sin(theta)
        if theta < math.pi - 1e-3:
            c = theta / (2.0 * sin_theta)
            w_bar = c * antisym
        else:
            # θ ≈ π: 反对称公式除以 sin θ 爆炸, 改从对称部分恢复转轴
            # (R = 2·aaᵀ - I → R[i][j] = 2·a_i·a_j, i≠j), 用最大的
            # 轴分量作符号参考 (对 float32 噪声最稳健)
            axis = mx.sqrt(mx.maximum((mx.diagonal(R) + 1.0) / 2.0, 0.0))
            axis_l = axis.tolist()
            R_l = R.tolist()
            ref = max(range(3), key=lambda i: abs(axis_l[i]))
            if ref == 0:
                axis_l[1] = math.copysign(axis_l[1], R_l[0][1])
                axis_l[2] = math.copysign(axis_l[2], R_l[0][2])
            elif ref == 1:
                axis_l[0] = math.copysign(axis_l[0], R_l[0][1])
                axis_l[2] = math.copysign(axis_l[2], R_l[1][2])
            else:
                axis_l[0] = math.copysign(axis_l[0], R_l[0][2])
                axis_l[1] = math.copysign(axis_l[1], R_l[1][2])
            w_bar = mx.array(axis_l, dtype=mx.float32) * theta

        # SO(3) 左雅可比的逆: v̄ = V̄⁻¹·t
        bx, by, bz = float(w_bar[0]), float(w_bar[1]), float(w_bar[2])
        wxm = mx.array(
            [
                [0.0, -bz, by],
                [bz, 0.0, -bx],
                [-by, bx, 0.0],
            ],
            dtype=mx.float32,
        )
        wx2 = mx.matmul(wxm, wxm, stream=mx.cpu)
        theta2 = theta * theta
        coeff = 1.0 / theta2 - (1.0 + cos_theta) / (2.0 * theta * sin_theta)
        V_inv = mx.eye(3) - 0.5 * wxm + coeff * wx2
        v_bar = mx.matmul(V_inv, t, stream=mx.cpu)

    w_l = (w_bar / 2.0).tolist()
    v_l = (v_bar / 2.0).tolist()
    return velocity_bivector(
        (w_l[0], w_l[1], w_l[2]),
        (v_l[0], v_l[1], v_l[2]),
    )


def interpolate_motor(M1: Multivector, M2: Multivector, t: float) -> Multivector:
    """Interpolate between two motors.

    M(t) = M1 * exp(t * log(M1^{-1} * M2))

    Args:
        M1: Start motor.
        M2: End motor.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated motor.
    """
    M1_rev = reverse(M1)
    delta = gp(M1_rev, M2)
    log_delta = log_motor(delta)
    return gp(M1, exp_bivector(log_delta, t))


def motor_to_matrix(M: Multivector) -> list[list[float]]:
    """Convert a CGA motor to a 4x4 homogeneous transformation matrix.

    The motor M acts on conformal points as: p' = M p M̃.
    This extracts the equivalent 4x4 matrix [R | t; 0 1].

    This is useful for traditional rendering pipelines that expect
    4x4 matrices.
    """
    # Transform the origin to get the translation
    origin_t = apply_motor(E0, M)
    tx = float(origin_t.values[1])
    ty = float(origin_t.values[2])
    tz = float(origin_t.values[3])

    # Transform unit points to extract the rotation matrix columns
    px = point(1, 0, 0)
    py = point(0, 1, 0)
    pz = point(0, 0, 1)

    px_t = apply_motor(px, M)
    py_t = apply_motor(py, M)
    pz_t = apply_motor(pz, M)

    r00 = float(px_t.values[1]) - tx
    r10 = float(px_t.values[2]) - ty
    r20 = float(px_t.values[3]) - tz

    r01 = float(py_t.values[1]) - tx
    r11 = float(py_t.values[2]) - ty
    r21 = float(py_t.values[3]) - tz

    r02 = float(pz_t.values[1]) - tx
    r12 = float(pz_t.values[2]) - ty
    r22 = float(pz_t.values[3]) - tz

    return [
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def velocity_bivector(
    angular: tuple[float, float, float], linear: tuple[float, float, float]
) -> Multivector:
    """Create a velocity bivector from angular and linear velocity.

    In CGA, the velocity bivector V represents the twist:
    V = ω + v ∧ e∞
    where ω = ωx*e23 + ωy*e31 + ωz*e12 is the angular velocity bivector,
    and v = vx*e1 + vy*e2 + vz*e3 is the linear velocity.

    The equation of motion is: dM/dt = -1/2 * V * M

    Args:
        angular: (ωx, ωy, ωz) angular velocity.
        linear: (vx, vy, vz) linear velocity.

    Returns:
        Velocity bivector.
    """
    wx, wy, wz = angular
    vx_val, vy_val, vz_val = linear

    vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    # Angular part (bivector):
    vals[6] = wz  # e12
    vals[7] = -wy  # e13 (from ωy*e31 = -ωy*e13)
    vals[10] = wx  # e23

    rot = Multivector(vals)
    tv = Multivector.vector(vx_val, vy_val, vz_val)
    return rot + op(tv, EINF)


def extract_velocity(
    M_current: Multivector, M_previous: Multivector, dt: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Extract angular and linear velocity from two consecutive motors.

    V ≈ 2 * log(M_prev^{-1} * M_curr) / dt
    (delta = exp(-½V·dt) → log(delta) = ½V·dt → V = 2·log(delta)/dt)

    Frame convention: the relative motor M_prev^{-1}·M_curr is expressed
    in the PREVIOUS BODY frame, so the returned twist is the body-frame
    velocity at the previous time step — NOT a world-frame twist.
    Transform it with M_previous before feeding world-frame consumers.

    Args:
        M_current: Current motor.
        M_previous: Previous motor.
        dt: Time step; must be > 0.

    Returns:
        (angular_velocity, linear_velocity) as ((ωx,ωy,ωz), (vx,vy,vz)),
        expressed in the previous body frame.

    Raises:
        ValueError: If dt <= 0.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    M_prev_rev = reverse(M_previous)
    delta = gp(M_prev_rev, M_current)
    log_delta = log_motor(delta)
    V = log_delta * (2.0 / dt)

    vals = V.values
    # Extract from bivector components
    wx = float(vals[10])  # e23
    wy = -float(vals[7])  # e13 (negated because e31 = -e13)
    wz = float(vals[6])  # e12
    # v∧e∞ 分量槽位: (i, 4) = e_i∧e∞
    vx = float(vals[9])  # e1∧e∞
    vy = float(vals[12])  # e2∧e∞
    vz = float(vals[14])  # e3∧e∞

    return ((wx, wy, wz), (vx, vy, vz))
