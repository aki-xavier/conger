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
    mv_zeros,
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


def apply_motor(M: Multivector, obj: Multivector) -> Multivector:
    """Apply a motor transformation to a CGA object.

    O' = M O M̃  where M̃ is the reverse of M.

    Args:
        M: A motor.
        obj: Any CGA multivector (point, line, plane, sphere, etc.).

    Returns:
        The transformed object.
    """
    M_rev = reverse(M)
    return gp(gp(M, obj), M_rev)


def exp_bivector(B: Multivector, scale: float = 1.0) -> Multivector:
    """Exponentiate a bivector: exp(-scale * B).

    For a bivector B in CGA, exp(-B) is a motor (when B represents a
    velocity bivector × dt/2).

    exp(-s*B) = cos(s*|B|) - sin(s*|B|) * B/|B|  (when B^2 < 0, i.e. elliptic)
             = cosh(s*|B|) - sinh(s*|B|) * B/|B|  (when B^2 > 0, i.e. hyperbolic)
             = 1 - s*B  (when B^2 = 0)

    For a general velocity bivector V in CGA:
    V = ω + v ∧ e∞  where ω is angular velocity bivector, v is linear velocity.
    exp(-dt/2 * V) = T * R  (a motor).
    """
    # Compute B^2 = B * B (geometric product)
    B2 = gp(B, B)
    B2_scalar = float(B2.values[0])  # scalar part of B^2

    if abs(B2_scalar) < 1e-12:
        # Check if B itself is zero
        B_vals = B.values
        bulk_sq = float(B_vals[6] ** 2 + B_vals[7] ** 2 + B_vals[10] ** 2)
        # 平移 bivector v∧e∞ 的分量槽位: e_i∧e+ 与 e_i∧e- 成对出现
        trans_sq = float(sum(B_vals[i] ** 2 for i in (8, 9, 11, 12, 13, 14)))
        if bulk_sq < 1e-12 and trans_sq < 1e-12:
            return mv_scalar(1.0)
        # B^2 = 0 but B ≠ 0: exp(-s*B) = 1 - s*B
        return mv_scalar(1.0) - B * scale

    if B2_scalar < 0:
        # Elliptic case: B^2 = -α^2 (α real)
        alpha = math.sqrt(-B2_scalar) * abs(scale)
        if alpha < 1e-12:
            return mv_scalar(1.0)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        B_norm = B / math.sqrt(-B2_scalar)
        return mv_scalar(cos_a) - B_norm * (sin_a * abs(scale))
    else:
        # Hyperbolic case: B^2 = α^2 (α real)
        alpha = math.sqrt(B2_scalar) * abs(scale)
        cosh_a = math.cosh(alpha)
        sinh_a = math.sinh(alpha)
        B_norm = B / math.sqrt(B2_scalar)
        return mv_scalar(cosh_a) - B_norm * (sinh_a * abs(scale))


def log_motor(M: Multivector) -> Multivector:
    """Logarithm of a motor: compute the bivector whose exponential is M.

    Useful for interpolation and velocity extraction.
    """
    # Extract scalar and bivector parts
    s = float(M.values[0])
    B = M.grade(2)

    B2 = gp(B, B)
    B2_scalar = float(B2.values[0])

    if B2_scalar <= 0:
        # Elliptic
        alpha = math.sqrt(-B2_scalar)
        if alpha < 1e-12:
            return mv_zeros()
        theta = math.atan2(alpha, s)
        return B * (-theta / alpha)
    else:
        # Hyperbolic
        alpha = math.sqrt(B2_scalar)
        theta = math.atanh(alpha / s) if abs(s) > abs(alpha) else 1.0
        return B * (-theta / alpha)


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
    return gp(M1, exp_bivector(log_delta, -t))


def motor_to_matrix(M: Multivector) -> list[list[float]]:
    """Convert a CGA motor to a 4x4 homogeneous transformation matrix.

    The motor M acts on conformal points as: p' = M p M̃.
    This extracts the equivalent 4x4 matrix [R | t; 0 1].

    This is useful for traditional rendering pipelines that expect
    4x4 matrices.
    """
    # Transform the origin to get the translation
    origin_t = apply_motor(M, E0)
    tx = float(origin_t.values[1])
    ty = float(origin_t.values[2])
    tz = float(origin_t.values[3])

    # Transform unit points to extract the rotation matrix columns
    px = point(1, 0, 0)
    py = point(0, 1, 0)
    pz = point(0, 0, 1)

    px_t = apply_motor(M, px)
    py_t = apply_motor(M, py)
    pz_t = apply_motor(M, pz)

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

    V ≈ -2 * log(M_prev^{-1} * M_curr) / dt

    Args:
        M_current: Current motor.
        M_previous: Previous motor.
        dt: Time step.

    Returns:
        (angular_velocity, linear_velocity) as ((ωx,ωy,ωz), (vx,vy,vz)).
    """
    M_prev_rev = reverse(M_previous)
    delta = gp(M_prev_rev, M_current)
    log_delta = log_motor(delta)
    V = log_delta * (-2.0 / dt)

    vals = V.values
    # Extract from bivector components
    wx = float(vals[10])  # e23
    wy = -float(vals[7])  # e13 (negated because e31 = -e13)
    wz = float(vals[6])  # e12
    # v∧e∞ 的系数 = e_i∧e+ 与 e_i∧e- 两槽之和 (e∞ = e+ + e-)
    vx = float(vals[8] + vals[9])  # e1∧e∞
    vy = float(vals[11] + vals[12])  # e2∧e∞
    vz = float(vals[13] + vals[14])  # e3∧e∞

    return ((wx, wy, wz), (vx, vy, vz))
