"""CGA Algebra Operations.

Core operations: geometric product, inner product, outer product,
reverse, dual, norm, and constructors for CGA primitives.

All operations use precomputed product tables for GPU-accelerated
computation via MLX.

表示约定 (统一): 所有原语构造器返回**直接形式** (join 表示)——
  point (grade 1) / point_pair (grade 2) / line, circle (grade 3) /
  plane, sphere (grade 4)
关联判据统一为 `op(up(p), X) = 0` (点 p 在原语 X 上)。
距离函数接受直接形式, 内部自行取对偶; meet 用于求交。
"""

import math

import mlx.core as mx

from cga.multivector import (
    GP_MASK,
    GP_NONZERO_I,
    GP_NONZERO_J,
    GRADE_INDICES,
    GRADE_MASKS,
    NUM_COMPONENTS,
    Multivector,
)


def gp(a: Multivector, b: Multivector) -> Multivector:
    """Geometric product of two multivectors.

    Uses the precomputed 32x32x32 GP_MASK with sparse (i,j) indexing.
    result[k] = sum_{i,j} GP_MASK[i,j,k] * a[i] * b[j]
    """
    a_vals = a.values
    b_vals = b.values

    # Compute all a_i * b_j products for non-zero GP pairs
    prod = a_vals[GP_NONZERO_I] * b_vals[GP_NONZERO_J]  # shape (N,)

    # Gather the corresponding mask rows
    mask_rows = GP_MASK[GP_NONZERO_I, GP_NONZERO_J, :]  # shape (N, 32)

    # Weighted sum: result[k] = sum_n mask_rows[n,k] * prod[n]
    result = (mask_rows * prod[:, None]).sum(axis=0)

    return Multivector(result)


def ip(a: Multivector, b: Multivector) -> Multivector:
    """Left contraction (inner product) a ⌋ b.

    For blades: grade_k ⌋ grade_m = <grade_k * grade_m>_{|m-k|} if m >= k, else 0.
    """
    full_gp = gp(a, b)
    a_grade = _dominant_grade(a)
    b_grade = _dominant_grade(b)
    target_grade = abs(b_grade - a_grade) if b_grade >= a_grade else -1

    if target_grade < 0:
        return Multivector.zeros()

    return Multivector(full_gp.values * GRADE_MASKS[target_grade])


def op(a: Multivector, b: Multivector) -> Multivector:
    """Outer product a ∧ b.

    For blades: grade_k ∧ grade_m = <grade_k * grade_m>_{k+m}.
    """
    full_gp = gp(a, b)
    a_grade = _dominant_grade(a)
    b_grade = _dominant_grade(b)
    target_grade = a_grade + b_grade

    if target_grade >= 6:
        return Multivector.zeros()

    return Multivector(full_gp.values * GRADE_MASKS[target_grade])


def reverse(a: Multivector) -> Multivector:
    """Reverse involution: reverses the order of basis vectors in each blade.

    For a grade-k blade: rev(A_k) = (-1)^{k(k-1)/2} A_k.
    """
    vals = mx.array(a.values)  # copy
    for g in range(6):
        sign = (-1) ** (g * (g - 1) // 2)
        if sign == -1:
            for idx in GRADE_INDICES[g]:
                vals[idx] = -vals[idx]
    return Multivector(vals)


def grade_involution(a: Multivector) -> Multivector:
    """Grade involution: negates odd-grade components."""
    vals = mx.array(a.values)
    for g in [1, 3, 5]:
        for idx in GRADE_INDICES[g]:
            vals[idx] = -vals[idx]
    return Multivector(vals)


def conjugate(a: Multivector) -> Multivector:
    """Clifford conjugate: reverse + grade involution."""
    return grade_involution(reverse(a))


def dual(a: Multivector) -> Multivector:
    """Hodge dual: multiply by the inverse pseudoscalar I^{-1}.

    In CGA, I = e1230∞, and I^2 = -1, so I^{-1} = -I.
    dual(A) = A * I^{-1} = -A * I.
    """
    I_vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    I_vals[31] = 1.0
    I_inv = Multivector(-I_vals)
    return gp(a, I_inv)


def meet(a: Multivector, b: Multivector) -> Multivector:
    """Intersection of two direct-form primitives: A ∨ B = (A* ∧ B*)*.

    例: meet(plane, plane) = line; meet(line, sphere) = point_pair.
    """
    return dual(op(dual(a), dual(b)))


def undual(a: Multivector) -> Multivector:
    """对偶的逆: dual(dual(x)) = -x (因 I⁻² = I² = -1), 故 undual = -dual。

    从直接形式还原对偶形式 (n + d·e∞ / up(c) − ½ρ²e∞) 时使用。
    """
    return -dual(a)


def norm(a: Multivector) -> float:
    """Euclidean norm: sqrt(|scalar_part(a * reverse(a))|)."""
    rev = reverse(a)
    prod = gp(a, rev)
    s = float(prod.values[0])
    return math.sqrt(abs(s))


def normalize(a: Multivector) -> Multivector:
    """Normalize to unit norm."""
    n = norm(a)
    if n < 1e-12:
        return Multivector.zeros()
    return a / n


def bulk(a: Multivector) -> Multivector:
    """Extract the Euclidean (bulk) part: grades containing no e0 or einf."""
    euc_indices = [0, 1, 2, 3, 6, 7, 10, 16]
    vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    for idx in euc_indices:
        vals[idx] = a.values[idx]
    return Multivector(vals)


def weight(a: Multivector) -> Multivector:
    """Extract the conformal (weight) part: grades involving e0/einf."""
    return a - bulk(a)


def _dominant_grade(a: Multivector) -> int:
    """Find the grade with the largest component magnitude."""
    max_mag = 0.0
    best_grade = 0
    for g in range(6):
        mag = float(mx.sum(mx.abs(a.values[GRADE_INDICES[g]])).item())
        if mag > max_mag:
            max_mag = mag
            best_grade = g
    return best_grade


# ── Basis vectors ──────────────────────────────────────────────────────────

E1 = Multivector.vector(1, 0, 0, 0, 0)
E2 = Multivector.vector(0, 1, 0, 0, 0)
E3 = Multivector.vector(0, 0, 1, 0, 0)
E0 = Multivector.vector(0, 0, 0, 1, 0)
EINF = Multivector.vector(0, 0, 0, 0, 1)


# ── CGA Object Constructors (统一返回直接/join 形式) ───────────────────────


def point(x: float, y: float, z: float) -> Multivector:
    """Create a conformal point from Euclidean coordinates (grade 1).

    p = e0 + x*e1 + y*e2 + z*e3 + 0.5*(x^2+y^2+z^2)*e∞
    This satisfies p·p = 0 (null vector property of conformal points).
    """
    r2 = x * x + y * y + z * z
    return Multivector.vector(x, y, z, 1.0, 0.5 * r2)


def point_pair(p1: Multivector, p2: Multivector) -> Multivector:
    """Create a point pair (0-sphere) from two conformal points (grade 2).

    Pp = p1 ∧ p2
    """
    return op(p1, p2)


def line(p1: Multivector, p2: Multivector) -> Multivector:
    """Create a line from two conformal points (grade 3).

    L = p1 ∧ p2 ∧ e∞
    """
    return op(op(p1, p2), EINF)


def plane(normal_vec: tuple[float, float, float], distance: float) -> Multivector:
    """Create a plane from a normal vector and distance from origin (grade 4).

    直接形式 = dual(n + d*e∞), n 为单位法向量。
    """
    nx, ny, nz = normal_vec
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl > 1e-12:
        nx, ny, nz = nx / nl, ny / nl, nz / nl
    return dual(Multivector.vector(nx, ny, nz, 0.0, distance))


def sphere(center: tuple[float, float, float], radius: float) -> Multivector:
    """Create a sphere from center and radius (grade 4).

    直接形式 = dual(up(c) - 0.5*r^2*e∞)。
    """
    cx, cy, cz = center
    pc = point(cx, cy, cz)
    half_r2 = 0.5 * radius * radius
    correction = Multivector.vector(0, 0, 0, 0, half_r2)
    return dual(pc - correction)


def circle(
    center: tuple[float, float, float],
    radius: float,
    normal: tuple[float, float, float],
) -> Multivector:
    """Create a circle as sphere ∩ plane (grade 3, 直接形式).

    C = dual(对偶球 ∧ 对偶平面) —— 即 meet 后再取一致表示。
    """
    cx, cy, cz = center
    pc = point(cx, cy, cz)
    s_dual = pc - Multivector.vector(0, 0, 0, 0, 0.5 * radius * radius)
    nx, ny, nz = normal
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl > 1e-12:
        nx, ny, nz = nx / nl, ny / nl, nz / nl
    d = cx * nx + cy * ny + cz * nz
    p_dual = Multivector.vector(nx, ny, nz, 0.0, d)
    return dual(op(s_dual, p_dual))


def mv_scalar_inline(s: float) -> Multivector:
    """Create a scalar multivector (inline helper)."""
    vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    vals[0] = s
    return Multivector(vals)


# ── Distance functions (接受直接形式, 内部取对偶) ──────────────────────────


def dist_point_point(p1: Multivector, p2: Multivector) -> float:
    """Euclidean distance between two conformal points.

    d(p1, p2) = sqrt(-2 * p1·p2)
    """
    dot = float(ip(p1, p2).values[0])
    d_sq = -2.0 * dot
    return math.sqrt(max(0.0, d_sq))


def dist_point_plane(p: Multivector, pi: Multivector) -> float:
    """Signed distance from a conformal point to a (direct-form) plane.

    d = p·π* / |π*_bulk|, π* 为对偶平面。
    """
    pi_dual = undual(pi)
    dot = float(ip(p, pi_dual).values[0])
    n_bulk = norm(bulk(pi_dual))
    if n_bulk < 1e-12:
        return float("inf")
    return dot / n_bulk


def dist_point_sphere(p: Multivector, s: Multivector) -> float:
    """Signed distance from a point to a (direct-form) sphere surface.

    对偶球 s* = up(c) − ½ρ²e∞ (weight w = −s*·e∞, 归一化时 w=1):
    ρ² = s*²/w², c = s* 的欧氏部分 / w。
    Positive = outside, negative = inside.
    """
    s_dual = undual(s)
    s_vals = s_dual.values
    w = -float(ip(s_dual, EINF).values[0])
    cx = float(s_vals[1]) / w
    cy = float(s_vals[2]) / w
    cz = float(s_vals[3]) / w
    rho_sq = float(gp(s_dual, s_dual).values[0]) / (w * w)
    r = math.sqrt(max(0.0, rho_sq))
    pc = point(cx, cy, cz)
    return dist_point_point(p, pc) - r
