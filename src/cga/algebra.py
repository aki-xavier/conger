"""CGA Algebra Operations.

Core operations: geometric product, inner product, outer product,
reverse, dual, norm, and constructors for CGA primitives.

All operations use precomputed product tables for GPU-accelerated
computation via MLX.

表示约定 (与 simu.cga 一致):
  - 点/点对/线为**直接 (join) 形式**: point (grade 1) / point_pair
    (grade 2) / line (grade 3 = p1∧p2∧e∞); 关联判据 op(p, X) = 0。
  - 平面/球/圆为**对偶形式**: plane, sphere (grade 1 向量) / circle
    (grade 2); 关联判据 ip(p, X) = 0。
  距离函数直接读取对偶形式的系数 (float64); meet 接受直接形式输入。
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
    NUM_GRADES,
    Multivector,
)


def _grade_signs(sign_of_grade) -> mx.array:
    """Build a per-component ±1 mask from a per-grade sign function."""
    vals = [1.0] * NUM_COMPONENTS
    for g in range(NUM_GRADES):
        for idx in GRADE_INDICES[g]:
            vals[idx] = float(sign_of_grade(g))
    return mx.array(vals, dtype=mx.float32)


# Sign masks for the involutions, precomputed once at module load.
_REVERSE_MASK = _grade_signs(lambda g: (-1) ** (g * (g - 1) // 2))
_INVOLUTION_MASK = _grade_signs(lambda g: -1 if g % 2 else 1)


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

    Linear extension of the blade rule over all grade pairs:
        a ⌋ b = Σ_{r<=s} < <a>_r * <b>_s >_{s-r}
    Correct for general mixed-grade multivectors.
    """
    result = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    for ga in range(NUM_GRADES):
        a_g = a.values * GRADE_MASKS[ga]
        if not bool(mx.any(a_g != 0).item()):
            continue
        for gb in range(ga, NUM_GRADES):
            b_g = b.values * GRADE_MASKS[gb]
            if not bool(mx.any(b_g != 0).item()):
                continue
            prod = gp(Multivector(a_g), Multivector(b_g))
            result = result + prod.values * GRADE_MASKS[gb - ga]
    return Multivector(result)


def op(a: Multivector, b: Multivector) -> Multivector:
    """Outer product a ∧ b.

    Linear extension of the blade rule over all grade pairs:
        a ∧ b = Σ_{r,s} < <a>_r * <b>_s >_{r+s}
    Correct for general mixed-grade multivectors.
    """
    result = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    for ga in range(NUM_GRADES):
        a_g = a.values * GRADE_MASKS[ga]
        if not bool(mx.any(a_g != 0).item()):
            continue
        for gb in range(NUM_GRADES - ga):
            b_g = b.values * GRADE_MASKS[gb]
            if not bool(mx.any(b_g != 0).item()):
                continue
            prod = gp(Multivector(a_g), Multivector(b_g))
            result = result + prod.values * GRADE_MASKS[ga + gb]
    return Multivector(result)


def reverse(a: Multivector) -> Multivector:
    """Reverse involution: reverses the order of basis vectors in each blade.

    For a grade-k blade: rev(A_k) = (-1)^{k(k-1)/2} A_k.
    """
    return Multivector(a.values * _REVERSE_MASK)


def grade_involution(a: Multivector) -> Multivector:
    """Grade involution: negates odd-grade components."""
    return Multivector(a.values * _INVOLUTION_MASK)


def conjugate(a: Multivector) -> Multivector:
    """Clifford conjugate: reverse + grade involution."""
    return grade_involution(reverse(a))


def dual(a: Multivector) -> Multivector:
    """Hodge dual: multiply by the inverse pseudoscalar I⁻¹.

    Orientation convention (与 simu.cga 一致): I = e123 ∧ e∞ ∧ e0,
    matching the `clifford` library's conformal pseudoscalar e12345
    (since e∞ ∧ e0 = +e45 in its e4/e5 basis). With this orientation
    I² = -1, so I⁻¹ = -I and dual(A) = A · I⁻¹.
    """
    # I = e123∧e∞∧e0 = -(canonical blade 31);  I⁻¹ = -I = +blade31.
    I_inv_vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    I_inv_vals[31] = 1.0
    return gp(a, Multivector(I_inv_vals))


def meet(a: Multivector, b: Multivector) -> Multivector:
    """Intersection of two direct-form primitives: A ∨ B = (A* ∧ B*)*.

    输入需为直接形式; 对偶形式的原语 (plane/sphere/circle) 先过
    dual() 再传入。例: meet(dual(π1), dual(π2)) = 交线 (直接形式)。
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


# ── Basis vectors ──────────────────────────────────────────────────────────

E1 = Multivector.vector(1, 0, 0, 0, 0)
E2 = Multivector.vector(0, 1, 0, 0, 0)
E3 = Multivector.vector(0, 0, 1, 0, 0)
E0 = Multivector.vector(0, 0, 0, 1, 0)
EINF = Multivector.vector(0, 0, 0, 0, 1)


# ── CGA Object Constructors (与 simu.cga 约定一致) ─────────────────────────
#
# 点/点对/线: 直接 (join) 形式; 平面/球/圆: 对偶形式。


def point(x: float, y: float, z: float) -> Multivector:
    """Create a conformal point from Euclidean coordinates (grade 1).

    p = e0 + x*e1 + y*e2 + z*e3 + 0.5*(x^2+y^2+z^2)*e∞
    This satisfies p·p = 0 (null vector property of conformal points).
    """
    r2 = x * x + y * y + z * z
    return Multivector.vector(x, y, z, 1.0, 0.5 * r2)


def point_pair(p1: Multivector, p2: Multivector) -> Multivector:
    """Create a point pair (0-sphere) from two conformal points (grade 2).

    Pp = p1 ∧ p2  (直接形式)
    """
    return op(p1, p2)


def line(p1: Multivector, p2: Multivector) -> Multivector:
    """Create a line from two conformal points (grade 3, 直接形式).

    L = p1 ∧ p2 ∧ e∞
    """
    return op(op(p1, p2), EINF)


def plane(normal_vec: tuple[float, float, float], distance: float) -> Multivector:
    """Create a plane from a normal vector and distance from origin
    (grade 1, 对偶形式).

    π = n + d*e∞  where n = nx*e1 + ny*e2 + nz*e3, 单位法向量。
    """
    nx, ny, nz = normal_vec
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl <= 1e-12:
        raise ValueError(f"plane normal vector is zero or degenerate: {normal_vec}")
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    return Multivector.vector(nx, ny, nz, 0.0, distance)


def sphere(center: tuple[float, float, float], radius: float) -> Multivector:
    """Create a sphere from center and radius (grade 1, 对偶形式).

    s = up(c) - 0.5*r^2*e∞  where up(c) is the conformal center point.
    """
    cx, cy, cz = center
    pc = point(cx, cy, cz)
    half_r2 = 0.5 * radius * radius
    correction = Multivector.vector(0, 0, 0, 0, half_r2)
    return pc - correction


def circle(
    center: tuple[float, float, float],
    radius: float,
    normal: tuple[float, float, float],
) -> Multivector:
    """Create a circle as intersection of sphere and plane (grade 2, 对偶形式).

    C = sphere ∧ plane  (对偶球 ∧ 对偶平面)
    """
    s = sphere(center, radius)
    d = center[0] * normal[0] + center[1] * normal[1] + center[2] * normal[2]
    p = plane(normal, d)  # plane() 负责法向量归一化与退化检查
    return op(s, p)


def mv_scalar_inline(s: float) -> Multivector:
    """Create a scalar multivector (inline helper)."""
    vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
    vals[0] = s
    return Multivector(vals)


# ── Distance functions (float64 欧氏公式) ──────────────────────────────────
#
# 不走 float32 的 conformal 内积: sqrt(-2·p1·p2) 在远原点时灾难性
# 抵消 (实测 (1000,0,0)-(1001,0,0) 得 0.0)。null 基下 conformal 权重
# 即 e0 系数 (槽 4), 显式存储, 无基换算抵消。


def _euclidean_coords(p: Multivector) -> tuple[float, float, float]:
    """Extract Euclidean coordinates from a conformal point.

    Reads the e1/e2/e3 coefficients and normalizes them by the e0
    (homogeneous weight) coefficient, in float64.
    """
    w = float(p.values[4])  # e0 coefficient
    if abs(w) < 1e-12:
        raise ValueError("multivector has no e0 component; not a finite point")
    return (
        float(p.values[1]) / w,
        float(p.values[2]) / w,
        float(p.values[3]) / w,
    )


def dist_point_point(p1: Multivector, p2: Multivector) -> float:
    """Euclidean distance between two conformal points.

    权重归一欧氏坐标直接作差 (float64), 见模块注释。
    """
    x1, y1, z1 = _euclidean_coords(p1)
    x2, y2, z2 = _euclidean_coords(p2)
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def dist_point_plane(p: Multivector, pi: Multivector) -> float:
    """Signed distance from a conformal point to a (对偶形式) plane.

    欧氏公式 (n·x − d)/|n|, 平面 π = n + d·e∞ 的 (n, d) 与点的权重
    归一坐标, 全程 float64。
    """
    x, y, z = _euclidean_coords(p)
    nx, ny, nz = float(pi.values[1]), float(pi.values[2]), float(pi.values[3])
    d = float(pi.values[5])  # e∞ coefficient
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl < 1e-12:
        return float("inf")
    return (nx * x + ny * y + nz * z - d) / nl


def dist_point_sphere(p: Multivector, s: Multivector) -> float:
    """Signed distance from a point to a (对偶形式) sphere surface.

    对偶球 s = w·(up(c) − ½ρ²e∞): 球心 c = v/w, ρ² = |c|² − 2f/w
    (v = 欧氏部分, w = e0 系数, f = e∞ 系数), float64。
    Positive = outside, negative = inside.
    """
    w = float(s.values[4])  # e0 coefficient
    if abs(w) < 1e-12:
        raise ValueError("sphere multivector has no e0 component")
    v1, v2, v3 = float(s.values[1]), float(s.values[2]), float(s.values[3])
    f = float(s.values[5])  # e∞ coefficient
    cx, cy, cz = v1 / w, v2 / w, v3 / w
    rho_sq = (v1 * v1 + v2 * v2 + v3 * v3) / (w * w) - 2.0 * f / w
    r = math.sqrt(max(0.0, rho_sq))
    x, y, z = _euclidean_coords(p)
    dx, dy, dz = x - cx, y - cy, z - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz) - r
