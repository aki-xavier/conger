"""Multivector representation for 5D Conformal Geometric Algebra.

A multivector in 5D CGA has 32 components organized by grade:
  Grade 0: 1 scalar
  Grade 1: 5 vectors   {e1, e2, e3, e+, e-}
  Grade 2: 10 bivectors
  Grade 3: 10 trivectors
  Grade 4: 5 quadvectors
  Grade 5: 1 pseudoscalar

内部基取**正交**基 {e1, e2, e3, e+, e-} (度规 diag(+,+,+,+,-)),
null 向量作为派生对象: e0 = (e- - e+)/2, einf = e- + e+ (e0·einf = -1)。

为什么不用 {e1,e2,e3,e0,einf} 作基: e0 与 einf 非正交, 积表递归
A·(b₁·b₂…) = A·b₁·b₂ 仅对正交基成立; 非正交基下 blade e0∧einf 会
混入标量成分, 污染一切 grade 投影 (ip/op/dual/meet)。

公开接口 (Multivector.vector 的 v0/ve 参数) 仍以 e0/einf 系数表达,
内部自动换算到 e+/e- 基。All components are stored in MLX arrays.
"""

import mlx.core as mx

# ── Basis blade definitions ────────────────────────────────────────────────

# Canonical ordering of the 32 basis blades
# Each blade is a tuple of basis vector indices: 0=e1, 1=e2, 2=e3, 3=e+, 4=e-
_BASIS_BLADES: list[tuple[int, ...]] = [
    # Grade 0 (1)
    (),
    # Grade 1 (5)
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    # Grade 2 (10)
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    # Grade 3 (10)
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
    # Grade 4 (5)
    (0, 1, 2, 3),
    (0, 1, 2, 4),
    (0, 1, 3, 4),
    (0, 2, 3, 4),
    (1, 2, 3, 4),
    # Grade 5 (1)
    (0, 1, 2, 3, 4),
]

NUM_COMPONENTS = 32
NUM_GRADES = 6

# Map blade tuple -> index for fast lookup
_BLADE_TO_IDX = {blade: i for i, blade in enumerate(_BASIS_BLADES)}

# Grade of each blade index
_BLADE_GRADE = [len(blade) for blade in _BASIS_BLADES]

# Indices grouped by grade
GRADE_INDICES: list[list[int]] = [[] for _ in range(NUM_GRADES)]
for i, g in enumerate(_BLADE_GRADE):
    GRADE_INDICES[g].append(i)

# Grade sizes: [1, 5, 10, 10, 5, 1]
GRADE_SIZES = [len(g) for g in GRADE_INDICES]

# Slice ranges for each grade in the flat array
_GRADE_SLICES: list[tuple[int, int]] = []
offset = 0
for size in GRADE_SIZES:
    _GRADE_SLICES.append((offset, offset + size))
    offset += size

# ── Metric for basis vectors ───────────────────────────────────────────────
# Indices: 0=e1, 1=e2, 2=e3, 3=e+, 4=e-
# e1^2=e2^2=e3^2=e+^2=+1, e-^2=-1; 正交 (非对角元全 0)
_VECTOR_METRIC = mx.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, -1],
    ],
    dtype=mx.float32,
)


def _parity(seq: list[int]) -> int:
    """Compute the permutation parity to sort seq. Returns (-1)^swaps."""
    arr = list(seq)
    swaps = 0
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
    return -1 if swaps % 2 else 1


def _compute_gp(
    blade_a: tuple[int, ...], blade_b: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Compute geometric product of two basis blades.

    基向量正交, 故 blade 的几何积 = 向量依次相乘, 递归:
    For blade A and vector v:
      A * v = A ⌋ v + A ∧ v
    where:
      A ⌋ v = Σ_{i} (-1)^{|A|-i} g(a_i, v) * (a_1 ∧ ... ∧ â_i ∧ ... ∧ a_k)

    Then A * B = A * (b_1 * b_2 * ... * b_q) applied recursively.

    Returns list of (sign, result_blade_idx).
    """
    # Accumulate results as {blade_tuple: accumulated_sign}
    results = {blade_a: 1.0}

    # Multiply each vector from blade_b through the current results
    for bv in blade_b:
        new_results = {}
        for cur_blade, cur_sign in results.items():
            cur_list = list(cur_blade)
            k = len(cur_list)

            # Term 1: Inner product (contraction) — for each vector in cur_blade
            for i in range(k):
                metric_val = float(_VECTOR_METRIC[cur_list[i], bv])
                if metric_val == 0:
                    continue
                # Remove the i-th vector
                contracted = tuple(cur_list[:i] + cur_list[i + 1 :])
                # Sign: (-1)^{k-i-1} * metric
                term_sign = cur_sign * metric_val * ((-1) ** (k - i - 1))
                if contracted in new_results:
                    new_results[contracted] += term_sign
                else:
                    new_results[contracted] = term_sign

            # Term 2: Outer product — append bv; sorting parity handles sign
            wedge_blade = tuple(cur_list + [bv])
            wedge_sign = cur_sign  # sign handled by canonicalization parity
            if wedge_blade in new_results:
                new_results[wedge_blade] += wedge_sign
            else:
                new_results[wedge_blade] = wedge_sign

        results = new_results

    # Canonicalize results: sort blades and compute sign
    final = []
    for blade, sign in results.items():
        if abs(sign) < 1e-12:
            continue
        # Sort to canonical form
        blade_list = list(blade)
        parity = _parity(blade_list)
        blade_list.sort()
        canon_blade = tuple(blade_list)
        canon_sign = round(sign * parity)
        if canon_sign == 0:
            continue
        if canon_blade in _BLADE_TO_IDX:
            idx = _BLADE_TO_IDX[canon_blade]
            # Merge with existing entries for the same blade
            merged = False
            for fi, (fs, fidx) in enumerate(final):
                if fidx == idx:
                    final[fi] = (fs + canon_sign, idx)
                    merged = True
                    break
            if not merged:
                final.append((canon_sign, idx))

    # Filter zero results
    return [(s, idx) for s, idx in final if s != 0]


def _build_gp_table() -> list[list[list[tuple[int, int]]]]:
    """Build the geometric product multiplication table."""
    gp_table = [[[] for _ in range(NUM_COMPONENTS)] for _ in range(NUM_COMPONENTS)]

    for i in range(NUM_COMPONENTS):
        for j in range(NUM_COMPONENTS):
            gp_table[i][j] = _compute_gp(_BASIS_BLADES[i], _BASIS_BLADES[j])

    return gp_table


# Build the table once at module load
_GP_TABLE = _build_gp_table()

# ── Precomputed sparse GP arrays for MLX ───────────────────────────────────

# Maximum terms in any product
_max_terms = max(len(terms) for row in _GP_TABLE for terms in row)

# Build dense index/sign arrays padded with zeros using Python lists (one-time cost)
_signs_list = [
    [[0.0] * _max_terms for _ in range(NUM_COMPONENTS)] for _ in range(NUM_COMPONENTS)
]
_indices_list = [
    [[0] * _max_terms for _ in range(NUM_COMPONENTS)] for _ in range(NUM_COMPONENTS)
]
_counts_list = [[0] * NUM_COMPONENTS for _ in range(NUM_COMPONENTS)]

for i in range(NUM_COMPONENTS):
    for j in range(NUM_COMPONENTS):
        terms = _GP_TABLE[i][j]
        _counts_list[i][j] = len(terms)
        for k, (sign, dst) in enumerate(terms):
            _signs_list[i][j][k] = float(sign)
            _indices_list[i][j][k] = dst

GP_SIGNS = mx.array(_signs_list, dtype=mx.float32)
GP_INDICES = mx.array(_indices_list, dtype=mx.int32)
GP_COUNTS = mx.array(_counts_list, dtype=mx.int32)

# ── Grade projection masks ─────────────────────────────────────────────────

_GRADE_MASKS = []
for g in range(NUM_GRADES):
    vals = [1.0 if i in GRADE_INDICES[g] else 0.0 for i in range(NUM_COMPONENTS)]
    _GRADE_MASKS.append(mx.array(vals, dtype=mx.float32))

GRADE_MASKS = _GRADE_MASKS

# ── Flattened GP table for scatter_add-based geometric product ─────────────

# GP_MASK[i,j,k] = sign if GP_TABLE[i][j] contributes to k, else 0
_mask_list = [
    [[0.0] * NUM_COMPONENTS for _ in range(NUM_COMPONENTS)]
    for _ in range(NUM_COMPONENTS)
]
for i in range(NUM_COMPONENTS):
    for j in range(NUM_COMPONENTS):
        for sign, dst in _GP_TABLE[i][j]:
            _mask_list[i][j][dst] += float(sign)
GP_MASK = mx.array(_mask_list, dtype=mx.float32)

# Non-zero (i,j) pairs for sparse computation
_nz_i = []
_nz_j = []
for i in range(NUM_COMPONENTS):
    for j in range(NUM_COMPONENTS):
        if _GP_TABLE[i][j]:
            _nz_i.append(i)
            _nz_j.append(j)
GP_NONZERO_I = mx.array(_nz_i, dtype=mx.int32)
GP_NONZERO_J = mx.array(_nz_j, dtype=mx.int32)


# ── Multivector class ──────────────────────────────────────────────────────


class Multivector:
    """A 32-component multivector in 5D CGA, backed by an MLX array.

    内部存储为正交基 {e1,e2,e3,e+,e-} 下的系数; e0/einf 系数
    与内部系数的换算见 vector() / e0_coeff() / einf_coeff()。
    """

    __slots__ = ("values",)

    def __init__(self, values: mx.array | None = None):
        if values is None:
            self.values = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        elif isinstance(values, mx.array):
            self.values = values
        else:
            arr = mx.array(values, dtype=mx.float32)
            if arr.shape != (NUM_COMPONENTS,):
                raise ValueError(f"Expected shape (32,), got {arr.shape}")
            self.values = arr

    @staticmethod
    def zeros() -> Multivector:
        return Multivector(mx.zeros(NUM_COMPONENTS, dtype=mx.float32))

    @staticmethod
    def scalar(s: float) -> Multivector:
        vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        vals[0] = s
        return Multivector(vals)

    @staticmethod
    def vector(
        v1: float, v2: float, v3: float, v0: float = 0.0, ve: float = 0.0
    ) -> Multivector:
        """Vector from Euclidean (v1,v2,v3) + e0/einf coefficients (v0/ve).

        换算: v0·e0 + ve·einf = (ve − v0/2)·e+ + (ve + v0/2)·e-
        (由 e0 = (e- − e+)/2, einf = e- + e+ 解出)。
        """
        vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        vals[1] = v1
        vals[2] = v2
        vals[3] = v3
        vals[4] = ve - v0 / 2.0  # e+ 系数
        vals[5] = ve + v0 / 2.0  # e- 系数
        return Multivector(vals)

    @staticmethod
    def bivector(components: list[float]) -> Multivector:
        vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        for i, v in enumerate(components):
            idx = GRADE_INDICES[2][i]
            vals[idx] = v
        return Multivector(vals)

    def grade(self, g: int) -> Multivector:
        mask = GRADE_MASKS[g]
        return Multivector(self.values * mask)

    def scalar_part(self) -> float:
        return float(self.values[0])

    def vector_part(self) -> mx.array:
        start, end = _GRADE_SLICES[1]
        return self.values[start:end]

    def euclidean_vector(self) -> tuple[float, float, float]:
        return (float(self.values[1]), float(self.values[2]), float(self.values[3]))

    def e0_coeff(self) -> float:
        """e0 系数 = slot(e-) − slot(e+)。"""
        return float(self.values[5] - self.values[4])

    def einf_coeff(self) -> float:
        """einf 系数 = (slot(e+) + slot(e-)) / 2。"""
        return float((self.values[4] + self.values[5]) / 2.0)

    def bivector_part(self) -> mx.array:
        start, end = _GRADE_SLICES[2]
        return self.values[start:end]

    @property
    def is_null(self) -> bool:
        return bool(mx.all(mx.abs(self.values) < 1e-10).item())

    def __add__(self, other: Multivector) -> Multivector:
        return Multivector(self.values + other.values)

    def __sub__(self, other: Multivector) -> Multivector:
        return Multivector(self.values - other.values)

    def __mul__(self, scalar: float) -> Multivector:
        return Multivector(self.values * scalar)

    def __rmul__(self, scalar: float) -> Multivector:
        return Multivector(self.values * scalar)

    def __truediv__(self, scalar: float) -> Multivector:
        return Multivector(self.values / scalar)

    def __neg__(self) -> Multivector:
        return Multivector(-self.values)

    def __repr__(self) -> str:
        parts = []
        for g in range(NUM_GRADES):
            for idx in GRADE_INDICES[g]:
                v = float(self.values[idx])
                if abs(v) > 1e-10:
                    blade_name = _blade_name(idx)
                    if blade_name == "1":
                        parts.append(f"{v:.4f}")
                    else:
                        parts.append(f"{v:+.4f}*{blade_name}")
        if not parts:
            return "Multivector(0)"
        return "Multivector(" + " ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Multivector):
            return False
        return bool(mx.allclose(self.values, other.values, atol=1e-6).item())

    def copy(self) -> Multivector:
        return Multivector(mx.array(self.values))


def _blade_name(idx: int) -> str:
    blade = _BASIS_BLADES[idx]
    if not blade:
        return "1"
    names = {0: "e1", 1: "e2", 2: "e3", 3: "e+", 4: "e-"}
    return "".join(names[v] for v in blade)


def mv_zeros() -> Multivector:
    return Multivector.zeros()


def mv_scalar(s: float) -> Multivector:
    return Multivector.scalar(s)


def mv_vector(
    v1: float, v2: float, v3: float, v0: float = 0.0, ve: float = 0.0
) -> Multivector:
    return Multivector.vector(v1, v2, v3, v0, ve)


def mv_bivector(components: list[float]) -> Multivector:
    return Multivector.bivector(components)


def stack_mv(mvs: list[Multivector]) -> mx.array:
    return mx.stack([mv.values for mv in mvs])


def unstack_mv(arr: mx.array) -> list[Multivector]:
    return [Multivector(arr[i]) for i in range(arr.shape[0])]
