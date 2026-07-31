"""Multivector representation for 5D Conformal Geometric Algebra.

A multivector in 5D CGA has 32 components organized by grade:
  Grade 0: 1 scalar
  Grade 1: 5 vectors   {e1, e2, e3, e0, einf}
  Grade 2: 10 bivectors
  Grade 3: 10 trivectors
  Grade 4: 5 quadvectors
  Grade 5: 1 pseudoscalar

基取 null 基 {e1, e2, e3, e0, e∞} (与 simu.cga 一致): e0² = e∞² = 0,
e0·e∞ = -1。e0 与 e∞ 非正交, 故 blade 的几何积不能用正交基的递归
公式——积表构建时对 blade_b 的全排列做反对称化 (见 _compute_gp)。
代价是建表稍慢 (一次性), 收益是 conformal 权重 (e0 系数) 显式存储,
远原点坐标提取无基换算抵消。All components are stored in MLX arrays.
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
# Indices: 0=e1, 1=e2, 2=e3, 3=e0, 4=einf
# e1^2=e2^2=e3^2=1, e0^2=einf^2=0, e0·einf = einf·e0 = -1
_VECTOR_METRIC = mx.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, -1],
        [0, 0, 0, -1, 0],
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

    blade_b is a WEDGE product b_1 ∧ ... ∧ b_q.  Because e0 and einf are not
    orthogonal (e0·einf = -1), the wedge is NOT the sequential geometric
    product of its vectors; it is the antisymmetrized geometric product:

        b_1 ∧ ... ∧ b_q = (1/q!) Σ_σ sign(σ) b_σ(1) * ... * b_σ(q)

    Each permuted vector sequence is multiplied through blade_a with the
    recursive formula A * v = A ⌋ v + A ∧ v, where the contraction is

        A ⌋ v = Σ_i (-1)^{|A|-i} g(a_i, v) * (a_1 ∧ ... ∧ â_i ∧ ... ∧ a_k)

    Coefficients must come out integral (blades are signed permutation
    products); a fractional result means the table builder itself is wrong,
    so we raise instead of silently rounding.

    Returns list of (sign, result_blade_idx).
    """
    import itertools
    import math

    # Accumulate results as {blade_tuple: accumulated_coeff}
    results: dict[tuple[int, ...], float] = {}
    q = len(blade_b)
    for perm in itertools.permutations(blade_b):
        perm_sign = _parity(list(perm))  # blade_b is sorted: parity of σ
        # Multiply blade_a by the vectors of perm, sequentially.
        partial = {blade_a: 1.0}
        for bv in perm:
            new_partial: dict[tuple[int, ...], float] = {}
            for cur_blade, cur_sign in partial.items():
                cur_list = list(cur_blade)
                k = len(cur_list)

                # Term 1: contraction — for each vector in cur_blade
                for i in range(k):
                    metric_val = float(_VECTOR_METRIC[cur_list[i], bv])
                    if metric_val == 0:
                        continue
                    contracted = tuple(cur_list[:i] + cur_list[i + 1 :])
                    term_sign = cur_sign * metric_val * ((-1) ** (k - i - 1))
                    new_partial[contracted] = (
                        new_partial.get(contracted, 0.0) + term_sign
                    )

                # Term 2: outer product — append bv; canonicalization parity
                # handles the sign later
                wedge_blade = tuple(cur_list + [bv])
                new_partial[wedge_blade] = new_partial.get(wedge_blade, 0.0) + cur_sign

            partial = new_partial

        scale = perm_sign / math.factorial(q)
        for blade, coef in partial.items():
            results[blade] = results.get(blade, 0.0) + scale * coef

    # Canonicalize results: sort blades, accumulate fractional coefficients,
    # round only after all contributions are merged.
    coef_by_idx: dict[int, float] = {}
    for blade, sign in results.items():
        if abs(sign) < 1e-12:
            continue
        blade_list = list(blade)
        parity = _parity(blade_list)
        blade_list.sort()
        canon_blade = tuple(blade_list)
        idx = _BLADE_TO_IDX.get(canon_blade)
        if idx is None:
            continue
        coef_by_idx[idx] = coef_by_idx.get(idx, 0.0) + sign * parity

    final = []
    for idx, coef in coef_by_idx.items():
        rounded = round(coef)
        if abs(coef - rounded) < 1e-9 and rounded != 0:
            final.append((rounded, idx))
        elif abs(coef) >= 1e-9:
            raise ArithmeticError(
                f"non-integer blade coefficient {coef} for blade index {idx}"
            )
    return final


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

    系数存储于 null 基 {e1, e2, e3, e0, e∞} 下 (槽 4 = e0, 槽 5 = e∞)。
    """

    __slots__ = ("values",)

    def __init__(self, values: mx.array | None = None):
        if values is None:
            self.values = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        elif isinstance(values, mx.array):
            if values.shape != (NUM_COMPONENTS,):
                raise ValueError(f"Expected shape (32,), got {values.shape}")
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
        """Vector from Euclidean (v1,v2,v3) + e0/einf coefficients (v0/ve)."""
        vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        vals[1] = v1
        vals[2] = v2
        vals[3] = v3
        vals[4] = v0
        vals[5] = ve
        return Multivector(vals)

    @staticmethod
    def bivector(components: list[float]) -> Multivector:
        if len(components) != 10:
            raise ValueError(f"Expected 10 bivector components, got {len(components)}")
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
        """e0 (原点) 系数 —— conformal 权重, 显式存储于槽 4。"""
        return float(self.values[4])

    def einf_coeff(self) -> float:
        """e∞ (无穷远点) 系数, 显式存储于槽 5。"""
        return float(self.values[5])

    def bivector_part(self) -> mx.array:
        start, end = _GRADE_SLICES[2]
        return self.values[start:end]

    @property
    def is_zero(self) -> bool:
        """是否为零 multivector (所有分量≈0)。

        这不是 CGA 的 null 性 (v·v = 0)——conformal point 等非零向量
        也是 null; 判 null 请用 gp(v, v) 的标量部。
        """
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

    def __hash__(self) -> int:
        # 精确表示的哈希; __eq__ 是近似比较 (atol=1e-6), 故"近似相等
        # 但非逐位相同"的 multivector 会有不同哈希——作 dict key/set
        # 成员时请自行量化或取整后再用。
        return hash(tuple(self.values.tolist()))

    def copy(self) -> Multivector:
        return Multivector(mx.array(self.values))


def _blade_name(idx: int) -> str:
    blade = _BASIS_BLADES[idx]
    if not blade:
        return "1"
    names = {0: "e1", 1: "e2", 2: "e3", 3: "e0", 4: "e∞"}
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
