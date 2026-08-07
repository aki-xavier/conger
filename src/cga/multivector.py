"""Multivector representation for 5D Conformal Geometric Algebra.

A multivector in 5D CGA has 32 components organized by grade:
  Grade 0: 1 scalar
  Grade 1: 5 vectors   {e1, e2, e3, e0, einf}
  Grade 2: 10 bivectors
  Grade 3: 10 trivectors
  Grade 4: 5 quadvectors
  Grade 5: 1 pseudoscalar

基取 null 基 {e1, e2, e3, e0, e∞}: e0² = e∞² = 0,
e0·e∞ = -1。e0 与 e∞ 非正交, 故 blade 的几何积不能用正交基的递归
公式——积表构建时对 blade_b 的全排列做反对称化 (见 _compute_gp)。
代价是建表稍慢 (一次性), 收益是 conformal 权重 (e0 系数) 显式存储,
远原点坐标提取无基换算抵消。All components are stored in MLX arrays.
"""

import math

import mlx.core as mx

# ── Basis blade definitions ────────────────────────────────────────────────

# Canonical ordering of the 32 basis blades (公开: 互操作/工具脚本
# 的合法元数据, 如 compare_clifford.py 的基变换)
# Each blade is a tuple of basis vector indices: 0=e1, 1=e2, 2=e3, 3=e0, 4=e∞
BASIS_BLADES: list[tuple[int, ...]] = [
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
_BLADE_TO_IDX = {blade: i for i, blade in enumerate(BASIS_BLADES)}

# Grade of each blade index
_BLADE_GRADE = [len(blade) for blade in BASIS_BLADES]

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
            gp_table[i][j] = _compute_gp(BASIS_BLADES[i], BASIS_BLADES[j])

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

    # ── 代数运算 (实现即验证过的原 cga.algebra 函数, 逐字搬入) ──────

    def gp(self, other: Multivector) -> Multivector:
        """几何积。result[k] = Σ GP_MASK[i,j,k]·self_i·other_j,
        用预计算的稀疏非零 (i,j) 对索引。"""
        prod = self.values[GP_NONZERO_I] * other.values[GP_NONZERO_J]
        mask_rows = GP_MASK[GP_NONZERO_I, GP_NONZERO_J, :]  # (N, 32)
        return Multivector((mask_rows * prod[:, None]).sum(axis=0))

    def ip(self, other: Multivector) -> Multivector:
        """内积 (fat dot / Hestenes, 与 clifford 库的 | 算子一致)。

        blade 规则对全 grade 对的线性扩张:
            A|B = Σ_{r,s≥1} ⟨ ⟨A⟩_r * ⟨B⟩_s ⟩_|r−s|
        含标量 (grade 0) 的项为零 (Hestenes 规则, 与 clifford 实测
        一致: 1|e12 = 0)。r>s 时非零 (对称内积, 与左收缩的区别);
        向量与 blade 间 (r=1≤s) 与左收缩相同, 故关联判据
        p.ip(X) = 0 的行为不受影响。对一般混合 grade multivector
        正确。"""
        result = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        for ga in range(1, NUM_GRADES):
            a_g = self.values * GRADE_MASKS[ga]
            if not bool(mx.any(a_g != 0).item()):
                continue
            for gb in range(1, NUM_GRADES):
                b_g = other.values * GRADE_MASKS[gb]
                if not bool(mx.any(b_g != 0).item()):
                    continue
                prod = Multivector(a_g).gp(Multivector(b_g))
                result = result + prod.values * GRADE_MASKS[abs(gb - ga)]
        return Multivector(result)

    def op(self, other: Multivector) -> Multivector:
        """外积 self ∧ other。

        blade 规则对全 grade 对的线性扩张:
            a ∧ b = Σ_{r,s} < <a>_r * <b>_s >_{r+s}
        对一般混合 grade multivector 正确。"""
        result = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        for ga in range(NUM_GRADES):
            a_g = self.values * GRADE_MASKS[ga]
            if not bool(mx.any(a_g != 0).item()):
                continue
            for gb in range(NUM_GRADES - ga):
                b_g = other.values * GRADE_MASKS[gb]
                if not bool(mx.any(b_g != 0).item()):
                    continue
                prod = Multivector(a_g).gp(Multivector(b_g))
                result = result + prod.values * GRADE_MASKS[ga + gb]
        return Multivector(result)

    def reverse(self) -> Multivector:
        """反转 involution: grade-k blade 乘 (-1)^{k(k-1)/2}。"""
        return Multivector(self.values * _REVERSE_MASK)

    def grade_involution(self) -> Multivector:
        """Grade involution: 奇 grade 分量取负。"""
        return Multivector(self.values * _INVOLUTION_MASK)

    def conjugate(self) -> Multivector:
        """Clifford 共轭: reverse + grade involution。"""
        return self.reverse().grade_involution()

    def dual(self) -> Multivector:
        """Hodge 对偶: 乘逆伪标量 I⁻¹。

        定向约定: I = e123 ∧ e∞ ∧ e0, 与 `clifford` 库的 conformal
        伪标量 e12345 一致 (e∞ ∧ e0 = +e45)。此定向下 I² = −1,
        故 I⁻¹ = −I, dual(A) = A · I⁻¹。
        """
        # I = e123∧e∞∧e0 = -(canonical blade 31);  I⁻¹ = -I = +blade31.
        I_inv_vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        I_inv_vals[31] = 1.0
        return self.gp(Multivector(I_inv_vals))

    def undual(self) -> Multivector:
        """对偶的逆: dual(dual(x)) = −x (因 I⁻² = I² = −1), 故 undual = −dual。

        从直接形式还原对偶形式 (n + d·e∞ / up(c) − ½ρ²e∞) 时使用。
        """
        return -self.dual()

    def meet(self, other: Multivector) -> Multivector:
        """两个直接形式原语的交: self ∨ other = (self* ∧ other*)*。

        输入需为直接形式; 对偶形式的原语 (plane/sphere/circle) 先过
        dual() 再传入。例: π1.dual().meet(π2.dual()) = 交线 (直接形式)。
        """
        return self.dual().op(other.dual()).dual()

    def norm(self) -> float:
        """欧氏范数: sqrt(|⟨self · reverse(self)⟩₀|)。"""
        s = float(self.gp(self.reverse()).values[0])
        return math.sqrt(abs(s))

    def normalized(self) -> Multivector:
        """归一化到单位范数。"""
        n = self.norm()
        if n < 1e-12:
            return Multivector.zeros()
        return self / n

    def bulk(self) -> Multivector:
        """欧氏 (bulk) 部分: 不含 e0/e∞ 的分量。"""
        euc_indices = [0, 1, 2, 3, 6, 7, 10, 16]
        vals = mx.zeros(NUM_COMPONENTS, dtype=mx.float32)
        for idx in euc_indices:
            vals[idx] = self.values[idx]
        return Multivector(vals)

    def weight(self) -> Multivector:
        """Conformal (weight) 部分: 含 e0/e∞ 的分量。"""
        return self - self.bulk()


def _blade_name(idx: int) -> str:
    blade = BASIS_BLADES[idx]
    if not blade:
        return "1"
    names = {0: "e1", 1: "e2", 2: "e3", 3: "e0", 4: "e∞"}
    return "".join(names[v] for v in blade)
