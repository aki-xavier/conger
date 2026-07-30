"""Conformal Geometric Algebra (CGA) core.

CGA in 5D embeds Euclidean 3D space into a conformal space with basis
{e1, e2, e3, e0, einf} where e0 is the origin and einf is the point at infinity.

This enables unified representation of points, lines, planes, circles, spheres,
and rigid-body transformations (motors).

表示约定: 所有原语构造器返回直接 (join) 形式, 关联判据统一为
op(up(p), X) = 0; 距离函数与 meet 内部处理对偶。
"""

from cga.algebra import (
    circle,
    dist_point_plane,
    dist_point_point,
    dual,
    gp,
    ip,
    line,
    meet,
    norm,
    normalize,
    op,
    plane,
    point,
    point_pair,
    reverse,
    sphere,
)
from cga.motors import apply_motor, motor, motor_to_matrix, rotor, translator
from cga.multivector import (
    Multivector,
    mv_bivector,
    mv_scalar,
    mv_vector,
    mv_zeros,
)

__all__ = [
    "Multivector",
    "apply_motor",
    "circle",
    "dist_point_plane",
    "dist_point_point",
    "dual",
    "gp",
    "ip",
    "line",
    "meet",
    "motor",
    "motor_to_matrix",
    "mv_bivector",
    "mv_scalar",
    "mv_vector",
    "mv_zeros",
    "norm",
    "normalize",
    "op",
    "plane",
    "point",
    "point_pair",
    "reverse",
    "rotor",
    "sphere",
    "translator",
]
