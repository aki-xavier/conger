"""Conformal Geometric Algebra (CGA) core.

CGA in 5D embeds Euclidean 3D space into a conformal space with basis
{e1, e2, e3, e0, einf} where e0 is the origin and einf is the point at infinity.

This enables unified representation of points, lines, planes, circles, spheres,
and rigid-body transformations (motors).

OOP 表面:
  - Multivector: 32 分量多重向量, 代数运算全是方法
    (gp/ip/op/reverse/dual/meet/norm/...)。
  - 图元类: Point / PointPair / Line (直接形式, 关联判据 p.op(X) = 0),
    Plane / Sphere / Circle (对偶形式, 关联判据 p.ip(X) = 0)。
  - Motor: 刚体变换 versor, O' = M.apply(O); exp/log/插值/速度提取
    都是类方法或方法。
"""

from cga.algebra import (
    E0,
    E1,
    E2,
    E3,
    EINF,
    Circle,
    Line,
    Plane,
    Point,
    PointPair,
    Sphere,
)
from cga.motors import Motor
from cga.multivector import Multivector

__all__ = [
    "E0",
    "E1",
    "E2",
    "E3",
    "EINF",
    "Circle",
    "Line",
    "Motor",
    "Multivector",
    "Plane",
    "Point",
    "PointPair",
    "Sphere",
]
