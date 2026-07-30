"""CGA package self-check: python -m cga

验证核心代数与原语/versor 的正确性。统一约定: 构造器返回直接形式,
关联判据 op(up(p), X) = 0; versor 作用 apply_motor(M, obj)。
"""

import math

import mlx.core as mx
import numpy as np

from cga import (
    Multivector,
    apply_motor,
    circle,
    dist_point_plane,
    dist_point_point,
    gp,
    line,
    meet,
    motor,
    motor_to_matrix,
    op,
    plane,
    point,
    rotor,
    sphere,
    translator,
)
from cga.algebra import dist_point_sphere

_ok = 0


def check(name: str, cond: bool) -> None:
    global _ok
    assert cond, f"FAIL: {name}"
    _ok += 1
    print(f"  ok  {name}")


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) < tol


def vmax(mv: Multivector) -> float:
    return float(mx.abs(mv.values).max().item())


def main() -> None:
    # null 性 + 距离
    p1, p2 = point(0, 0, 0), point(1, 0, 0)
    check("point is null", close(gp(p1, p1).values[0], 0))
    check("dist_point_point", close(dist_point_point(p1, point(3, 4, 0)), 5.0))

    # 统一关联判据 op(up, X) = 0 (所有原语均为直接形式)
    L = line(p1, p2)
    check("point on line", close(vmax(op(point(2, 0, 0), L)), 0))
    check("point off line", vmax(op(point(0, 1, 0), L)) > 1e-3)

    pi = plane((0, 0, 1), 2.0)  # z = 2 平面
    check("point on plane", close(vmax(op(point(0.3, -0.7, 2), pi)), 0))
    check("point off plane", vmax(op(point(0, 0, 0), pi)) > 1e-3)

    s = sphere((1, 2, 3), 2.0)
    check("point on sphere", close(vmax(op(point(3, 2, 3), s)), 0))
    check("point off sphere", vmax(op(point(0, 0, 0), s)) > 1e-3)

    c = circle((0, 0, 0), 1.0, (0, 0, 1))
    check("point on circle", close(vmax(op(point(0, 1, 0), c)), 0))
    check("point off circle", vmax(op(point(0, 0, 1), c)) > 1e-3)

    # 距离函数 (接受直接形式)
    check("dist_point_plane", close(dist_point_plane(point(0, 0, 5), pi), 3.0))
    check("dist_point_sphere on", close(dist_point_sphere(point(3, 2, 3), s), 0))
    check("dist_point_sphere out", dist_point_sphere(point(5, 2, 3), s) > 0)
    check("dist_point_sphere in", dist_point_sphere(point(1, 2, 3), s) < 0)

    # meet: 两平面交线 / 线球交点对
    pi2 = plane((0, 1, 0), 1.0)  # y = 1 平面
    Lm = meet(pi, pi2)  # 交线: y=1, z=2, 沿 x 方向
    check(
        "meet(plane,plane) = line",
        close(vmax(op(point(0, 1, 2), Lm)), 0)
        and close(vmax(op(point(5, 1, 2), Lm)), 0),
    )
    Lz = line(point(0, 0, -2), point(0, 0, 2))  # z 轴
    unit_s = sphere((0, 0, 0), 1.0)
    PPm = meet(Lz, unit_s)  # 交于 (0,0,±1)
    check(
        "meet(line,sphere) = point pair",
        close(vmax(op(point(0, 0, 1), PPm)), 0)
        and close(vmax(op(point(0, 0, -1), PPm)), 0),
    )

    # motor: 平移 / 旋转 / 复合 (参数序: apply_motor(M, obj))
    T = translator((1, 2, 3))
    check("translator", close(dist_point_point(apply_motor(T, p1), point(1, 2, 3)), 0))
    R = rotor((0, 0, 1), math.pi / 2)
    check("rotor 90° z", close(dist_point_point(apply_motor(R, p2), point(0, 1, 0)), 0))
    M = motor((0, 0, 1), math.pi / 2, (1, 0, 0))
    check(
        "motor rot+trans",
        close(dist_point_point(apply_motor(M, p2), point(1, 1, 0)), 0),
    )

    # versor 保持关联: 平移后的线仍过平移后的点
    Lm2 = apply_motor(T, L)
    check(
        "motor preserves incidence",
        close(vmax(op(apply_motor(T, point(2, 0, 0)), Lm2)), 0),
    )

    # versor 保持 meet: 先交后变换 = 先变换后交
    lhs = apply_motor(T, meet(pi, pi2))
    rhs = meet(apply_motor(T, pi), apply_motor(T, pi2))
    a, b = lhs.values, rhs.values
    scale = float(mx.abs(b).max().item())
    check(
        "versor preserves meet",
        bool(mx.allclose(a / scale, b / scale, atol=1e-4).item())
        or bool(mx.allclose(a / scale, -b / scale, atol=1e-4).item()),
    )

    # motor_to_matrix 与 sandwich 作用一致
    Hm = np.array(motor_to_matrix(M))
    p_h = Hm @ np.array([1, 0, 0, 1.0])
    check("motor_to_matrix", np.allclose(p_h[:3], [1, 1, 0], atol=1e-4))

    print(f"\nall {_ok} checks passed")


if __name__ == "__main__":
    main()
