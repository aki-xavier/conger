"""CGA package self-check: python -m cga

验证核心代数与原语/versor 的正确性。约定 (与 simu.cga 一致):
点/点对/线为直接形式, 关联判据 op(p, X) = 0; 平面/球/圆为对偶
形式, 关联判据 ip(p, X) = 0; versor 作用 apply_motor(obj, M);
meet 接受直接形式输入。
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
    dist_point_sphere,
    dual,
    gp,
    ip,
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
from cga.motors import (
    exp_bivector,
    extract_velocity,
    identity_motor,
    interpolate_motor,
    log_motor,
)

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

    # 关联判据: 线 (直接形式) op(p, X) = 0
    L = line(p1, p2)
    check("point on line", close(vmax(op(point(2, 0, 0), L)), 0))
    check("point off line", vmax(op(point(0, 1, 0), L)) > 1e-3)

    # 关联判据: 平面/球/圆 (对偶形式) ip(p, X) = 0
    pi = plane((0, 0, 1), 2.0)  # z = 2 平面
    check("point on plane", close(vmax(ip(point(0.3, -0.7, 2), pi)), 0))
    check("point off plane", vmax(ip(point(0, 0, 0), pi)) > 1e-3)

    s = sphere((1, 2, 3), 2.0)
    check("point on sphere", close(vmax(ip(point(3, 2, 3), s)), 0))
    check("point off sphere", vmax(ip(point(0, 0, 0), s)) > 1e-3)

    c = circle((0, 0, 0), 1.0, (0, 0, 1))
    check("point on circle", close(vmax(ip(point(0, 1, 0), c)), 0))
    check("point off circle", vmax(ip(point(0, 0, 1), c)) > 1e-3)

    # 退化输入守卫
    try:
        plane((0, 0, 0), 1.0)
        raise AssertionError("FAIL: degenerate plane accepted")
    except ValueError:
        check("degenerate plane raises", True)
    try:
        circle((0, 0, 0), 1.0, (0, 0, 0))
        raise AssertionError("FAIL: degenerate circle accepted")
    except ValueError:
        check("degenerate circle raises", True)
    try:
        Multivector.bivector([1.0, 2.0])
        raise AssertionError("FAIL: short bivector accepted")
    except ValueError:
        check("bivector length check", True)

    # 距离函数 (float64 欧氏公式)
    check("dist_point_plane", close(dist_point_plane(point(0, 0, 5), pi), 3.0))
    check("dist_point_sphere on", close(dist_point_sphere(point(3, 2, 3), s), 0))
    check("dist_point_sphere out", dist_point_sphere(point(5, 2, 3), s) > 0)
    check("dist_point_sphere in", dist_point_sphere(point(1, 2, 3), s) < 0)

    # meet (直接形式输入; 对偶原语先过 dual()):
    # 两平面交线 / 线球交点对
    pi2 = plane((0, 1, 0), 1.0)  # y = 1 平面
    Lm = meet(dual(pi), dual(pi2))  # 交线: y=1, z=2, 沿 x 方向
    check(
        "meet(plane,plane) = line",
        close(vmax(op(point(0, 1, 2), Lm)), 0)
        and close(vmax(op(point(5, 1, 2), Lm)), 0),
    )
    Lz = line(point(0, 0, -2), point(0, 0, 2))  # z 轴
    unit_s = sphere((0, 0, 0), 1.0)
    PPm = meet(Lz, dual(unit_s))  # 交于 (0,0,±1)
    check(
        "meet(line,sphere) = point pair",
        close(vmax(op(point(0, 0, 1), PPm)), 0)
        and close(vmax(op(point(0, 0, -1), PPm)), 0),
    )

    # motor: 平移 / 旋转 / 复合 (参数序: apply_motor(obj, M))
    T = translator((1, 2, 3))
    check("translator", close(dist_point_point(apply_motor(p1, T), point(1, 2, 3)), 0))
    R = rotor((0, 0, 1), math.pi / 2)
    check("rotor 90° z", close(dist_point_point(apply_motor(p2, R), point(0, 1, 0)), 0))
    M = motor((0, 0, 1), math.pi / 2, (1, 0, 0))
    check(
        "motor rot+trans",
        close(dist_point_point(apply_motor(p2, M), point(1, 1, 0)), 0),
    )

    # versor 保持关联: 平移后的线仍过平移后的点
    Lm2 = apply_motor(L, T)
    check(
        "motor preserves incidence",
        close(vmax(op(apply_motor(point(2, 0, 0), T), Lm2)), 0),
    )

    # versor 保持 meet: 先交后变换 = 先变换后交
    lhs = apply_motor(meet(dual(pi), dual(pi2)), T)
    rhs = meet(dual(apply_motor(pi, T)), dual(apply_motor(pi2, T)))
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

    # exp/log roundtrip 与 motor 插值 (exp_bivector: exp(-s·B), s 带符号)
    R90 = rotor((0, 0, 1), math.pi / 2)
    R45 = rotor((0, 0, 1), math.pi / 4)
    B90 = log_motor(R90)
    check(
        "exp∘log roundtrip",
        close(
            dist_point_point(apply_motor(p2, exp_bivector(B90)), apply_motor(p2, R90)),
            0,
        ),
    )
    check(
        "exp_bivector scale=0.5 = half motor",
        close(
            dist_point_point(
                apply_motor(p2, exp_bivector(B90, 0.5)), apply_motor(p2, R45)
            ),
            0,
        ),
    )
    check(
        "exp_bivector negative scale = inverse",
        close(
            dist_point_point(
                apply_motor(p2, exp_bivector(B90, -0.5)),
                apply_motor(p2, rotor((0, 0, 1), -math.pi / 4)),
            ),
            0,
        ),
    )
    check(
        "interpolate_motor midpoint",
        close(
            dist_point_point(
                apply_motor(p2, interpolate_motor(identity_motor(), R90, 0.5)),
                apply_motor(p2, R45),
            ),
            0,
        ),
    )

    # 远原点距离 (float32 conformal 内积会灾难性抵消 → 0, 回归检查)
    check(
        "dist far from origin",
        close(dist_point_point(point(1000, 0, 0), point(1001, 0, 0)), 1.0, tol=1e-2),
    )

    # 螺旋运动 (非零节距) exp∘log roundtrip
    M_screw = motor((0, 0, 1), 0.4, (0.3, -0.2, 0.1))
    M_rt = exp_bivector(log_motor(M_screw))
    check(
        "screw exp∘log roundtrip",
        close(
            dist_point_point(apply_motor(p2, M_rt), apply_motor(p2, M_screw)),
            0,
        ),
    )

    # extract_velocity: 纯平移幅值 + 纯旋转符号 (body frame)
    dt = 0.1
    (w_v, v_v) = extract_velocity(
        gp(translator((0.03, 0.0, 0.0)), identity_motor()), identity_motor(), dt
    )
    check(
        "extract_velocity translation",
        close(v_v[0], 0.3) and close(v_v[1], 0.0) and close(w_v[2], 0.0),
    )
    (w_r, v_r) = extract_velocity(
        gp(rotor((0, 0, 1), 0.2), identity_motor()), identity_motor(), dt
    )
    check("extract_velocity rotation sign", close(w_r[2], 2.0))

    print(f"\nall {_ok} checks passed")


if __name__ == "__main__":
    main()
