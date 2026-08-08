"""CGA 包自检: python -m cga

验证核心代数与原语/versor 的正确性 (OOP API)。约定:
点/点对/线为直接形式, 关联判据 p.op(X) = 0; 平面/球/圆为对偶
形式, 关联判据 p.ip(X) = 0; versor 作用 M.apply(obj);
meet 接受直接形式输入。
"""

import math

import mlx.core as mx
import numpy as np

from cga import (
    Circle,
    Line,
    Motor,
    Multivector,
    Plane,
    Point,
    Sphere,
)

_ok = 0


def check(name: str, cond: bool) -> None:
    """断言一条检查并计数。"""
    global _ok
    assert cond, f"FAIL: {name}"
    _ok += 1
    print(f"  ok  {name}")


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    """|a−b| < tol。"""
    return abs(float(a) - float(b)) < tol


def vmax(mv: Multivector) -> float:
    """分量的最大绝对值。"""
    return float(mx.abs(mv.values).max().item())


def main() -> None:
    """全部自检: 代数 / 图元 / versor / exp-log / 距离。"""
    # null 性 + 距离
    p1, p2 = Point(0, 0, 0), Point(1, 0, 0)
    check("point is null", close(p1.gp(p1).values[0], 0))
    check("dist_point_point", close(p1.dist(Point(3, 4, 0)), 5.0))

    # 关联判据: 线 (直接形式) op(p, X) = 0
    L = Line(p1, p2)
    check("point on line", close(vmax(Point(2, 0, 0).op(L)), 0))
    check("point off line", vmax(Point(0, 1, 0).op(L)) > 1e-3)

    # 关联判据: 平面/球/圆 (对偶形式) ip(p, X) = 0
    pi = Plane((0, 0, 1), 2.0)  # z = 2 平面
    check("point on plane", close(vmax(Point(0.3, -0.7, 2).ip(pi)), 0))
    check("point off plane", vmax(Point(0, 0, 0).ip(pi)) > 1e-3)

    s = Sphere((1, 2, 3), 2.0)
    check("point on sphere", close(vmax(Point(3, 2, 3).ip(s)), 0))
    check("point off sphere", vmax(Point(0, 0, 0).ip(s)) > 1e-3)

    c = Circle((0, 0, 0), 1.0, (0, 0, 1))
    check("point on circle", close(vmax(Point(0, 1, 0).ip(c)), 0))
    check("point off circle", vmax(Point(0, 0, 1).ip(c)) > 1e-3)
    # 非单位法向: d 须按单位法向计算 (回归: Plane 归一化 n 但不缩放 d)
    cnu = Circle((1, 2, 3), 2.0, (0, 0, 2))
    check("circle non-unit normal", close(vmax(Point(3, 2, 3).ip(cnu)), 0))

    # 退化输入守卫
    try:
        Plane((0, 0, 0), 1.0)
        raise AssertionError("FAIL: degenerate plane accepted")
    except ValueError:
        check("degenerate plane raises", True)
    try:
        Circle((0, 0, 0), 1.0, (0, 0, 0))
        raise AssertionError("FAIL: degenerate circle accepted")
    except ValueError:
        check("degenerate circle raises", True)
    try:
        Multivector.bivector([1.0, 2.0])
        raise AssertionError("FAIL: short bivector accepted")
    except ValueError:
        check("bivector length check", True)

    # 距离函数 (float64 欧氏公式)
    check("dist_point_plane", close(Point(0, 0, 5).dist(pi), 3.0))
    check("dist_point_plane (plane 侧)", close(pi.dist(Point(0, 0, 5)), 3.0))
    check("dist_point_sphere on", close(Point(3, 2, 3).dist(s), 0))
    check("dist_point_sphere out", Point(5, 2, 3).dist(s) > 0)
    check("dist_point_sphere in", Point(1, 2, 3).dist(s) < 0)

    # meet (直接形式输入; 对偶原语先过 dual()):
    # 两平面交线 / 线球交点对
    pi2 = Plane((0, 1, 0), 1.0)  # y = 1 平面
    Lm = pi.dual().meet(pi2.dual())  # 交线: y=1, z=2, 沿 x 方向
    check(
        "meet(plane,plane) = line",
        close(vmax(Point(0, 1, 2).op(Lm)), 0) and close(vmax(Point(5, 1, 2).op(Lm)), 0),
    )
    Lz = Line(Point(0, 0, -2), Point(0, 0, 2))  # z 轴
    unit_s = Sphere((0, 0, 0), 1.0)
    PPm = Lz.meet(unit_s.dual())  # 交于 (0,0,±1)
    check(
        "meet(line,sphere) = point pair",
        close(vmax(Point(0, 0, 1).op(PPm)), 0)
        and close(vmax(Point(0, 0, -1).op(PPm)), 0),
    )

    # motor: 平移 / 旋转 / 复合 (作用: M.apply(obj))
    T = Motor.translator((1, 2, 3))
    check("translator", close(T.apply(p1).dist(Point(1, 2, 3)), 0))
    R = Motor.rotor((0, 0, 1), math.pi / 2)
    check("rotor 90° z", close(R.apply(p2).dist(Point(0, 1, 0)), 0))
    M = Motor((0, 0, 1), math.pi / 2, (1, 0, 0))
    check(
        "motor rot+trans",
        close(M.apply(p2).dist(Point(1, 1, 0)), 0),
    )

    # versor 保持关联: 平移后的线仍过平移后的点
    Lm2 = T.apply(L)
    check(
        "motor preserves incidence",
        close(vmax(T.apply(Point(2, 0, 0)).op(Lm2)), 0),
    )

    # versor 保持 meet: 先交后变换 = 先变换后交
    lhs = T.apply(pi.dual().meet(pi2.dual()))
    rhs = T.apply(pi).dual().meet(T.apply(pi2).dual())
    a, b = lhs.values, rhs.values
    scale = float(mx.abs(b).max().item())
    check(
        "versor preserves meet",
        bool(mx.allclose(a / scale, b / scale, atol=1e-4).item())
        or bool(mx.allclose(a / scale, -b / scale, atol=1e-4).item()),
    )

    # to_matrix 与 sandwich 作用一致
    Hm = np.array(M.to_matrix())
    p_h = Hm @ np.array([1, 0, 0, 1.0])
    check("motor_to_matrix", np.allclose(p_h[:3], [1, 1, 0], atol=1e-4))

    # exp/log roundtrip 与 motor 插值 (Motor.exp(B, s): exp(-s·B), s 带符号)
    R90 = Motor.rotor((0, 0, 1), math.pi / 2)
    R45 = Motor.rotor((0, 0, 1), math.pi / 4)
    B90 = R90.log()
    check(
        "exp∘log roundtrip",
        close(Motor.exp(B90).apply(p2).dist(R90.apply(p2)), 0),
    )
    check(
        "Motor.exp scale=0.5 = half motor",
        close(Motor.exp(B90, 0.5).apply(p2).dist(R45.apply(p2)), 0),
    )
    check(
        "Motor.exp negative scale = inverse",
        close(
            Motor.exp(B90, -0.5)
            .apply(p2)
            .dist(Motor.rotor((0, 0, 1), -math.pi / 4).apply(p2)),
            0,
        ),
    )
    check(
        "interpolate midpoint",
        close(
            Motor.identity().interpolate(R90, 0.5).apply(p2).dist(R45.apply(p2)),
            0,
        ),
    )

    # 远原点距离 (float32 conformal 内积会灾难性抵消 → 0, 回归检查)
    check(
        "dist far from origin",
        close(Point(1000, 0, 0).dist(Point(1001, 0, 0)), 1.0, tol=1e-2),
    )

    # 螺旋运动 (非零节距) exp∘log roundtrip
    M_screw = Motor((0, 0, 1), 0.4, (0.3, -0.2, 0.1))
    M_rt = Motor.exp(M_screw.log())
    check(
        "screw exp∘log roundtrip",
        close(M_rt.apply(p2).dist(M_screw.apply(p2)), 0),
    )

    # extract_velocity: 纯平移幅值 + 纯旋转符号 (body frame)
    dt = 0.1
    ID = Motor.identity()
    Mv = Motor.translator((0.03, 0.0, 0.0))
    (w_v, v_v) = Motor.extract_velocity(Mv.gp(ID), ID, dt)
    check(
        "extract_velocity translation",
        close(v_v[0], 0.3) and close(v_v[1], 0.0) and close(w_v[2], 0.0),
    )
    (w_r, v_r) = Motor.extract_velocity(Motor.rotor((0, 0, 1), 0.2).gp(ID), ID, dt)
    check("extract_velocity rotation sign", close(w_r[2], 2.0))

    print(f"\nall {_ok} checks passed")


if __name__ == "__main__":
    main()
