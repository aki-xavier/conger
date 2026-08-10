"""CGA 立方体建模 → 2D 渲染 (demo_cube.py, 2026-08-10)。

建模: 6 个 CGA Plane blade (外向法向, 半空间 n̂·x ≤ d), 立方体中心
(0,0,3) 半边长 0.6 —— 闭凸多面体 = 半空间交 (render.py 标注的
"面片范围留钩" 的最小实现: 无限平面 + 半空间裁剪)。

渲染: 与 cga/render.py 同相机约定 (X 右 / Y 下 / Z 前, 相机在原点,
col = fx·X/Z + cx)。每像素光线 → 6 平面最近命中 (t = d/(n̂·dir)) →
命中点须在全部 6 半空间内 (立方体表面判定) → z-buffer 最近面。
motor (Motor) 是世界→相机变换的换视角入口 (versor 共轭 blade)。

着色: 面片色盘 + 固定方向 Lambert (同 render.py 约定, 非光学声明)。
输出: artifacts/cube.png (正视 + 旋转视角)。
"""

import math

import matplotlib.pyplot as plt
import mlx.core as mx

from cga import Motor, Plane
from utils import Utils

# CJK 字体: matplotlib 默认字体无中文字形 (中文标题会渲染成方框)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

CENTER = (0.0, 0.0, 3.0)  # 立方体中心 (相机前 3m)
HALF = 0.6  # 半边长
K = (500.0, 500.0, 320.0, 240.0)  # (fx, fy, cx, cy)
SHAPE = (480, 640)

PALETTE = [
    (244, 67, 54), (33, 150, 243), (76, 175, 80),
    (255, 193, 7), (156, 39, 176), (0, 188, 212),
]
_LIGHT_RAW = mx.array([0.3, 0.6, 1.0], dtype=mx.float32)
_LIGHT = _LIGHT_RAW / mx.linalg.norm(_LIGHT_RAW)


def make_cube() -> list[Plane]:
    """6 个外向法向半空间: n̂·x ≤ d。"""
    cx, cy, cz = CENTER
    return [
        Plane((1.0, 0.0, 0.0), cx + HALF),    # 右
        Plane((-1.0, 0.0, 0.0), -(cx - HALF)),  # 左
        Plane((0.0, 1.0, 0.0), cy + HALF),    # 下 (Y 向下, 视觉下方)
        Plane((0.0, -1.0, 0.0), -(cy - HALF)),  # 上
        Plane((0.0, 0.0, 1.0), cz + HALF),    # 远面 (z = cz+HALF, 法向 +z 向外)
        Plane((0.0, 0.0, -1.0), -(cz - HALF)),  # 近面 (z = cz−HALF, 法向 −z 向外)
    ]


def plane_nd(blade: Plane) -> tuple[mx.array, float]:
    """Plane blade → (单位法向 (3,), 距离 d), n̂·x = d。"""
    n = blade.values[1:4]
    return n, float(blade.values[5])


def render_cube(
    planes: list[Plane], motor: Motor | None = None,
) -> mx.array:
    """半空间裁剪渲染 → (H,W,3) uint8 rgb (z-buffer 最近有效面)。"""
    H, W = SHAPE
    fx, fy, cx, cy = K
    yy, xx = mx.meshgrid(
        mx.arange(H, dtype=mx.float32), mx.arange(W, dtype=mx.float32),
        indexing="ij",
    )
    dirs = mx.stack(
        [(xx - cx) / fx, (yy - cy) / fy, mx.ones_like(xx)], axis=-1
    )
    bl = [motor.apply(p) if motor is not None else p for p in planes]
    best = mx.full((H, W), float("inf"))
    best_rgb = mx.zeros((H, W, 3), dtype=mx.uint8)
    for i, b in enumerate(bl):
        n, d = plane_nd(b)
        denom = mx.sum(dirs * n[None, None, :], axis=-1)
        t = d / denom  # 光线命中参数 (从原点, t = Z)
        hit = t > 0.1  # 相机前
        # 半空间裁剪: 命中点在全部 6 面半空间内 (n̂·p ≤ d + ε)
        p = dirs * t[..., None]
        inside = mx.full((H, W), True, dtype=mx.bool_)
        for b2 in bl:
            n2, d2 = plane_nd(b2)
            inside = inside & (
                mx.sum(p * n2[None, None, :], axis=-1) <= d2 + 1e-3
            )
        valid = hit & inside & (t < best)
        # Lambert 明暗
        sh = mx.maximum(
            mx.sum(mx.array(PALETTE[i], dtype=mx.float32) / 255.0
                   * mx.array(_LIGHT) * n[None, None, :], axis=-1),
            0.15,
        )
        rgb = mx.stack(
            [mx.full((H, W), PALETTE[i][k] * sh) for k in range(3)], axis=-1
        )
        best = mx.where(valid, t, best)
        best_rgb = mx.where(valid[..., None], rgb, best_rgb)
    return best_rgb


def rotation_motor(axis: tuple[float, float, float], angle_deg: float) -> Motor:
    """绕**立方体中心**旋转 angle_deg: M = T(c)·R·T(−c) —— 绕原点会
    把立方体甩出画面中心且视角不对 (实测绕 Y 28° 只见近面)。
    绕自身中心 = 3/4 姿态, 三个面可见。"""
    ax, ay, az = axis
    nl = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / nl, ay / nl, az / nl
    th = math.radians(angle_deg)
    c, s, t = math.cos(th), math.sin(th), 1.0 - math.cos(th)
    R = mx.array([
        [c + ax * ax * t, ax * ay * t - az * s, ax * az * t + ay * s],
        [ay * ax * t + az * s, c + ay * ay * t, ay * az * t - ax * s],
        [az * ax * t - ay * s, az * ay * t + ax * s, c + az * az * t],
    ], dtype=mx.float32)
    rot = Motor.from_matrix(R, (0.0, 0.0, 0.0))
    cx, cy, cz = CENTER
    t_c = Motor.translator((cx, cy, cz))
    t_n = Motor.translator((-cx, -cy, -cz))
    return Motor._wrap(t_c.gp(rot).gp(t_n))  # type: ignore[reportPrivateUsage]


def main() -> None:
    root = Utils.project_root()
    cube = make_cube()
    # 正视 (只看到前表面方块的投影范围) + 绕 Y 轴 28° (看到 3 面)
    views = [
        ("绕 Y 30° (3/4 姿态)", rotation_motor((0.0, 1.0, 0.0), 30.0)),
        ("绕 X -20° + Y 30°", None),
        ("正视 (仅近面, 透视参考)", None),
    ]
    # 绕中心双轴: motor 直接 gp 组合 (T·R₁·T⁻¹)·(T·R₂·T⁻¹) = T·R₁R₂·T⁻¹
    m2 = Motor._wrap(  # type: ignore[reportPrivateUsage]
        rotation_motor((1.0, 0.0, 0.0), -20.0).gp(
            rotation_motor((0.0, 1.0, 0.0), 30.0)
        )
    )
    views[1] = ("绕 X -20° + Y 30°", m2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, motor) in zip(axes, views):
        rgb = render_cube(cube, motor)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out = root / "artifacts/cube.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
    # 断言: 中心光线命中前表面 z=3.6; 立方体透视投影宽 = 前后角点
    # x/z 范围 (中心 0 → 后角 0.6/2.4=0.25 → 250px 半宽 → 500px 全宽?
    # 不对: 左右对称, 全宽 = 2·500·(0.6/2.4) = 250px)
    rgb0 = render_cube(cube)
    row = rgb0.shape[0] // 2
    mask = mx.any(rgb0[row] != 0, axis=-1)
    nz = Utils.nonzero(mask)  # MLX 无布尔索引 → 项目共享技巧
    w_px = int(nz[-1]) - int(nz[0]) + 1
    expect = int(2 * K[0] * HALF / (CENTER[2] - HALF))
    assert abs(w_px - expect) < 3, f"投影宽 {w_px} vs 期望 {expect}"
    print(f"正视投影宽 {w_px}px (期望 {expect}px, 前后角点透视) ✓")


if __name__ == "__main__":
    main()
