import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from gabor import GaborWavelet
from utils import Utils


@dataclass(slots=True)
class GaborScale:
    """Circular spectrum statistics over orientations for a single scale.

    Gabor orientations are axial (θ and θ+π are identical), so all
    circular statistics are computed on doubled angles ψ = 2θ and
    mapped back to orientation units at the end.

    CGA 视角: per-scale 统计给出原语的尺寸参数——阶跃全尺度响应 (α=2),
    细线的特征尺度=线宽, 光栅的特征尺度=λ; 尺度维即尺寸测量轴,
    因此方向统计必须 per-scale, 不可跨尺度先合并。
    """

    es: list[mx.array]  # energies per orientation at this scale
    thetas: list[float]  # orientation angle in rad, uniform in [0, π)
    e_ref: float = 1e-4  # 能量可靠性参考 (flat 底噪量级, 输入需归一化 [0,1])
    sum_e: mx.array | None = None  # total energy over orientations
    safe_e: mx.array | None = None
    rho: mx.array | None = None  # 能量可靠性 sum_e/(sum_e+e_ref): 退化区降权
    mean_dir: mx.array | None = None  # 圆均值方向, rad in [0, π)
    resultant: mx.array | None = None  # R = |m₁| ∈ [0,1]: 1=单一方向, 0=各向同性
    r2: mx.array | None = None  # |m₂| ∈ [0,1]: 第二谐波——角点/十字（正交方向对）强度
    # 圆矩分量 (复矩 m₁ = m1c + i·m1s, m₂ 同理), 供跨尺度相干聚合
    m1c: mx.array | None = None
    m1s: mx.array | None = None
    m2c: mx.array | None = None
    m2s: mx.array | None = None

    def __post_init__(self):
        total = self.es[0]
        for e in self.es[1:]:
            total = total + e
        self.sum_e = total
        self.safe_e = mx.maximum(self.sum_e, 1e-12)
        self.rho = self.sum_e / (self.sum_e + self.e_ref)

        self.calc_circular_features()

    def calc_circular_features(self):
        # 方向分布的傅里叶展开（ψ = 2θ 的圆上）：m₁ = Σ p·e^{iψ},
        # m₂ = Σ p·e^{i2ψ}。(R, r2) 平面：R≈1 单方向(边缘)；
        # R≈0 且 r2≈1 正交方向对(角点/十字)；两者≈0 各向同性。
        self.m1c = mx.zeros_like(self.sum_e)
        self.m1s = mx.zeros_like(self.sum_e)
        self.m2c = mx.zeros_like(self.sum_e)
        self.m2s = mx.zeros_like(self.sum_e)
        for theta, e in zip(self.thetas, self.es, strict=True):
            p = e / self.safe_e
            self.m1c = self.m1c + p * math.cos(2 * theta)
            self.m1s = self.m1s + p * math.sin(2 * theta)
            self.m2c = self.m2c + p * math.cos(4 * theta)
            self.m2s = self.m2s + p * math.sin(4 * theta)

        self.resultant = mx.sqrt(self.m1c * self.m1c + self.m1s * self.m1s)
        self.mean_dir = mx.remainder(0.5 * mx.arctan2(self.m1s, self.m1c), math.pi)
        self.r2 = mx.sqrt(self.m2c * self.m2c + self.m2s * self.m2s)


# ── 原语假设场: CGA 前向推理的测量接口 ────────────────────────────────────
#
# 每像素一组带权几何假设 (位置即像素坐标, 不重复存储)。知觉组织层消费
# 这些场做 meet/join 聚合 (edgel → image line/circle, junction → point),
# 无需再碰原始谱特征; weight 即 meet/join 的测量权重。


@dataclass(slots=True)
class EdgelField:
    """有向线元场 — 图像平面内 line 原语的逐像素关联证据。

    跨尺度相干聚合 (能量加权矩之和 = 合并方向谱之矩, 精确恒等);
    weight = ρ·R: 能量可靠性 × 方向集中度。
    """

    dir: mx.array  # (H, W) 方向, rad in [0, π)
    resultant: mx.array  # (H, W) R
    weight: mx.array  # (H, W)


@dataclass(slots=True)
class JunctionField:
    """结点场 — point 原语 (角点/十字) 证据; T/X 结点承载遮挡拓扑。

    weight = ρ·r2·(1−R): 第二谐波强度, 且排除单方向像素 (那是 edgel 的
    职责)——(R, r2) 平面上结点即 R≈0 且 r2≈1 区域。
    """

    r2: mx.array  # (H, W)
    weight: mx.array  # (H, W)


@dataclass(slots=True)
class TextureField:
    """纹理场 — plane 表面周期标记的参数 (λ, θ) + 显著度。

    shape-from-texture: 倾斜平面上 λ 沿 tilt 方向压缩, (λ, θ) 即 plane
    姿态约束。residual = 鼓包存在性证据 (非 1/f 程度), weight = ρ·R。
    """

    lam: mx.array  # (H, W) 波长 px = 1/bump_freq (主导方向)
    dir: mx.array  # (H, W) 方向, rad in [0, π)
    residual: mx.array  # (H, W) 鼓包显著度 (主导方向)
    weight: mx.array  # (H, W)


@dataclass(slots=True)
class GaborDirection:
    """Orientation-domain statistics → weighted primitive hypothesis fields."""

    gw: GaborWavelet
    e_ref: float = 1e-4
    scales: list[GaborScale] = field(default_factory=list)
    edgel: EdgelField | None = None
    junction: JunctionField | None = None
    texture: TextureField | None = None

    def __post_init__(self):
        for s in range(len(self.gw.lams)):
            es = [ori.es[s] for ori in self.gw.oris]
            self.scales.append(
                GaborScale(es=es, thetas=self.gw.thetas, e_ref=self.e_ref)
            )
        self.calc_fields()

    def calc_fields(self):
        # 跨尺度相干聚合: m_global = Σ_s (E_s/E)·m_s —— 与"先合并方向谱
        # 再取矩"严格相等, 故 per-scale 表示不损失任何全局信息
        total_e = self.scales[0].sum_e
        m1c = self.scales[0].m1c * self.scales[0].sum_e
        m1s = self.scales[0].m1s * self.scales[0].sum_e
        m2c = self.scales[0].m2c * self.scales[0].sum_e
        m2s = self.scales[0].m2s * self.scales[0].sum_e
        for sc in self.scales[1:]:
            total_e = total_e + sc.sum_e
            m1c = m1c + sc.m1c * sc.sum_e
            m1s = m1s + sc.m1s * sc.sum_e
            m2c = m2c + sc.m2c * sc.sum_e
            m2s = m2s + sc.m2s * sc.sum_e
        safe = mx.maximum(total_e, 1e-12)
        m1c, m1s, m2c, m2s = m1c / safe, m1s / safe, m2c / safe, m2s / safe

        rho = total_e / (total_e + self.e_ref)
        R = mx.sqrt(m1c * m1c + m1s * m1s)
        r2 = mx.sqrt(m2c * m2c + m2s * m2s)
        direction = mx.remainder(0.5 * mx.arctan2(m1s, m1c), math.pi)

        self.edgel = EdgelField(dir=direction, resultant=R, weight=rho * R)
        self.junction = JunctionField(r2=r2, weight=rho * r2 * (1.0 - R))

        # texture: 主导方向 (argmax_θ Σ_s e) 的谱鼓包参数 —— 前向模型
        # 中光栅 → bump at 1/λ, 故 λ = 1/bump_freq
        e_ori = mx.stack([o.sum_e for o in self.gw.oris])  # (O, H, W)
        dom = mx.argmax(e_ori, axis=0, keepdims=True)

        def at_dom(name: str) -> mx.array:
            f = mx.stack([getattr(o, name) for o in self.gw.oris])  # (O,H,W)
            return mx.take_along_axis(f, dom, axis=0)[0]

        self.texture = TextureField(
            lam=1.0 / mx.maximum(at_dom("bump_freq"), 1e-6),
            dir=direction,
            residual=at_dom("residual"),
            weight=rho * R,
        )

    def visualize(
        self,
        scale: int,
        out_path: str | Path,
        dpi: int = 150,
    ):
        """Render one scale's orientation feature maps to an image.

        Args:
            scale: index into ``self.scales`` / ``self.gw.lams``.
            out_path: save the figure here (e.g. ``"scale.png"``).
            dpi: Save resolution.
        """
        sc = self.scales[scale]
        plots = [
            ("original", "gray", self.gw.img),
            ("mean_dir", "twilight", sc.mean_dir),  # circular colormap
            ("resultant", "viridis", sc.resultant),
            ("r2", "viridis", sc.r2),
            ("rho", "viridis", sc.rho),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path, dpi=dpi)

    def visualize_fields(self, out_path: str | Path, dpi: int = 150):
        """Render the aggregated primitive hypothesis fields."""
        plots = [
            ("original", "gray", self.gw.img),
            ("edgel.dir", "twilight", self.edgel.dir),
            ("edgel.weight", "viridis", self.edgel.weight),
            ("junction.weight", "viridis", self.junction.weight),
            ("texture.lam", "viridis", self.texture.lam),
            ("texture.weight", "viridis", self.texture.weight),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path, dpi=dpi)


if __name__ == "__main__":
    # natural images (downloaded from picsum.photos)
    from PIL import Image

    from color import Color

    for img_id in [10, 1015, 1016, 1018, 1035]:
        img = Image.open(Utils.out_dir() / f"images/nat{img_id}.jpg")
        arr = Color.image_to_mlx(img.convert("L"))
        gr = GaborDirection(GaborWavelet(arr))
        path = Utils.out_dir() / "artifacts" / f"nat{img_id}_direction.png"
        print(path)
        gr.visualize(scale=2, out_path=path)
        path = Utils.out_dir() / "artifacts" / f"nat{img_id}_fields.png"
        print(path)
        gr.visualize_fields(out_path=path)
