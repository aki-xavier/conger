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
    """

    es: list[mx.array]  # energies per orientation at this scale
    thetas: list[float]  # orientation angle in rad, uniform in [0, π)
    sum_e: mx.array | None = None  # total energy over orientations
    safe_e: mx.array | None = None
    mean_dir: mx.array | None = None  # 圆均值方向, rad in [0, π)
    resultant: mx.array | None = None  # R = |m₁| ∈ [0,1]: 1=单一方向, 0=各向同性
    r2: mx.array | None = None  # |m₂| ∈ [0,1]: 第二谐波——角点/十字（正交方向对）强度
    flatness: mx.array | None = None  # 方向能量的几何/算术均值比

    def __post_init__(self):
        total = self.es[0]
        for e in self.es[1:]:
            total = total + e
        self.sum_e = total
        self.safe_e = mx.maximum(self.sum_e, 1e-12)

        self.calc_circular_features()

    def calc_circular_features(self):
        # 方向分布的傅里叶展开（ψ = 2θ 的圆上）：m₁ = Σ p·e^{iψ},
        # m₂ = Σ p·e^{i2ψ}。(R, r2) 平面：R≈1 单方向(边缘)；
        # R≈0 且 r2≈1 正交方向对(角点/十字)；两者≈0 各向同性。
        c1 = mx.zeros_like(self.sum_e)
        s1 = mx.zeros_like(self.sum_e)
        c2 = mx.zeros_like(self.sum_e)
        s2 = mx.zeros_like(self.sum_e)
        for theta, e in zip(self.thetas, self.es, strict=True):
            p = e / self.safe_e
            c1 = c1 + p * math.cos(2 * theta)
            s1 = s1 + p * math.sin(2 * theta)
            c2 = c2 + p * math.cos(4 * theta)
            s2 = s2 + p * math.sin(4 * theta)

        self.resultant = mx.sqrt(c1 * c1 + s1 * s1)
        self.mean_dir = mx.remainder(0.5 * mx.arctan2(s1, c1), math.pi)
        self.r2 = mx.sqrt(c2 * c2 + s2 * s2)

        # ── flatness over orientation energies ──────────────────────
        # geometric mean via log space: exp((1/K) Σ log(E_k))
        log_sum = mx.log(self.es[0])
        for e in self.es[1:]:
            log_sum = log_sum + mx.log(e)
        geom = mx.exp(log_sum / len(self.es))
        assert self.sum_e is not None
        self.flatness = geom / mx.maximum(self.sum_e / len(self.es), 1e-12)


@dataclass(slots=True)
class GaborRotation:
    """Orientation-domain statistics: one GaborScale per scale of a bank."""

    gw: GaborWavelet
    scales: list[GaborScale] = field(default_factory=list)

    def __post_init__(self):
        for s in range(len(self.gw.lams)):
            es = [ori.es[s] for ori in self.gw.oris]
            self.scales.append(GaborScale(es=es, thetas=self.gw.thetas))

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
            ("flatness", "viridis", sc.flatness),
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
        gr = GaborRotation(GaborWavelet(arr))
        path = Utils.out_dir() / "artifacts" / f"nat{img_id}_rotation.png"
        print(path)
        gr.visualize(scale=2, out_path=path)
