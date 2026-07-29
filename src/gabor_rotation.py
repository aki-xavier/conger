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
    resultant: mx.array | None = None  # R ∈ [0,1]: 1=单一方向, 0=各向同性
    skewness: mx.array | None = None  # Batschelet 圆偏度
    kurtosis: mx.array | None = None  # Batschelet 圆峰度（有符号）
    flatness: mx.array | None = None  # 方向能量的几何/算术均值比

    def __post_init__(self):
        total = self.es[0]
        for e in self.es[1:]:
            total = total + e
        self.sum_e = total
        self.safe_e = mx.maximum(self.sum_e, 1e-12)

        self.calc_circular_features()

    def calc_circular_features(self):
        # ── first trigonometric moment on doubled angles ψ = 2θ ─────
        c1 = mx.zeros_like(self.sum_e)
        s1 = mx.zeros_like(self.sum_e)
        for theta, e in zip(self.thetas, self.es, strict=True):
            p = e / self.safe_e
            c1 = c1 + p * math.cos(2 * theta)
            s1 = s1 + p * math.sin(2 * theta)

        r = mx.sqrt(c1 * c1 + s1 * s1)
        psi_bar = mx.arctan2(s1, c1)

        self.resultant = r
        self.mean_dir = mx.remainder(0.5 * psi_bar, math.pi)

        # ── second centered trigonometric moment → skew / kurt ──────
        # m2 = Σ p · exp(i·2(ψ−ψ̄)); Batschelet definitions.
        c2 = mx.zeros_like(self.sum_e)
        s2 = mx.zeros_like(self.sum_e)
        for theta, e in zip(self.thetas, self.es, strict=True):
            p = e / self.safe_e
            d = 4 * theta - 2 * psi_bar  # 2(ψ − ψ̄), ψ = 2θ
            c2 = c2 + p * mx.cos(d)
            s2 = s2 + p * mx.sin(d)

        self.skewness = s2 / mx.maximum((1.0 - r) ** 1.5, 1e-12)
        self.kurtosis = c2 / mx.maximum((1.0 - r) ** 2, 1e-12)

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
            ("skewness", "RdBu_r", sc.skewness),  # signed → diverging
            ("kurtosis", "RdBu_r", sc.kurtosis),  # signed → diverging
            ("flatness", "viridis", sc.flatness),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path, dpi=dpi)


if __name__ == "__main__":
    tasks = [
        ("signal03", Utils.synthesize_signal03()),  # vertical grating θ=0
        ("signal04", Utils.synthesize_signal04()),  # isotropic noise
        ("signal05", Utils.synthesize_signal05()),  # grating | smooth boundary
    ]
    for name, img in tasks:
        gw = GaborWavelet(img)
        gr = GaborRotation(gw)
        r_means = [f"{float(mx.mean(sc.resultant).item()):.3f}" for sc in gr.scales]
        print(f"{name}: lams={[round(lam, 1) for lam in gw.lams]}")
        print(f"  mean resultant R per scale: {r_means}")

        path = Utils.out_dir() / "artifacts" / (name + "_rotation.png")
        print(f"  {path}")
        gr.visualize(scale=0, out_path=path)
