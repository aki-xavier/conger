import math
from dataclasses import dataclass

import mlx.core as mx

from utils import Utils


@dataclass(slots=True)
class GaborOrientation:
    """All Gabor scales for a single orientation.

    Owns its responses, wavelengths, and orientation metadata directly.
    Callers no longer need a parent wavelet to use an orientation.

    Args:
        index: Orientation index.
        theta: Orientation angle in radians.
        wavelengths: Centre wavelength for each scale.
        responses: Complex Gabor response at each scale for this orientation.
    """

    theta: float
    scales: list[mx.array]


@dataclass(slots=True)
class GaborWavelet:
    thetas: list[float]  # orientation angle in rad
    lams: list[float]  # wavelength
    oris: list[GaborOrientation]
    img: mx.array
    lam_min: float = 3.0  # min wavelength
    height: int = 0
    width: int = 0
    scale_size: int = 0
    ori_size: int = 8
    bandwidth: float = 1.0  # used to create gabor kernel
    gamma: float = 0.5  # used to create gabor kernel
    adaptive_pad: bool = False
    pad: int = 0
    fft: mx.array | None = None
    xgrid: mx.array | None = None
    ygrid: mx.array | None = None
    dc: mx.array | None = None

    def __post_init__(self):
        if self.img.ndim != 2:
            raise ValueError(f"img must be 2D, got shape {self.img.shape}")
        if self.ori_size < 1:
            raise ValueError(f"num_orientations must be >= 1, got {self.ori_size}")
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")
        if self.gamma <= 0:
            raise ValueError(f"aspect_ratio must be > 0, got {self.gamma}")

        self.height, self.width = self.img.shape

        if self.scale_size <= 0:
            lam_max = self.lam_max()
            s = round(math.log2(lam_max / self.lam_min)) + 1
            self.scale_size = max(4, s)

        self.calc_pad()
        self.calc_lams()
        self.calc_thetas()
        self.calc_dc()
        self.calc_oris()

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.height, self.width) / 2.0

    def calc_pad(self):
        # ── self-adaptive padding to avoid FFT wraparound ────────────
        if self.adaptive_pad:
            self.pad = int(self.lam_max())
            H_pad = self.height + 2 * self.pad
            W_pad = self.width + 2 * self.pad
            padded = mx.pad(
                self.img,
                [(self.pad, self.pad), (self.pad, self.pad)],
                mode="edge",
            )
            self.fft = mx.fft.fft2(padded)
            self.xgrid, self.ygrid = Utils.freqgrid((H_pad, W_pad))
        else:
            self.fft = mx.fft.fft2(self.img)
            self.xgrid, self.ygrid = Utils.freqgrid((self.height, self.width))

    def calc_lams(self):
        lam_min = self.lam_min
        lam_max = self.lam_max()
        if self.scale_size == 1:
            self.lams.append(lam_min)
        else:
            for i in range(self.scale_size):
                lam = lam_min * 2.0 ** (
                    i * math.log2(lam_max / lam_min) / (self.scale_size - 1)
                )
                self.lams.append(lam)

    def calc_thetas(self):
        for i in range(self.ori_size):
            theta = i * math.pi / self.ori_size
            self.thetas.append(theta)

    def calc_dc(self):
        # DC / lowpass channel: narrow Gaussian captures local mean intensity.
        # σ scales with the coarsest wavelength so the cutoff sits below the
        # Gabor bank's lowest band for any image size.
        sigma_spatial = self.lam_max() / 8.0
        sigma_f = 1.0 / (2.0 * math.pi * sigma_spatial)

        assert self.xgrid is not None
        assert self.ygrid is not None
        r2 = self.xgrid**2 + self.ygrid**2
        h_dc = mx.exp(-0.5 * r2 / sigma_f**2)
        dc = mx.real(mx.fft.ifft2(self.fft * h_dc))

        # ── crop padding ────────────────────────────────────────────
        if self.pad > 0:
            dc = dc[
                self.pad : self.pad + self.height,
                self.pad : self.pad + self.width,
            ]

        self.dc = dc

    def calc_oris(self):
        for _o, theta in enumerate(self.thetas):
            scales: list[mx.array] = []
            for lam in self.lams:
                kernel = self.gabor_kernel(lam, theta)
                resp_f = self.fft * kernel
                resp = mx.fft.ifft2(resp_f)
                if self.pad > 0:
                    resp = resp[
                        self.pad : self.pad + self.height,
                        self.pad : self.pad + self.width,
                    ]
                scales.append(resp)
            go = GaborOrientation(
                theta=theta,
                scales=scales,
            )
            self.oris.append(go)

    def resp_at(self, scale_idx: int, ori_idx: int) -> mx.array:
        return self.oris[ori_idx].scales[scale_idx]

    def gabor_kernel(self, lam: float, theta: float) -> mx.array:
        f0 = 2.0 / lam
        if self.pad > 0:
            # frequency grid is denser with padding — scale f0 down
            f0 *= self.width / (self.width + 2 * self.pad)
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )

        sigma_f = sigma_f_rel * f0

        assert self.xgrid is not None
        assert self.ygrid is not None
        u = self.xgrid * math.cos(theta) + self.ygrid * math.sin(theta)
        v = -self.xgrid * math.sin(theta) + self.ygrid * math.cos(theta)
        du = u - f0
        return mx.exp(-0.5 * (du**2 / sigma_f**2 + v**2 / (sigma_f / self.gamma) ** 2))
