import math

import mlx.core as mx

from utils import Utils


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

    def __init__(
        self,
        index: int,
        theta: float,
        wavelengths: list[float],
        responses: list[mx.array],
    ):
        self.index = index
        self.theta = theta
        self.wavelengths = wavelengths
        self._responses = responses
        self.scale_size = len(responses)

    def response_at(self, scale_index: int) -> mx.array:
        """Complex Gabor response at the given scale for this orientation."""
        if not (0 <= scale_index < self.scale_size):
            raise ValueError(
                f"scale must be in [0, {self.scale_size}), got {scale_index}"
            )
        return self._responses[scale_index]

    def amplitude_at(self, scale_index: int) -> mx.array:
        """Amplitude of the Gabor response at the given scale."""
        return mx.abs(self.response_at(scale_index))

    def pixel_response(self, x: int, y: int) -> list[float]:
        """Amplitude at pixel (x, y) across all scales for this orientation."""
        return [
            float(self.amplitude_at(s)[y, x].item()) for s in range(self.scale_size)
        ]

    def orientation_spectrum(self, x: int, y: int) -> tuple[list[float], list[float]]:
        """Frequency and amplitude at (x, y) across all scales."""
        freqs = [1.0 / self.wavelengths[s] for s in range(self.scale_size)]
        amps = [
            float(self.amplitude_at(s)[y, x].item()) for s in range(self.scale_size)
        ]
        return freqs, amps


class GaborWavelet:
    """2D Gabor wavelet filter bank via frequency-domain convolution.

    Decomposes an image into multiple scales x orientations using Gabor
    filters with octave-spaced wavelengths and uniform orientation sampling.

    Args:
        img: 2D input image (real or complex).  The filter bank takes the
            2D FFT internally, multiplies by each Gabor kernel in the
            frequency domain, then IFFTs to get spatial responses.
    """

    lam_min: float = 3.0
    """Finest usable wavelength (pixels). Slightly above Nyquist so the
    sinusoid has enough support to be orientation-selective."""

    def __init__(
        self,
        img: mx.array,
        scale_size: int | None = None,
        orientation_size: int = 8,
        bandwidth: float = 1.0,
        gamma: float = 0.5,
        adaptive_pad: bool = False,
    ):
        if img.ndim != 2:
            raise ValueError(f"img must be 2D, got shape {img.shape}")
        if scale_size is not None and scale_size < 1:
            raise ValueError(f"num_scales must be >= 1, got {scale_size}")
        if orientation_size < 1:
            raise ValueError(f"num_orientations must be >= 1, got {orientation_size}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
        if gamma <= 0:
            raise ValueError(f"aspect_ratio must be > 0, got {gamma}")

        self.img = img
        self.H, self.W = img.shape

        if scale_size is None:
            scale_size = self.default_scale_size(self.H, self.W)
        self.scale_size = scale_size
        self.orientation_size = orientation_size
        self.bandwidth = bandwidth
        self.gamma = gamma

        # ── self-adaptive padding to avoid FFT wraparound ────────────
        if adaptive_pad:
            self._pad = int(self.lam_max())
        else:
            self._pad = 0

        if self._pad > 0:
            H_pad = self.H + 2 * self._pad
            W_pad = self.W + 2 * self._pad
            padded = mx.pad(
                img, [(self._pad, self._pad), (self._pad, self._pad)], mode="edge"
            )
            self._F = mx.fft.fft2(padded)
            self.X, self.Y = Utils.freqgrid((H_pad, W_pad))
        else:
            self._F = mx.fft.fft2(img)
            self.X, self.Y = Utils.freqgrid((self.H, self.W))

        self.wavelengths: list[float] = []
        self._orientations: list[GaborOrientation] = []
        self._energy_cache: mx.array | None = None
        self._compute()

    @property
    def orientations(self) -> list[float]:
        """Orientation angles in radians."""
        return [go.theta for go in self._orientations]

    @staticmethod
    def default_scale_size(H: int, W: int) -> int:
        """Band count the size-adaptive auto derivation picks for an H×W
        image: octaves between lam_min and lam_max = min(H, W)/2, floored
        at 4. Exposed separately so a caller that builds a filter bank on
        a CROP of a larger frame (foveal_attention._engine_kwargs) can
        align the crop's band count with the frame-level bank — a frozen
        GMM fixes the feature dimension, so crop and frame must agree on
        it (that agreement is what makes the foveal column-alignment
        promise possible at all)."""
        lam_max = min(H, W) / 2.0
        return max(4, round(math.log2(lam_max / GaborWavelet.lam_min)) + 1)

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.H, self.W) / 2.0

    @property
    def energy_tensor(self) -> mx.array:
        """|R_{s,o}|² for all scales and orientations.  Shape (S, O, H, W).
        Lazily computed and cached.
        """
        if self._energy_cache is None:
            S, n_ori = self.scale_size, self.orientation_size
            layers = []
            for s in range(S):
                ori_layers = []
                for o in range(n_ori):
                    ori_layers.append(mx.abs(self._orientations[o]._responses[s]) ** 2)
                layers.append(mx.stack(ori_layers, axis=0))
            self._energy_cache = mx.stack(layers, axis=0)
        return self._energy_cache

    def amplitude_at(self, scale_index: int, orientation_index: int) -> mx.array:
        return self._orientations[orientation_index].amplitude_at(scale_index)

    def response_at(self, scale_index: int, orientation_index: int) -> mx.array:
        return self._orientations[orientation_index].response_at(scale_index)

    def orientation_spectrum(
        self, x: int, y: int, orientation_index: int
    ) -> tuple[list[float], list[float]]:
        go = self._orientations[orientation_index]
        freqs = [1.0 / go.wavelengths[s] for s in range(go.scale_size)]
        amps = [float(go.amplitude_at(s)[y, x].item()) for s in range(go.scale_size)]
        return freqs, amps

    def smoothness_map(self) -> mx.array:
        dc_energy = mx.abs(self.dc_response) ** 2
        gabor_energy = self.energy_tensor.sum(axis=(0, 1))
        ref_dc = max(
            float(mx.median(dc_energy.reshape(-1)).item()),
            1e-12,
        )
        return ref_dc / (ref_dc + gabor_energy)

    def _compute(self):
        self.wavelengths = self._octave_wavelengths()
        thetas = [
            i * math.pi / self.orientation_size for i in range(self.orientation_size)
        ]

        # DC / lowpass channel: narrow Gaussian captures local mean intensity.
        # σ scales with the coarsest wavelength so the cutoff sits below the
        # Gabor bank's lowest band for any image size.
        sigma_spatial = self.lam_max() / 8.0
        sigma_f = 1.0 / (2.0 * math.pi * sigma_spatial)
        r2 = self.X**2 + self.Y**2
        h_dc = mx.exp(-0.5 * r2 / sigma_f**2)
        self.dc_response = mx.real(mx.fft.ifft2(self._F * h_dc))

        # ── crop padding ────────────────────────────────────────────
        pad = self._pad
        if pad > 0:
            self.dc_response = self.dc_response[pad : pad + self.H, pad : pad + self.W]

        self._orientations = []
        for o, theta in enumerate(thetas):
            ori_responses: list[mx.array] = []
            for lam in self.wavelengths:
                kernel_F = self.gabor_kernel_freq(lam, theta)
                resp_F = self._F * kernel_F
                resp = mx.fft.ifft2(resp_F)
                if pad > 0:
                    resp = resp[pad : pad + self.H, pad : pad + self.W]
                ori_responses.append(resp)
            go = GaborOrientation(
                index=o,
                theta=theta,
                wavelengths=list(self.wavelengths),
                responses=ori_responses,
            )
            self._orientations.append(go)

    def gabor_kernel_freq(self, lam: float, theta: float) -> mx.array:
        f0 = 2.0 / lam
        if self._pad > 0:
            # frequency grid is denser with padding — scale f0 down
            f0 *= self.W / (self.W + 2 * self._pad)
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )
        sigma_f = sigma_f_rel * f0

        u = self.X * math.cos(theta) + self.Y * math.sin(theta)
        v = -self.X * math.sin(theta) + self.Y * math.cos(theta)
        du = u - f0
        g = mx.exp(-0.5 * (du**2 / sigma_f**2 + v**2 / (sigma_f / self.gamma) ** 2))
        return g

    def _octave_wavelengths(self) -> list[float]:
        lam_min = self.lam_min
        lam_max = self.lam_max()
        if self.scale_size == 1:
            return [lam_min]
        return [
            lam_min * 2.0 ** (i * math.log2(lam_max / lam_min) / (self.scale_size - 1))
            for i in range(self.scale_size)
        ]
