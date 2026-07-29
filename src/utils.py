import math
from pathlib import Path

import matplotlib
import mlx.core as mx

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Utils:
    @staticmethod
    def out_dir() -> Path:
        d = Path(__file__).resolve().parent.parent
        return d

    @staticmethod
    def fftfreq(n: int) -> mx.array:
        """MLX version of np.fft.fftfreq(n), in cycles/sample (Nyquist = 0.5)."""
        k = mx.arange(n, dtype=mx.float32)
        half = (n + 1) // 2
        k = mx.where(k < half, k, k - n)
        return k / n

    @staticmethod
    def freqgrid(shape: tuple[int, ...]) -> list[mx.array]:
        """Generate normalized frequency grids for a given height and width."""
        height, width = shape
        x = Utils.fftfreq(width)
        y = Utils.fftfreq(height)
        return mx.meshgrid(x, y)

    @staticmethod
    def standard_normal_pdf(
        amp: float, sigma: float, x: mx.array, y: mx.array
    ) -> mx.array:
        amp = abs(amp)
        sigma = abs(sigma)
        exponent = -0.5 * (x**2 + y**2) / (sigma**2)
        ret = amp * mx.exp(exponent)
        return mx.array(ret)

    @staticmethod
    def grid_shape(n: int) -> tuple[int, int]:
        if n <= 0:
            return (0, 0)
        # 从 sqrt(n) 向下找第一个能整除 n 的数（不含 1，否则质数会退化成单行）
        for rows in range(int(math.sqrt(n)), 1, -1):
            if n % rows == 0:
                cols = n // rows
                return (rows, cols)
        # n 是质数等无法整除的情况：向上取整，倾向 rows < cols（横向布局）
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return (rows, cols)

    @staticmethod
    def visualize(plots: list[tuple[str, str, mx.array]]):
        rows, cols = Utils.grid_shape(len(plots))
        fig, axes = plt.subplots(rows, cols, squeeze=False)
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                ax = axes[row][col]
                ax.set_xticks([])
                ax.set_yticks([])
                if idx >= len(plots):  # grid has more slots than plots
                    ax.axis("off")
                    continue
                title, cmap, data = plots[idx]
                im = ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return fig

    @staticmethod
    def normalize(arr: mx.array) -> mx.array:
        arr_min = mx.min(arr)
        arr_max = mx.max(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    @staticmethod
    def invert(mlx_arr: mx.array) -> mx.array:
        return 1.0 - mlx_arr

    @staticmethod
    def synthesize_signal01(size: int = 300) -> mx.array:
        left = mx.zeros((size, size // 2), dtype=mx.float32)
        right = mx.ones((size, size - size // 2), dtype=mx.float32)
        return mx.concatenate([left, right], axis=1)

    @staticmethod
    def synthesize_signal02(size: int = 300) -> mx.array:
        black = mx.zeros((size, int(size * 0.45)), dtype=mx.float32)
        x_end = int(size * 0.55)
        ramp = mx.linspace(0.0, 1.0, x_end - int(size * 0.45))
        ramp_2d = mx.repeat(ramp.reshape(1, -1), size, axis=0)
        white = mx.ones((size, size - x_end), dtype=mx.float32)
        return mx.concatenate([black, ramp_2d, white], axis=1)

    @staticmethod
    def make_grating(shape, wavelength, angle_rad, phase=0.0) -> mx.array:  # type: ignore
        """Sinusoidal grating at given wavelength (px) and angle (rad), in [0, 1]."""
        H, W = shape
        y = mx.arange(H, dtype=mx.float32)
        x = mx.arange(W, dtype=mx.float32)
        yy, xx = mx.meshgrid(y, x, indexing="ij")
        xr = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
        s = mx.sin(2 * math.pi * xr / wavelength + phase).astype(mx.float32)
        return (s + 1.0) * 0.5

    @staticmethod
    def make_texture_composite(size: int = 128):
        """4-quadrant composite for Gabor clustering test.
        TL: λ=8, 0° | TR: λ=24, 45° | BL: uniform noise | BR: flat 0.5.
        """
        H, W = size, size
        hh, hw = H // 2, W // 2
        tl = Utils.make_grating((hh, hw), wavelength=8.0, angle_rad=0.0)
        tr = Utils.make_grating((hh, hw), wavelength=24.0, angle_rad=math.radians(45))
        key = mx.random.key(42)
        bl = mx.random.uniform(shape=(hh, hw), key=key)
        br = mx.full((hh, hw), 0.5, dtype=mx.float32)
        top = mx.concatenate([tl, tr], axis=1)
        bot = mx.concatenate([bl, br], axis=1)
        return mx.concatenate([top, bot], axis=0)

    @staticmethod
    def synthesize_signal03(size: int = 300) -> mx.array:
        """Sinusoidal grating — fine texture, no edge.

        Vertical grating at 8 px wavelength fills the entire image.
        """
        return Utils.make_grating((size, size), wavelength=8.0, angle_rad=0.0)

    @staticmethod
    def synthesize_signal04(size: int = 300) -> mx.array:
        """Random Gaussian noise — broadband texture.

        Normalized to [0, 1] after clipping to 2σ.
        """
        noise = mx.random.normal(shape=(size, size), key=mx.random.key(0))
        return Utils.normalize(mx.clip(noise, -2.0, 2.0))

    @staticmethod
    def synthesize_signal05(
        size: int = 300, wavelength: float = 16.0, boundary: float = 0.5
    ) -> mx.array:
        """Vertical grating on the left, smooth on the right — direction-specific
        continuity drop at the boundary.

        Left half: vertical-stripe grating (θ=0° texture).
        Right half: uniform gray (no texture).

        At the boundary, θ=0° continuity drops sharply (texture →
        smooth). At θ=90° no texture exists in either half → no drop.

        Args:
            size: Image side length (square).
            wavelength: Grating wavelength in pixels.
            boundary: Horizontal position of the boundary (0..1).

        Returns:
            (size, size) float32 array.
        """
        grating = Utils.make_grating((size, size), wavelength=wavelength, angle_rad=0.0)
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x < boundary, 1.0, 0.0)
        mask = mask.reshape(1, -1)
        return grating * mask

    @staticmethod
    def synthesize_signal06(
        size: int = 300, wavelength: float = 16.0, boundary: float = 0.5
    ) -> mx.array:
        """Smooth on the left, grating on the right — Type B: smooth→texture.

        Left half: uniform gray. Right half: grating. At the boundary,
        all spectral metrics jump simultaneously as energy shifts
        from a single coarse scale to a specific matching scale.

        Args:
            size: Image side length (square).
            wavelength: Grating wavelength in pixels.
            boundary: Position of the boundary (0..1).

        Returns:
            (size, size) float32 array.
        """
        grating = Utils.make_grating((size, size), wavelength=wavelength, angle_rad=0.0)
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x >= boundary, 1.0, 0.0)
        mask = mask.reshape(1, -1)
        return grating * mask + 0.5 * mx.where(x < boundary, 1.0, 0.0).reshape(1, -1)

    @staticmethod
    def synthesize_signal07(
        size: int = 300,
        wavelength1: float = 16.0,
        wavelength2: float = 8.0,
        boundary: float = 0.5,
    ) -> mx.array:
        """Grating frequency change — Type D: texture→texture (different scale).

        Left half: grating at wavelength1. Right half: grating at
        wavelength2. Slope and res_scale jump; the fit residual stays
        similar (both halves are single-scale).

        Args:
            size: Image side length (square).
            wavelength1: Left grating wavelength in pixels.
            wavelength2: Right grating wavelength in pixels.
            boundary: Position of the boundary (0..1).

        Returns:
            (size, size) float32 array.
        """
        g1 = Utils.make_grating((size, size), wavelength=wavelength1, angle_rad=0.0)
        g2 = Utils.make_grating((size, size), wavelength=wavelength2, angle_rad=0.0)
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x < boundary, 1.0, 0.0).reshape(1, -1)
        return g1 * mask + g2 * (1.0 - mask)

    @staticmethod
    def synthesize_signal08(
        size: int = 300,
        left_val: float = 0.8,
        right_val: float = 0.2,
        boundary: float = 0.5,
    ) -> mx.array:
        """Bright→dark smooth — Type E: illumination change only.

        Left half: uniform bright. Right half: uniform dark. All
        spectral shape metrics remain constant (same coarse-scale
        energy). Only absolute pixel intensity changes.

        Args:
            size: Image side length (square).
            left_val: Left half intensity (0..1).
            right_val: Right half intensity (0..1).
            boundary: Position of the boundary (0..1).

        Returns:
            (size, size) float32 array.
        """
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x < boundary, 1.0, 0.0).reshape(1, -1)
        return (
            mx.full((size, size), right_val, dtype=mx.float32)
            + (left_val - right_val) * mask
        )

    @staticmethod
    def synthesize_signal09(
        size: int = 300, wavelength: float = 16.0, boundary: float = 0.5
    ) -> mx.array:
        """Noise on the left, grating on the right — Type C: noise→texture.

        Noise has edge-like spectral properties (energy spread across
        all scales → poor power-law fit). Grating has energy at
        a single matching scale. All metrics jump at the boundary.

        Args:
            size: Image side length (square).
            wavelength: Right grating wavelength in pixels.
            boundary: Position of the boundary (0..1).

        Returns:
            (size, size) float32 array.
        """
        noise = mx.random.normal(shape=(size, size), key=mx.random.key(42))
        noise = Utils.normalize(mx.clip(noise, -2.0, 2.0))
        grating = Utils.make_grating((size, size), wavelength=wavelength, angle_rad=0.0)
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x < boundary, 1.0, 0.0).reshape(1, -1)
        return noise * mask + grating * (1.0 - mask)

    @staticmethod
    def corrcoef(a: mx.array, b: mx.array) -> float:
        """Pearson correlation coefficient between two 1-D arrays."""
        a_c = a - a.mean()
        b_c = b - b.mean()
        cov = (a_c * b_c).mean()
        std_a = mx.sqrt((a_c**2).mean())
        std_b = mx.sqrt((b_c**2).mean())
        return float((cov / mx.maximum(std_a * std_b, 1e-12)).item())

    @staticmethod
    def make_step_edge(shape: tuple[int, int]) -> mx.array:
        """Vertical step edge: left half 0, right half 1."""
        _, W = shape
        arr = mx.zeros(shape, dtype=mx.float32)
        arr[:, W // 2 :] = 1.0
        return arr

    @staticmethod
    def make_smooth_patch(shape: tuple[int, int]) -> mx.array:
        """Uniform mid-gray patch with tiny noise to avoid all-zeros."""
        rng = mx.random.key(7)
        return mx.full(shape, 0.5, dtype=mx.float32) + (
            mx.random.normal(shape=shape, key=rng) * 1e-4
        ).astype(mx.float32)

    @staticmethod
    def make_hue_grating(
        shape: tuple[int, int],
        wavelength: float,
        angle_rad: float = 0.0,
        sat: float = 1.0,
    ) -> mx.array:
        """Periodic hue variation (sinusoidal) at given wavelength and angle."""
        H, W = shape
        Y, X = mx.meshgrid(
            mx.arange(H, dtype=mx.float32),
            mx.arange(W, dtype=mx.float32),
            indexing="ij",
        )
        cx, cy = W / 2, H / 2
        xr = (X - cx) * math.cos(angle_rad) + (Y - cy) * math.sin(angle_rad)
        hue = (math.pi / 2) * mx.sin(2 * math.pi * xr / wavelength)
        hue = hue.astype(mx.float32)
        saturation = mx.full((H, W), sat, dtype=mx.float32)
        return saturation * mx.exp(1j * hue)

    @staticmethod
    def make_luminance_edge(shape: tuple[int, int]) -> mx.array:
        """Luminance-only step edge: left half 0.3, right half 0.7."""
        H, W = shape
        lum = mx.where(mx.arange(W).reshape(1, -1) < W // 2, 0.3, 0.7)
        lum = mx.broadcast_to(lum, (H, W)).astype(mx.float32)
        rgb = mx.stack([lum, lum, lum], axis=-1).astype(mx.float32)
        from color import Color  # lazy — avoids circular import

        hsl = Color.rgb_to_hsl(mx.array(rgb))
        return Color.hsl_to_complex(hsl)

    @staticmethod
    def make_hue_step_edge(
        shape: tuple[int, int],
        hue1_deg: float = 0,
        hue2_deg: float = 180,
        sat: float = 1.0,
    ) -> mx.array:
        """Vertical step edge: left half hue1, right half hue2."""
        H, W = shape
        hue1_rad = math.radians(hue1_deg)
        hue2_rad = math.radians(hue2_deg)
        hue = mx.where(mx.arange(W).reshape(1, -1) < W // 2, hue1_rad, hue2_rad)
        hue = mx.broadcast_to(hue, (H, W)).astype(mx.float32)
        saturation = mx.full((H, W), sat, dtype=mx.float32)
        return saturation * mx.exp(1j * hue)

    @staticmethod
    def make_uniform_hsl(
        shape: tuple[int, int], hue_deg: float = 0, sat: float = 0.5
    ) -> mx.array:
        """Uniform hue–saturation patch with tiny noise to avoid all-zeros."""
        rng = mx.random.key(13)
        hue_rad = math.radians(hue_deg)
        hue = mx.full(shape, hue_rad, dtype=mx.float32)
        sat_arr = mx.full(shape, sat, dtype=mx.float32) + (
            mx.random.normal(shape=shape, key=rng) * 1e-4
        ).astype(mx.float32)
        return sat_arr * mx.exp(1j * hue)
