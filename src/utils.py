import math
from pathlib import Path

import mlx.core as mx

# def label_coherence(labels, Ny, Nx):
#     """4-neighbour agreement rate (0..2) of a label map."""
#     lab = labels.reshape(Ny, Nx)
#     dh = (lab[:, 1:] == lab[:, :-1]).astype(mx.float32).mean()
#     dv = (lab[1:, :] == lab[:-1, :]).astype(mx.float32).mean()
#     return float(dh + dv)


# def remap_accuracy(labels_pred, gt):
#     """Best-match accuracy over all K! label permutations."""
#     lab = mx.array(labels_pred, dtype=mx.int32)
#     g = mx.array(gt, dtype=mx.int32)
#     best = 0.0
#     best_perm = None
#     for perm in permutations(range(int(mx.max(lab).item()) + 1)):
#         lut = mx.array(perm, dtype=mx.int32)
#         acc = float(mx.equal(lut[lab], g).astype(mx.float32).mean().item())
#         if acc > best:
#             best, best_perm = acc, perm
#     return best, best_perm


class Utils:
    @staticmethod
    def fftfreq(n: int) -> mx.array:
        """MLX version of fftfreq(n), normalized to cycles/sample."""
        k = mx.arange(n, dtype=mx.float32)
        half = (n + 1) // 2
        k = mx.where(k < half, k, k - n)
        return 2 * k / n

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
    def remove_dc(spectrum: mx.array) -> mx.array:
        """Return a copy of the frequency-domain array with the DC component zeroed.

        The original array is not modified (MLX arrays are immutable).
        """
        result = mx.array(spectrum)
        result[0, 0] = 0
        return result

    @staticmethod
    def normalize(arr: mx.array) -> mx.array:
        arr_min = mx.min(arr)
        arr_max = mx.max(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    @staticmethod
    def invert(mlx_arr: mx.array) -> mx.array:
        return 1.0 - mlx_arr

    @staticmethod
    def out_dir(folder: str) -> Path:
        d = Path(__file__).resolve().parent.parent / folder
        d.mkdir(parents=True, exist_ok=True)
        return d

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
        """Sinusoidal grating at given wavelength (px) and angle (rad)."""
        H, W = shape
        y = mx.arange(H, dtype=mx.float32)
        x = mx.arange(W, dtype=mx.float32)
        yy, xx = mx.meshgrid(y, x, indexing="ij")
        xr = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
        return mx.sin(2 * math.pi * xr / wavelength + phase).astype(mx.float32)

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
        all six spectral metrics jump simultaneously as energy shifts
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
        wavelength2. Only centroid and rolloff jump; variance and
        entropy remain similar (both are single-scale).

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

        Left half: uniform bright. Right half: uniform dark. All six
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
        all scales → high variance and entropy). Grating has energy at
        a single matching scale. All six metrics jump at the boundary.

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
    def make_pattern_complex(
        shape: tuple[int, int], l_case: str, hs_case: str
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build an HSL image and return (hs_complex, rgb, L).

        Args:
            shape: (H, W) of the output.
            l_case: Lightness case — ``"edge"``, ``"texture"``, or ``"smooth"``.
            hs_case: Hue–saturation case — ``"edge"``, ``"texture"``, or ``"smooth"``.

        Returns:
            (hs_complex, rgb, L) — all mlx arrays.
        """
        H, W = shape

        # ── lightness map ──
        if l_case == "edge":
            l = mx.where(mx.arange(W).reshape(1, -1) < W // 2, 0.3, 0.8)
            l = mx.broadcast_to(l, (H, W)).astype(mx.float32)
        elif l_case == "texture":
            lam = 16.0
            x = mx.arange(W, dtype=mx.float32).reshape(1, -1)
            l = 0.55 + 0.25 * mx.sin(2 * math.pi * (x - W / 2) / lam)
            l = mx.broadcast_to(l, (H, W)).astype(mx.float32)
        elif l_case == "smooth":
            rng = mx.random.key(17)
            l = (
                mx.full((H, W), 0.6, dtype=mx.float32)
                + mx.random.normal(shape=(H, W), key=rng).astype(mx.float32) * 1e-4
            )
        else:
            raise ValueError(f"unknown l_case: {l_case}")

        # ── hue map ──
        if hs_case == "edge":
            hue_rad = mx.where(mx.arange(W).reshape(1, -1) < W // 2, 0.0, math.pi)
            hue_rad = mx.broadcast_to(hue_rad, (H, W)).astype(mx.float32)
            hue_norm = hue_rad / (2 * math.pi)
            sat = mx.full((H, W), 1.0, dtype=mx.float32)
        elif hs_case == "texture":
            lam = 16.0
            x = mx.arange(W, dtype=mx.float32).reshape(1, -1)
            hue_norm = 0.5 + 0.25 * mx.sin(2 * math.pi * (x - W / 2) / lam)
            hue_norm = mx.broadcast_to(hue_norm, (H, W)).astype(mx.float32)
            sat = mx.full((H, W), 1.0, dtype=mx.float32)
        elif hs_case == "smooth":
            hue_rad = mx.full((H, W), math.radians(60), dtype=mx.float32)
            hue_norm = hue_rad / (2 * math.pi)
            rng = mx.random.key(31)
            sat = (
                mx.full((H, W), 0.5, dtype=mx.float32)
                + mx.random.normal(shape=(H, W), key=rng).astype(mx.float32) * 1e-4
            )
        else:
            raise ValueError(f"unknown hs_case: {hs_case}")

        hsl = mx.stack([hue_norm, sat, l], axis=-1).astype(mx.float32)
        from color import Color  # lazy — avoids circular import

        rgb = Color.hsl_to_rgb(mx.array(hsl))
        hsl_round = Color.rgb_to_hsl(rgb)
        return Color.hsl_to_complex(hsl_round), rgb, l

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
        l = mx.where(mx.arange(W).reshape(1, -1) < W // 2, 0.3, 0.7)
        l = mx.broadcast_to(l, (H, W)).astype(mx.float32)
        rgb = mx.stack([l, l, l], axis=-1).astype(mx.float32)
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
