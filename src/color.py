import mlx.core as mx
import numpy as np
from PIL import Image


class Color:
    @staticmethod
    def image_to_mlx(image: Image.Image) -> mx.array:
        # Normalize to [0, 1]; numpy bridge required because mx.array
        # does not accept PIL.Image directly.
        return mx.array(np.asarray(image, dtype=np.float32) / 255.0)

    @staticmethod
    def lab_to_rgb(lab_image: mx.array) -> mx.array:
        """
        Converts a CIELAB (L*a*b*) image to sRGB color space using Apple MLX.

        Args:
            lab_image: An mx.array of shape (..., 3) containing L*a*b* values.

        Returns:
            An mx.array of shape (..., 3) with RGB values clipped to the range [0, 1].
        """
        L = lab_image[..., 0]
        a = lab_image[..., 1]
        b = lab_image[..., 2]

        # 1. LAB to XYZ
        f_y = (L + 16.0) / 116.0
        f_x = (a / 500.0) + f_y
        f_z = f_y - (b / 200.0)

        f_xyz = mx.stack([f_x, f_y, f_z], axis=-1)

        epsilon = 0.008856

        # Calculate the cube. We use multiplication to avoid potential NaN
        # issues with negative bases in floating-point power operations.
        f_xyz_cubed = f_xyz * f_xyz * f_xyz

        # Inverse transfer function for LAB to XYZ
        xyz_normalized = mx.where(
            f_xyz_cubed > epsilon, f_xyz_cubed, (f_xyz - (16.0 / 116.0)) / 7.787
        )

        # D65 standard illuminant reference values
        white_point = mx.array([0.95047, 1.00000, 1.08883])
        xyz = xyz_normalized * white_point

        # 2. XYZ to Linear RGB
        # Inverse of the sRGB to XYZ conversion matrix
        xyz_to_rgb_matrix = mx.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )

        linear_rgb = mx.matmul(xyz, xyz_to_rgb_matrix.T)

        # 3. Linear RGB to sRGB (Gamma Correction)
        # Prevent NaNs during the power operation in the where-clause
        safe_linear_rgb = mx.maximum(linear_rgb, 1e-6)

        mask_srgb = linear_rgb > 0.0031308
        srgb = mx.where(
            mask_srgb,
            1.055 * mx.power(safe_linear_rgb, 1.0 / 2.4) - 0.055,
            12.92 * linear_rgb,
        )

        # 4. Gamut Clipping
        # LAB space is strictly larger than sRGB. If the LAB values correspond
        # to a color outside the sRGB gamut, we must clip it to [0, 1].
        rgb_clipped = mx.clip(srgb, 0.0, 1.0)

        return rgb_clipped

    @staticmethod
    def rgb_to_lab(rgb_image: mx.array) -> mx.array:
        """
        Converts an sRGB image to CIELAB (L*a*b*) color space using Apple MLX.

        Args:
            rgb_image: An mx.array of shape (..., 3) with RGB values in the
                range [0, 1].

        Returns:
            An mx.array of shape (..., 3) containing the L*a*b* values.
            l: [0, 100], a: [-128, 127], b: [-128, 127]
        """
        # 1. sRGB to Linear RGB (Inverse Gamma Correction)
        mask_linear = rgb_image > 0.04045
        linear_rgb = mx.where(
            mask_linear, mx.power((rgb_image + 0.055) / 1.055, 2.4), rgb_image / 12.92
        )

        # 2. Linear RGB to XYZ (Using D65 Reference White)
        xyz_matrix = mx.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )

        # Matrix multiplication applied to the last dimension
        xyz = mx.matmul(linear_rgb, xyz_matrix.T)

        # 3. XYZ to L*a*b*
        # D65 standard illuminant reference values
        white_point = mx.array([0.95047, 1.00000, 1.08883])
        xyz_normalized = xyz / white_point

        epsilon = 0.008856

        # Prevent NaNs during the power operation. Since MLX may evaluate
        # both branches of mx.where, a 0 or negative value could break mx.power.
        safe_xyz = mx.maximum(xyz_normalized, 1e-6)

        mask_lab = xyz_normalized > epsilon
        f_xyz = mx.where(
            mask_lab,
            mx.power(safe_xyz, 1.0 / 3.0),
            (7.787 * xyz_normalized) + (16.0 / 116.0),
        )

        # Split the channels for the final calculations
        f_x = f_xyz[..., 0]
        f_y = f_xyz[..., 1]
        f_z = f_xyz[..., 2]

        # Calculate L*, a*, b*
        L = mx.maximum(0.0, 116.0 * f_y - 16.0)
        a = 500.0 * (f_x - f_y)
        b = 200.0 * (f_y - f_z)

        # Stack channels back together
        lab = mx.stack([L, a, b], axis=-1)

        return lab

    @staticmethod
    def rgb_to_hsl(rgb_image: mx.array) -> mx.array:
        """
        Converts an sRGB image to HSL color space using Apple MLX.

        Args:
            rgb_image: An mx.array of shape (..., 3) with RGB values in the
                range [0, 1].

        Returns:
            An mx.array of shape (..., 3) containing H in [0, 1), S in [0, 1],
            L in [0, 1].
        """
        if rgb_image.ndim not in (2, 3):
            raise ValueError(f"rgb_image must be 2-D or 3-D, got ndim={rgb_image.ndim}")
        if rgb_image.ndim == 2:
            rgb_image = mx.stack([rgb_image] * 3, axis=-1)

        r = rgb_image[..., 0]
        g = rgb_image[..., 1]
        b = rgb_image[..., 2]

        cmax = mx.maximum(mx.maximum(r, g), b)
        cmin = mx.minimum(mx.minimum(r, g), b)
        delta = cmax - cmin

        # Lightness
        light = (cmax + cmin) / 2.0

        # Hue (same as HSV)
        hue = mx.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0)  # type: ignore
        mask_g = (cmax == g) & (delta > 0)  # type: ignore
        mask_b = (cmax == b) & (delta > 0)  # type: ignore

        hue = mx.where(mask_r, 60.0 * (((g - b) / delta) % 6), hue)
        hue = mx.where(mask_g, 60.0 * (((b - r) / delta) + 2.0), hue)
        hue = mx.where(mask_b, 60.0 * (((r - g) / delta) + 4.0), hue)
        hue = (hue % 360.0) / 360.0

        # Saturation
        sat = mx.where(
            delta > 0, delta / (1.0 - mx.abs(2.0 * light - 1.0)), mx.zeros_like(delta)
        )

        return mx.stack([hue, sat, light], axis=-1)

    @staticmethod
    def hsl_to_rgb(hsl_image: mx.array) -> mx.array:
        """
        Converts an HSL image to sRGB color space using Apple MLX.

        Args:
            hsl_image: An mx.array of shape (..., 3) with H in [0, 1),
                S in [0, 1], L in [0, 1].

        Returns:
            An mx.array of shape (..., 3) with RGB values clipped to the range [0, 1].
        """
        if hsl_image.ndim not in (2, 3):
            raise ValueError(f"hsl_image must be 2-D or 3-D, got ndim={hsl_image.ndim}")
        if hsl_image.ndim == 2:
            hsl_image = mx.stack([hsl_image] * 3, axis=-1)

        hue = hsl_image[..., 0] * 360.0
        sat = hsl_image[..., 1]
        light = hsl_image[..., 2]

        c = (1.0 - mx.abs(2.0 * light - 1.0)) * sat
        x = c * (1.0 - mx.abs((hue / 60.0) % 2.0 - 1.0))
        m = light - c / 2.0

        sector = (hue / 60.0).astype(mx.int32) % 6

        r1 = mx.zeros_like(hue)
        g1 = mx.zeros_like(hue)
        b1 = mx.zeros_like(hue)

        r1 = mx.where(sector == 0, c, r1)
        g1 = mx.where(sector == 0, x, g1)

        r1 = mx.where(sector == 1, x, r1)
        g1 = mx.where(sector == 1, c, g1)

        g1 = mx.where(sector == 2, c, g1)
        b1 = mx.where(sector == 2, x, b1)

        g1 = mx.where(sector == 3, x, g1)
        b1 = mx.where(sector == 3, c, b1)

        r1 = mx.where(sector == 4, x, r1)
        b1 = mx.where(sector == 4, c, b1)

        r1 = mx.where(sector == 5, c, r1)
        b1 = mx.where(sector == 5, x, b1)

        rgb = mx.stack([r1 + m, g1 + m, b1 + m], axis=-1)
        return mx.clip(rgb, 0.0, 1.0)

    @staticmethod
    def hsl_to_complex(hsl_image: mx.array) -> mx.array:
        """
        Converts an HSL image to a complex-valued representation using Apple MLX.

        Args:
            hsl_image: An mx.array of shape (..., 3) with H in [0, 1),
                S in [0, 1], L in [0, 1].

        Returns:
            A complex-valued mx.array of the same leading shape, computed as
            S · exp(i · H · 2π). The magnitude encodes saturation and the
            phase encodes hue — this is the standard hue–saturation embedding
            for clustering / GMM use.
        """
        hue = hsl_image[..., 0] * 2.0 * mx.pi
        sat = hsl_image[..., 1]

        return sat * mx.exp(1j * hue)
