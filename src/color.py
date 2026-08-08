import mlx.core as mx
import numpy as np
from PIL import Image


class Color:
    """颜色空间转换与图像读取 (MLX)。"""

    @staticmethod
    def image_to_mlx(image: Image.Image) -> mx.array:
        """PIL 图像 → [0,1] 的 mx 数组 (numpy 桥接: mx.array 不直接
        接受 PIL.Image)。"""
        return mx.array(np.asarray(image, dtype=np.float32) / 255.0)

    @staticmethod
    def lab_to_rgb(lab_image: mx.array) -> mx.array:
        """CIELAB (L*a*b*) 图像 → sRGB (MLX)。

        输入 (...,3) 的 L*a*b* 值; 返回 (...,3) 的 RGB, 裁剪到 [0,1]
        (LAB 色域严格大于 sRGB, 超色域值必须裁剪)。
        """
        L = lab_image[..., 0]
        a = lab_image[..., 1]
        b = lab_image[..., 2]

        # 1. LAB → XYZ
        f_y = (L + 16.0) / 116.0
        f_x = (a / 500.0) + f_y
        f_z = f_y - (b / 200.0)

        f_xyz = mx.stack([f_x, f_y, f_z], axis=-1)

        epsilon = 0.008856

        # 立方用连乘写: 避免浮点幂运算对负底数产生 NaN
        f_xyz_cubed = f_xyz * f_xyz * f_xyz

        # LAB→XYZ 逆转移函数
        xyz_normalized = mx.where(
            f_xyz_cubed > epsilon, f_xyz_cubed, (f_xyz - (16.0 / 116.0)) / 7.787
        )

        # D65 标准照明体参考白
        white_point = mx.array([0.95047, 1.00000, 1.08883])
        xyz = xyz_normalized * white_point

        # 2. XYZ → 线性 RGB (sRGB→XYZ 转换矩阵的逆)
        xyz_to_rgb_matrix = mx.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )

        linear_rgb = mx.matmul(xyz, xyz_to_rgb_matrix.T)

        # 3. 线性 RGB → sRGB (伽马校正)
        # where 两分支都会求值, 钳底防幂运算 NaN
        safe_linear_rgb = mx.maximum(linear_rgb, 1e-6)

        mask_srgb = linear_rgb > 0.0031308
        srgb = mx.where(
            mask_srgb,
            1.055 * mx.power(safe_linear_rgb, 1.0 / 2.4) - 0.055,
            12.92 * linear_rgb,
        )

        # 4. 色域裁剪
        rgb_clipped = mx.clip(srgb, 0.0, 1.0)

        return rgb_clipped

    @staticmethod
    def rgb_to_lab(rgb_image: mx.array) -> mx.array:
        """sRGB 图像 → CIELAB (L*a*b*) (MLX)。

        输入 (...,3) 的 RGB (值域 [0,1]); 返回 (...,3) 的 L*a*b*:
        L∈[0,100], a∈[−128,127], b∈[−128,127]。
        """
        # 入口钳负: where 两分支都求值, 微负输入的 2.4 次幂会 NaN
        rgb_image = mx.maximum(rgb_image, 0.0)
        # 1. sRGB → 线性 RGB (逆伽马校正)
        mask_linear = rgb_image > 0.04045
        linear_rgb = mx.where(
            mask_linear, mx.power((rgb_image + 0.055) / 1.055, 2.4), rgb_image / 12.92
        )

        # 2. 线性 RGB → XYZ (D65 参考白)
        xyz_matrix = mx.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )

        # 矩阵乘作用在最后一维
        xyz = mx.matmul(linear_rgb, xyz_matrix.T)

        # 3. XYZ → L*a*b*
        # D65 标准照明体参考白
        white_point = mx.array([0.95047, 1.00000, 1.08883])
        xyz_normalized = xyz / white_point

        epsilon = 0.008856

        # 幂运算防 NaN: mx.where 两分支都会求值, 0/负值会炸 mx.power
        safe_xyz = mx.maximum(xyz_normalized, 1e-6)

        mask_lab = xyz_normalized > epsilon
        f_xyz = mx.where(
            mask_lab,
            mx.power(safe_xyz, 1.0 / 3.0),
            (7.787 * xyz_normalized) + (16.0 / 116.0),
        )

        # 拆通道做最后计算
        f_x = f_xyz[..., 0]
        f_y = f_xyz[..., 1]
        f_z = f_xyz[..., 2]

        # 算 L*, a*, b*
        L = mx.maximum(0.0, 116.0 * f_y - 16.0)
        a = 500.0 * (f_x - f_y)
        b = 200.0 * (f_y - f_z)

        # 堆回通道
        lab = mx.stack([L, a, b], axis=-1)

        return lab

    @staticmethod
    def rgb_to_hsl(rgb_image: mx.array) -> mx.array:
        """sRGB 图像 → HSL (MLX)。

        输入 (...,3) RGB [0,1]; 返回 (...,3): H∈[0,1), S∈[0,1], L∈[0,1]。
        2D 输入按灰度复制三通道处理。
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

        # 亮度
        light = (cmax + cmin) / 2.0

        # 色相 (同 HSV)
        hue = mx.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0)  # type: ignore
        mask_g = (cmax == g) & (delta > 0)  # type: ignore
        mask_b = (cmax == b) & (delta > 0)  # type: ignore

        hue = mx.where(mask_r, 60.0 * (((g - b) / delta) % 6), hue)
        hue = mx.where(mask_g, 60.0 * (((b - r) / delta) + 2.0), hue)
        hue = mx.where(mask_b, 60.0 * (((r - g) / delta) + 4.0), hue)
        hue = (hue % 360.0) / 360.0

        # 饱和度
        sat = mx.where(
            delta > 0, delta / (1.0 - mx.abs(2.0 * light - 1.0)), mx.zeros_like(delta)
        )

        return mx.stack([hue, sat, light], axis=-1)

    @staticmethod
    def hsl_to_rgb(hsl_image: mx.array) -> mx.array:
        """HSL 图像 → sRGB (MLX)。

        输入 (...,3): H∈[0,1), S∈[0,1], L∈[0,1]; 返回 (...,3) RGB,
        裁剪到 [0,1]。2D 输入按灰度复制三通道处理。
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
        """HSL 图像 → 复数表示 (MLX): S·exp(i·H·2π)。

        幅值编码饱和度, 相位编码色相 —— 聚类/GMM 的标准
        色相-饱和度嵌入。输入 (...,3), 返回同前导形状的复数数组。
        """
        hue = hsl_image[..., 0] * 2.0 * mx.pi
        sat = hsl_image[..., 1]

        return sat * mx.exp(1j * hue)

    @staticmethod
    def gray_world_wb(rgb_image: mx.array) -> mx.array:
        """灰度世界白平衡 (prior.md 光学先验: 白平衡/光源颜色恒常性,
        Land & McCann Retinex 的工程形): 假设场景平均反射率为中性灰,
        逐通道增益 = 总平均/通道平均, 整体色偏被自动校正。
        输入 (...,3) [0,1]; 保总亮度, 输出裁剪回 [0,1]。"""
        means = mx.mean(rgb_image, axis=(-3, -2))  # (...,3) 通道均值
        gray = mx.mean(means, axis=-1, keepdims=True)
        gain = gray / mx.maximum(means, 1e-6)
        # 广播: (…,3) 输入的空间维在 −3/−2 → 增益补两维
        return mx.clip(rgb_image * gain[..., None, None, :], 0.0, 1.0)

    @staticmethod
    def log_chromaticity(rgb_image: mx.array, eps: float = 1e-3) -> mx.array:
        """对数色度 (光照不变特征, prior.md "归一化必须在特征层"):
        c1 = log(R/G), c2 = log(B/G) —— 强度缩放 I→λI 在对数域相消,
        阴影/曝光变化不改色度。输入 (...,3), 返回 (...,2)。"""
        safe = mx.maximum(rgb_image, eps)
        g = safe[..., 1:2]
        return mx.concatenate(
            [mx.log(safe[..., 0:1] / g), mx.log(safe[..., 2:3] / g)], axis=-1
        )


if __name__ == "__main__":
    # ── 光学先验包自检 ─────────────────────────────────────────────
    # 场景: 三块表面 (红/绿/蓝灰) 条带
    base = mx.zeros((32, 96, 3))
    base = base.at[:, :32].add(mx.array([0.7, 0.2, 0.2]))
    base = base.at[:, 32:64].add(mx.array([0.2, 0.6, 0.3]))
    base = base.at[:, 64:].add(mx.array([0.5, 0.5, 0.55]))

    # 1. 白平衡: 暖光源 (R×1.3, B×0.75) → 校正后通道均值近相等
    cast = base * mx.array([1.3, 1.0, 0.75])
    wb = Color.gray_world_wb(cast)
    means = mx.mean(wb, axis=(0, 1))
    spread = float(mx.max(means) - mx.min(means))
    assert spread < 0.05, f"校正后通道均值应近等: {means.tolist()}"
    print(f"1. 白平衡: 暖色偏校正, 通道均值散布 {spread:.4f} ✓")
    # 1b. 4D 批量输入广播 (P0 修复: gain 维度曾错位)
    wb4 = Color.gray_world_wb(mx.stack([cast, cast]))
    assert wb4.shape == (2, 32, 96, 3)
    assert float(mx.max(mx.abs(wb4[0] - wb))) < 1e-6, "批量应与单图一致"
    print("1b. 白平衡 4D 批量: 广播正确 ✓")

    # 2. 对数色度: 同表面两强度 → 色度相同 (阴影不变性)
    dark = base * 0.3
    c_bright = Color.log_chromaticity(base)
    c_dark = Color.log_chromaticity(dark)
    diff = float(mx.max(mx.abs(c_bright - c_dark)))
    assert diff < 1e-3, f"强度缩放不应改色度: {diff}"
    # 不同表面色度可区分
    gap = float(mx.abs(c_bright[16, 16] - c_bright[16, 48]).sum())
    assert gap > 0.5, f"红/绿表面色度应可分: {gap}"
    print(f"2. 对数色度: 强度不变 (差 {diff:.2e}), 表面可分 ({gap:.2f}) ✓")
