import math
from pathlib import Path

import matplotlib
import mlx.core as mx

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Utils:
    """图像合成 / 频率网格 / 可视化等杂项工具。"""

    @staticmethod
    def project_root() -> Path:
        """项目根目录 (src/ 的父目录)。"""
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def fftfreq(n: int) -> mx.array:
        """np.fft.fftfreq(n) 的 MLX 版, 单位 cycles/sample (Nyquist = 0.5)。"""
        k = mx.arange(n, dtype=mx.float32)
        half = (n + 1) // 2
        k = mx.where(k < half, k, k - n)
        return k / n

    @staticmethod
    def freqgrid(shape: tuple[int, ...]) -> list[mx.array]:
        """给定 (高, 宽) 生成归一化频率网格。"""
        height, width = shape
        x = Utils.fftfreq(width)
        y = Utils.fftfreq(height)
        return mx.meshgrid(x, y)

    @staticmethod
    def standard_normal_pdf(
        amp: float, sigma: float, x: mx.array, y: mx.array
    ) -> mx.array:
        """二维各向同性高斯: amp·exp(−(x²+y²)/(2σ²))。"""
        amp = abs(amp)
        sigma = abs(sigma)
        exponent = -0.5 * (x**2 + y**2) / (sigma**2)
        ret = amp * mx.exp(exponent)
        return mx.array(ret)

    @staticmethod
    def grid_shape(n: int) -> tuple[int, int]:
        """n 个面板的拼图网格 (rows, cols), 最接近正方形。"""
        if n <= 0:
            return (0, 0)
        # 在 rows·cols >= n 的约束下取最接近正方形的网格 (rows <= cols,
        # 横向布局)。不能强求整除: n=22 = 2×11 这类半质数, 最近正方形的
        # 整除分解是 2×11 —— 一条极宽条带; 放宽到 5×5 (空 3 格) 才正常。
        best = (1, n)
        rows = 1
        while rows <= (cols := math.ceil(n / rows)):
            if cols - rows < best[1] - best[0]:
                best = (rows, cols)
            rows += 1
        return best

    @staticmethod
    def visualize(plots: list[tuple[str, str, mx.array]]):
        """多面板拼图: (标题, colormap, 数据) 列表 → matplotlib fig。"""
        rows, cols = Utils.grid_shape(len(plots))
        fig, axes = plt.subplots(
            rows, cols, squeeze=False, figsize=(cols * 2.2, rows * 2.2)
        )
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                ax = axes[row][col]
                ax.set_xticks([])
                ax.set_yticks([])
                if idx >= len(plots):  # 网格槽位多于面板数
                    ax.axis("off")
                    continue
                title, cmap, data = plots[idx]
                im = ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return fig

    @staticmethod
    def normalize(arr: mx.array) -> mx.array:
        """线性归一化到 [0,1] (常数数组 → 全零, 防除零)。"""
        arr_min = mx.min(arr)
        arr_max = mx.max(arr)
        # 常数数组 → 全零 (max−min=0 时避免除零 NaN)
        return (arr - arr_min) / mx.maximum(arr_max - arr_min, 1e-12)

    @staticmethod
    def invert(mlx_arr: mx.array) -> mx.array:
        """反相: 1 − x。"""
        return 1.0 - mlx_arr

    @staticmethod
    def synthesize_signal01(size: int = 300) -> mx.array:
        """竖直阶跃边缘: 左半 0, 右半 1。"""
        left = mx.zeros((size, size // 2), dtype=mx.float32)
        right = mx.ones((size, size - size // 2), dtype=mx.float32)
        return mx.concatenate([left, right], axis=1)

    @staticmethod
    def synthesize_signal02(size: int = 300) -> mx.array:
        """黑 → 线性渐变 → 白 的灰度过渡带。"""
        black = mx.zeros((size, int(size * 0.45)), dtype=mx.float32)
        x_end = int(size * 0.55)
        ramp = mx.linspace(0.0, 1.0, x_end - int(size * 0.45))
        ramp_2d = mx.repeat(ramp.reshape(1, -1), size, axis=0)
        white = mx.ones((size, size - x_end), dtype=mx.float32)
        return mx.concatenate([black, ramp_2d, white], axis=1)

    @staticmethod
    def make_grating(shape, wavelength, angle_rad, phase=0.0) -> mx.array:  # type: ignore
        """给定波长 (px) 与角度 (rad) 的正弦光栅, 值域 [0,1]。"""
        H, W = shape
        y = mx.arange(H, dtype=mx.float32)
        x = mx.arange(W, dtype=mx.float32)
        yy, xx = mx.meshgrid(y, x, indexing="ij")
        xr = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
        s = mx.sin(2 * math.pi * xr / wavelength + phase).astype(mx.float32)
        return (s + 1.0) * 0.5

    @staticmethod
    def make_slanted_grid(
        shape: tuple[int, int], lam0: float, slant: float, tilt: float
    ) -> mx.array:
        """斜面方格纹理 (正交投影): 表面波长 lam0 的方格, slant=倾斜角
        (rad, 0=正面), tilt=图像内压缩方向 (rad)。沿 tilt 波长压缩为
        lam0·cos(slant), 垂直方向不变 —— shape-from-texture 的真值图。"""
        H, W = shape
        yy, xx = mx.meshgrid(
            mx.arange(H, dtype=mx.float32),
            mx.arange(W, dtype=mx.float32),
            indexing="ij",
        )
        xi = xx * math.cos(tilt) + yy * math.sin(tilt)
        eta = -xx * math.sin(tilt) + yy * math.cos(tilt)
        lam_par = lam0 * math.cos(slant)
        g = mx.sin(2 * math.pi * xi / lam_par) + mx.sin(2 * math.pi * eta / lam0)
        return ((g + 2.0) * 0.25).astype(mx.float32)

    @staticmethod
    def make_texture_composite(size: int = 128):
        """四象限合成图 (Gabor 聚类测试用): 左上 λ=8,0°; 右上
        λ=24,45°; 左下 均匀噪声; 右下 平坦 0.5。"""
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
        """正弦光栅 —— 纯细纹理, 无边缘 (全图 8px 波长竖直光栅)。"""
        return Utils.make_grating((size, size), wavelength=8.0, angle_rad=0.0)

    @staticmethod
    def synthesize_signal04(size: int = 300) -> mx.array:
        """高斯白噪声 —— 宽带纹理 (裁到 2σ 后归一化到 [0,1])。"""
        noise = mx.random.normal(shape=(size, size), key=mx.random.key(0))
        return Utils.normalize(mx.clip(noise, -2.0, 2.0))

    @staticmethod
    def synthesize_signal05(
        size: int = 300, wavelength: float = 16.0, boundary: float = 0.5
    ) -> mx.array:
        """左光栅右平坦 —— 边界处方向选择性连续性跌落。

        左半竖直光栅 (θ=0° 纹理), 右半均匀灰。边界处 θ=0° 方向连续
        性急降 (纹理→平坦); θ=90° 两侧都无纹理 → 无跌落。

        boundary 为边界的横向相对位置 (0..1)。返回 (size,size) float32。
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
        """左平坦右光栅 —— Type B: 平坦→纹理。

        边界处能量从单一粗尺度跳到特定匹配尺度, 所有谱指标同时跳变。
        boundary 为边界相对位置 (0..1)。返回 (size,size) float32。
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
        """光栅频率突变 —— Type D: 纹理→纹理 (不同尺度)。

        左半 wavelength1, 右半 wavelength2。slope 与 bump 跳变;
        拟合残差相近 (两半都是单尺度)。返回 (size,size) float32。
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
        """亮→暗平坦 —— Type E: 仅照明变化。

        谱形状指标不变 (同样的粗尺度能量), 只有像素绝对强度变。
        返回 (size,size) float32。
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
        """左噪声右光栅 —— Type C: 噪声→纹理。

        噪声的谱类似边缘 (能量散在全尺度 → 幂律拟合差); 光栅能量
        集中在单一匹配尺度。边界处所有指标跳变。返回 (size,size)
        float32。
        """
        noise = mx.random.normal(shape=(size, size), key=mx.random.key(42))
        noise = Utils.normalize(mx.clip(noise, -2.0, 2.0))
        grating = Utils.make_grating((size, size), wavelength=wavelength, angle_rad=0.0)
        x = mx.arange(size, dtype=mx.float32) / size
        mask = mx.where(x < boundary, 1.0, 0.0).reshape(1, -1)
        return noise * mask + grating * (1.0 - mask)

    @staticmethod
    def corrcoef(a: mx.array, b: mx.array) -> float:
        """两个一维数组的 Pearson 相关系数。"""
        a_c = a - a.mean()
        b_c = b - b.mean()
        cov = (a_c * b_c).mean()
        std_a = mx.sqrt((a_c**2).mean())
        std_b = mx.sqrt((b_c**2).mean())
        return float((cov / mx.maximum(std_a * std_b, 1e-12)).item())

    @staticmethod
    def make_step_edge(shape: tuple[int, int]) -> mx.array:
        """竖直阶跃边缘: 左半 0, 右半 1。"""
        _, W = shape
        arr = mx.zeros(shape, dtype=mx.float32)
        arr[:, W // 2 :] = 1.0
        return arr

    @staticmethod
    def make_corner(shape: tuple[int, int]) -> mx.array:
        """L 形角点: 右上象限亮 — 竖直边 (x=W/2, y<H/2) 与水平边
        (y=H/2, x>W/2) 交于 (W/2, H/2)。"""
        H, W = shape
        arr = mx.zeros(shape, dtype=mx.float32)
        arr[: H // 2, W // 2 :] = 1.0
        return arr

    @staticmethod
    def make_smooth_patch(shape: tuple[int, int]) -> mx.array:
        """中灰平坦块 + 微噪声 (防全零)。"""
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
        """给定波长与角度的正弦色相周期变化 (复数表示 S·e^{iH})。"""
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
        """仅亮度的阶跃边缘: 左半 0.3, 右半 0.7 (复数表示)。"""
        H, W = shape
        lum = mx.where(mx.arange(W).reshape(1, -1) < W // 2, 0.3, 0.7)
        lum = mx.broadcast_to(lum, (H, W)).astype(mx.float32)
        rgb = mx.stack([lum, lum, lum], axis=-1).astype(mx.float32)
        from color import Color  # 惰性导入 —— 防循环依赖

        hsl = Color.rgb_to_hsl(mx.array(rgb))
        return Color.hsl_to_complex(hsl)

    @staticmethod
    def make_hue_step_edge(
        shape: tuple[int, int],
        hue1_deg: float = 0,
        hue2_deg: float = 180,
        sat: float = 1.0,
    ) -> mx.array:
        """竖直色相阶跃: 左半 hue1, 右半 hue2 (度), 复数表示。"""
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
        """均匀色相-饱和度块 (度) + 微噪声 (防全零), 复数表示。"""
        rng = mx.random.key(13)
        hue_rad = math.radians(hue_deg)
        hue = mx.full(shape, hue_rad, dtype=mx.float32)
        sat_arr = mx.full(shape, sat, dtype=mx.float32) + (
            mx.random.normal(shape=shape, key=rng) * 1e-4
        ).astype(mx.float32)
        return sat_arr * mx.exp(1j * hue)
