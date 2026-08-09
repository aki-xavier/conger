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
    def nonzero(sel: mx.array) -> mx.array:
        """布尔掩码 → 扁平索引 (MLX 无布尔索引, argsort 技巧:
        选中位给原下标, 未选中给 N, 升序排序后前 k 个即索引)。"""
        flat = sel.reshape(-1)
        k = int(mx.sum(flat))
        key = mx.where(flat, mx.arange(flat.shape[0]), flat.shape[0])
        return mx.argsort(key)[:k]

    @staticmethod
    def grid(shape: tuple[int, ...]) -> tuple[mx.array, mx.array]:
        """(row, col) 坐标网格, 尾部维度广播。"""
        yy, xx = mx.meshgrid(
            mx.arange(shape[0], dtype=mx.float32),
            mx.arange(shape[1], dtype=mx.float32),
            indexing="ij",
        )
        while yy.ndim < len(shape):
            yy = yy[..., None]
            xx = xx[..., None]
        return yy, xx

    @staticmethod
    def logdet_spd(a: mx.array) -> mx.array:
        """ln|A|, 对称正定 (eigh 路径; 支持 (...,d,d) 批量 → (...))。
        特征值钳底 1e-12, 返回非精确值。"""
        ev = mx.linalg.eigh(a, stream=mx.cpu)[0]
        return mx.sum(mx.log(mx.maximum(ev, 1e-12)), axis=-1)

    @staticmethod
    def bhatt(
        m1: mx.array, c1: mx.array, m2: mx.array, c2: mx.array
    ) -> mx.array:
        """K 组 (μ,Σ) 对单候选的 Bhattacharyya 距离 (批量):
        d_B = ⅛ΔμᵀΣ̄⁻¹Δμ + ½ln(detΣ̄/√(detΣᵢ·detΣⱼ))。
        m1 (K,P), c1 (K,P,P), m2 (P,), c2 (P,P) → (K,)。"""
        dim = c1.shape[-1]
        if not (
            mx.all(mx.isfinite(c1)).item() and mx.all(mx.isfinite(c2)).item()
        ):
            # 退化输入 (NaN/inf 协方差) → 无限距离: eigh/inv 遇 NaN 抛
            # C++ 异常 (Abort trap), 此处兜底所有调用方 (scenegraph
            # _match / vbgmm 合并)。输入守卫在 scenegraph.accumulate。
            return mx.full((m1.shape[0],), float("inf"))
        cb = (c1 + c2) * 0.5
        inv = mx.linalg.inv(cb + 1e-9 * mx.eye(dim), stream=mx.cpu)
        dm = (m1 - m2)[:, :, None]  # (K,P,1)
        t1 = (dm.transpose(0, 2, 1) @ inv @ dm)[:, 0, 0] / 8.0
        t2 = 0.5 * (
            Utils.logdet_spd(cb)
            - 0.5 * (Utils.logdet_spd(c1) + Utils.logdet_spd(c2))
        )
        return t1 + t2

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
    def visualize(plots: list[tuple[str, str | None, mx.array]]):
        """多面板拼图: (标题, colormap, 数据) 列表 → matplotlib fig。
        cmap=None 表示 RGB 图 (不配 colorbar)。"""
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
                if cmap is not None:  # RGB 图 (cmap=None) 不配 colorbar
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
    def synthesize_signal04(size: int = 300) -> mx.array:
        """高斯白噪声 —— 宽带纹理 (裁到 2σ 后归一化到 [0,1])。"""
        noise = mx.random.normal(shape=(size, size), key=mx.random.key(0))
        return Utils.normalize(mx.clip(noise, -2.0, 2.0))






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






