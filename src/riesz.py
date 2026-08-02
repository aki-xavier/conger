import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx

from color import Color
from utils import Utils


@dataclass(slots=True)
class RieszScale:
    """单尺度单演小波响应: ψ 是各向同性带通, R₁ψ/R₂ψ 是它的
    Riesz 变换 (频域乘子 −j·ω/|ω|, 即 2D Hilbert 变换)。
    b0 偶对称、b1/b2 分别沿 x/y 奇对称, 三者构成正交三元组。"""

    b0: mx.array  # 带通响应 (偶)
    b1: mx.array  # Riesz-x 响应 (沿 x 奇)
    b2: mx.array  # Riesz-y 响应 (沿 y 奇)
    pad: int = 0
    amp: mx.array | None = None  # A = sqrt(b0²+b1²+b2²): 局部幅值
    phase: mx.array | None = None  # φ = atan2(|R|, b0): 局部相位 ∈ [0, π]
    ori: mx.array | None = None  # atan2(b2, b1): 结构法向 ∈ (−π, π]
    energy: mx.array | None = None  # A²

    def __post_init__(self):
        r2 = self.b1**2 + self.b2**2
        self.energy = self.b0**2 + r2
        self.amp = mx.sqrt(self.energy)
        self.phase = mx.arctan2(mx.sqrt(r2), self.b0)
        self.ori = mx.arctan2(self.b2, self.b1)

    def steer(self, theta: float) -> mx.array:
        """沿 θ 方向的一阶 Riesz 转向: cosθ·b1 + sinθ·b2。
        任意方向的奇对称滤波无需新卷积 —— 与 Gabor 多方向通道互为对偶:
        Gabor 用 N 个方向核逼近角度, Riesz 用 2 个基精确合成任意角度。"""
        return self.b1 * math.cos(theta) + self.b2 * math.sin(theta)


@dataclass(slots=True)
class RieszWavelet:
    img: mx.array
    lam_min: float = 3.0  # min wavelength
    height: int = 0
    width: int = 0
    scale_size: int = 0
    bandwidth: float = 1.0  # radial bandpass octave bandwidth
    adaptive_pad: bool = True
    pad: int = 0
    xgrid: mx.array | None = None
    ygrid: mx.array | None = None
    dc: mx.array | None = None
    fft: mx.array | None = None
    lams: list[float] = field(default_factory=list)  # wavelength
    kernels: list[mx.array] = field(default_factory=list)  # radial bandpass
    scales: list[RieszScale] = field(default_factory=list)

    def __post_init__(self):
        if self.img.ndim != 2:
            raise ValueError(f"img must be 2D, got shape {self.img.shape}")
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")

        self.height, self.width = self.img.shape

        if self.scale_size <= 0:
            lam_max = self.lam_max()
            s = round(math.log2(lam_max / self.lam_min)) + 1
            self.scale_size = max(4, s)

        self.calc_lams()
        self.calc_freqs()
        self.calc_kernels()
        self.calc_scales()

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.height, self.width) / 2.0

    def calc_lams(self):
        # lams 从长波长到短波长排列 (低频→高频), 与 kernels/scales 顺序一致
        lam_min = self.lam_min
        lam_max = self.lam_max()
        if self.scale_size == 1:
            self.lams.append(lam_max)
        else:
            for i in range(self.scale_size):
                lam = lam_max * 2.0 ** (
                    -i * math.log2(lam_max / lam_min) / (self.scale_size - 1)
                )
                self.lams.append(lam)

    def calc_freqs(self):
        # ── self-adaptive padding to avoid FFT wraparound ────────────
        h, w = self.height, self.width
        if self.adaptive_pad:
            self.pad = int(self.lam_max())
            h = self.height + 2 * self.pad
            w = self.width + 2 * self.pad
            padded = mx.pad(
                self.img,
                [(self.pad, self.pad), (self.pad, self.pad)],
                mode="edge",
            )
            self.fft = mx.fft.fft2(padded)
        else:
            self.fft = mx.fft.fft2(self.img)

        self.xgrid, self.ygrid = Utils.freqgrid((h, w))
        sigma_f = 0.5 / self.lam_max()

        dist = self.xgrid**2 + self.ygrid**2
        dc_kernel = mx.exp(-0.5 * dist / sigma_f**2)
        self.dc = self.fft * dc_kernel
        self.fft = self.fft - self.dc

    def calc_kernels(self):
        # 各向同性径向高斯带通, 与 gabor.py 同一核族; Riesz 框架下
        # 角度分解不再用方向核, 而用 Riesz 乘子 (见 calc_scales)。
        radius = mx.sqrt(self.xgrid**2 + self.ygrid**2)
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )
        for lam in self.lams:
            f0 = 1.0 / lam
            sigma_f = sigma_f_rel * f0
            kernel = mx.exp(-0.5 * (radius - f0) ** 2 / sigma_f**2)
            self.kernels.append(kernel)

    def calc_scales(self):
        # Riesz 乘子: R(ω) = −j·ω/|ω|。DC 处 0/0, 但带通核在
        # ω=0 处本已为零, 用 safe 半径防 NaN 即可。
        radius = mx.sqrt(self.xgrid**2 + self.ygrid**2)
        safe_r = mx.maximum(radius, 1e-12)
        m1 = (-1j) * self.xgrid / safe_r  # type: ignore # −j·ωx/|ω|
        m2 = (-1j) * self.ygrid / safe_r  # type: ignore # −j·ωy/|ω|

        for kernel in self.kernels:
            spec = self.fft * kernel
            # b0 是实函数 ↔ 频谱 Hermitian; Riesz 乘子保持 Hermitian,
            # b1/b2 也是实函数, 虚部只剩数值噪声, 取 real。
            b0 = mx.real(mx.fft.ifft2(spec))
            b1 = mx.real(mx.fft.ifft2(spec * m1))
            b2 = mx.real(mx.fft.ifft2(spec * m2))
            if self.pad > 0:
                b0 = b0[self.pad : -self.pad, self.pad : -self.pad]
                b1 = b1[self.pad : -self.pad, self.pad : -self.pad]
                b2 = b2[self.pad : -self.pad, self.pad : -self.pad]
            self.scales.append(RieszScale(b0=b0, b1=b1, b2=b2, pad=self.pad))

    def ifft2(self, arr: mx.array):
        ret = mx.real(mx.fft.ifft2(arr))
        if self.adaptive_pad:
            ret = ret[
                self.pad : self.pad + self.height,
                self.pad : self.pad + self.width,
            ]

        return ret

    def visualize(self, out_path: str | Path):
        plots = [("original", "gray", self.img), ("dc", "gray", self.ifft2(self.dc))]
        for idx, scale in enumerate(self.scales):
            lam = self.lams[idx]
            plots.append((f"s{idx} λ={lam:.1f} amp", "gray", scale.amp))
            plots.append((f"s{idx} λ={lam:.1f} phase", "twilight", scale.phase))
            plots.append((f"s{idx} λ={lam:.1f} ori", "hsv", scale.ori))

        fig = Utils.visualize(plots)
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    from PIL import Image

    # ── synthetic ground truth checks ────────────────────────────────
    # grating: 单频平面波, 法向=angle, 匹配尺度上 phase 线性爬坡、
    # amp 常数、ori 常数。
    angle = math.radians(30.0)
    grating = Utils.make_grating((256, 256), wavelength=16.0, angle_rad=angle)
    rw = RieszWavelet(grating)
    best = max(range(len(rw.scales)), key=lambda i: float(rw.scales[i].energy.mean()))
    sc = rw.scales[best]
    print(f"grating λ=16 @30°: 匹配尺度 s{best} (λ={rw.lams[best]:.1f})")
    print(f"  amp mean/std = {float(sc.amp.mean()):.4f}/{float(sc.amp.std()):.4f}")
    # 法向有 ±π 模糊 (Riesz 向量是带符号方向但 grating 无极性), 折到 mod π
    ori_mean = math.atan2(
        float(mx.mean(mx.sin(2 * sc.ori))),  # type: ignore
        float(mx.mean(mx.cos(2 * sc.ori))),  # type: ignore
    )
    print(f"  ori 圆均值(2θ) = {math.degrees(ori_mean) / 2:.2f}° (期望 30°)")

    # natural images
    for img_name in [
        "12.png",
        "nat10.jpg",
        "nat1015.jpg",
        "nat1016.jpg",
        "nat1018.jpg",
        "nat1035.jpg",
    ]:
        img = Image.open(Utils.project_root() / f"images/{img_name}")
        img = img.convert("L")
        arr = Color.image_to_mlx(img)
        rw3 = RieszWavelet(arr)
        path = Utils.project_root() / f"artifacts/riesz_{img_name}"
        print(path)
        rw3.visualize(path)
