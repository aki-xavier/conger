import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx

from color import Color
from utils import Utils


@dataclass(slots=True)
class GaborScale:
    spectra: list[mx.array]  # 复数频谱 per orientation at this scale
    thetas: list[float]  # orientation angle in rad, uniform in [0, π)
    pad: int = 0  # fft padding, cropped from spatial responses
    es: list[mx.array] = field(default_factory=list)  # energies per orientation
    sum_e: mx.array | None = None  # total energy over orientations
    safe_e: mx.array | None = None
    mean_dir: mx.array | None = None  # 圆均值方向, rad in [0, π)
    resultant: mx.array | None = None  # R = |m₁| ∈ [0,1]: 1=单一方向, 0=各向同性
    r2: mx.array | None = None  # |m₂| ∈ [0,1]: 第二谐波——角点/十字（正交方向对）强度

    def __post_init__(self):
        for spec in self.spectra:
            # 单边核 → 解析信号: |resp|² 是平滑包络能量 (相位不变,
            # 无载波振荡); resp 的复相角留作跨尺度一致性备用
            resp = mx.fft.ifft2(spec)
            if self.pad > 0:
                resp = resp[self.pad : -self.pad, self.pad : -self.pad]
            self.es.append(mx.abs(resp) ** 2)

        total = self.es[0]
        for e in self.es[1:]:
            total = total + e
        self.sum_e = total
        self.safe_e = mx.maximum(self.sum_e, 1e-12)

        # 圆统计 (θ∈[0,π), 角度翻倍): m₁→主方向, m₂→正交方向对
        m1_re = mx.zeros_like(self.safe_e)
        m1_im = mx.zeros_like(self.safe_e)
        m2_re = mx.zeros_like(self.safe_e)
        m2_im = mx.zeros_like(self.safe_e)
        for e, theta in zip(self.es, self.thetas):
            m1_re = m1_re + e * math.cos(2.0 * theta)
            m1_im = m1_im + e * math.sin(2.0 * theta)
            m2_re = m2_re + e * math.cos(4.0 * theta)
            m2_im = m2_im + e * math.sin(4.0 * theta)
        m1_re = m1_re / self.safe_e
        m1_im = m1_im / self.safe_e
        m2_re = m2_re / self.safe_e
        m2_im = m2_im / self.safe_e

        mean_dir = 0.5 * mx.arctan2(m1_im, m1_re)  # [−π/2, π/2]
        self.mean_dir = mx.where(mean_dir < 0, mean_dir + math.pi, mean_dir)
        self.resultant = mx.sqrt(m1_re**2 + m1_im**2)
        self.r2 = mx.sqrt(m2_re**2 + m2_im**2)


@dataclass(slots=True)
class GaborWavelet:
    img: mx.array
    lam_min: float = 3.0  # min wavelength
    height: int = 0
    width: int = 0
    scale_size: int = 0
    ori_size: int = 8
    bandwidth: float = 1.0  # used to create gabor kernel
    gamma: float = 1.0  # used to create gabor kernel; 1.0 gives enough
    # angular selectivity for the second orientation harmonic (r2) to survive
    adaptive_pad: bool = True
    pad: int = 0
    xgrid: mx.array | None = None
    ygrid: mx.array | None = None
    dc: mx.array | None = None
    fft: mx.array | None = None
    thetas: list[float] = field(default_factory=list)  # orientation angle in [0, pi]
    lams: list[float] = field(default_factory=list)  # wavelength
    ffts: list[mx.array] = field(default_factory=list)  # ffts
    scales: list[GaborScale] = field(default_factory=list)

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

        self.calc_lams()
        self.calc_thetas()
        self.calc_freqs()
        self.calc_ffts()
        self.calc_scales()

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.height, self.width) / 2.0

    def calc_lams(self):
        # lams 从长波长到短波长排列 (低频→高频), 与 ffts/scales 顺序一致
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

    def calc_thetas(self):
        for i in range(self.ori_size):
            theta = i * math.pi / self.ori_size
            self.thetas.append(theta)

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

    def calc_ffts(self):
        # 各向同性径向高斯核, 把剥离 DC 后的频谱按 1/lam 从低频到高频分带。
        radius = mx.sqrt(self.xgrid**2 + self.ygrid**2)
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )
        for lam in self.lams:
            f0 = 1.0 / lam
            sigma_f = sigma_f_rel * f0
            kernel = mx.exp(-0.5 * (radius - f0) ** 2 / sigma_f**2)
            self.ffts.append(self.fft * kernel)

    def calc_scales(self):
        # 各向异性分解 = 各向同性频带 × 纯角度权重:
        # 在频域极角 φ 上以 θ 为中心放高斯, 与半径无关, 故角选择性
        # 跨尺度严格一致 (尺度不变)。
        # 核是单边(mod 2π)的: 只取 θ 方向的正频率半边 → 空域响应为
        # 解析信号, |resp| 是平滑包络, 相位可用于跨尺度一致性。
        # (θ+π 通道是 θ 通道的共轭, thetas∈[0,π) 无冗余。)
        # σ_θ = σ_f_rel/γ: 切向宽 σ_f/γ 在 r=f0 处折算成的角度,
        # 沿用 bandwidth/gamma 两个旋钮但纯角度化解释。
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )
        sigma_th = sigma_f_rel / self.gamma
        phi = mx.arctan2(self.ygrid, self.xgrid)  # 频域极角, (−π, π]

        for band in self.ffts:
            spectra: list[mx.array] = []

            for theta in self.thetas:
                d = phi - theta
                d = d - 2.0 * math.pi * mx.floor(
                    d / (2.0 * math.pi) + 0.5
                )  # wrap mod 2π
                kernel = mx.exp(-0.5 * d**2 / sigma_th**2)
                spectra.append(band * kernel)

            gs = GaborScale(spectra=spectra, thetas=self.thetas, pad=self.pad)
            self.scales.append(gs)

    def ifft2(self, arr: mx.array):
        ret = mx.real(mx.fft.ifft2(arr))
        if self.adaptive_pad:
            ret = ret[
                self.pad : self.pad + self.height,
                self.pad : self.pad + self.width,
            ]

        return ret

    def visualize(self, out_path: str | Path):
        dc = self.ifft2(self.dc)

        plots = [
            ("original", "gray", self.img),
            ("dc", "gray", dc),
        ]

        for idx, scale in enumerate(self.scales):
            lam = self.lams[idx]
            plots.append((f"s{idx} λ={lam:.1f} sum_e", "gray", scale.sum_e))
            plots.append((f"s{idx} λ={lam:.1f} mean_dir", "hsv", scale.mean_dir))
            plots.append((f"s{idx} λ={lam:.1f} R", "viridis", scale.resultant))

        fig = Utils.visualize(plots)
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    from PIL import Image

    # natural images (downloaded from picsum.photos)
    for img_id in [10, 1015, 1016, 1018, 1035]:
        img = Image.open(Utils.project_root() / f"images/nat{img_id}.jpg")
        img = img.convert("L")
        arr = Color.image_to_mlx(img)
        gw = GaborWavelet(arr)
        path = Utils.project_root() / "artifacts" / f"nat{img_id}.png"
        print(path)
        gw.visualize(path)
