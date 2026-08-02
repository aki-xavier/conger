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
    radius: mx.array | None = None
    safe_r: mx.array | None = None
    m1: mx.array | None = None
    m2: mx.array | None = None
    dc: mx.array | None = None
    lams: list[float] = field(default_factory=list)  # wavelength
    dc_kernel: mx.array | None = None  # DC 低通核 (与图像内容无关)
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

        if self.adaptive_pad:
            self.pad = int(self.lam_max())

        h = self.height + 2 * self.pad
        w = self.width + 2 * self.pad

        self.xgrid, self.ygrid = Utils.freqgrid((h, w))
        sigma_f = 0.5 / self.lam_max()
        self.radius = self.xgrid**2 + self.ygrid**2
        self.safe_r = mx.maximum(mx.sqrt(self.radius), 1e-12)
        self.dc_kernel = mx.exp(-0.5 * self.radius / sigma_f**2)  # type: ignore

        # 各向同性径向高斯带通, 与 gabor.py 同一核族; Riesz 框架下
        # 角度分解不再用方向核, 而用 Riesz 乘子 (见 calc_scales)。
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )
        for lam in self.lams:
            f0 = 1.0 / lam
            sigma_f = sigma_f_rel * f0
            kernel = mx.exp(-0.5 * (mx.sqrt(self.radius) - f0) ** 2 / sigma_f**2)
            self.kernels.append(kernel)

        # Riesz 乘子: R(ω) = −j·ω/|ω|。DC 处 0/0, 但带通核在
        # ω=0 处本已为零, 用 safe 半径防 NaN 即可。
        self.m1 = (-1j) * self.xgrid / self.safe_r  # type: ignore # −j·ωx/|ω|
        self.m2 = (-1j) * self.ygrid / self.safe_r  # type: ignore # −j·ωy/|ω|

        self.update(self.img)

    def update(self, img: mx.array):
        """实时刷新: 网格/DC核/径向核只依赖形状与核参数, 初始化时
        算好后与图像内容无关; 逐帧只需重算图像相关部分
        (FFT, DC 剥离, 各尺度响应)。形状必须与初始化一致。"""
        if img.shape != (self.height, self.width):
            raise ValueError(
                f"img shape {img.shape} != init shape {(self.height, self.width)}"
            )

        self.img = img
        self.scales.clear()

        fft: mx.array
        if self.pad != 0:
            padded = mx.pad(
                img,
                [(self.pad, self.pad), (self.pad, self.pad)],
                mode="edge",
            )
            fft = mx.fft.fft2(padded)
        else:
            fft = mx.fft.fft2(img)

        self.dc = fft * self.dc_kernel
        fft = fft - self.dc

        for kernel in self.kernels:
            spec = fft * kernel
            # b0 是实函数 ↔ 频谱 Hermitian; Riesz 乘子保持 Hermitian,
            # b1/b2 也是实函数, 虚部只剩数值噪声, 取 real。
            b0 = mx.real(mx.fft.ifft2(spec))
            b1 = mx.real(mx.fft.ifft2(spec * self.m1))
            b2 = mx.real(mx.fft.ifft2(spec * self.m2))
            if self.pad > 0:
                b0 = b0[self.pad : -self.pad, self.pad : -self.pad]
                b1 = b1[self.pad : -self.pad, self.pad : -self.pad]
                b2 = b2[self.pad : -self.pad, self.pad : -self.pad]
            self.scales.append(RieszScale(b0=b0, b1=b1, b2=b2, pad=self.pad))

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.height, self.width) / 2.0

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


@dataclass(slots=True)
class RieszFeatures:
    """跨尺度谱统计特征 (逐像素)。

    把每个像素在 S 个尺度上的能量 e_s = amp² 看作尺度轴上的一个
    分布, 提取形状统计 (坐标 x = 距 lam_max 的倍频程数, log 间隔
    等距, 单位 octave):

      log_mag   log Σe_s          —— 局部对比度总量
      slope     log e_s 对 x 的最小二乘斜率 —— 幂律衰减 (1/f 型)
      residual  拟合 RMS 残差      —— 偏离幂律 = 有峰 (纹理/周期结构)
      bump      argmax_s e_s      —— 峰所在尺度 (归一化到 [0,1])
      centroid/spread/skew/kurt   —— 能量分布 p_s = e_s/Σe 的前四阶矩
      ori_R     跨尺度方向一致性   —— 能量加权 2θ 圆均值 resultant
      phase_coh 跨尺度相位一致性   —— amp 加权相位 resultant
                (等价于投影到平均相位: ΣA·cos(φ_s−φ̄)/ΣA)
    """

    rw: RieszWavelet
    log_e: mx.array | None = None  # (H, W, S)
    log_mag: mx.array | None = None
    slope: mx.array | None = None
    residual: mx.array | None = None
    bump: mx.array | None = None
    centroid: mx.array | None = None
    spread: mx.array | None = None
    skew: mx.array | None = None
    kurt: mx.array | None = None
    ori_R: mx.array | None = None
    phase_coh: mx.array | None = None

    def __post_init__(self):
        e = mx.stack([s.energy for s in self.rw.scales], axis=-1)  # (H,W,S)
        total = mx.sum(e, axis=-1)
        safe_total = mx.maximum(total, 1e-12)
        p = e / safe_total[..., None]
        self.log_e = mx.log(mx.maximum(e, 1e-12))
        self.log_mag = mx.log(safe_total)

        lam_max = max(self.rw.lams)
        x = mx.array([math.log2(lam_max / lam) for lam in self.rw.lams])
        n_scales = len(self.rw.lams)

        # ── 幂律拟合: y = log e 对 x (octave) 的逐像素线性回归 ──────
        y = self.log_e
        xc = x - mx.mean(x)
        var_x = float(mx.sum(xc**2))
        self.slope = mx.sum(xc * (y - mx.mean(y, axis=-1, keepdims=True)), axis=-1)
        self.slope = self.slope / var_x
        intercept = mx.mean(y, axis=-1) - self.slope * float(mx.mean(x))
        fit = intercept[..., None] + self.slope[..., None] * x  # type: ignore
        self.residual = mx.sqrt(mx.mean((y - fit) ** 2, axis=-1))
        self.bump = mx.argmax(e, axis=-1).astype(mx.float32) / max(n_scales - 1, 1)

        # ── 谱矩 (p 是概率分布, 矩在 octave 坐标上) ─────────────────
        mu = mx.sum(p * x, axis=-1)
        d = x - mu[..., None]
        var = mx.sum(p * d**2, axis=-1)
        sd = mx.sqrt(mx.maximum(var, 1e-12))
        self.centroid = mu
        self.spread = sd
        self.skew = mx.sum(p * d**3, axis=-1) / sd**3
        self.kurt = mx.sum(p * d**4, axis=-1) / sd**4

        # ── 跨尺度方向一致性: ori 是法向 (轴向), 用 2θ 圆统计 ──────
        ori = mx.stack([s.ori for s in self.rw.scales], axis=-1)
        m_re = mx.sum(e * mx.cos(2 * ori), axis=-1)
        m_im = mx.sum(e * mx.sin(2 * ori), axis=-1)
        self.ori_R = mx.sqrt(m_re**2 + m_im**2) / safe_total

        # ── 跨尺度相位一致性: 边缘上各尺度 φ≈π/2 对齐 → ≈1,
        # 纹理/噪声上相位随机 → ≈0 ────────────────────────────────
        a = mx.stack([s.amp for s in self.rw.scales], axis=-1)
        ph = mx.stack([s.phase for s in self.rw.scales], axis=-1)
        p_re = mx.sum(a * mx.cos(ph), axis=-1)
        p_im = mx.sum(a * mx.sin(ph), axis=-1)
        self.phase_coh = mx.sqrt(p_re**2 + p_im**2)
        self.phase_coh = self.phase_coh / mx.maximum(mx.sum(a, axis=-1), 1e-12)

    def visualize(self, out_path: str | Path):
        plots = [
            ("original", "gray", self.rw.img),
            ("log_mag", "viridis", self.log_mag),
            ("slope", "RdBu_r", self.slope),
            ("residual", "viridis", self.residual),
            ("bump", "viridis", self.bump),
            ("centroid", "viridis", self.centroid),
            ("skew", "RdBu_r", self.skew),
            ("ori_R", "viridis", self.ori_R),
            ("phase_coh", "viridis", self.phase_coh),
        ]
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

    # update(): 逐帧刷新应与全新初始化逐位一致
    import time

    step = Utils.make_step_edge((256, 256))
    t0 = time.perf_counter()
    rw.update(step)
    mx.eval(rw.scales[-1].energy)
    t1 = time.perf_counter()
    diff = float(mx.max(mx.abs(rw.scales[0].amp - RieszWavelet(step).scales[0].amp)))
    stale = float(mx.max(mx.abs(rw.img - step)))  # update 必须同步 self.img
    print(
        f"update(step): {1000 * (t1 - t0):.0f}ms, "
        f"与全新初始化 max|Δamp|={diff:.2e}, img 同步残差={stale:.2e}"
    )

    # ── 跨尺度特征: 三种原型信号的谱形状应显著不同 ──────────────────
    def show_feat(name: str, img: mx.array):
        f = RieszFeatures(RieszWavelet(img))
        print(
            f"{name}: slope={float(f.slope.mean()):+.2f} "
            f"resid={float(f.residual.mean()):.2f} "
            f"bump={float(f.bump.mean()):.2f} "
            f"spread={float(f.spread.mean()):.2f}oct "
            f"ori_R={float(f.ori_R.mean()):.2f} "
            f"phase_coh={float(f.phase_coh.mean()):.2f}"
        )

    print("── cross-scale features (图均值) ──")
    show_feat("grating λ=16", grating)
    show_feat("noise        ", Utils.synthesize_signal04(256))
    show_feat("step edge    ", Utils.make_step_edge((256, 256)))

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
        fpath = Utils.project_root() / f"artifacts/rieszfeat_{img_name}"
        RieszFeatures(rw3).visualize(fpath)
