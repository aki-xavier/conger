import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from color import Color
from utils import Utils


@dataclass(slots=True)
class GaborOri:
    resps: list[mx.array]
    es: list[mx.array] = field(default_factory=list)  # energies
    sum_e: mx.array | None = None
    safe_e: mx.array | None = None
    slope: mx.array | None = None  # 幂律谱斜率 α: log e ≈ a − α·log f
    residual: mx.array | None = None  # 幂律拟合残差范数（非 1/f 程度）
    bump_freq: mx.array | None = None  # 谱鼓包频率 cycles/sample (连续,
    # Nyquist=0.5 为固定上界便于归一化; 残差显著时有意义)
    pc: mx.array | None = None  # 跨尺度相位一致性 |Σresp| / Σ|resp|
    odd_frac: mx.array | None = None  # 奇对称占比: 1=阶跃, 0=细线

    def __post_init__(self):
        for s in self.resps:
            self.es.append(mx.abs(s) ** 2)

        total = self.es[0]
        for e in self.es[1:]:
            total = total + e
        self.sum_e = total

        self.safe_e = mx.maximum(self.sum_e, 1e-12)

    def calc_powerlaw(self, freqs: list[float]):
        """Per-pixel power-law fit of the band spectrum: log p ≈ a − α·log f.

        Produces three roughly-orthogonal features:
        slope      — spectral slope α (replaces centroid/variance/rolloff),
        residual   — RMSE of the fit in log space (non-1/f-ness),
        bump_freq  — frequency of the spectral bump: centroid of the
                     positive linear-domain excess p − p_fit over log f,
                     in cycles/sample (Nyquist = 0.5). Continuous
                     (sub-band precision); meaningful only where
                     residual is significant.
        """
        # normalized band energies, floored so log is finite (floored bands
        # count as strong negative deviations from a power law)
        p = [mx.maximum(e / self.safe_e, 1e-6) for e in self.es]
        y = mx.log(mx.stack(p))  # (S, H, W)
        x = mx.log(mx.array(freqs, dtype=mx.float32))  # (S,)

        x_c = x - x.mean()
        y_mean = y.mean(axis=0)
        denom = (x_c * x_c).sum()
        slope = (x_c.reshape(-1, 1, 1) * (y - y_mean)).sum(axis=0) / denom
        intercept = y_mean - slope * x.mean()

        fit = intercept + slope * x.reshape(-1, 1, 1)
        resid = y - fit
        self.slope = slope
        self.residual = mx.sqrt((resid * resid).mean(axis=0))

        # bump centroid 在线性 p 域计算: 对数域地板值会让拟合线跌穿地板,
        # 把 argmax/质心错误地甩到粗端
        w = mx.maximum(mx.stack(p) - mx.exp(fit), 0.0)  # 鼓包(线性域)
        wsum = w.sum(axis=0)
        bc = (w * x.reshape(-1, 1, 1)).sum(axis=0) / mx.maximum(wsum, 1e-12)
        self.bump_freq = mx.exp(bc)  # log f 质心 → 频率

    def calc_phase(self):
        """跨尺度相位特征: pc = 一致性, odd_frac = 奇偶类型(阶跃/细线)。

        单边频域核使空间响应为解析信号: 实部=偶对称(细线)通道,
        虚部=奇对称(阶跃)通道。边缘/细线处各尺度响应同相 → pc≈1;
        噪声相位随机 → pc 低。
        """
        s_sum = self.resps[0]
        a_sum = mx.abs(self.resps[0])
        for resp in self.resps[1:]:
            s_sum = s_sum + resp
            a_sum = a_sum + mx.abs(resp)
        mag = mx.abs(s_sum)
        # 三角不等式保证 pc≤1, minimum 仅作数值安全
        self.pc = mx.minimum(mag / mx.maximum(a_sum, 1e-12), 1.0)
        self.odd_frac = mx.abs(mx.imag(s_sum)) / mx.maximum(mag, 1e-12)


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
    fft: mx.array | None = None
    xgrid: mx.array | None = None
    ygrid: mx.array | None = None
    dc: mx.array | None = None
    h_dc: mx.array | None = None  # Gaussian lowpass kernel of the dc channel
    thetas: list[float] = field(default_factory=list)  # orientation angle in [0, pi]
    lams: list[float] = field(default_factory=list)  # wavelength
    oris: list[GaborOri] = field(default_factory=list)

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

        self.calc_pad()
        self.calc_lams()
        self.calc_thetas()
        self.calc_dc()
        self.calc_oris()

    def lam_max(self) -> float:
        """Coarsest supported wavelength for the image dimensions."""
        return min(self.height, self.width) / 2.0

    def calc_pad(self):
        # ── self-adaptive padding to avoid FFT wraparound ────────────
        if self.adaptive_pad:
            self.pad = int(self.lam_max())
            H_pad = self.height + 2 * self.pad
            W_pad = self.width + 2 * self.pad
            padded = mx.pad(
                self.img,
                [(self.pad, self.pad), (self.pad, self.pad)],
                mode="edge",
            )
            self.fft = mx.fft.fft2(padded)
            self.xgrid, self.ygrid = Utils.freqgrid((H_pad, W_pad))
        else:
            self.fft = mx.fft.fft2(self.img)
            self.xgrid, self.ygrid = Utils.freqgrid((self.height, self.width))

    def calc_lams(self):
        lam_min = self.lam_min
        lam_max = self.lam_max()
        if self.scale_size == 1:
            self.lams.append(lam_min)
        else:
            for i in range(self.scale_size):
                lam = lam_min * 2.0 ** (
                    i * math.log2(lam_max / lam_min) / (self.scale_size - 1)
                )
                self.lams.append(lam)

    def calc_thetas(self):
        for i in range(self.ori_size):
            theta = i * math.pi / self.ori_size
            self.thetas.append(theta)

    def calc_dc(self):
        # DC / lowpass channel: Gaussian lowpass captures local mean intensity.
        # sigma_f sits one octave below the coarsest Gabor band center
        # (f0 = 1/lam_max), so the filter passes DC fully and rolls off
        # before the bank's lowest band (gain ≈ 0.14 at that band's center).
        sigma_f = 0.5 / self.lam_max()

        assert self.xgrid is not None
        assert self.ygrid is not None
        r2 = self.xgrid**2 + self.ygrid**2
        h_dc = mx.exp(-0.5 * r2 / sigma_f**2)
        dc = mx.real(mx.fft.ifft2(self.fft * h_dc))

        # ── crop padding ────────────────────────────────────────────
        if self.pad > 0:
            dc = dc[
                self.pad : self.pad + self.height,
                self.pad : self.pad + self.width,
            ]

        self.h_dc = h_dc
        self.dc = dc

    def calc_oris(self):
        # band center frequencies in cycles/sample (Nyquist = 0.5),
        # matching gabor_kernel's f0 = 1/lam
        freqs: list[float] = []
        for lam in self.lams:
            freqs.append(1.0 / lam)

        # Gabor channels must not see the DC component: apply the
        # complementary Gaussian highpass (1 - h_dc) of the dc channel.
        # Done here rather than in calc_pad so calc_dc (which runs
        # earlier) keeps the mean.
        assert self.fft is not None
        assert self.h_dc is not None
        self.fft = self.fft * (1.0 - self.h_dc)

        for theta in self.thetas:
            resps: list[mx.array] = []
            for lam in self.lams:
                kernel = self.gabor_kernel(lam, theta)
                resp_f = self.fft * kernel
                resp = mx.fft.ifft2(resp_f)
                if self.pad > 0:
                    resp = resp[
                        self.pad : self.pad + self.height,
                        self.pad : self.pad + self.width,
                    ]
                resps.append(resp)

            go = GaborOri(resps=resps)
            go.calc_powerlaw(freqs)
            go.calc_phase()
            self.oris.append(go)

    def gabor_kernel(self, lam: float, theta: float) -> mx.array:
        # cycles/sample (Nyquist = 0.5); a wavelength-lam sinusoid sits at
        # 1/lam regardless of padding — padding only makes the grid denser.
        f0 = 1.0 / lam
        bw = self.bandwidth
        sigma_f_rel = (2.0**bw - 1.0) / (
            (2.0**bw + 1.0) * math.sqrt(2.0 * math.log(2.0))
        )

        sigma_f = sigma_f_rel * f0

        assert self.xgrid is not None
        assert self.ygrid is not None
        u = self.xgrid * math.cos(theta) + self.ygrid * math.sin(theta)
        v = -self.xgrid * math.sin(theta) + self.ygrid * math.cos(theta)
        du = u - f0
        return mx.exp(-0.5 * (du**2 / sigma_f**2 + v**2 / (sigma_f / self.gamma) ** 2))

    def get_ori_at(self, theta: float = 0) -> GaborOri:
        idx = self.thetas.index(theta)
        return self.oris[idx]

    def visualize(
        self,
        theta: float,
        out_path: str | Path,
        dpi: int = 150,
    ):
        """Render this orientation's spectral feature maps to an image.
        Args:
            out_path: save the figure here (e.g. ``"ori.png"``).
            title: Optional figure title (e.g. the orientation angle).
            cmap: Matplotlib colormap applied to every panel.
            dpi: Save resolution.

        Returns:
            The matplotlib ``Figure`` (caller may ``plt.show()`` it).
        """

        ori = self.get_ori_at(theta=theta)

        gabor = mx.real(mx.fft.ifft2(self.fft))

        if self.pad > 0:
            gabor = gabor[
                self.pad : self.pad + self.height,
                self.pad : self.pad + self.width,
            ]

        plots = [
            ("original", "gray", self.img),
            ("dc", "gray", self.dc),
            ("gabor", "coolwarm", gabor),
            ("slope", "viridis", ori.slope),
            ("residual", "viridis", ori.residual),
            ("bump_freq", "viridis", ori.bump_freq),
            ("pc", "viridis", ori.pc),
            ("odd_frac", "viridis", ori.odd_frac),
        ]

        fig = Utils.visualize(plots)
        fig.savefig(out_path, dpi=dpi)


if __name__ == "__main__":
    from PIL import Image

    tasks = [
        ("signal01", Utils.synthesize_signal01()),
        ("signal02", Utils.synthesize_signal02()),
        ("signal03", Utils.synthesize_signal03()),
        ("signal04", Utils.synthesize_signal04()),
        ("signal05", Utils.synthesize_signal05()),
        ("signal06", Utils.synthesize_signal06()),
        ("signal07", Utils.synthesize_signal07()),
        ("signal08", Utils.synthesize_signal08()),
        ("signal09", Utils.synthesize_signal09()),
        # 补充类别覆盖: 纯色 / 粗纹理
        # (细纹理=signal03+04, 强边缘=signal01, 弱边缘=signal08——
        #  形状特征对比度不变, 单一阶跃对比度样本即可)
        ("flat", Utils.make_smooth_patch((300, 300))),
        ("coarse_tex", Utils.make_grating((300, 300), 64.0, 0.0)),
    ]
    for task in tasks:
        name, img = task
        gw = GaborWavelet(img)
        path = Utils.out_dir() / "artifacts" / (name + ".png")
        print(path)
        gw.visualize(theta=0, out_path=path)

    img = Image.open(Utils.out_dir() / "images/12.png")
    img = img.convert("L")
    arr = Color.image_to_mlx(img)
    gw = GaborWavelet(arr)
    path = Utils.out_dir() / "artifacts/signal12.png"
    print(path)
    gw.visualize(theta=0, out_path=path)

    # natural images (downloaded from picsum.photos)
    for img_id in [10, 1015, 1016, 1018, 1035]:
        img = Image.open(Utils.out_dir() / f"images/nat{img_id}.jpg")
        img = img.convert("L")
        arr = Color.image_to_mlx(img)
        gw = GaborWavelet(arr)
        path = Utils.out_dir() / "artifacts" / f"nat{img_id}.png"
        print(path)
        gw.visualize(theta=0, out_path=path)
