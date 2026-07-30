import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from gabor import GaborWavelet
from gabor_rotation import GaborRotation
from utils import Utils


@dataclass(slots=True)
class EdgeConfidence:
    """边缘置信度: 方向统计 × 尺度谱统计 × 相位统计的乘积门控。

    特征全部取自 GaborOri 特征层(每像素主方向 = 能量最大的方向通道),
    本层只做门控组合, 不做特征提取。四类证据(均在 [0,1], 连乘):

    aniso      — 方向一致性: 各尺度 R (resultant) 按尺度能量加权。
                 边缘 R≈1; 角点/十字 R≈0 (正交方向对消); 噪声 R 低且乱。
    gate_slope — 谱斜率门: 阶跃边缘的小波响应近似尺度不变,
                 主方向上 log e ≈ a − α·log f 的 α≈0 (FFT 环绕压粗频带
                 + 直流高通 + 有限长边缘 v 窗损失致正倾, 实测 ≈0.3)。
                 单频纹理能量集中于单个频带, |slope| 远大于 0。
                 注意: 模糊边缘/斜坡谱衰减快, 会被本门压制——本指标
                 度量的是锐利阶跃性, 非广义边界性。
    gate_resid — 残差门: 边缘沿主方向是干净的宽带衰减(residual 小);
                 纹理的谱有尖峰, 幂律拟合残差大。residual 对任意干净
                 幂律都为零(抓不到指数错误的谱), 与 slope 门互补。
    gate_phase — pc · odd_frac: pc 是跨尺度相位一致性, 否决相位随机
                 的噪声; odd_frac 是奇对称占比, 阶跃≈1, 细线≈0。
                 单频纹理只有一个活跃频带, pc 退化为 1, 纹理仍交给
                 slope/resid 门否决。
    """

    gw: GaborWavelet
    slope_center: float = 0.3  # 阶跃边缘实测 slope (环绕/高通/v窗致正倾)
    slope_sigma: float = 0.6
    resid_sigma: float = 0.8

    aniso: mx.array | None = None  # 能量加权的方向一致性
    gate_slope: mx.array | None = None
    gate_resid: mx.array | None = None
    gate_phase: mx.array | None = None
    conf: mx.array | None = None

    def __post_init__(self):
        gr = GaborRotation(self.gw)

        # ── 方向一致性: 按各尺度能量加权的 R ─────────────────────
        total = gr.scales[0].sum_e
        assert total is not None
        for sc in gr.scales[1:]:
            total = total + sc.sum_e
        safe_total = mx.maximum(total, 1e-12)
        aniso = mx.zeros_like(total)
        for sc in gr.scales:
            aniso = aniso + (sc.sum_e / safe_total) * sc.resultant

        # ── 主方向 (每像素能量最大的方向通道) ────────────────────
        sum_e_stack = mx.stack([o.sum_e for o in self.gw.oris])  # (K,H,W)
        dom = mx.argmax(sum_e_stack, axis=0, keepdims=True)  # (1,H,W)

        def gather(feat: str) -> mx.array:
            stack = mx.stack([getattr(o, feat) for o in self.gw.oris])
            return mx.take_along_axis(stack, dom, axis=0)[0]

        slope_dom = gather("slope")
        resid_dom = gather("residual")
        pc_dom = gather("pc")
        odd_dom = gather("odd_frac")

        # ── 门控 ─────────────────────────────────────────────────
        z = (slope_dom - self.slope_center) / self.slope_sigma
        self.gate_slope = mx.exp(-0.5 * z * z)
        zr = resid_dom / self.resid_sigma
        self.gate_resid = mx.exp(-0.5 * zr * zr)
        self.gate_phase = pc_dom * odd_dom

        self.aniso = aniso
        self.conf = aniso * self.gate_slope * self.gate_resid * self.gate_phase

    def visualize(
        self,
        out_path: str | Path,
        dpi: int = 150,
    ):
        plots = [
            ("original", "gray", self.gw.img),
            ("aniso", "viridis", self.aniso),
            ("gate_slope", "viridis", self.gate_slope),
            ("gate_resid", "viridis", self.gate_resid),
            ("gate_phase", "viridis", self.gate_phase),
            ("conf", "viridis", self.conf),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path, dpi=dpi)


def make_square(size: int = 300) -> mx.array:
    """中央白色方块: 用于检验角点抑制(边缘高、角点低)。"""
    img = mx.zeros((size, size), dtype=mx.float32)
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    return img


def make_line(size: int = 300, width: int = 2) -> mx.array:
    """中央竖直细线: 用于检验阶跃/细线区分(细线应被相位门否决)。"""
    img = mx.zeros((size, size), dtype=mx.float32)
    img[:, size // 2 - width // 2 : size // 2 + width // 2] = 1.0
    return img


def make_blurred_step(size: int = 300, sigma: float = 4.0) -> mx.array:
    """高斯模糊的阶跃边缘: slope 门会压制它(锐利阶跃性的设计取舍)。"""
    img = Utils.synthesize_signal01(size)
    xg, yg = Utils.freqgrid((size, size))
    r2 = xg**2 + yg**2
    sigma_f = 1.0 / (2.0 * math.pi * sigma)  # 高斯模糊的频域标准差
    h = mx.exp(-0.5 * r2 / sigma_f**2)
    return mx.real(mx.fft.ifft2(mx.fft.fft2(img) * h))


if __name__ == "__main__":
    from PIL import Image

    from color import Color

    tasks = [
        ("step01", Utils.synthesize_signal01()),
        ("blur_step", make_blurred_step()),
        ("ramp02", Utils.synthesize_signal02()),
        ("grat03", Utils.synthesize_signal03()),
        ("noise04", Utils.synthesize_signal04()),
        ("square", make_square()),
        ("line", make_line()),
    ]
    for name, img in tasks:
        ec = EdgeConfidence(GaborWavelet(img))
        path = Utils.out_dir() / "artifacts" / f"{name}_edge.png"
        print(path, f"mean_conf={float(mx.mean(ec.conf)):.4f}")
        ec.visualize(out_path=path)

    # 定量: 锐利阶跃 conf 应高; 平坦区/细线应低; 模糊阶跃被压制(取舍)
    for name, img in [
        ("step01", Utils.synthesize_signal01()),
        ("blur_step", make_blurred_step()),
    ]:
        ec = EdgeConfidence(GaborWavelet(img))
        assert ec.conf is not None
        edge_col = float(mx.mean(ec.conf[:, 148:153]))
        interior = float(mx.mean(ec.conf[:, 30:120]))
        print(f"{name}: edge={edge_col:.4f} interior={interior:.4f}")
    ec = EdgeConfidence(GaborWavelet(make_line()))
    assert ec.conf is not None
    print(f"line: center={float(mx.mean(ec.conf[:, 148:153])):.4f}")

    # natural images (downloaded from picsum.photos)
    for img_id in [10, 1015, 1016, 1018, 1035]:
        img = Image.open(Utils.out_dir() / f"images/nat{img_id}.jpg")
        arr = Color.image_to_mlx(img.convert("L"))
        ec = EdgeConfidence(GaborWavelet(arr))
        path = Utils.out_dir() / "artifacts" / f"nat{img_id}_edge.png"
        print(path, f"mean_conf={float(mx.mean(ec.conf)):.4f}")
        ec.visualize(out_path=path)
