"""Riesz 自检与自然图特征可视化: python src/riesz_selftest.py。"""

import math
import time

import mlx.core as mx

from color import Color
from riesz import RieszWavelet
from utils import Utils


class RieszSelfTest:
    """Riesz 前端自检 (合成 GT + update 一致性 + 特征 + 自然图)。"""

    @staticmethod
    def run() -> None:
        from PIL import Image

        # ── synthetic ground truth checks ────────────────────────────────
        # grating: 单频平面波, 法向=angle, 匹配尺度上 phase 线性爬坡、
        # amp 常数、ori 常数。
        angle = math.radians(30.0)
        grating = Utils.make_grating((256, 256), wavelength=16.0, angle_rad=angle)
        rw = RieszWavelet(grating)
        best = max(
            range(len(rw.scales)), key=lambda i: float(rw.scales[i].energy.mean())
        )
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

        step = Utils.make_step_edge((256, 256))
        t0 = time.perf_counter()
        rw.update(step)
        mx.eval(rw.scales[-1].energy)
        t1 = time.perf_counter()
        fresh = RieszWavelet(step).scales[0].amp
        diff = float(mx.max(mx.abs(rw.scales[0].amp - fresh)))
        stale = float(mx.max(mx.abs(rw.img - step)))  # update 必须同步 self.img
        print(
            f"update(step): {1000 * (t1 - t0):.0f}ms, "
            f"与全新初始化 max|Δamp|={diff:.2e}, img 同步残差={stale:.2e}"
        )

        # ── 跨尺度特征: 三种原型信号的谱形状应显著不同 ──────────────────
        def show_feat(name: str, img: mx.array):
            """打印一张图的谱特征图均值 (六指标)。"""
            f = RieszWavelet(img).features()
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

        # mean_ori: grating 上应等于法向 30° (mod π)
        f_g = RieszWavelet(grating).features()
        mo = 0.5 * math.atan2(
            float(mx.mean(mx.sin(2 * f_g.mean_ori))),  # type: ignore
            float(mx.mean(mx.cos(2 * f_g.mean_ori))),  # type: ignore
        )
        print(f"mean_ori 圆均值(2θ) = {math.degrees(mo):.2f}° (期望 30°)")

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
            rw3.visualize_features(rw3.features(), fpath)


if __name__ == "__main__":
    RieszSelfTest.run()
