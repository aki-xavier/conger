"""SceneReconstructor: 特征/帧对 → 完整 cga.Scene 参数化重建。

这里“完整”指当前场景族的全部可控自由度: kind / u / v / s / z /
图元色相 / 光色 / 光向。相机、背景、环境光、材质和渲染 rig 由
Codebook 的固定配置提供 (训练与推理同 renderer)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
from cga.engine import PerspectiveCamera, Renderer, Scene

from codebook import Codebook
from stereo import StereoDepth

if TYPE_CHECKING:
    from inverse_app import InverseApp
    from mixture_spn import MixtureSPN
    from riesz import RieszWavelet


class SceneReconstructor:
    """完整场景参数的头拆分、物理量反参数化与 cga.Scene 构造。"""

    CAT_SIZES = (
        Codebook.N_KIND,
        Codebook.N_HUE,
        len(Codebook.LIGHT_COLORS),
        len(Codebook.LIGHT_DIRS),
    )

    @staticmethod
    def s_proxy(stats: mx.array) -> mx.array:
        """表观尺寸代理: √(area/π)·zc/FX (形状系数留给模型残差学)。"""
        return mx.sqrt(stats[:, 2] / math.pi) * (
            Codebook.CAM_Z - stats[:, 0]
        ) / Codebook.FX

    @classmethod
    def split_cat(cls, cat_p: mx.array) -> tuple[mx.array, ...]:
        """拼接场景后验 (N,21) → kind/hue/lcol/ldir 四个 (N,C_j)。"""
        out, lo = [], 0
        for nc in cls.CAT_SIZES:
            out.append(cat_p[:, lo : lo + nc])
            lo += nc
        return tuple(out)

    @classmethod
    def params(
        cls,
        t_pred: mx.array,  # (N,4) 残差参数化的 u,v,s−ŝ,z−ẑ
        cat_p: mx.array,  # (N,21) 拼接场景因子后验
        stats: mx.array,  # (N,3) [ẑ, 视差, 掩码面积]
    ) -> tuple[tuple[float, ...], ...]:
        """模型输出 → Codebook.to_scene 参数 (kind,u,v,s,z,hue,lcol,ldir)。"""
        probs = [mx.argmax(p, axis=1).astype(mx.int32) for p in cls.split_cat(cat_p)]
        s = t_pred[:, 2] + cls.s_proxy(stats)
        z = t_pred[:, 3] + stats[:, 0]
        rows = []
        for i in range(t_pred.shape[0]):
            rows.append(
                (
                    float(probs[0][i]),
                    float(t_pred[i, 0]),
                    float(t_pred[i, 1]),
                    float(s[i]),
                    float(z[i]),
                    float(probs[1][i]),
                    float(probs[2][i]),
                    float(probs[3][i]),
                )
            )
        return tuple(rows)

    @classmethod
    def physical_targets(
        cls, t_pred: mx.array, stats: mx.array
    ) -> mx.array:
        """残差目标 → 物理连续目标 [u,v,s,z] (评估/可视化用)。"""
        return mx.concatenate(
            [
                t_pred[:, :2],
                (t_pred[:, 2] + cls.s_proxy(stats))[:, None],
                (t_pred[:, 3] + stats[:, 0])[:, None],
            ],
            axis=1,
        )

    @staticmethod
    def scenes(
        params: tuple[tuple[float, ...], ...], codebook: Codebook
    ) -> tuple[Scene, ...]:
        """场景参数 → cga.Scene 对象组 (含预测光照)。"""
        return tuple(codebook.to_scene(p) for p in params)

    @staticmethod
    def _masked_mse(
        observed: mx.array, candidate: mx.array, weights: mx.array
    ) -> float:
        """前景加权 RGB MSE (左右两视图同一权重的观测侧定义)。"""
        d = (
            observed[..., :3].astype(mx.float32)
            - candidate[..., :3].astype(mx.float32)
        )
        num = mx.sum(weights * mx.sum(d * d, axis=2))
        den = mx.maximum(mx.sum(weights), 1e-8)
        return float(num / den)

    @classmethod
    def refine_appearance(
        cls,
        codebook: Codebook,
        base_params: tuple[float, ...],
        fl: mx.array,
        fr: mx.array,
        renderer: Renderer | None = None,
        cam_l: PerspectiveCamera | None = None,
        cam_r: PerspectiveCamera | None = None,
    ) -> tuple[tuple[float, ...], float]:
        """固定 kind/u/v/s/z, 用渲染残差联合精炼 hue/lcol/ldir。

        候选 = 6 图元色相 × 3 光色 × 3 光向; 每候选同时渲染左右视图,
        以观测前景权重计算 RGB MSE。这个方法把反照率×光照的联合歧义
        交还给正向渲染模型裁决, 而不是让共享 SPN 责任度独立猜三个
        边缘类别。返回 (最佳完整参数, 最佳残差)。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        best, best_score = base_params, float("inf")
        for hue in range(Codebook.N_HUE):
            for lcol in range(len(Codebook.LIGHT_COLORS)):
                for ldir in range(len(Codebook.LIGHT_DIRS)):
                    prm = base_params[:5] + (
                        float(hue), float(lcol), float(ldir)
                    )
                    scene = codebook.to_scene(prm)
                    cl = renderer.render(scene, cam_l)
                    cr = renderer.render(scene, cam_r)
                    score = 0.5 * (
                        cls._masked_mse(fl, cl, wl)
                        + cls._masked_mse(fr, cr, wr)
                    )
                    if score < best_score:
                        best, best_score = prm, score
        return best, best_score

    @staticmethod
    def frame_features(
        app: InverseApp,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
    ) -> tuple[mx.array, mx.array, RieszWavelet | None]:
        """左/右帧 → (模型特征 (1,V), 立体统计 (1,3), Riesz 工作区)。"""
        vec, rw = app.extractor.of_frame(fl, rw)
        z_hat, d, area = StereoDepth().estimate(fl, fr)
        vec = mx.concatenate([vec, mx.array([z_hat, area / 1000.0])])
        return vec[None, :], mx.array(
            [[z_hat, d, area]], dtype=mx.float32
        ), rw

    @classmethod
    def from_frames(
        cls,
        app: InverseApp,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
        refine: bool = True,
    ) -> tuple[Scene, tuple[float, ...], mx.array]:
        """左/右二维图像 → 完整 cga.Scene (含精炼光照)。

        返回 (scene, 场景参数, SPN 场景因子后验)。渲染 rig 保持
        Codebook.make_renderer 的训练配置; refine=True 时 hue/lcol/ldir
        由候选渲染残差联合精炼, SPN 后验仍随返回值保留。"""
        f, stats, _ = cls.frame_features(app, fl, fr, rw)
        t, cat_p, _ = net.predict(f)
        prm = cls.params(t, cat_p, stats)[0]
        if refine:
            prm, _ = cls.refine_appearance(app.codebook, prm, fl, fr)
        return app.codebook.to_scene(prm), prm, cat_p

    @staticmethod
    def rig() -> tuple[Renderer, PerspectiveCamera, PerspectiveCamera]:
        """公开训练 rig: 调用方可用它获取模型输入的左右二维图像。"""
        return Codebook.make_renderer()
