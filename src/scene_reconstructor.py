"""SceneReconstructor: 特征/帧对 → 完整 cga.Scene 参数化重建。

这里“完整”指当前场景族的全部可控自由度: kind / u / v / s / z /
图元色相 / 光色 / 光向。相机、背景、环境光、材质和渲染 rig 由
Codebook 的固定配置提供 (训练与推理同 renderer)。推理输出不是单个
MAP 点, 而是 StructuredHypothesis: MAP Scene + 候选渲染残差 + 联合后验。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
from cga.engine import PerspectiveCamera, Renderer, Scene

from codebook import Codebook
from composite_geometry import CompositeGeometry
from lateral_composite_geometry import LateralCompositeGeometry
from stereo import StereoDepth
from stereo_layers import StereoLayers
from structured_hypothesis import HypothesisCandidate, StructuredHypothesis

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
    def novelty_metrics(
        cat_p: mx.array,
        r: mx.array,
        cat_sizes: tuple[int, ...],
        render_residual: float | None,
    ) -> tuple[float, float, float]:
        """→ (责任度新颖性, 归一化后验熵, 综合新颖性诊断分)。"""
        max_r = float(mx.max(r)) + 1e-12
        responsibility_novelty = -math.log(max_r) / math.log(r.shape[1])
        lo = 0
        ent = 0.0
        for nc in cat_sizes:
            p = cat_p[lo : lo + nc]
            ent -= float(mx.sum(p * mx.log(mx.maximum(p, 1e-12)))) / math.log(nc)
            lo += nc
        posterior_entropy = ent / len(cat_sizes)
        render_term = 0.0 if render_residual is None else math.log1p(render_residual)
        return (
            responsibility_novelty,
            posterior_entropy,
            responsibility_novelty + posterior_entropy + render_term,
        )

    @staticmethod
    def s_proxy(kind: int | mx.array, stats: mx.array) -> mx.array:
        """kind-conditioned 表观尺寸代理。

        sphere/cylinder 在当前 rig 的可见轮廓按圆盘面积 A≈πs_img²;
        box 正面按 A≈(2s_img)²。掩码面积受光照/阈值影响, 剩余偏差由
        SPN 的 s 残差学习, 不再让 box/cylinder 共享球代理的系统偏差。"""
        q = mx.sqrt(stats[:, 2]) * (Codebook.CAM_Z - stats[:, 0]) / Codebook.FX
        if isinstance(kind, int):
            coef = 0.5 if kind == 2 else 1.0 / math.sqrt(math.pi)
            return q * coef
        coef = mx.where(
            kind.astype(mx.int32) == 2,
            0.5,
            1.0 / math.sqrt(math.pi),
        )
        return q * coef

    @classmethod
    def split_cat(cls, cat_p: mx.array) -> tuple[mx.array, ...]:
        """拼接场景后验 (N,15) → kind/hue/lcol/ldir 四个 (N,C_j)。"""
        out, lo = [], 0
        for nc in cls.CAT_SIZES:
            out.append(cat_p[:, lo : lo + nc])
            lo += nc
        return tuple(out)

    @classmethod
    def params(
        cls,
        t_pred: mx.array,  # (N,4) 残差参数化的 u,v,s−ŝ,z−ẑ
        cat_p: mx.array,  # (N,15) 拼接场景因子后验
        stats: mx.array,  # (N,3) [ẑ, 视差, 掩码面积]
    ) -> tuple[tuple[float, ...], ...]:
        """模型输出 → Codebook.to_scene 参数 (kind,u,v,s,z,hue,lcol,ldir)。"""
        probs = [mx.argmax(p, axis=1).astype(mx.int32) for p in cls.split_cat(cat_p)]
        s = t_pred[:, 2] + cls.s_proxy(probs[0], stats)
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
        cls, t_pred: mx.array, stats: mx.array, kind: mx.array
    ) -> mx.array:
        """残差目标 → 物理连续目标 [u,v,s,z] (评估/可视化用)。"""
        return mx.concatenate(
            [
                t_pred[:, :2],
                (t_pred[:, 2] + cls.s_proxy(kind, stats))[:, None],
                (t_pred[:, 3] + stats[:, 0])[:, None],
            ],
            axis=1,
        )

    @staticmethod
    def targets_from_params(
        params: tuple[tuple[float, ...], ...]
    ) -> mx.array:
        """完整场景参数 → 物理连续目标 [u,v,s,z] (评估最终 Scene)。"""
        return mx.array([p[1:5] for p in params], dtype=mx.float32)

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
        d = observed[..., :3].astype(mx.float32) - candidate[..., :3].astype(mx.float32)
        num = mx.sum(weights * mx.sum(d * d, axis=2))
        den = mx.maximum(mx.sum(weights), 1e-8)
        return float(num / den)

    @staticmethod
    def appearance_candidates(
        base_params: tuple[float, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """固定 kind/u/v/s/z 的 54 个 hue×lcol×ldir 候选。"""
        out = []
        for hue in range(Codebook.N_HUE):
            for lcol in range(len(Codebook.LIGHT_COLORS)):
                for ldir in range(len(Codebook.LIGHT_DIRS)):
                    out.append(base_params[:5] + (float(hue), float(lcol), float(ldir)))
        return tuple(out)

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
    ) -> tuple[tuple[float, ...], float, mx.array]:
        """固定 kind/u/v/s/z, 用渲染残差联合精炼 hue/lcol/ldir。

        候选 = 6 图元色相 × 3 光色 × 3 光向; 每候选同时渲染左右视图,
        以观测前景权重计算 RGB MSE。返回 (最佳完整参数, 最佳残差,
        全部 54 候选残差)。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        scores = []
        candidates = cls.appearance_candidates(base_params)
        for prm in candidates:
            scene = codebook.to_scene(prm)
            cl = renderer.render(scene, cam_l)
            cr = renderer.render(scene, cam_r)
            scores.append(
                0.5 * (cls._masked_mse(fl, cl, wl) + cls._masked_mse(fr, cr, wr))
            )
        score_arr = mx.array(scores, dtype=mx.float32)
        best_i = int(mx.argmin(score_arr))
        return candidates[best_i], float(score_arr[best_i]), score_arr

    @classmethod
    def refine_scene(
        cls,
        codebook: Codebook,
        base_params: tuple[float, ...],
        kind_p: mx.array,
        stats: mx.array,
        fl: mx.array,
        fr: mx.array,
        kind_topk: int = 3,
        renderer: Renderer | None = None,
        cam_l: PerspectiveCamera | None = None,
        cam_r: PerspectiveCamera | None = None,
    ) -> tuple[
        tuple[float, ...], tuple[tuple[float, ...], ...], mx.array, mx.array, float
    ]:
        """top-k kind × 54 外观候选的联合渲染后验。

        结构评分沿用共享几何, 避免把 kind 选择过度耦合到面积代理偏差;
        候选返回前再按各自 kind 重校准 s (保留 SPN 学到的残差)。后验
        log 形式 = −残差/T + log P(kind|SPN)。温度
        T=max(2·best_residual,1) 用最佳残差估计观测噪声尺度并随
        StructuredHypothesis 返回; 该后验表达候选间相对置信度, 不声称绝对
        校准。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        kind_topk = max(1, min(kind_topk, Codebook.N_KIND))
        order = mx.argsort(kind_p)[::-1][:kind_topk].tolist()
        stats = stats[None, :] if stats.ndim == 1 else stats
        s_resid = base_params[3] - float(
            cls.s_proxy(int(base_params[0]), stats)[0]
        )
        params, scores, weights = [], [], []
        for k in order:
            base = (float(k),) + base_params[1:]
            _, _, block_scores = cls.refine_appearance(
                codebook, base, fl, fr, renderer, cam_l, cam_r
            )
            params.extend(cls.appearance_candidates(base))
            scores.append(block_scores)
            weights.extend([float(kind_p[k])] * block_scores.shape[0])
        score_arr = mx.concatenate(scores)
        weight_arr = mx.maximum(mx.array(weights, dtype=mx.float32), 1e-12)
        temperature = max(2.0 * float(mx.min(score_arr)), 1.0)
        logp = -score_arr / temperature + mx.log(weight_arr)
        posterior = mx.exp(logp - mx.logsumexp(logp))
        calibrated = tuple(
            p[:3]
            + (float(cls.s_proxy(int(p[0]), stats)[0]) + s_resid,)
            + p[4:]
            for p in params
        )
        best_i = int(mx.argmax(posterior))
        return calibrated[best_i], calibrated, score_arr, posterior, temperature

    @staticmethod
    def frame_features(
        app: InverseApp,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
    ) -> tuple[mx.array, mx.array, RieszWavelet | None]:
        """左/右帧 → (模型特征 (1,V), 立体统计 (1,3), Riesz 工作区)。"""
        vec, rw = app.extractor.of_frame(fl, rw)
        if app.codebook.GEOMETRY_FAMILY == "lateral":
            stat = LateralCompositeGeometry.estimate(fl, fr)
            vec = mx.concatenate(
                [vec, StereoLayers.scaled(mx.array([stat]))[0]]
            )
            return vec[None, :], mx.array([stat], dtype=mx.float32), rw
        if app.codebook.USES_COMPOSITE_STATS:
            stat = CompositeGeometry.estimate(fl, fr)
            vec = mx.concatenate(
                [vec, StereoLayers.scaled(mx.array([stat]))[0]]
            )
            return vec[None, :], mx.array([stat], dtype=mx.float32), rw
        if app.codebook.USES_LAYER_STATS:
            stat = StereoLayers().estimate(fl, fr)
            vec = mx.concatenate(
                [vec, StereoLayers.scaled(mx.array([stat]))[0]]
            )
            return vec[None, :], mx.array([stat], dtype=mx.float32), rw
        z_hat, d, area = StereoDepth().estimate(fl, fr)
        vec = mx.concatenate([vec, mx.array([z_hat, area / 1000.0])])
        return vec[None, :], mx.array([[z_hat, d, area]], dtype=mx.float32), rw

    @classmethod
    def from_frames(
        cls,
        app: InverseApp,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
        refine: bool = True,
        kind_topk: int = 3,
    ) -> StructuredHypothesis:
        """左/右二维图像 → StructuredHypothesis (MAP Scene + 候选联合后验)。

        渲染 rig 保持 Codebook.make_renderer 的训练配置。refine=True 时
        kind_topk 个结构候选 × 54 个外观候选进入渲染残差联合后验;
        SPN 原始后验仍随返回值保留。"""
        f, stats, _ = cls.frame_features(app, fl, fr, rw)
        t, cat_p, r = net.predict(f)
        prm = cls.params(t, cat_p, stats)[0]
        rn, ent, novelty = cls.novelty_metrics(
            cat_p[0], r, cls.CAT_SIZES, None
        )
        if not refine:
            return StructuredHypothesis(
                scene=app.codebook.to_scene(prm),
                params=prm,
                spn_posterior=cat_p[0],
                geometry_family=app.codebook.GEOMETRY_FAMILY,
                template_delta=app.codebook.TEMPLATE_LINEAGE.delta,
                candidate_params=(prm,),
                hypotheses=(HypothesisCandidate(prm, 1.0, None),),
                responsibility_max=float(mx.max(r)),
                posterior_entropy=ent,
                residual=None,
                complexity=app.codebook.TEMPLATE_COMPLEXITY,
                novelty_score=novelty,
            )
        prm, candidates, scores, posterior, temperature = cls.refine_scene(
            app.codebook,
            prm,
            cat_p[0, : Codebook.N_KIND],
            stats,
            fl,
            fr,
            kind_topk=kind_topk,
        )
        order = mx.argsort(posterior)[::-1][:5].tolist()
        hypotheses = tuple(
            HypothesisCandidate(candidates[i], float(posterior[i]), float(scores[i]))
            for i in order
        )
        best_residual = float(mx.min(scores))
        rn, ent, novelty = cls.novelty_metrics(
            cat_p[0], r, cls.CAT_SIZES, best_residual
        )
        return StructuredHypothesis(
            scene=app.codebook.to_scene(prm),
            params=prm,
            spn_posterior=cat_p[0],
            geometry_family=app.codebook.GEOMETRY_FAMILY,
                template_delta=app.codebook.TEMPLATE_LINEAGE.delta,
            candidate_params=candidates,
            candidate_scores=scores,
            candidate_posterior=posterior,
            candidate_temperature=temperature,
            hypotheses=hypotheses,
            responsibility_max=float(mx.max(r)),
            posterior_entropy=ent,
            residual=best_residual,
            complexity=app.codebook.TEMPLATE_COMPLEXITY,
            novelty_score=novelty,
        )

    @staticmethod
    def rig() -> tuple[Renderer, PerspectiveCamera, PerspectiveCamera]:
        """公开训练 rig: 调用方可用它获取模型输入的左右二维图像。"""
        return Codebook.make_renderer()
