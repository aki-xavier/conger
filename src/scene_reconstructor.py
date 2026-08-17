"""SceneReconstructor: 特征/帧对 → 完整 cga.Scene 参数化重建。

这里“完整”指当前场景族的全部可控自由度: kind / u / v / s / z /
图元色相 / 光色 / 光向。相机、背景、环境光、材质和渲染 rig 由
Codebook 的固定配置提供 (训练与推理同 renderer)。推理输出不是单个
MAP 点, 而是 StructuredHypothesis: MAP Scene + 候选渲染残差 + 联合后验。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import mlx.core as mx
from cga.engine import (  # pyright: ignore[reportMissingImports]
    PerspectiveCamera,
    Renderer,
    Scene,
)

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

    @classmethod
    def cat_sizes(cls, n_textures: int = 0) -> tuple[int, ...]:
        """cat_sizes 随纹理自由度扩展: 默认 (3,6,3,3); textured → (3,6,3,3,n_tex)。"""
        base = cls.CAT_SIZES
        return base + (n_textures,) if n_textures > 0 else base
    # 物理下限钳制 (与 LayeredReconstructor 对称): 负残差 + 缩水面积代理
    # 会把 s 压成负值 → cga 几何拒绝 radius≤0; z 越界会经 unproject 产生
    # zc<0 的镜像翻转。只做崩溃/野值防护, 不改变正常样本估计。
    S_FLOOR = 0.05
    Z_MIN = 0.5
    Z_MAX = Codebook.CAM_Z - 0.5

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
    def split_cat(
        cls, cat_p: mx.array, cat_sizes: tuple[int, ...] | None = None
    ) -> tuple[mx.array, ...]:
        """拼接场景后验 (N,ΣC) → 各因子 (N,C_j)。"""
        out, lo = [], 0
        for nc in (cat_sizes or cls.CAT_SIZES):
            out.append(cat_p[:, lo : lo + nc])
            lo += nc
        return tuple(out)

    @classmethod
    def params(
        cls,
        t_pred: mx.array,  # (N,4) 残差参数化 u,v,s−ŝ,z−ẑ
        cat_p: mx.array,  # (N,ΣC) 拼接场景因子后验
        stats: mx.array,  # (N,3) [ẑ, 视差, 掩码面积]
        cat_sizes: tuple[int, ...] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """模型输出 → Codebook.to_scene 参数元组。

        参数顺序: kind,u,v,s,z,hue,lcol,ldir[,tex_id,roughness];
        返回每行一个场景的参数元组。
        """
        sizes = cat_sizes or cls.CAT_SIZES
        probs = [
            mx.argmax(p, axis=1).astype(mx.int32) for p in cls.split_cat(cat_p, sizes)
        ]
        s = mx.maximum(t_pred[:, 2] + cls.s_proxy(probs[0], stats), cls.S_FLOOR)
        z = mx.clip(t_pred[:, 3] + stats[:, 0], cls.Z_MIN, cls.Z_MAX)
        tex = len(sizes) >= 5
        rows = []
        for i in range(t_pred.shape[0]):
            row = [
                float(probs[0][i]),
                float(t_pred[i, 0]),
                float(t_pred[i, 1]),
                float(s[i]),
                float(z[i]),
                float(probs[1][i]),
                float(probs[2][i]),
                float(probs[3][i]),
            ]
            if tex:
                row += [float(probs[4][i]), 0.55]  # tex_id, roughness (固定 0.55)
            rows.append(tuple(row))
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
    def marginal_appearance(
        posterior: mx.array,
        factor: str,
        n_hue: int = Codebook.N_HUE,
        n_lcol: int | None = None,
        n_ldir: int | None = None,
    ) -> mx.array:
        """联合外观后验 (hue×lcol×ldir 行主序) → 单因子边缘。

        对 nuisance 因子求和得到不变估计: 反照率 (hue) 对光照 (lcol/ldir)
        边缘化, 光照对反照率边缘化 —— 分析-合成的因果不变估计器
        (见 docs/architecture.md §9.3 路线 ①)。
        """
        n_lcol = len(Codebook.LIGHT_COLORS) if n_lcol is None else n_lcol
        n_ldir = len(Codebook.LIGHT_DIRS) if n_ldir is None else n_ldir
        p = mx.reshape(posterior, (n_hue, n_lcol, n_ldir))
        if factor == "hue":
            return mx.sum(p, axis=(1, 2))
        if factor == "lcol":
            return mx.sum(p, axis=(0, 2))
        if factor == "ldir":
            return mx.sum(p, axis=(0, 1))
        raise ValueError(f"未知外观因子: {factor} (期望 hue/lcol/ldir)")

    @staticmethod
    def marginal_joint(
        posterior: mx.array,
        factor: str,
        n_kind: int,
        n_hue: int = Codebook.N_HUE,
        n_lcol: int | None = None,
        n_ldir: int | None = None,
    ) -> mx.array:
        """top-k kind × 外观候选联合后验 → 单因子边缘。

        后验行主序 (kind, hue, lcol, ldir); 对 nuisance 因子求和得到不变
        估计。factor ∈ {kind, hue, lighting}: `lighting` 返回 (lcol,ldir)
        的**联合**边缘 (展平), 因为光照两因子有投影歧义, 不拆开。
        """
        n_lcol = len(Codebook.LIGHT_COLORS) if n_lcol is None else n_lcol
        n_ldir = len(Codebook.LIGHT_DIRS) if n_ldir is None else n_ldir
        p = mx.reshape(posterior, (n_kind, n_hue, n_lcol, n_ldir))
        if factor == "kind":
            return mx.sum(p, axis=(1, 2, 3))
        if factor == "hue":
            return mx.sum(p, axis=(0, 2, 3))
        if factor == "lighting":
            return mx.reshape(mx.sum(p, axis=(0, 1)), (-1,))
        raise ValueError(f"未知场景因子: {factor} (期望 kind/hue/lighting)")

    @staticmethod
    def decoupled_map(
        posterior: mx.array,
        n_kind: int,
        n_hue: int = Codebook.N_HUE,
        n_lcol: int | None = None,
        n_ldir: int | None = None,
    ) -> tuple[int, int, int, int]:
        """联合后验 → 解耦 MAP: kind/hue 各自边缘 argmax, 光照保持联合。

        返回 (kind_idx, hue, lcol, ldir); kind_idx 是后验 kind 维下标
        (调用方映射回实际 kind)。反照率 (hue) 对光照**联合**边缘化 =
        因果不变估计 (反照率↔光照是干净的可分离机制); 但光照内部 (lcol,
        ldir) 是同一机制的联合变量, 两者有投影歧义, 拆开边缘化会破坏其
        联合可识别性 (全量实测 lcol 0.994→0.870), 故 (lcol,ldir) 取联合
        argmax。
        """
        n_lcol = len(Codebook.LIGHT_COLORS) if n_lcol is None else n_lcol
        n_ldir = len(Codebook.LIGHT_DIRS) if n_ldir is None else n_ldir
        kind_idx = int(
            mx.argmax(
                SceneReconstructor.marginal_joint(
                    posterior, "kind", n_kind, n_hue, n_lcol, n_ldir
                )
            )
        )
        hue = int(
            mx.argmax(
                SceneReconstructor.marginal_joint(
                    posterior, "hue", n_kind, n_hue, n_lcol, n_ldir
                )
            )
        )
        lighting = SceneReconstructor.marginal_joint(
            posterior, "lighting", n_kind, n_hue, n_lcol, n_ldir
        )
        li = int(mx.argmax(lighting))
        return kind_idx, hue, li // n_ldir, li % n_ldir

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
        marginalize: bool = False,
    ) -> tuple[
        tuple[float, ...], tuple[tuple[float, ...], ...], mx.array, mx.array, float
    ]:
        """top-k kind × 54 外观候选的联合渲染后验。

        结构评分沿用共享几何, 避免把 kind 选择过度耦合到面积代理偏差;
        候选返回前再按各自 kind 重校准 s (保留 SPN 学到的残差)。后验
        log 形式 = −残差/T + log P(kind|SPN)。温度
        T=max(2·best_residual,1) 用最佳残差估计观测噪声尺度并随
        StructuredHypothesis 返回; 该后验表达候选间相对置信度, 不声称绝对
        校准。marginalize=True 时 MAP 改为各因子边缘 argmax 的解耦估计
        (因果不变估计, 见 marginal_joint), 否则沿用联合 argmax。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        kind_topk = max(1, min(kind_topk, Codebook.N_KIND))
        order = cast(list, mx.argsort(kind_p)[::-1][:kind_topk].tolist())
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
        if marginalize:
            # 解耦边缘 MAP: 反照率对光照、光照对反照率/几何分别边缘化
            ki, hi, ci, di = cls.decoupled_map(posterior, len(order))
            kind = int(order[ki])
            best = (
                float(kind),
                base_params[1],
                base_params[2],
                float(cls.s_proxy(kind, stats)[0]) + s_resid,
                base_params[4],
                float(hi),
                float(ci),
                float(di),
            )
        else:
            best_i = int(mx.argmax(posterior))
            best = calibrated[best_i]
        return best, calibrated, score_arr, posterior, temperature

    @staticmethod
    def em_refine(
        app: InverseApp,
        prm: tuple[float, ...],
        fl: mx.array,
        fr: mx.array,
    ) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
        """几何↔光照 ECM 精炼 (u,v,s,z), kind 固定。

        返回 (refined_prm, trajectory)。app.cfg.em_refine 为 False 时
        直接返回原 prm + None 轨迹。外观 (hue/lcol/ldir) 沿用 prm。
        """
        if not app.cfg.em_refine:
            return prm, None
        from generic_em import EMLoop
        from scene_em_refiner import SceneEMRefiner

        kind = int(prm[0])
        refiner = SceneEMRefiner(
            app.codebook,
            kind,
            fl,
            fr,
            appearance_topk=app.cfg.em_appearance_topk,
        )
        em = EMLoop(
            refiner,
            max_iters=app.cfg.em_max_iters,
            tol=app.cfg.em_tolerance,
        )
        res = em.run((fl, fr), tuple(prm[1:5]))
        return (float(kind), *res.params, prm[5], prm[6], prm[7]), res.trajectory

    @staticmethod
    def frame_features(
        app: InverseApp,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
    ) -> tuple[mx.array, mx.array, RieszWavelet | None]:
        """左/右帧 → (模型特征 (1,V), 立体统计 (1,3), Riesz 工作区)。"""
        vec, rw = app.extractor.of_frame(fl, rw)
        if str(app.codebook.GEOMETRY_FAMILY) == "lateral":
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
        sizes = cls.cat_sizes(app.cfg.n_textures)
        prm = cls.params(t, cat_p, stats, sizes)[0]
        rn, ent, novelty = cls.novelty_metrics(
            cat_p[0], r, sizes, None
        )
        if not refine or app.cfg.textured:
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
            marginalize=app.cfg.appearance_marginalize,
        )
        # 推理期几何↔光照 ECM 精炼 (§7.1): 默认关闭。kind 固定, 只精炼
        # 连续几何 (u,v,s,z); 外观 (hue/lcol/ldir) 沿用 refine_scene 的 MAP。
        prm, em_trajectory = cls.em_refine(app, prm, fl, fr)
        order = cast(list, mx.argsort(posterior)[::-1][:5].tolist())
        hypotheses = tuple(
            HypothesisCandidate(candidates[i], float(posterior[i]), float(scores[i]))
            for i in order
        )
        best_residual = float(mx.min(scores))
        rn, ent, novelty = cls.novelty_metrics(
            cat_p[0], r, sizes, best_residual
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
            em_trajectory=em_trajectory,
        )

    @staticmethod
    def rig() -> tuple[Renderer, PerspectiveCamera, PerspectiveCamera]:
        """公开训练 rig: 调用方可用它获取模型输入的左右二维图像。"""
        return Codebook.make_renderer()
