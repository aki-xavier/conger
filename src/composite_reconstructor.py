"""CompositeReconstructor: 附着组合模板的部分感知参数解码。

CompositeGeometry 提供 base/part 的 [u,v,z,area] 锚点; MixtureSPN 只学习
相对这些锚点的有界残差。这样组合物仍是一个结构专家, 但几何估计不再
被单一全局质心/面积稀释。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from layered_reconstructor import LayeredReconstructor
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth
from structured_hypothesis import HypothesisCandidate, StructuredHypothesis

if TYPE_CHECKING:
    from inverse_app import InverseApp
    from mixture_spn import MixtureSPN
    from riesz import RieszWavelet


class CompositeReconstructor(LayeredReconstructor):
    """双层参数宽度 + base/part 附着关系的几何/类别解码。"""

    CAT_SIZES = CompositeCodebook.CAT_SIZES
    RESIDUAL_SCALE = (1.0,) * 8  # 组合两部分都有模板/视差锚点

    @classmethod
    def params(
        cls,
        t_pred: mx.array,
        cat_p: mx.array,
        stats: mx.array | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """8 维残差 + 6 离散头 MAP → 14 维组合场景参数。"""
        return super().params(t_pred, cat_p, stats)

    @staticmethod
    def _topk(p: mx.array, k: int) -> list[int]:
        """单头后验 → top-k 下标。"""
        k = max(1, min(k, p.shape[0]))
        return cast(list[int], mx.argsort(p)[::-1][:k].tolist())

    @classmethod
    def refine_scene(
        cls,
        codebook: CompositeCodebook,
        base_params: tuple[float, ...],
        cat_p: mx.array,
        fl: mx.array,
        fr: mx.array,
        kind_topk: int = 2,
        hue_topk: int = 1,
        light_topk: int = 1,
        renderer=None,
        cam_l=None,
        cam_r=None,
    ) -> tuple[
        tuple[float, ...], tuple[tuple[float, ...], ...], mx.array, mx.array, float
    ]:
        """组合几何固定, top-k kind/hue/light 候选的渲染残差联合后验。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        heads = cls.split_cat(cat_p[None, :])
        tops = [
            cls._topk(heads[0][0], kind_topk),
            cls._topk(heads[1][0], kind_topk),
            cls._topk(heads[2][0], hue_topk),
            cls._topk(heads[3][0], hue_topk),
            cls._topk(heads[4][0], light_topk),
            cls._topk(heads[5][0], light_topk),
        ]
        candidates, weights, scores = [], [], []
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        for k0 in tops[0]:
            for k1 in tops[1]:
                for h0 in tops[2]:
                    for h1 in tops[3]:
                        for lc in tops[4]:
                            for ld in tops[5]:
                                prm = (
                                    float(k0), *base_params[1:5], float(h0),
                                    float(k1), *base_params[7:11], float(h1),
                                    float(lc), float(ld),
                                )
                                scene = codebook.to_scene(prm)
                                cl = renderer.render(scene, cam_l)
                                cr = renderer.render(scene, cam_r)
                                score = 0.5 * (
                                    SceneReconstructor._masked_mse(fl, cl, wl)
                                    + SceneReconstructor._masked_mse(fr, cr, wr)
                                )
                                candidates.append(prm)
                                scores.append(score)
                                weights.append(
                                    float(heads[0][0, k0])
                                    * float(heads[1][0, k1])
                                    * float(heads[2][0, h0])
                                    * float(heads[3][0, h1])
                                    * float(heads[4][0, lc])
                                    * float(heads[5][0, ld])
                                )
        score_arr = mx.array(scores, dtype=mx.float32)
        weight_arr = mx.maximum(mx.array(weights, dtype=mx.float32), 1e-12)
        temperature = max(2.0 * float(mx.min(score_arr)), 1.0)
        logp = -score_arr / temperature + mx.log(weight_arr)
        posterior = mx.exp(logp - mx.logsumexp(logp))
        best_i = int(mx.argmax(posterior))
        return (
            candidates[best_i], tuple(candidates), score_arr, posterior, temperature
        )

    @classmethod
    def from_frames(
        cls,
        app: InverseApp,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
        refine: bool = False,
    ) -> StructuredHypothesis:
        """左/右二维图像 → 附着组合 StructuredHypothesis。"""
        f, stats, _ = SceneReconstructor.frame_features(app, fl, fr, rw)
        t, cat_p, r = net.predict(f)
        prm = cls.params(t, cat_p, stats)[0]
        candidates = (prm,)
        scores = None
        posterior = None
        temperature = None
        hypotheses = (HypothesisCandidate(prm, 1.0, None),)
        residual = None
        if refine:
            prm, candidates, scores, posterior, temperature = cls.refine_scene(
                app.codebook,
                prm,
                cat_p[0],
                fl,
                fr,
            )
            order = cast(list[int], mx.argsort(posterior)[::-1][:5].tolist())
            hypotheses = tuple(
                HypothesisCandidate(
                    candidates[i], float(posterior[i]), float(scores[i])
                )
                for i in order
            )
            residual = float(mx.min(scores))
        _, ent, novelty = SceneReconstructor.novelty_metrics(
            cat_p[0], r, cls.CAT_SIZES, residual
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
            factor_sizes=CompositeCodebook.CAT_SIZES,
            factor_indices=CompositeCodebook.CLASS_IDX,
            responsibility_max=float(mx.max(r)),
            posterior_entropy=ent,
            residual=residual,
            complexity=CompositeCodebook.TEMPLATE_COMPLEXITY,
            novelty_score=novelty,
        )

    @staticmethod
    def targets_from_params(
        params: tuple[tuple[float, ...], ...]
    ) -> mx.array:
        """组合场景参数 → 连续评估目标 (8 维)。"""
        return LayeredReconstructor.targets_from_params(params)
