"""CompositeReconstructor: 附着组合模板的 SPN 输出解码。

第一层组合模板使用全局立体锚点作特征, 但连续目标仍直读 8 个几何量;
不把 LayeredCodebook 的逐层遮挡锚点误套到单一组合物上。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from composite_codebook import CompositeCodebook
from layered_reconstructor import LayeredReconstructor
from scene_reconstructor import SceneReconstructor
from structured_hypothesis import HypothesisCandidate, StructuredHypothesis

if TYPE_CHECKING:
    from inverse_app import InverseApp
    from mixture_spn import MixtureSPN
    from riesz import RieszWavelet


class CompositeReconstructor:
    """双层参数宽度 + 单一组合物语义的几何/类别解码。"""

    CAT_SIZES = CompositeCodebook.CAT_SIZES

    @classmethod
    def params(
        cls,
        t_pred: mx.array,
        cat_p: mx.array,
    ) -> tuple[tuple[float, ...], ...]:
        """8 连续直读目标 + 6 离散头 MAP → 14 维组合场景参数。"""
        return LayeredReconstructor.params(t_pred, cat_p, stats=None)

    @classmethod
    def from_frames(
        cls,
        app: InverseApp,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
    ) -> StructuredHypothesis:
        """左/右二维图像 → 附着组合 StructuredHypothesis。"""
        f, _, _ = SceneReconstructor.frame_features(app, fl, fr, rw)
        t, cat_p, r = net.predict(f)
        prm = cls.params(t, cat_p)[0]
        _, ent, novelty = SceneReconstructor.novelty_metrics(
            cat_p[0], r, cls.CAT_SIZES, None
        )
        return StructuredHypothesis(
            scene=app.codebook.to_scene(prm),
            params=prm,
            spn_posterior=cat_p[0],
            candidate_params=(prm,),
            hypotheses=(HypothesisCandidate(prm, 1.0, None),),
            factor_sizes=CompositeCodebook.CAT_SIZES,
            factor_indices=CompositeCodebook.CLASS_IDX,
            responsibility_max=float(mx.max(r)),
            posterior_entropy=ent,
            residual=None,
            complexity=CompositeCodebook.TEMPLATE_COMPLEXITY,
            novelty_score=novelty,
        )

    @staticmethod
    def targets_from_params(
        params: tuple[tuple[float, ...], ...]
    ) -> mx.array:
        """组合场景参数 → 连续评估目标 (8 维)。"""
        return LayeredReconstructor.targets_from_params(params)
