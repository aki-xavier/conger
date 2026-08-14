"""LayeredReconstructor: 双图元遮挡场景的参数解码与 SceneEstimate。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from layered_codebook import LayeredCodebook
from scene_estimate import SceneEstimate, SceneHypothesis
from scene_reconstructor import SceneReconstructor

if TYPE_CHECKING:
    from inverse_app import InverseApp
    from mixture_spn import MixtureSPN
    from riesz import RieszWavelet


class LayeredReconstructor:
    """双层 SPN 输出 → 完整 cga.Scene 参数/对象。

    第一版遮挡路径不做渲染残差精炼: 双层遮挡下外观候选为
    kind²×hue²×lcol×ldir=2916, 需要先建立分层几何/遮挡校验。
    返回 SceneEstimate 保留 SPN 联合后验, 避免过早 argmax。"""

    CAT_SIZES = LayeredCodebook.CAT_SIZES

    @classmethod
    def split_cat(cls, cat_p: mx.array) -> tuple[mx.array, ...]:
        """拼接后验 (N,24) → k0,k1,h0,h1,lcol,ldir 六个头。"""
        out, lo = [], 0
        for nc in cls.CAT_SIZES:
            out.append(cat_p[:, lo : lo + nc])
            lo += nc
        return tuple(out)

    @classmethod
    def params(
        cls, t_pred: mx.array, cat_p: mx.array
    ) -> tuple[tuple[float, ...], ...]:
        """连续目标 + 离散头 MAP → 14 维双层场景参数。"""
        probs = [mx.argmax(p, axis=1).astype(mx.int32) for p in cls.split_cat(cat_p)]
        rows = []
        for i in range(t_pred.shape[0]):
            t = t_pred[i].tolist()
            rows.append(
                (
                    float(probs[0][i]), t[0], t[1], t[2], t[3],
                    float(probs[2][i]),
                    float(probs[1][i]), t[4], t[5], t[6], t[7],
                    float(probs[3][i]),
                    float(probs[4][i]), float(probs[5][i]),
                )
            )
        return tuple(rows)

    @classmethod
    def from_frames(
        cls,
        app: InverseApp,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
        rw: RieszWavelet | None = None,
    ) -> SceneEstimate:
        """左/右二维图像 → 双层 SceneEstimate (SPN 后验, 无渲染精炼)。"""
        f, _, _ = SceneReconstructor.frame_features(app, fl, fr, rw)
        t, cat_p, _ = net.predict(f)
        prm = cls.params(t, cat_p)[0]
        return SceneEstimate(
            scene=app.codebook.to_scene(prm),
            params=prm,
            spn_posterior=cat_p[0],
            candidate_params=(prm,),
            hypotheses=(SceneHypothesis(prm, 1.0, None),),
            factor_sizes=LayeredCodebook.CAT_SIZES,
            factor_indices=LayeredCodebook.CLASS_IDX,
        )

    @staticmethod
    def targets_from_params(
        params: tuple[tuple[float, ...], ...]
    ) -> mx.array:
        """双层场景参数 → 连续评估目标 (8 维)。"""
        idx = LayeredCodebook.TARGET_IDX
        return mx.array([[p[j] for j in idx] for p in params], dtype=mx.float32)
