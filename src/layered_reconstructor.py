"""LayeredReconstructor: 双图元遮挡场景的参数解码与 StructuredHypothesis。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from codebook import Codebook
from layered_codebook import LayeredCodebook
from scene_reconstructor import SceneReconstructor
from structured_hypothesis import HypothesisCandidate, StructuredHypothesis

if TYPE_CHECKING:
    from inverse_app import InverseApp
    from mixture_spn import MixtureSPN
    from riesz import RieszWavelet


class LayeredReconstructor:
    """双层 SPN 输出 → 完整 cga.Scene 参数/对象。

    第一版遮挡路径不做渲染残差精炼: 双层遮挡下外观候选为
    kind²×hue²×lcol×ldir=2916, 需要先建立分层几何/遮挡校验。
    返回 StructuredHypothesis 保留 SPN 联合后验, 避免过早 argmax。"""

    CAT_SIZES = LayeredCodebook.CAT_SIZES
    # 前层通常完整可见, 允许全部残差; 后层被遮挡, 低密度 SPN 的
    # s/z 残差会放大可见面积歧义, 先采用双目锚点 (实测优于校准)
    RESIDUAL_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    # 遮挡锚点残差有界, 但可见面积代理在强遮挡下仍可能过小 → 残差到
    # 负下限时 s 可塌成负值 (cga 几何拒绝 radius≤0)。这里只做物理下限
    # 钳制防崩溃, 不改变正常样本估计。
    S_FLOOR = 0.05
    Z_MIN = 0.5
    Z_MAX = Codebook.CAM_Z - 0.5

    @classmethod
    def split_cat(cls, cat_p: mx.array) -> tuple[mx.array, ...]:
        """拼接后验 (N,24) → k0,k1,h0,h1,lcol,ldir 六个头。"""
        out, lo = [], 0
        for nc in cls.CAT_SIZES:
            out.append(cat_p[:, lo : lo + nc])
            lo += nc
        return tuple(out)

    @staticmethod
    def _proxy(kind: int, stats: mx.array, off: int) -> float:
        """一层 [u,v,z,area] 统计 → kind-conditioned 几何尺寸代理。"""
        st = mx.array(
            [[stats[off + 2], 0.0, stats[off + 3]]], dtype=mx.float32
        )
        return float(SceneReconstructor.s_proxy(kind, st)[0])

    @classmethod
    def residual_targets(
        cls, t: mx.array, classes: mx.array, stats: mx.array
    ) -> mx.array:
        """物理连续目标 − 逐层双目观测锚点 (训练用)。"""
        out = []
        for i in range(t.shape[0]):
            st = stats[i]
            p0 = cls._proxy(int(classes[i, 0]), st, 0)
            p1 = cls._proxy(int(classes[i, 1]), st, 4)
            vals = [
                t[i, 0] - st[0],
                t[i, 1] - st[1],
                t[i, 2] - p0,
                t[i, 3] - st[2],
                t[i, 4] - st[4],
                t[i, 5] - st[5],
                t[i, 6] - p1,
                t[i, 7] - st[6],
            ]
            out.append(
                [v * s for v, s in zip(vals, cls.RESIDUAL_SCALE, strict=True)]
            )
        return mx.array(out, dtype=mx.float32)

    @classmethod
    def params(
        cls,
        t_pred: mx.array,
        cat_p: mx.array,
        stats: mx.array | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """连续残差/直读目标 + 离散头 MAP → 14 维双层场景参数。"""
        probs = [mx.argmax(p, axis=1).astype(mx.int32) for p in cls.split_cat(cat_p)]
        rows = []
        for i in range(t_pred.shape[0]):
            t = t_pred[i].tolist()
            if stats is None:
                geom = t
            else:
                st = stats[i]
                # 遮挡锚点校正有界: 低密度 SPN 的野性残差不应覆盖
                # 逐层视差提供的物理量 (阈值比训练分布边距宽一档)
                r = [
                    min(max(t[0], -25.0), 25.0),
                    min(max(t[1], -25.0), 25.0),
                    min(max(t[2], -0.25), 0.25),
                    min(max(t[3], -0.5), 0.5),
                    min(max(t[4], -25.0), 25.0),
                    min(max(t[5], -25.0), 25.0),
                    min(max(t[6], -0.25), 0.25),
                    min(max(t[7], -0.5), 0.5),
                ]
                r = [
                    v * s
                    for v, s in zip(r, cls.RESIDUAL_SCALE, strict=True)
                ]
                geom = [
                    r[0] + float(st[0]),
                    r[1] + float(st[1]),
                    max(r[2] + cls._proxy(int(probs[0][i]), st, 0), cls.S_FLOOR),
                    min(max(r[3] + float(st[2]), cls.Z_MIN), cls.Z_MAX),
                    r[4] + float(st[4]),
                    r[5] + float(st[5]),
                    max(r[6] + cls._proxy(int(probs[1][i]), st, 4), cls.S_FLOOR),
                    min(max(r[7] + float(st[6]), cls.Z_MIN), cls.Z_MAX),
                ]
            rows.append(
                (
                    float(probs[0][i]), geom[0], geom[1], geom[2], geom[3],
                    float(probs[2][i]),
                    float(probs[1][i]), geom[4], geom[5], geom[6], geom[7],
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
    ) -> StructuredHypothesis:
        """左/右二维图像 → 双层 StructuredHypothesis (SPN 后验, 无渲染精炼)。"""
        f, stats, _ = SceneReconstructor.frame_features(app, fl, fr, rw)
        t, cat_p, r = net.predict(f)
        prm = cls.params(t, cat_p, stats)[0]
        rn, ent, novelty = SceneReconstructor.novelty_metrics(
            cat_p[0], r, cls.CAT_SIZES, None
        )
        return StructuredHypothesis(
            scene=app.codebook.to_scene(prm),
            params=prm,
            spn_posterior=cat_p[0],
            geometry_family=app.codebook.GEOMETRY_FAMILY,
            template_delta=app.codebook.TEMPLATE_LINEAGE.delta,
            candidate_params=(prm,),
            hypotheses=(HypothesisCandidate(prm, 1.0, None),),
            factor_sizes=LayeredCodebook.CAT_SIZES,
            factor_indices=LayeredCodebook.CLASS_IDX,
            responsibility_max=float(mx.max(r)),
            posterior_entropy=ent,
            residual=None,
            complexity=app.codebook.TEMPLATE_COMPLEXITY,
            novelty_score=novelty,
        )

    @staticmethod
    def targets_from_params(
        params: tuple[tuple[float, ...], ...]
    ) -> mx.array:
        """双层场景参数 → 连续评估目标 (8 维)。"""
        idx = LayeredCodebook.TARGET_IDX
        return mx.array([[p[j] for j in idx] for p in params], dtype=mx.float32)
