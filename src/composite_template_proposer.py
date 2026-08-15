"""CompositeTemplateProposer: 残差驱动的附着组合模板提案。

未知结构出生时不直接训练所有几何组合; 先用现有最佳估计作为底座,
枚举少量 part kind/hue × 尺度比例 × 横向偏移, 通过同一 cga renderer
的左右图残差筛选候选。提案只进入 StructureBirthRequest, 训练仍需显式触发。
"""

from __future__ import annotations

import math

import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from inverse_config import InverseConfig
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth
from structure_birth import StructureCase
from template_proposal import TemplateProposal


class CompositeTemplateProposer:
    """单/双层最佳估计 → attached_on_top 组合模板候选。"""

    def __init__(
        self,
        complexity_weight: float = 1.0,
        ratios: tuple[float, ...] = (0.45, 0.60),
        lateral_ratios: tuple[float, ...] = (-0.20, 0.0, 0.20),
        part_kinds: tuple[int, ...] | None = None,
        part_hues: tuple[int, ...] | None = None,
        max_cases: int = 4,
        max_proposals: int = 5,
    ):
        self.complexity_weight = complexity_weight
        self.ratios = ratios
        self.lateral_ratios = lateral_ratios
        self.part_kinds = part_kinds or tuple(range(Codebook.N_KIND))
        self.part_hues = part_hues or tuple(range(Codebook.N_HUE))
        self.max_cases = max_cases
        self.max_proposals = max_proposals
        self.codebook = CompositeCodebook(InverseConfig(scene_family="composite"))

    @staticmethod
    def _base_from_params(params: tuple[float, ...]) -> tuple[float, ...] | None:
        """单物体 8 维或双图元 14 维估计 → (k,u,v,s,z,hue,lcol,ldir)。"""
        if len(params) == 8:
            return params
        if len(params) == 14:
            return params[:6] + params[12:14]
        return None

    @staticmethod
    def _attach(
        base: tuple[float, ...],
        part_kind: int,
        part_hue: int,
        ratio: float,
        lateral_ratio: float,
    ) -> tuple[float, ...]:
        """底座 + attached_on_top 关系 → 14 维 composite 参数。"""
        k0, u0, v0, s0, z0, h0, lcol, ldir = base
        s1 = s0 * ratio
        z1 = z0
        x0, y0 = Codebook.unproject(u0, v0, z0)
        x1 = x0 + lateral_ratio * (s0 + s1)
        y1 = y0 + s0 + s1 - 0.05 * min(s0, s1)
        zc1 = Codebook.CAM_Z - z1
        u1 = (Codebook.W - 1) / 2.0 + x1 * Codebook.FX / zc1
        v1 = (Codebook.H - 1) / 2.0 - y1 * Codebook.FY / zc1
        return (
            float(k0), float(u0), float(v0), float(s0), float(z0), float(h0),
            float(part_kind), float(u1), float(v1), float(s1), float(z1),
            float(part_hue), float(lcol), float(ldir),
        )

    def _render_residual(
        self,
        params: tuple[float, ...],
        fl: mx.array,
        fr: mx.array,
        renderer,
        cam_l,
        cam_r,
    ) -> float:
        """候选 composite 重渲染左右视图 → 前景加权 RGB MSE。"""
        scene = self.codebook.to_scene(params)
        cl = renderer.render(scene, cam_l)
        cr = renderer.render(scene, cam_r)
        wl = StereoDepth.foreground_weights(fl)
        wr = StereoDepth.foreground_weights(fr)
        return 0.5 * (
            SceneReconstructor._masked_mse(fl, cl, wl)
            + SceneReconstructor._masked_mse(fr, cr, wr)
        )

    def _propose_case(
        self, case: StructureCase, case_index: int
    ) -> list[TemplateProposal]:
        base = self._base_from_params(tuple(float(x) for x in case.params))
        if base is None or not hasattr(case.fl, "shape"):
            return []
        fl = case.fl
        fr = case.fr
        baseline = min(case.residuals.values()) if case.residuals else math.inf
        renderer, cam_l, cam_r = Codebook.make_renderer()
        out = []
        for part_kind in self.part_kinds:
            for part_hue in self.part_hues:
                for ratio in self.ratios:
                    for lateral in self.lateral_ratios:
                        params = self._attach(
                            base, part_kind, part_hue, ratio, lateral
                        )
                        if not CompositeCodebook._inside(
                            params[7], params[8], params[9], params[10]
                        ):
                            continue
                        residual = self._render_residual(
                            params, fl, fr, renderer, cam_l, cam_r
                        )
                        score = (
                            residual
                            + self.complexity_weight
                            * CompositeCodebook.TEMPLATE_COMPLEXITY
                        )
                        out.append(
                            TemplateProposal(
                                family="composite",
                                operation="attach",
                                params=params,
                                residual=residual,
                                complexity=CompositeCodebook.TEMPLATE_COMPLEXITY,
                                score=score,
                                metadata={
                                    "relation": CompositeCodebook.RELATION,
                                    "base_kind": int(base[0]),
                                    "part_kind": part_kind,
                                    "part_hue": part_hue,
                                    "ratio": ratio,
                                    "lateral_ratio": lateral,
                                    "case_index": case_index,
                                    "residual_gain": baseline - residual,
                                },
                            )
                        )
        return out

    def propose(
        self, cases: tuple[StructureCase, ...]
    ) -> tuple[TemplateProposal, ...]:
        """聚合前 max_cases 个出生样本, 返回统一排序的 top 提案。"""
        proposals = []
        for i, case in enumerate(cases[: self.max_cases]):
            proposals.extend(self._propose_case(case, i))
        proposals.sort(key=lambda p: p.score)
        return tuple(proposals[: self.max_proposals])
