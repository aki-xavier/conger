"""StructureGeometry: 结构门控的观测级几何兼容性证据。

渲染残差只回答“哪个专家当前参数画得像”, 不直接回答“观测更像哪种
结构”。这里从同一左右图提取三类低维结构证据: 单模板紧致性、前后层
视差分离、attached_on_top 组合接触关系, 供 StructureGate 作为
MDL/几何惩罚项。
"""

from __future__ import annotations

import math

import mlx.core as mx

from codebook import Codebook
from composite_geometry import CompositeGeometry
from joint_layer_optimizer import JointLayerOptimizer
from lateral_codebook import LateralCompositeCodebook
from lateral_composite_geometry import LateralCompositeGeometry
from stereo import StereoDepth
from stereo_layers import StereoLayers
from utils import Utils


class StructureGeometry:
    """左右图 → 各结构族的几何代价/证据偏移 (越小越兼容)。"""

    @staticmethod
    def _iou_cost(template: mx.array, fg: mx.array) -> float:
        inter = mx.sum((template & fg).astype(mx.float32))
        union = mx.sum((template | fg).astype(mx.float32))
        return 1.0 - float(inter / mx.maximum(union, 1.0))

    @classmethod
    def _single_cost(cls, fg: mx.array, split_score: float | None) -> float:
        """单图元代价: 单模板 IoU + 细长结构/可拆分组合惩罚。"""
        fgd = JointLayerOptimizer._down_mask(fg)
        idx = Utils.nonzero(fgd.reshape(-1))
        if idx.shape[0] < 8:
            return 1.0
        h, w = fgd.shape
        xs, ys = idx % w, idx // w
        height = float(mx.max(ys) - mx.min(ys) + 1)
        width = float(mx.max(xs) - mx.min(xs) + 1)
        aspect = max(height / max(width, 1.0), width / max(height, 1.0))
        tmpl = JointLayerOptimizer._bbox_template(fgd, 0.0)
        cost = cls._iou_cost(tmpl.mask(h, w), fgd)
        # 当前单图元投影近似各向同性; 组合/双物体更容易出现细长外接框
        cost += 0.5 * min(max(aspect - 1.35, 0.0), 1.0)
        if split_score is not None:
            cost += 0.5 * min(max(0.35 - split_score, 0.0), 1.0)
        return min(max(cost, 0.0), 1.0)

    @staticmethod
    def _layered_cost(fl: mx.array, fr: mx.array, fg: mx.array) -> float:
        """双层代价: 有效视差必须能分出物理上有间距的前后层。"""
        disp, _, valid = StereoLayers.disparity_map(fl, fr)
        valid = valid & fg
        fw = StereoDepth.foreground_weights(fl)
        clustered = StereoLayers._cluster_layers(disp, fw, valid)
        if clustered is None:
            return 1.0
        _, c_front, c_back = clustered
        separation = c_front[2] - c_back[2]
        spatial = math.hypot(
            c_front[0] - c_back[0], c_front[1] - c_back[1]
        )
        # layered 不仅视差分离, 前后层中心也应可分辨; composite 的
        # base/part 中心被附着关系约束, 防止纹理噪声把单组合物拆成层
        separation_cost = min(max((2.5 - separation) / 2.5, 0.0), 1.0)
        spatial_cost = min(max((20.0 - spatial) / 20.0, 0.0), 1.0)
        return max(separation_cost, spatial_cost)

    @classmethod
    def _composite_cost(
        cls,
        fl: mx.array,
        fr: mx.array,
        split: tuple[float, object, object] | None,
    ) -> float:
        """组合代价: 接触线模板得分 + base/part 深度接近性。"""
        if split is None:
            return 1.0
        split_cost = split[0]
        st = CompositeGeometry.estimate(fl, fr)
        depth_gap = abs(st[2] - st[6])
        depth_cost = min(max((depth_gap - 0.15) / 0.5, 0.0), 1.0)
        return min(max(split_cost + 0.7 * depth_cost, 0.0), 1.0)

    @classmethod
    def _lateral_cost(cls, fl: mx.array, fr: mx.array, fg: mx.array) -> float:
        """横向组合代价: 垂直分隔模板得分 + 两部件深度接近性。"""
        split = LateralCompositeGeometry.split_score(fg)
        if split is None:
            return 1.0
        split_cost = split[0]
        st = LateralCompositeGeometry.estimate(fl, fr)
        depth_gap = abs(st[2] - st[6])
        depth_cost = min(max((depth_gap - 0.15) / 0.5, 0.0), 1.0)
        return min(max(split_cost + 0.7 * depth_cost, 0.0), 1.0)

    @staticmethod
    def geometry_stats(
        family: str, fl: mx.array, fr: mx.array
    ) -> tuple[float, ...] | None:
        """按基础结构族返回 [u,v,z,area]×2 统计 (single 无)。"""
        if family == "layered":
            return StereoLayers.estimate(fl, fr)
        if family == "composite":
            return CompositeGeometry.estimate(fl, fr)
        if family == "lateral":
            return LateralCompositeGeometry.estimate(fl, fr)
        return None

    @classmethod
    def delta_cost(
        cls,
        family: str,
        delta: dict[str, object] | None,
        stats: tuple[float, ...] | None,
        fl: mx.array | None = None,
        fr: mx.array | None = None,
    ) -> float:
        """子模板 delta 与观测几何的匹配/特异性代价。"""
        if not delta or stats is None:
            return 0.0
        u0, _, z0, a0, u1, _, z1, a1 = stats
        # area 是 π·r_px²; 除以 π 后开方 = 像素半径, 再换算世界半径 s
        # (sum 分母不抵消 sqrt(π), 必须修掉, 否则横向归一化间隔偏小 1/√π)
        q0 = (
            math.sqrt(max(a0, 1e-8) / math.pi)
            * (Codebook.CAM_Z - z0)
            / Codebook.FX
        )
        q1 = (
            math.sqrt(max(a1, 1e-8) / math.pi)
            * (Codebook.CAM_Z - z1)
            / Codebook.FX
        )
        observed_ratio = q1 / max(q0, 1e-8)
        cost = 0.0

        def range_term(
            key: str, observed: float, default: tuple[float, float]
        ) -> float:
            if key not in delta:
                return 0.0
            lo, hi = (float(x) for x in delta[key])
            outside = max(lo - observed, observed - hi, 0.0)
            if outside > 0.0:
                # 带外: 只按距离惩罚, 不给窄带特异性奖励 (否则窄带会
                # 在观测明显不匹配时仍吃到 log(width/default) 负证据)
                return 4.0 * outside
            width = max(hi - lo, 1e-6)
            default_width = default[1] - default[0]
            # 带内: 窄支持集获得负对数证据
            return 0.25 * math.log(width / default_width)

        cost += range_term("scale_ratio", observed_ratio, (0.35, 0.75))
        x_gap = abs(
            (u1 - u0) * (Codebook.CAM_Z - (z0 + z1) / 2.0) / Codebook.FX
        )
        lateral = x_gap / max(q0 + q1, 1e-8)
        relation = str(delta.get("relation", ""))
        if relation in {"mirror", "repeat"}:
            cost += cls.lateral_gap_cost(relation, delta, fl, fr)
        else:
            lateral_default = (
                (-0.25, 0.25)
                if relation == "attach"
                else (-0.75, 0.75)
            )
            cost += range_term(
                "lateral_ratio", lateral, lateral_default
            )
        return cost

    @classmethod
    def lateral_gap_cost(
        cls,
        relation: str,
        delta: dict[str, object] | None,
        fl: mx.array | None = None,
        fr: mx.array | None = None,
    ) -> float:
        """mirror/repeat 横向间隔判别证据 (越小越符合 relation)。

        用 kind 感知近端盖校正后的世界归一化间隔
        g = |x1-x0|/(s0+s1) (`LateralCompositeGeometry.corrected_gap`),
        消掉圆柱端盖投影的表观半径偏置, 再按 relation 的 spacing_factor
        还原 period; period 应落在学习到的 period_ratio 带内。带外按距离
        惩罚, 且当同一间隔按另一操作还原反而落入可行 period 带时, 额外
        加交叉判别惩罚。
        """
        if relation not in {"mirror", "repeat"} or fl is None or fr is None:
            return 0.0
        kind = 1  # 默认 cylinder
        if delta and delta.get("part_kinds"):
            kind = int(tuple(delta["part_kinds"])[0])
        g = LateralCompositeGeometry.corrected_gap(fl, fr, kind)
        if g is None:
            return 0.0  # 无法可靠分割 → 中性, 不误导门控
        spacing = LateralCompositeCodebook.spacing_factor(relation)
        other_spacing = LateralCompositeCodebook.spacing_factor(
            "repeat" if relation == "mirror" else "mirror"
        )
        learned = delta.get("period_ratio", ()) if delta else ()
        if isinstance(learned, (list, tuple)) and len(learned) == 2:
            lo, hi = float(learned[0]), float(learned[1])
        else:
            lo, hi = LateralCompositeCodebook.PART_PERIOD_RANGE
        own_p = g / spacing
        other_p = g / other_spacing
        own_out = max(lo - own_p, own_p - hi, 0.0)
        cost = 4.0 * own_out
        lo_feas, hi_feas = LateralCompositeCodebook.PART_PERIOD_RANGE
        if own_out > 0.0 and lo_feas <= other_p <= hi_feas:
            cost += 1.0
        return min(cost, 2.0)

    @classmethod
    def costs(cls, fl: mx.array, fr: mx.array) -> dict[str, float]:
        """同一观测 → single/layered/composite 三个几何代价。"""
        fw = StereoDepth.foreground_weights(fl)
        fg = fw > 0.01
        split = CompositeGeometry.split_score(fg)
        split_score = None if split is None else split[0]
        single = cls._single_cost(fg, split_score)
        layered = cls._layered_cost(fl, fr, fg)
        composite = cls._composite_cost(fl, fr, split)
        lateral = cls._lateral_cost(fl, fr, fg)
        # 横向组合证据明显强于 attach → 组合结构高代价: 左右并排样本会
        # 让 CompositeGeometry 退化出伪水平接触线, 把横向间隔误判成
        # attach 的零横向偏移, 必须在此拒绝。
        if lateral < composite - 0.15 and lateral < 0.4:
            composite = max(composite, 1.6)
        # 强结构证据作为负对数证据偏移: 只在没有前后层证据时奖励单模板,
        # 只有视差/空间双层都明确时奖励 layered, 只有接触模板很稳时奖励
        # composite。偏移用于打破渲染残差歧义, 阈值由三族基准样本标定。
        layered_raw = layered
        if single < 0.35 and layered_raw > 0.5:
            single -= 1.3
        lateral_blocks_layer = lateral < 0.35 and layered_raw > 0.05
        if not lateral_blocks_layer:
            if layered < composite - 0.03 and layered < single + 0.05:
                layered -= 1.0
            if layered + 0.05 < min(single, composite):
                layered -= 2.0
        if composite < 0.10:
            composite -= 0.2
        if lateral < 0.10:
            lateral -= 0.2
        return {
            "single": single,
            "layered": layered,
            "composite": composite,
            "lateral": lateral,
        }
