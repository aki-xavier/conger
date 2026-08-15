"""StructureGeometry: 结构门控的观测级几何兼容性证据。

渲染残差只回答“哪个专家当前参数画得像”, 不直接回答“观测更像哪种
结构”。这里从同一左右图提取三类低维结构证据: 单模板紧致性、前后层
视差分离、attached_on_top 组合接触关系, 供 StructureGate 作为
MDL/几何惩罚项。
"""

from __future__ import annotations

import math

import mlx.core as mx

from composite_geometry import CompositeGeometry
from joint_layer_optimizer import JointLayerOptimizer
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
