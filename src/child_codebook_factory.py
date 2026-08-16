"""ChildCodebookFactory: ChildTemplateSpec → 可训练子 Codebook 类。

支持 TemplateGrammar 的 attach/layer/mirror/repeat 子模板。动态类的
TEMPLATE_VARIANT 进入缓存指纹, TEMPLATE_LINEAGE 保留 parent/delta。
"""

from __future__ import annotations

from composite_codebook import CompositeCodebook
from lateral_codebook import LateralCompositeCodebook
from layered_codebook import LayeredCodebook
from template_lineage import ChildTemplateSpec, TemplateLineage


class ChildCodebookFactory:
    """把学习到的 delta 约束转换为显式场景族类。"""

    @staticmethod
    def _discrete_sets(spec: ChildTemplateSpec) -> tuple[tuple, tuple]:
        c = spec.constraints
        return (
            tuple(c.get("part_kinds", CompositeCodebook.PART_KINDS)),
            tuple(c.get("part_hues", CompositeCodebook.PART_HUES)),
        )

    @classmethod
    def _attach(cls, spec: ChildTemplateSpec) -> type[CompositeCodebook]:
        c = spec.constraints
        part_kinds, part_hues = cls._discrete_sets(spec)
        attrs = {
            "SCALE_RATIO": tuple(
                c.get("scale_ratio", CompositeCodebook.SCALE_RATIO)
            ),
            "LATERAL_RANGE": tuple(
                c.get("lateral_ratio", CompositeCodebook.LATERAL_RANGE)
            ),
            "DEPTH_JITTER": tuple(
                c.get("depth_jitter", CompositeCodebook.DEPTH_JITTER)
            ),
            "PART_KINDS": part_kinds,
            "PART_HUES": part_hues,
            "N_COMBO": (
                len(CompositeCodebook.BASE_KINDS)
                * len(part_kinds)
                * len(CompositeCodebook.BASE_HUES)
                * len(part_hues)
                * CompositeCodebook.N_LIGHT_COLORS
                * CompositeCodebook.N_LIGHT_DIRS
            ),
            "TEMPLATE_VARIANT": spec.name,
            "TEMPLATE_LINEAGE": spec.lineage(),
            "__module__": "child_codebook_factory",
        }
        return type(spec.name, (CompositeCodebook,), attrs)

    @classmethod
    def _layer(cls, spec: ChildTemplateSpec) -> type[LayeredCodebook]:
        c = spec.constraints
        part_kinds, part_hues = cls._discrete_sets(spec)
        # 后层必须至少部分可见: 近对齐 (lateral≈0) 会让后层投影完全落在
        # 前层投影内被遮挡, 退化成单物体 (layer child → single 混淆根因)。
        # 强制横向偏移幅度 ≥0.35 (后层投影越出前层), 保证层结构可观测。
        lateral = tuple(c.get("lateral_ratio", (-0.75, 0.75)))
        if max(abs(lateral[0]), abs(lateral[1])) < 0.35:
            lateral = (0.35, 0.7)
        lineage = spec.lineage()
        if lateral != tuple(c.get("lateral_ratio", (-0.75, 0.75))):
            # 加宽后的横向范围写回血缘 delta, 使门控 delta_cost 与采样一致
            # (否则 delta_cost 仍用旧 ~0 lateral 约束惩罚子模板自己采出的样本)
            delta = dict(spec.constraints)
            delta["lateral_ratio"] = lateral
            lineage = TemplateLineage(
                family=spec.name,
                parent_family=spec.parent_family,
                operation=spec.operation,
                complexity=spec.complexity,
                generation=spec.generation,
                delta=delta,
            )
        attrs = {
            "PART_SCALE_RANGE": tuple(c.get("scale_ratio", (0.35, 0.75))),
            "PART_LATERAL_RANGE": lateral,
            "DEPTH_GAP_RANGE": tuple(c.get("depth_gap", (0.7, 1.4))),
            "PART_KINDS": part_kinds,
            "PART_HUES": part_hues,
            "N_COMBO": (
                len(LayeredCodebook.BASE_KINDS)
                * len(part_kinds)
                * len(LayeredCodebook.BASE_HUES)
                * len(part_hues)
                * LayeredCodebook.N_LIGHT_COLORS
                * LayeredCodebook.N_LIGHT_DIRS
            ),
            "TEMPLATE_VARIANT": spec.name,
            "STEREO_V": "sl8c1",  # 受限 layer 的全残差目标契约
            "TEMPLATE_LINEAGE": lineage,
            "__module__": "child_codebook_factory",
        }
        return type(spec.name, (LayeredCodebook,), attrs)

    @classmethod
    def _lateral(cls, spec: ChildTemplateSpec) -> type[LateralCompositeCodebook]:
        c = spec.constraints
        part_kinds, part_hues = cls._discrete_sets(spec)
        spacing = LateralCompositeCodebook.spacing_factor(spec.operation)
        attrs = {
            "SCALE_RATIO": tuple(c.get("scale_ratio", (0.35, 0.75))),
            "PART_PERIOD_RANGE": tuple(c.get("period_ratio", (0.15, 0.25))),
            "SPACING_FACTOR": spacing,
            "RELATION": spec.operation,
            "BASE_KINDS": part_kinds,
            "PART_KINDS": part_kinds,
            "BASE_HUES": part_hues,
            "PART_HUES": part_hues,
            "N_COMBO": (
                len(part_kinds)
                * len(part_hues)
                * LateralCompositeCodebook.N_LIGHT_COLORS
                * LateralCompositeCodebook.N_LIGHT_DIRS
            ),
            "TEMPLATE_VARIANT": spec.name,
            "TEMPLATE_LINEAGE": spec.lineage(),
            "__module__": "child_codebook_factory",
        }
        return type(spec.name, (LateralCompositeCodebook,), attrs)

    @classmethod
    def build(cls, spec: ChildTemplateSpec) -> type:
        """按 operation 物化对应子 Codebook。"""
        if spec.operation == "attach":
            return cls._attach(spec)
        if spec.operation == "layer":
            return cls._layer(spec)
        if spec.operation in {"mirror", "repeat"}:
            return cls._lateral(spec)
        raise ValueError(f"未知子模板操作: {spec.operation}")
