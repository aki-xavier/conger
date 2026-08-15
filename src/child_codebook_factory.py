"""ChildCodebookFactory: ChildTemplateSpec → 可训练子 Codebook 类。

第一阶段只物化 attach 子模板: 约束来自 TemplateDeltaLearner 的
ChildTemplateSpec, 类仍继承 CompositeCodebook 的 renderer/参数契约;
SAMPLE/缓存指纹由 TEMPLATE_VARIANT 区分。
"""

from __future__ import annotations

from composite_codebook import CompositeCodebook
from template_lineage import ChildTemplateSpec


class ChildCodebookFactory:
    """把学习到的 delta 约束转换为显式场景族类。"""

    @staticmethod
    def build(spec: ChildTemplateSpec) -> type[CompositeCodebook]:
        """attach ChildTemplateSpec → 动态 CompositeCodebook 子类。"""
        if spec.operation != "attach":
            raise ValueError(f"暂不支持由 {spec.operation} 物化子 Codebook")
        c = spec.constraints
        scale_ratio = tuple(c.get("scale_ratio", CompositeCodebook.SCALE_RATIO))
        lateral_range = tuple(c.get("lateral_ratio", CompositeCodebook.LATERAL_RANGE))
        depth_jitter = tuple(c.get("depth_jitter", CompositeCodebook.DEPTH_JITTER))
        part_kinds = tuple(c.get("part_kinds", CompositeCodebook.PART_KINDS))
        part_hues = tuple(c.get("part_hues", CompositeCodebook.PART_HUES))
        n_combo = (
            len(CompositeCodebook.BASE_KINDS)
            * len(part_kinds)
            * len(CompositeCodebook.BASE_HUES)
            * len(part_hues)
            * CompositeCodebook.N_LIGHT_COLORS
            * CompositeCodebook.N_LIGHT_DIRS
        )
        attrs = {
            "SCALE_RATIO": scale_ratio,
            "LATERAL_RANGE": lateral_range,
            "DEPTH_JITTER": depth_jitter,
            "PART_KINDS": part_kinds,
            "PART_HUES": part_hues,
            "N_COMBO": n_combo,
            "TEMPLATE_VARIANT": spec.name,
            "TEMPLATE_LINEAGE": spec.lineage(),
            "__module__": "child_codebook_factory",
        }
        return type(spec.name, (CompositeCodebook,), attrs)
