"""TemplateLineage 测试: 模板父/子血缘契约。"""

from codebook import Codebook
from composite_codebook import CompositeCodebook
from expert_registry import ExpertRegistry
from layered_codebook import LayeredCodebook
from template_lineage import TemplateLineage


class _LineageExpert:
    def __init__(self, lineage: TemplateLineage):
        self._lineage = lineage

    def lineage(self) -> TemplateLineage:
        return self._lineage


def test_codebook_template_lineages() -> None:
    """single 是根; layered 继承 single; composite 继承 layered。"""
    single = Codebook.TEMPLATE_LINEAGE
    layered = LayeredCodebook.TEMPLATE_LINEAGE
    composite = CompositeCodebook.TEMPLATE_LINEAGE
    assert single.is_root
    assert layered.parent_family == "single"
    assert composite.parent_family == "layered"
    assert composite.delta["relation"] == "attached_on_top"
    assert (single.generation, layered.generation, composite.generation) == (0, 1, 2)


def test_registry_lineage_tree() -> None:
    """注册表应暴露专家血缘和直接子模板查询。"""
    registry = ExpertRegistry(
        {
            "single": _LineageExpert(Codebook.TEMPLATE_LINEAGE),
            "layered": _LineageExpert(LayeredCodebook.TEMPLATE_LINEAGE),
            "composite": _LineageExpert(CompositeCodebook.TEMPLATE_LINEAGE),
        }
    )
    lineages = registry.lineages()
    assert lineages["composite"].signature() == "layered->composite:attach"
    assert registry.children_of("single") == ("layered",)
    assert registry.children_of("layered") == ("composite",)
    assert registry.children_of("composite") == ()
