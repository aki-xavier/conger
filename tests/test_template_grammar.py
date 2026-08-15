"""TemplateGrammar 测试: 有界组合空间与操作约束。"""

import pytest

from composite_template_proposer import CompositeTemplateProposer
from template_grammar import TemplateGrammar


def test_template_grammar_depth_bounds() -> None:
    """depth=1 只有 primitive; depth=2 加入有界二元组合。"""
    g1 = TemplateGrammar(
        operations=("attach", "layer", "mirror", "repeat"), max_depth=1
    )
    assert len(g1.rules()) == 3
    assert all(r.operation == "primitive" for r in g1.rules())

    g2 = TemplateGrammar(
        operations=("attach", "layer", "mirror", "repeat"), max_depth=2
    )
    assert len(g2.primitives()) == 3
    assert len(g2.composites()) == 24  # 9 attach + 9 layer + 3 mirror + 3 repeat
    assert len(g2.rules()) == 27


def test_template_grammar_operator_constraints() -> None:
    """mirror/repeat 只允许同 kind; 未知操作 fail closed。"""
    g = TemplateGrammar(operations=("mirror", "repeat"))
    assert all(r.base_kind == r.part_kind for r in g.composites())
    with pytest.raises(ValueError):
        TemplateGrammar(operations=("boolean_union",))


def test_proposer_uses_bounded_grammar() -> None:
    """提案器暴露完整 depth≤2 文法; mirror 参数保持同 kind/hue/depth。"""
    proposer = CompositeTemplateProposer(
        operations=("attach", "layer", "mirror", "repeat")
    )
    assert len(proposer.grammar.composites()) == 24
    mirror = next(
        r
        for r in proposer.grammar.composites()
        if r.operation == "mirror" and r.base_kind == 1
    )
    base = (1.0, 64.0, 72.0, 0.4, 3.2, 2.0, 0.0, 1.0)
    params = proposer._params_for_rule(base, mirror, 5, 0.8, 0.2)
    assert params[0] == params[6] == 1.0
    assert params[5] == params[11] == 2.0
    assert params[4] == params[10] == 3.2
    assert params[7] > params[1]
