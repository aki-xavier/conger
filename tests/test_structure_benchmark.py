"""StructureBenchmark 汇总逻辑测试。"""

from typing import cast

from structure_benchmark import StructureBenchmark, StructureCaseResult


def _result(true: str, pred: str, p: float) -> StructureCaseResult:
    names = ("single", "layered", "composite")
    return StructureCaseResult(
        true=true,
        predicted=pred,
        posterior={k: (p if k == pred else (1.0 - p) / 2) for k in names},
        residuals={k: 1.0 for k in names},
        scores={k: 1.0 for k in names},
        needs_new_structure=False,
    )


def test_structure_benchmark_summary() -> None:
    """accuracy/confusion/posterior_mean 应从逐样本结果聚合。"""
    results = (
        _result("single", "single", 0.8),
        _result("layered", "composite", 0.6),
        _result("composite", "composite", 0.7),
    )
    out = StructureBenchmark.summarize(results)
    confusion = cast(dict[str, dict[str, int]], out["confusion"])
    assert out["n"] == 3
    assert abs(cast(float, out["accuracy"]) - 2.0 / 3.0) < 1e-12
    assert confusion["layered"]["composite"] == 1
    assert set(out["posterior_mean"]) == {"single", "layered", "composite"}
