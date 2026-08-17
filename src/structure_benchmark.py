"""StructureBenchmark: 跨场景结构族的真实模型门控评估。

默认加载已训练的 single/layered/composite 专家, 对每个结构族采样若干
场景, 渲染左右图后调用 ExpertRegistry.decide, 输出结构分类准确率、
混淆矩阵、残差和后验摘要。该入口用于结构专家联合标定, 不参与训练。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from codebook import Codebook
from expert_registry import ExpertRegistry
from inverse_config import InverseConfig


@dataclass(frozen=True)
class StructureCaseResult:
    """一个真实结构样本的门控结果。"""

    true: str
    predicted: str
    posterior: dict[str, float]
    residuals: dict[str, float]
    scores: dict[str, float]
    needs_new_structure: bool


class StructureBenchmark:
    """跨结构族采样 → 渲染 → 三专家联合门控 → 汇总指标。"""

    def __init__(
        self,
        registry: ExpertRegistry,
        samples_per_family: int = 3,
        seed: int = 777,
    ):
        if samples_per_family < 1:
            raise ValueError("samples_per_family 必须 >=1")
        self.registry = registry
        self.samples_per_family = samples_per_family
        self.seed = seed

    def cases(self) -> tuple[tuple[str, mx.array, mx.array], ...]:
        """按结构族生成确定性评估帧对。"""
        renderer, cam_l, cam_r = Codebook.make_renderer()
        out = []
        for family_i, (name, expert) in enumerate(self.registry.experts.items()):
            cb = expert.app.codebook
            p = cb.sample(1, self.seed + family_i)
            for row in cast(list, p[: self.samples_per_family].tolist()):
                scene = cb.to_scene(tuple(float(x) for x in row))
                out.append(
                    (
                        name,
                        renderer.render(scene, cam_l),
                        renderer.render(scene, cam_r),
                    )
                )
        return tuple(out)

    @staticmethod
    def summarize(
        results: tuple[StructureCaseResult, ...],
    ) -> dict[str, object]:
        """门控记录 → accuracy/confusion/mean posterior/ECE 校准。"""
        assert results, "至少需要一条门控结果"
        confusion: dict[str, dict[str, int]] = {}
        correct = 0
        posterior_sum: dict[str, float] = {}
        for r in results:
            confusion.setdefault(r.true, {})
            confusion[r.true][r.predicted] = (
                confusion[r.true].get(r.predicted, 0) + 1
            )
            correct += r.predicted == r.true
            for k, v in r.posterior.items():
                posterior_sum[k] = posterior_sum.get(k, 0.0) + v
        n = len(results)
        return {
            "n": n,
            "accuracy": correct / n,
            "confusion": confusion,
            "posterior_mean": {
                k: v / n for k, v in posterior_sum.items()
            },
            "ece": StructureBenchmark._ece(results),
        }

    @staticmethod
    def _ece(results: tuple[StructureCaseResult, ...], bins: int = 10) -> float:
        """winner 置信度的期望校准误差 (ECE, 值越小越校准)。"""
        if not results:
            return 0.0
        acc = [0.0] * bins
        conf = [0.0] * bins
        cnt = [0] * bins
        for r in results:
            c = r.posterior[r.predicted]
            idx = min(int(c * bins), bins - 1)
            acc[idx] += 1.0 if r.predicted == r.true else 0.0
            conf[idx] += c
            cnt[idx] += 1
        total = len(results)
        ece = 0.0
        for i in range(bins):
            if cnt[i] == 0:
                continue
            ece += (cnt[i] / total) * abs(conf[i] / cnt[i] - acc[i] / cnt[i])
        return ece

    def run(self) -> dict[str, object]:
        """执行评估并打印逐样本与汇总结果。"""
        results = []
        for i, (true, fl, fr) in enumerate(self.cases(), 1):
            decision = self.registry.decide(fl, fr)
            result = StructureCaseResult(
                true=true,
                predicted=decision.estimate.structure_id,
                posterior=decision.posterior,
                residuals=decision.residuals,
                scores=decision.scores,
                needs_new_structure=decision.needs_new_structure,
            )
            results.append(result)
            print(
                f"[{i}] true={true} pred={result.predicted} "
                f"posterior={self._fmt(result.posterior)} "
                f"residual={self._fmt(result.residuals)}"
            )
        summary = self.summarize(tuple(results))
        print(f"accuracy: {summary['accuracy']:.3f} ({summary['n']} cases)")
        print(f"confusion: {summary['confusion']}")
        print(f"posterior_mean: {summary['posterior_mean']}")
        print(f"ece: {summary['ece']:.3f}")
        return {"results": tuple(results), **summary}

    @staticmethod
    def _fmt(values: Mapping[str, float]) -> str:
        return "{" + ", ".join(f"{k}:{v:.2f}" for k, v in values.items()) + "}"


def main() -> None:
    """CLI: python src/structure_benchmark.py [--samples-per-family N]。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--geometry-weight", type=float, default=5000.0)
    ap.add_argument("--temperature-scale", type=float, default=1.0)
    ap.add_argument(
        "--no-single-refine",
        action="store_true",
        help="单物体专家跳过 162 候选渲染精炼 (快速门控压力测试)",
    )
    args = ap.parse_args()
    configs = {
        "single": InverseConfig(
            scene_family="single",
            refine_appearance=not args.no_single_refine,
        ),
        "layered": InverseConfig(scene_family="layered", replicates=1),
        "composite": InverseConfig(scene_family="composite", replicates=1),
    }
    registry = ExpertRegistry.from_configs(
        configs,
        geometry_weight=args.geometry_weight,
        temperature_scale=args.temperature_scale,
    )
    StructureBenchmark(
        registry,
        samples_per_family=args.samples_per_family,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
