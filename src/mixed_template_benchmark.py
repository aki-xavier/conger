"""MixedTemplateBenchmark: 多子模板混合注册表联合门控。

从 registry manifests 恢复 attach/layer/mirror/repeat 子模板, 与
single/layered/composite 父模板一起对混合场景流门控, 输出混淆矩阵、
winner 后验和正确/错误置信摘要。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from codebook import Codebook
from expert_registry import ExpertRegistry
from inverse_config import InverseConfig
from structure_benchmark import StructureBenchmark, StructureCaseResult


class MixedTemplateBenchmark:
    """父模板 + 多动态子模板的混合场景门控基准。"""

    def __init__(self, samples_per_family: int = 1, seed: int = 777):
        if samples_per_family < 1:
            raise ValueError("samples_per_family 必须 >=1")
        self.samples_per_family = samples_per_family
        self.seed = seed

    @staticmethod
    def manifest_paths() -> tuple[Path, ...]:
        root = ExpertRegistry.default_manifest_path().parent
        return tuple(
            root / name
            for name in (
                "registry_manifest.json",
                "registry_manifest_layer.json",
                "registry_manifest_mirror.json",
                "registry_manifest_repeat.json",
            )
            if (root / name).exists()
        )

    @classmethod
    def load_registry(cls) -> ExpertRegistry:
        """加载三个父模板 + 全部 manifest 中已训练的动态子模板。"""
        configs = {
            "single": InverseConfig(
                scene_family="single", refine_appearance=False
            ),
            "layered": InverseConfig(scene_family="layered", replicates=1),
            "composite": InverseConfig(scene_family="composite", replicates=1),
        }
        registry = ExpertRegistry.from_configs(configs)
        for path in cls.manifest_paths():
            registry.load_manifest(path, missing_ok=True)
        return registry

    def run(self) -> dict[str, object]:
        registry = self.load_registry()
        renderer, cam_l, cam_r = Codebook.make_renderer()
        results = []
        for family_i, (name, expert) in enumerate(registry.experts.items()):
            cb = expert.app.codebook
            params = cb.sample(1, self.seed + family_i)
            for row in params[: self.samples_per_family].tolist():
                scene = cb.to_scene(tuple(float(x) for x in row))
                fl = renderer.render(scene, cam_l)
                fr = renderer.render(scene, cam_r)
                decision = registry.decide(fl, fr)
                result = StructureCaseResult(
                    true=name,
                    predicted=decision.estimate.structure_id,
                    posterior=decision.posterior,
                    residuals=decision.residuals,
                    scores=decision.scores,
                    needs_new_structure=decision.needs_new_structure,
                )
                results.append(result)
                print(
                    f"true={name} pred={result.predicted} "
                    f"winner_p={result.posterior[result.predicted]:.3f} "
                    f"true_score={result.scores[name]:.1f} "
                    f"pred_score={result.scores[result.predicted]:.1f}"
                )
        summary = StructureBenchmark.summarize(tuple(results))
        correct_p = [
            r.posterior[r.predicted] for r in results if r.predicted == r.true
        ]
        wrong_p = [
            r.posterior[r.predicted] for r in results if r.predicted != r.true
        ]
        summary["winner_posterior_mean"] = sum(
            r.posterior[r.predicted] for r in results
        ) / len(results)
        summary["correct_posterior_mean"] = (
            sum(correct_p) / len(correct_p) if correct_p else None
        )
        summary["wrong_posterior_mean"] = (
            sum(wrong_p) / len(wrong_p) if wrong_p else None
        )
        print(f"experts: {list(registry.experts)}")
        print(f"accuracy: {summary['accuracy']:.3f} ({summary['n']} cases)")
        print(f"confusion: {summary['confusion']}")
        print(f"winner_posterior_mean: {summary['winner_posterior_mean']:.3f}")
        print(f"correct_posterior_mean: {summary['correct_posterior_mean']}")
        print(f"wrong_posterior_mean: {summary['wrong_posterior_mean']}")
        return {"results": tuple(results), **summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-family", type=int, default=1)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    MixedTemplateBenchmark(
        samples_per_family=args.samples_per_family,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
