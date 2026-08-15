"""ChildOperationBenchmark: layer/mirror/repeat 子模板真实闭环。

每种操作都执行: 渲染受控漂移样本 → 残差提案 → delta 学习 → 动态
Codebook → 显式训练注册 → 同分布 held-out 父子门控。用于验证多操作
子模板学习, 不作为大样本精度基准。
"""

from __future__ import annotations

import argparse

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_template_proposer import CompositeTemplateProposer
from expert_registry import ExpertRegistry, SceneExpert
from inverse_config import InverseConfig
from structure_birth import StructureBirthRequest, StructureCase


class ChildOperationBenchmark:
    """单一模板操作的 proposal→child→gate 闭环。"""

    def __init__(self, operation: str, seed: int, replicates: int):
        if operation not in {"layer", "mirror", "repeat"}:
            raise ValueError(f"未知子模板操作 {operation}")
        self.operation = operation
        self.seed = seed
        self.replicates = replicates
        self.parent_family = "layered" if operation == "layer" else "composite"
        self.proposer = CompositeTemplateProposer(
            ratios=(0.45, 0.60),
            lateral_ratios=(0.0,) if operation == "layer" else (0.20,),
            part_kinds=(1,),
            part_hues=(2,),
            operations=(operation,),
            max_proposals=2,
        )
        self.renderer, self.cam_l, self.cam_r = Codebook.make_renderer()

    def gt_params(self, base: tuple[float, ...]) -> tuple[float, ...]:
        if self.operation == "layer":
            return self.proposer._layer(base, 1, 2, 0.45, 0.0)
        return self.proposer._lateral(base, self.operation, 0.45, 0.20)

    def render_case(self, base: tuple[float, ...]) -> StructureCase:
        gt = self.gt_params(base)
        cb = CompositeCodebook(InverseConfig(scene_family="composite"))
        scene = cb.to_scene(gt)
        fl = self.renderer.render(scene, self.cam_l)
        fr = self.renderer.render(scene, self.cam_r)
        return StructureCase(
            fl=fl,
            fr=fr,
            residuals={self.parent_family: 1000.0},
            posterior={self.parent_family: 1.0},
            params=base,
            structure_id=self.parent_family,
        )

    def request(self) -> StructureBirthRequest:
        bases = (
            (1.0, 68.0, 90.0, 0.42, 3.15, 2.0, 0.0, 1.0),
            (1.0, 78.0, 88.0, 0.48, 3.35, 2.0, 0.0, 1.0),
        )
        cases = tuple(self.render_case(base) for base in bases)
        return StructureBirthRequest(
            cases=cases,
            residual_mean=1000.0,
            best_posterior_mean=1.0,
            reason=f"{self.operation} child benchmark",
            proposals=self.proposer.propose(cases),
        )

    def run(self) -> None:
        request = self.request()
        parent_cfg = InverseConfig(
            scene_family=self.parent_family, replicates=1
        )
        parent = SceneExpert.from_config(self.parent_family, parent_cfg)
        registry = ExpertRegistry({self.parent_family: parent})
        manifest_path = ExpertRegistry.default_manifest_path().with_name(
            f"registry_manifest_{self.operation}.json"
        )
        registry.enable_child_template_learning(manifest_path=manifest_path)
        pending = registry.observe_birth_request(request)
        assert pending, f"{self.operation} 出生请求未产生子模板"
        child_cfg = InverseConfig(
            scene_family=self.parent_family,
            replicates=self.replicates,
        )
        reg = registry.confirm_child_template(
            pending[0].name, cfg=child_cfg
        )
        print("child:", reg.spec.name)
        print("operation:", self.operation)
        print("constraints:", reg.spec.constraints)
        print("lineage:", reg.expert.lineage().signature())

        child_cb = reg.codebook_cls(child_cfg)
        eval_params = child_cb.sample(1, self.seed)[:3]
        wins = 0
        for i, row in enumerate(eval_params.tolist(), 1):
            scene = child_cb.to_scene(tuple(float(x) for x in row))
            fl = self.renderer.render(scene, self.cam_l)
            fr = self.renderer.render(scene, self.cam_r)
            out = registry.decide(fl, fr)
            wins += out.estimate.structure_id == reg.spec.name
            print(
                f"[{i}] pred={out.estimate.structure_id} "
                f"posterior={out.posterior} residuals={out.residuals}"
            )
        print(f"child wins: {wins}/{eval_params.shape[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operation", choices=("layer", "mirror", "repeat"), required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--replicates", type=int, default=2)
    args = ap.parse_args()
    ChildOperationBenchmark(
        args.operation, args.seed, args.replicates
    ).run()


if __name__ == "__main__":
    main()
