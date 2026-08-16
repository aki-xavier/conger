"""ChildTemplateBenchmark: 数据驱动 attach 子模板的真实样本闭环。

流程: 渲染受约束 composite 漂移样本 → CompositeTemplateProposer 生成
出生提案 → TemplateDeltaLearner 估计约束 → ChildCodebookFactory 物化
子模板 → 显式训练注册 → 父/子结构联合门控。默认子模板只含 162 组合,
用于验证机制闭环而非追求大规模精度。
"""

from __future__ import annotations

import argparse

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_template_proposer import CompositeTemplateProposer
from expert_registry import ExpertRegistry, SceneExpert
from inverse_config import InverseConfig
from structure_birth import StructureBirthRequest, StructureCase


def _case(
    proposer: CompositeTemplateProposer,
    base: tuple[float, ...],
    renderer,
    cam_l,
    cam_r,
) -> StructureCase:
    """按受约束 attach 子分布渲染一个未知结构证据样本。"""
    gt = proposer._attach(base, part_kind=1, part_hue=2, ratio=0.45, lateral_ratio=0.0)
    cb = CompositeCodebook(proposer.codebook.cfg)
    scene = cb.to_scene(gt)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    return StructureCase(
        fl=fl,
        fr=fr,
        residuals={"composite": 1000.0},
        posterior={"composite": 1.0},
        params=base,
        structure_id="composite",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=991)
    ap.add_argument("--replicates", type=int, default=4)
    args = ap.parse_args()
    renderer, cam_l, cam_r = Codebook.make_renderer()
    proposer = CompositeTemplateProposer(
        ratios=(0.45, 0.60),
        lateral_ratios=(0.0,),
        part_kinds=(1,),
        part_hues=(2,),
        max_proposals=4,
    )
    bases = [
        (0.0, 68.0, 91.0, 0.42, 3.15, 1.0, 0.0, 1.0),
        (0.0, 78.0, 88.0, 0.48, 3.35, 4.0, 0.0, 1.0),
    ]
    cases = tuple(
        _case(proposer, base, renderer, cam_l, cam_r) for base in bases
    )
    request = StructureBirthRequest(
        cases=cases,
        residual_mean=1000.0,
        best_posterior_mean=1.0,
        reason="child template benchmark",
        proposals=proposer.propose(cases),
    )
    parent_cfg = InverseConfig(scene_family="composite", replicates=1)
    parent = SceneExpert.from_config("composite", parent_cfg)
    registry = ExpertRegistry({"composite": parent})
    registry.enable_child_template_learning(
        manifest_path=ExpertRegistry.default_manifest_path()
    )
    pending = registry.observe_birth_request(request)
    assert pending, "出生请求未产生可学习子模板"
    child_cfg = InverseConfig(scene_family="composite", replicates=args.replicates)
    reg = registry.confirm_child_template(
        pending[0].name, cfg=child_cfg
    )
    print("child:", reg.spec.name)
    print("constraints:", reg.spec.constraints)
    print("lineage:", reg.expert.lineage().signature())

    # 同子分布 held-out 样本: 父 composite 与学习到的子模板联合门控
    child_cb = reg.codebook_cls(child_cfg)
    eval_params = child_cb.sample(1, args.seed)[:3]
    wins = 0
    for i, row in enumerate(eval_params.tolist(), 1):
        scene = child_cb.to_scene(tuple(float(x) for x in row))
        fl = renderer.render(scene, cam_l)
        fr = renderer.render(scene, cam_r)
        out = registry.decide(fl, fr)
        wins += out.estimate.structure_id == reg.spec.name
        print(
            f"[{i}] pred={out.estimate.structure_id} "
            f"posterior={out.posterior} residuals={out.residuals}"
        )
    print(f"child wins: {wins}/{eval_params.shape[0]}")


if __name__ == "__main__":
    main()
