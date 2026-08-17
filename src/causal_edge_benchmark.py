"""CausalEdgeBenchmark: 真实提案闭环上的结构级因果边验证 (路线 ③)。

不变性因果发现需要「每环境多样本」: 这里环境 = 光照条件 (lcol,ldir),
每个环境内多个不同底座几何但同一 attach ratio 的样本。实测 delta 取自
观测帧几何证据 (`CompositeTemplateProposer._observed_delta`), 用
`CausalDeltaLearner` 按光照分组验证「attach → scale_ratio/lateral_ratio」
边的一致度。机制真实 → 实测 ratio 跨光照稳定 → agreement 高 (因果);
`--drift` 让 ratio 逐环境漂移 → agreement 低 (伪相关对照)。
"""

from __future__ import annotations

import argparse

from causal_edge import CausalDeltaLearner
from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_template_proposer import CompositeTemplateProposer
from inverse_config import InverseConfig
from structure_birth import StructureCase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--n-per-env", type=int, default=3)
    ap.add_argument("--drift", action="store_true", help="让 ratio 逐环境漂移 (伪相关对照)")
    args = ap.parse_args()

    renderer, cam_l, cam_r = Codebook.make_renderer()
    cb = CompositeCodebook(InverseConfig(scene_family="composite"))
    proposer = CompositeTemplateProposer(
        operations=("attach",),
        ratios=(0.45,),
        lateral_ratios=(0.0,),
        part_kinds=(1,),
        part_hues=(2,),
        max_proposals=16,
    )

    # 环境 = 光照 (lcol, ldir); 每环境多个不同底座几何 (同一 ratio)
    rows = Codebook.sample(1, args.seed).tolist()
    envs: dict[tuple[int, int], list[tuple]] = {}
    for r in rows:
        key = (int(r[6]), int(r[7]))
        envs.setdefault(key, []).append(tuple(r))
    env_keys = list(envs)[:3]  # 取 3 个光照环境
    drift = [0.40, 0.45, 0.50][: len(env_keys)] if args.drift else None

    cases = []
    for ei, key in enumerate(env_keys):
        ratio = drift[ei] if drift else 0.45
        for base in envs[key][: args.n_per_env]:
            gt = proposer._attach(base, part_kind=1, part_hue=2, ratio=ratio, lateral_ratio=0.0)
            scene = cb.to_scene(gt)
            fl = renderer.render(scene, cam_l)
            fr = renderer.render(scene, cam_r)
            cases.append(
                StructureCase(
                    fl=fl,
                    fr=fr,
                    residuals={"composite": 1000.0},
                    posterior={"composite": 1.0},
                    params=base,
                    structure_id="composite",
                )
            )

    proposals = proposer.propose(tuple(cases))
    # 环境键从提案参数取光照 (14 维: lcol=12, ldir=13)
    edges = CausalDeltaLearner().learn(
        proposals, env_key=lambda p: (int(p.params[12]), int(p.params[13]))
    )
    if not edges:
        print("无因果边 (实测 delta 无法可靠估计)")
        return
    print(f"{len(cases)} 样本 / {len(env_keys)} 光照环境 / drift={bool(drift)}")
    for e in edges:
        print(
            f"  {e.parent_family} --{e.operation}--> {e.target}: "
            f"agreement={e.agreement:.3f} causal={e.is_causal} "
            f"envs={e.n_envs} pooled={e.pooled_range}"
        )


if __name__ == "__main__":
    main()
