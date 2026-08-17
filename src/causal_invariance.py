"""CausalInvariance: 池外光照探针 + 边缘估计 + 不变性打分 (路线 ①)。

因果不变性: 反照率 (hue) → 图像 的映射应对光照 (lcol/ldir) 不变;
光照 → 图像 的映射应对反照率/几何不变。这里测量「分析-合成精炼」
(§3, 靠 re-render 全光照候选做 do-搜索) 在 held-out 光照干预下反照率
估计的稳定性 —— 按构造它应不变; 相关密度 (MixtureSPN) 则退化。探针
量化这个差距, 是路线 ②③ 共同的验收基础设施 (见 docs/architecture.md §9)。
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from codebook import Codebook
from scene_reconstructor import SceneReconstructor


@dataclass(frozen=True)
class LightingHoldout:
    """光照分类水平 → 训练/池外划分 (干预外设计)。

    holdout_colors/dirs 每个都持有一个水平; 「池外」组合 = 含任一池外
    水平的光照组合 (9 组合里 5 个池外, 4 个训练), 模仿真实干预分布漂移。
    """

    train_colors: tuple[int, ...]
    train_dirs: tuple[int, ...]
    holdout_colors: tuple[int, ...]
    holdout_dirs: tuple[int, ...]

    @classmethod
    def split(
        cls,
        n_colors: int = len(Codebook.LIGHT_COLORS),
        n_dirs: int = len(Codebook.LIGHT_DIRS),
        holdout_color: int = len(Codebook.LIGHT_COLORS) - 1,
        holdout_dir: int = len(Codebook.LIGHT_DIRS) - 1,
    ) -> LightingHoldout:
        colors = tuple(range(n_colors))
        dirs = tuple(range(n_dirs))
        return cls(
            train_colors=tuple(c for c in colors if c != holdout_color),
            train_dirs=tuple(d for d in dirs if d != holdout_dir),
            holdout_colors=(holdout_color,),
            holdout_dirs=(holdout_dir,),
        )

    def in_support(self, lcol: int, ldir: int) -> bool:
        """该光照组合是否完全落在训练支持集内。"""
        return lcol in self.train_colors and ldir in self.train_dirs

    def holdout(self, lcol: int, ldir: int) -> bool:
        """该光照组合是否含任一池外水平。"""
        return lcol in self.holdout_colors or ldir in self.holdout_dirs


def invariance_score(group_accuracies: Iterable[float]) -> float:
    """跨分组的准确率 → 不变性分数 = 最差组准确率 (瓶颈)。

    因果不变性要求估计对干预变量分组后每组都准; 任一分组崩塌即不变性
    被破坏。返回 [0,1], 1 = 完全不变。
    """
    vals = list(group_accuracies)
    if not vals:
        return 0.0
    return float(min(vals))


@dataclass(frozen=True)
class InvarianceReport:
    """一次不变性探针的完整结果。"""

    factor: str
    in_support_accuracy: float
    holdout_accuracy: float
    invariance_score: float
    gap: float
    per_group_accuracy: dict[tuple[int, int], float]
    n_groups: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "in_support_accuracy": self.in_support_accuracy,
            "holdout_accuracy": self.holdout_accuracy,
            "invariance_score": self.invariance_score,
            "gap": self.gap,
            "per_group_accuracy": self.per_group_accuracy,
            "n_groups": self.n_groups,
        }


class InvarianceProbe:
    """分析-合成精炼在 held-out 光照干预下的反照率不变性探针。

    每个场景固定几何, 用 `SceneReconstructor.refine_appearance` 枚举全
    光照候选得到联合后验, 再对光照边缘化得到反照率估计 (因果不变估计),
    按光照分组统计准确率。`summarize` 是纯统计部分 (可脱离渲染单测),
    `run` 用真实 renderer 端到端测量。
    """

    @classmethod
    def summarize(
        cls,
        groups: dict[Hashable, list[tuple[int, int]]],
        factor: str,
        holdout: LightingHoldout,
    ) -> InvarianceReport:
        """分组 (真值, 预测) 对 → 不变性报告。分组键须是 (lcol, ldir)。"""
        per_group: dict[tuple[int, int], float] = {}
        in_support: list[float] = []
        held_out: list[float] = []
        for key, pairs in groups.items():
            lcol, ldir = cast(tuple[int, int], key)
            acc = (
                sum(1 for t, p in pairs if t == p) / len(pairs)
                if pairs
                else 0.0
            )
            per_group[(lcol, ldir)] = acc
            (in_support if holdout.in_support(lcol, ldir) else held_out).append(acc)
        in_acc = sum(in_support) / len(in_support) if in_support else 0.0
        out_acc = sum(held_out) / len(held_out) if held_out else 0.0
        return InvarianceReport(
            factor=factor,
            in_support_accuracy=in_acc,
            holdout_accuracy=out_acc,
            invariance_score=invariance_score(per_group.values()),
            gap=in_acc - out_acc,
            per_group_accuracy=per_group,
            n_groups=len(per_group),
        )

    @classmethod
    def run(
        cls,
        codebook: Codebook,
        scenes: Sequence[Sequence[float]],
        holdout: LightingHoldout,
        factor: str = "hue",
        renderer=None,
        cam_l=None,
        cam_r=None,
    ) -> InvarianceReport:
        """真实渲染 + 分析-合成 + 边缘化的端到端不变性测量。"""
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row in scenes:
            prm = tuple(float(x) for x in row)
            scene = codebook.to_scene(prm)
            fl = renderer.render(scene, cam_l)
            fr = renderer.render(scene, cam_r)
            base = prm[:5]  # kind,u,v,s,z 固定; 只精炼 hue/lcol/ldir
            _, _, score_arr = SceneReconstructor.refine_appearance(
                codebook, base, fl, fr, renderer, cam_l, cam_r
            )
            temperature = max(2.0 * float(mx.min(score_arr)), 1.0)
            logp = -score_arr / temperature
            posterior = mx.exp(logp - mx.logsumexp(logp))
            key = (int(prm[6]), int(prm[7]))  # (lcol, ldir)
            true = int(prm[5] if factor == "hue" else prm[6 if factor == "lcol" else 7])
            marginal = SceneReconstructor.marginal_appearance(posterior, factor)
            pred = int(mx.argmax(marginal))
            groups.setdefault(key, []).append((true, pred))
        return cls.summarize(groups, factor, holdout)
