"""SceneEMRefiner: 单物体几何↔光照的 ECM 精炼 (GenericEM 实例)。

§7.1 设计: 场景拆成几何 G=(u,v,s,z) 与外观 A=(hue,lcol,ldir)。

  E 步 (responsibilities): 固定 G, 枚举 54 外观候选 (hue×lcol×ldir),
     渲染残差 → 软后验 q(A|G)。
  M 步 (maximize): 固定 q(A), 对几何做坐标搜索, 期望残差局部最小化。
     默认只搜 u/v (§7.1 下一步 ①): s/z 有投影歧义 (大而远 ≡ 小而近),
     贪心搜索会拖坏 s; freeze=(False,False,True,True) 冻结 s/z。

这是推理期分析-合成迭代 (把反照率×光照歧义交回正向模型), 不恢复已
退役的 SPN 学习 EM (§2.2); 几何 G 是参数 θ, 外观 A 是隐变量 Z。kind
在 ECM 外固定 (来自 SPN 的 MAP/top-1), 不参与连续几何极大化。

成本: 每轮 E 步 54×2 渲染, M 步 2·|非冻结维| 几何候选 × top-k 外观 ×
2 渲染 (默认只搜 u/v → 4 候选; 四维全搜为 8)。默认不接主链路
(§7.1: 单物体验证稳定后再启用)。
"""

from __future__ import annotations

import mlx.core as mx

from codebook import Codebook
from scene_reconstructor import SceneReconstructor
from stereo import StereoDepth


class SceneEMRefiner:
    """几何↔光照 ECM 的 GenerativeModel 适配。"""

    def __init__(
        self,
        codebook: Codebook,
        kind: int,
        fl: mx.array,
        fr: mx.array,
        appearance_topk: int = 3,
        deltas: tuple[float, float, float, float] = (2.0, 2.0, 0.05, 0.1),
        freeze: tuple[bool, bool, bool, bool] = (False, False, True, True),
        renderer=None,
        cam_l=None,
        cam_r=None,
    ):
        self.codebook = codebook
        self.kind = kind
        self.fl, self.fr = fl, fr
        self.appearance_topk = appearance_topk
        self.deltas = deltas
        self.freeze = freeze
        if renderer is None or cam_l is None or cam_r is None:
            renderer, cam_l, cam_r = Codebook.make_renderer()
        self.renderer, self.cam_l, self.cam_r = renderer, cam_l, cam_r
        self.wl = StereoDepth.foreground_weights(fl)
        self.wr = StereoDepth.foreground_weights(fr)
        self._appearances = [
            (hue, lcol, ldir)
            for hue in range(Codebook.N_HUE)
            for lcol in range(len(Codebook.LIGHT_COLORS))
            for ldir in range(len(Codebook.LIGHT_DIRS))
        ]

    # ── 渲染残差 (正向模型) ───────────────────────────────────────

    def _residual(
        self, geometry: tuple[float, ...], appearance: tuple[int, int, int]
    ) -> float:
        u, v, s, z = geometry
        hue, lcol, ldir = appearance
        prm = (
            float(self.kind), float(u), float(v), float(s), float(z),
            float(hue), float(lcol), float(ldir),
        )
        scene = self.codebook.to_scene(prm)
        cl = self.renderer.render(scene, self.cam_l)
        cr = self.renderer.render(scene, self.cam_r)
        return 0.5 * (
            SceneReconstructor._masked_mse(self.fl, cl, self.wl)
            + SceneReconstructor._masked_mse(self.fr, cr, self.wr)
        )

    # ── GenerativeModel 接口 ──────────────────────────────────────

    def responsibilities(
        self, geometry: tuple[float, ...], observation, temperature: float = 1.0
    ) -> mx.array:
        """E 步: 固定几何, 外观候选渲染残差 → q(A|G) (54,)。"""
        scores = mx.array([self._residual(geometry, a) for a in self._appearances])
        t = max(2.0 * float(mx.min(scores)), 1.0) * temperature
        logp = -scores / t
        return mx.exp(logp - mx.logsumexp(logp))

    def maximize(
        self,
        q: mx.array,
        observation,
        geometry: tuple[float, ...],
        damping: float = 0.0,
    ) -> tuple[float, ...]:
        """M 步: 固定 q(A), 对几何坐标搜索期望残差最小 (冻结维跳过)。"""
        order = mx.argsort(q)[::-1][: self.appearance_topk].tolist()
        top_q = q[order] / mx.sum(q[order])

        def expected(g: tuple[float, ...]) -> float:
            return sum(
                float(top_q[i]) * self._residual(g, self._appearances[j])
                for i, j in enumerate(order)
            )

        cur = list(geometry)
        for i, delta in enumerate(self.deltas):
            if self.freeze[i]:
                continue
            best = expected(tuple(cur))
            for sign in (-1.0, 1.0):
                cand = list(cur)
                cand[i] += sign * delta
                e = expected(tuple(cand))
                if e < best:
                    best, cur = e, cand
        new = tuple(cur)
        if damping > 0.0:
            new = tuple(
                (1.0 - damping) * g + damping * o
                for g, o in zip(new, geometry, strict=True)
            )
        return new

    def log_likelihood(self, geometry: tuple[float, ...], observation) -> float:
        """观测对数似然代理: 负最佳外观残差 (越小越不像)。"""
        return -float(min(self._residual(geometry, a) for a in self._appearances))

    def sample(
        self, geometry: tuple[float, ...], rng=None
    ) -> tuple[mx.array, mx.array]:
        """正向模型: 几何 + 随机外观 → 左右帧 (合成验证用)。"""
        import random

        r = random.Random(rng if rng is not None else 0)
        hue, lcol, ldir = r.choice(self._appearances)
        u, v, s, z = geometry
        prm = (
            float(self.kind), float(u), float(v), float(s), float(z),
            float(hue), float(lcol), float(ldir),
        )
        scene = self.codebook.to_scene(prm)
        return (
            self.renderer.render(scene, self.cam_l),
            self.renderer.render(scene, self.cam_r),
        )
