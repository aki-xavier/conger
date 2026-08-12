"""SequenceRunner: 多帧运动先验 (贝叶斯前向滤波, prior.md 时间先验)。"""

from __future__ import annotations

import math

import mlx.core as mx

from code_bayes import CodeBayes
from codebook import Codebook
from demo_config import DemoConfig
from feature_extractor import FeatureExtractor
from riesz import RieszWavelet
from spn import SPN


class SequenceRunner:
    """多帧运动先验 (prior.md 运动与时间先验): 贝叶斯前向滤波。"""

    def __init__(
        self, cfg: DemoConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def gen_sequence(self, seed: int, n_frames: int) -> list[tuple[int, ...]]:
        """运动序列: 起始码随机, gx/gy 每帧 ±1 格随机游走 (运动连续性),
        kind/size/z 固定 (物体属性不变, prior.md 时间一致性/不变性假设)。"""
        cb = self.codebook
        key = mx.random.key(seed)
        code = list(
            cb.idx_to_code(int(mx.random.randint(0, cb.N_CODES, shape=(1,), key=key)))
        )
        seq = [tuple(code)]
        for _ in range(1, n_frames):
            key, k1, k2 = mx.random.split(key, 3)
            dx = int(mx.random.randint(-1, 2, shape=(1,), key=k1))
            dy = int(mx.random.randint(-1, 2, shape=(1,), key=k2))
            code[1] = min(cb.N_GX - 1, max(0, code[1] + dx))
            code[2] = min(cb.N_GY - 1, max(0, code[2] + dy))
            seq.append(tuple(code))
        return seq

    def temporal_preds(self) -> list[list[int]]:
        """转移图: T(c'|c) 高 ⟺ 同 kind/size/z 且 |Δgx|+|Δgy|≤1 (运动连续性)。
        返回每个 c' 的前驱列表 (P(c_t|c_{t-1}) 非零的 c_{t-1})。"""
        cb = self.codebook
        preds: list[list[int]] = [[] for _ in range(cb.N_CODES)]
        for c in range(cb.N_CODES):
            k, gx, gy, s, z = cb.idx_to_code(c)
            for dgx in (-1, 0, 1):
                for dgy in (-1, 0, 1):
                    nx, ny = gx + dgx, gy + dgy
                    if 0 <= nx < cb.N_GX and 0 <= ny < cb.N_GY:
                        preds[cb.code_to_idx((k, nx, ny, s, z))].append(c)
        return preds

    def run(
        self,
        net: SPN | CodeBayes,
        mu: mx.array,
        sd: mx.array,
        n_seqs: int,
        n_frames: int,
        seq_seed: int,
    ) -> None:
        """序列推理: 逐帧后验, 对比单帧 MAP vs 贝叶斯前向滤波
        (马尔可夫时间先验: P(c_t|c_{t-1}) 同属性+邻域)。"""
        cb = self.codebook
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        codes = cb.all_codes()
        preds = self.temporal_preds()
        log_off = math.log(0.01)
        keys = ("code", "kind", "gx", "gy", "size", "z")
        acc_single: dict[str, float] = {k: 0.0 for k in keys}
        acc_filter: dict[str, float] = {k: 0.0 for k in keys}
        total = 0
        for s in range(n_seqs):
            seq = self.gen_sequence(seq_seed + s, n_frames)
            prev_post: mx.array | None = None
            for code in seq:
                scene = cb.to_scene(code)
                vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
                x = (vec - mu) / sd  # (1, V), 训练预处理统计
                like = net.posterior(x, codes)[0]  # (K,) log 似然
                pred1 = int(mx.argmax(like))
                if prev_post is not None:
                    # 贝叶斯滤波: P(c_t|I) ∝ P(I_t|c_t)·Σ_{c_{t-1}} T·P(c_{t-1})
                    agg = mx.full((cb.N_CODES,), log_off)
                    for c in range(cb.N_CODES):
                        agg[c] = mx.logsumexp(prev_post[preds[c]])
                    post_f = like + agg
                    post_f = post_f - mx.logsumexp(post_f)
                else:
                    post_f = like
                pred2 = int(mx.argmax(post_f))
                acc_single["code"] += pred1 == cb.code_to_idx(code)
                acc_filter["code"] += pred2 == cb.code_to_idx(code)
                c1, c2 = cb.idx_to_code(pred1), cb.idx_to_code(pred2)
                for name, ci in (
                    ("kind", 0), ("gx", 1), ("gy", 2), ("size", 3), ("z", 4)
                ):
                    acc_single[name] += c1[ci] == code[ci]
                    acc_filter[name] += c2[ci] == code[ci]
                prev_post = post_f
                total += 1
        fmt = "  ".join(f"{k} {acc_single[k]/total:.3f}" for k in keys)
        fmt2 = "  ".join(f"{k} {acc_filter[k]/total:.3f}" for k in keys)
        print(f"  单帧    : {fmt}")
        print(f"  时序滤波: {fmt2}")
        assert acc_filter["code"] > acc_single["code"], "时序先验应提升码准确率"
        print(
            f"  码准确率: 单帧 {acc_single['code']/total:.3f} → "
            f"滤波 {acc_filter['code']/total:.3f}"
        )
        print("demo_inverse: 序列自检 ✓")
