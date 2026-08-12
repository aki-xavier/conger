"""DataBuilder: 数据构建 (含缓存) 与标准化。"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from riesz import RieszWavelet


class DataBuilder:
    """数据构建 (含缓存) 与标准化。"""

    def __init__(
        self, cfg: InverseConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def cache_tag(self, n_train: int, n_test: int) -> str:
        """配置指纹 → 缓存文件名 (任一相关配置变化 → 新缓存)。"""
        cfg = self.cfg
        cb = self.codebook
        feat_tag = "".join(f"{s[:2]}{c[:2]}" for s, c in cfg.feat_spec)
        col_tag = "".join(f"{c:x}" for c in cfg.kind_colors)
        eq_tag = "eqn" if cfg.equal_luma else "std"
        occ_tag = "occ" if cfg.occlusion else "noc"
        lt_tag = "ml" if cfg.multi_light else "sl"
        res_tag = "fr" if cfg.full_res else "pl"
        return (
            f"inv_{cb.H}x{cb.W}_g{cb.N_GX}x{cb.N_GY}_{feat_tag}_{col_tag}_"
            f"{eq_tag}_{occ_tag}_{lt_tag}_{res_tag}_{n_train}_{n_test}.safetensors"
        )

    def build(
        self, n_train: int, n_test: int, use_cache: bool
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """→ (Xtr, Ctr, Xte, Cte): 特征 (n, n_feat) + 码 (n, 5), 均 float32。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / self.cache_tag(n_train, n_test)
        if use_cache and path.exists():
            d = mx.load(str(path))
            return d["Xtr"], d["Ctr"], d["Xte"], d["Cte"]

        cb = self.codebook
        tr = mx.random.randint(
            0, cb.N_CODES, shape=(n_train,), key=mx.random.key(42)
        ).tolist()
        te = mx.random.randint(
            0, cb.N_CODES, shape=(n_test,), key=mx.random.key(99)
        ).tolist()
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None

        def feats_of(idxs: list[int]) -> mx.array:
            nonlocal rw
            out = []
            for n, i in enumerate(idxs):
                # 多光照训练: 每帧轮流取方向池 (确定性, 缓存可复现)
                light = (
                    cb.LIGHT_DIRS[n % len(cb.LIGHT_DIRS)]
                    if self.cfg.multi_light
                    else None
                )
                scene = cb.to_scene(cb.idx_to_code(i), light=light)
                vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
                # 逐帧立即求值: MLX 惰性求值会把数千帧的计算图累积到
                # 一次性 eval, 超 Metal 显存上限
                mx.eval(vec)
                out.append(vec)
            return mx.stack(out)

        x_tr = feats_of(tr)
        x_te = feats_of(te)
        c_tr = mx.array([list(cb.idx_to_code(i)) for i in tr], dtype=mx.float32)
        c_te = mx.array([list(cb.idx_to_code(i)) for i in te], dtype=mx.float32)
        mx.save_safetensors(
            str(path), {"Xtr": x_tr, "Ctr": c_tr, "Xte": x_te, "Cte": c_te}
        )
        print(f"数据缓存 → {path.name}")
        return x_tr, c_tr, x_te, c_te

    @staticmethod
    def standardize(
        x_tr: mx.array, x_te: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """逐特征 z-score (训练集统计) → (z_tr, z_te, mu, sd)。

        mu/sd 随模型保存: 加载模型推理必须用同一统计。"""
        mu = x_tr.mean(axis=0, keepdims=True)
        sd = mx.maximum(x_tr.std(axis=0, keepdims=True), 1e-6)
        return (x_tr - mu) / sd, (x_te - mu) / sd, mu, sd
