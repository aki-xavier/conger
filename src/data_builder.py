"""DataBuilder: 连续场景数据构建 (含缓存)。无标准化 —— 对角高斯
按维学 σ, 天然免尺度; 目标各维单位不同也无关 (各自独立高斯)。"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from riesz import RieszWavelet


class DataBuilder:
    """连续参数采样 → 渲染 → 全分辨率特征 (含缓存)。"""

    def __init__(
        self, cfg: InverseConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def cache_tag(self, n_train: int, n_interp: int, n_extrap: int) -> str:
        """配置指纹 → 缓存文件名 (任一相关配置变化 → 新缓存)。"""
        cfg = self.cfg
        cb = self.codebook
        feat_tag = "".join(f"{s[:2]}{c[:2]}" for s, c in cfg.feat_spec)
        col_tag = "".join(f"{c:x}" for c in cfg.kind_colors)
        eq_tag = "eqn" if cfg.equal_luma else "std"
        occ_tag = "occ" if cfg.occlusion else "noc"
        lt_tag = "ml" if cfg.multi_light else ("tl" if cfg.test_light else "sl")
        return (
            f"mix_{cb.H}x{cb.W}_{feat_tag}_{col_tag}_{eq_tag}_{occ_tag}_"
            f"{lt_tag}_{n_train}_{n_interp}_{n_extrap}.safetensors"
        )

    def feats_of(self, params: mx.array) -> mx.array:
        """参数行 (n,5) → 渲染 → 特征 (n, n_feat)。"""
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        out = []
        rows = params.tolist()
        for n, p in enumerate(rows):
            # 多光照训练: 每帧轮流取方向池 (确定性, 缓存可复现)
            light = (
                Codebook.LIGHT_DIRS[n % len(Codebook.LIGHT_DIRS)]
                if self.cfg.multi_light
                else None
            )
            scene = self.codebook.to_scene(p, light=light)
            vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
            # 逐帧立即求值: MLX 惰性求值会把数千帧的计算图累积到
            # 一次性 eval, 超 Metal 显存上限
            mx.eval(vec)
            out.append(vec)
        return mx.stack(out)

    def build(
        self, n_train: int, n_interp: int, n_extrap: int, use_cache: bool
    ) -> tuple[mx.array, ...]:
        """→ (Ftr, Ptr, Fi, Pi, Fe, Pe): 特征 (n,V) + 参数 (n,5), float32。
        三分裂: 训练 (范围内) / 插值测试 (范围内, 独立种子) / 外推测试
        (s,z 支撑集外)。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / self.cache_tag(n_train, n_interp, n_extrap)
        if use_cache and path.exists():
            d = mx.load(str(path))
            return d["Ftr"], d["Ptr"], d["Fi"], d["Pi"], d["Fe"], d["Pe"]

        cb = self.codebook
        p_tr = cb.sample(n_train, mx.random.key(42))
        p_ti = cb.sample(n_interp, mx.random.key(99))
        p_te = cb.sample(n_extrap, mx.random.key(7), extrap=True)
        f_tr = self.feats_of(p_tr)
        f_ti = self.feats_of(p_ti)
        f_te = self.feats_of(p_te)
        mx.save_safetensors(
            str(path),
            {"Ftr": f_tr, "Ptr": p_tr, "Fi": f_ti, "Pi": p_ti,
             "Fe": f_te, "Pe": p_te},
        )
        print(f"数据缓存 → {path.name}")
        return f_tr, p_tr, f_ti, p_ti, f_te, p_te
