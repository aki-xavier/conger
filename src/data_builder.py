"""DataBuilder: 数据构建 (含缓存)。离散因子全笛卡尔积 × R 复制,
连续因子逐样本随机。无标准化 —— 对角高斯按维学 σ, 天然免尺度。"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from riesz import RieszWavelet


class DataBuilder:
    """组合覆盖采样 → 渲染 → 全分辨率特征 (含缓存)。"""

    def __init__(
        self, cfg: InverseConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def cache_tag(self, r_tr: int, r_i: int, r_e: int) -> str:
        """配置指纹 → 缓存文件名 (任一相关配置变化 → 新缓存)。"""
        cfg = self.cfg
        cb = self.codebook
        feat_tag = "".join(f"{s[:2]}{c[:2]}" for s, c in cfg.feat_spec)
        lvl_tag = (
            f"k{cb.N_KIND}h{cb.N_HUE}c{len(cb.LIGHT_COLORS)}d{len(cb.LIGHT_DIRS)}"
        )
        eq_tag = "eqn" if cfg.equal_luma else "std"
        occ_tag = "occ" if cfg.occlusion else "noc"
        return (
            f"mix_{cb.H}x{cb.W}_{feat_tag}_{lvl_tag}_{eq_tag}_{occ_tag}_"
            f"{r_tr}_{r_i}_{r_e}.safetensors"
        )

    def feats_of(self, params: mx.array) -> mx.array:
        """参数行 (n,8) → 渲染 → 特征 (n, n_feat)。"""
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        out = []
        for p in params.tolist():
            scene = self.codebook.to_scene(p)
            vec, rw = self.extractor.of_frame(renderer.render(scene, cam), rw)
            # 逐帧立即求值: MLX 惰性求值会把数千帧的计算图累积到
            # 一次性 eval, 超 Metal 显存上限
            mx.eval(vec)
            out.append(vec)
        return mx.stack(out)

    def build(
        self, r_train: int, r_interp: int, r_extrap: int, use_cache: bool
    ) -> tuple[mx.array, ...]:
        """→ (Ftr, Ptr, Fi, Pi, Fe, Pe): 特征 (360R, V) + 参数 (360R, 8)。
        三分裂: 训练 (范围内) / 插值测试 (范围内, 独立种子) / 外推测试
        (s,z 支撑集外)。R = 每离散组合的连续复制数。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / self.cache_tag(r_train, r_interp, r_extrap)
        if use_cache and path.exists():
            d = mx.load(str(path))
            return d["Ftr"], d["Ptr"], d["Fi"], d["Pi"], d["Fe"], d["Pe"]

        cb = self.codebook
        p_tr = cb.sample(r_train, mx.random.key(42))
        p_ti = cb.sample(r_interp, mx.random.key(99))
        p_te = cb.sample(r_extrap, mx.random.key(7), extrap=True)
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

    @staticmethod
    def targets(p: mx.array) -> mx.array:
        """参数 (n,8) → 目标 (n,6) [u,v,s,z,cosH,sinH]。
        色相环形量走 (cos,sin) 两维 (与复数色度特征同构)。
        色恒常: 彩光下色相不可观测是推理侧的事 (预测自然退回
        白光关联先验); 标签本身是真实反照率色相, 且与光色独立
        采样, 簇内均值不受彩光样本污染 —— 无需掩码 (实测: 目标
        不进 E 步后, 掩码在数学上无谓)。"""
        h = p[:, 5] * (2.0 * math.pi / Codebook.N_HUE)
        return mx.concatenate(
            [p[:, 1:5], mx.cos(h)[:, None], mx.sin(h)[:, None]], axis=1
        )
