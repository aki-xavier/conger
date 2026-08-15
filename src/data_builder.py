"""DataBuilder: 数据构建 (含缓存)。离散因子全笛卡尔积 × R 复制,
连续因子逐样本随机。无标准化 —— 对角高斯按维学 σ, 天然免尺度。"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from stereo import StereoDepth
from stereo_layers import StereoLayers


class DataBuilder:
    """组合覆盖采样 → 渲染 → 全分辨率特征 (含缓存)。"""

    def __init__(
        self, cfg: InverseConfig, codebook: Codebook, extractor: FeatureExtractor
    ):
        self.cfg = cfg
        self.codebook = codebook
        self.extractor = extractor

    def cache_tag(self) -> str:
        """配置指纹 → 缓存文件名干 (任一相关配置变化 → 新缓存)。"""
        cfg = self.cfg
        cb = self.codebook
        feat_tag = "".join(f"{s[:2]}{c[:2]}" for s, c in cfg.feat_spec)
        lvl_tag = (
            f"k{cb.N_KIND}h{cb.N_HUE}c{len(cb.LIGHT_COLORS)}d{len(cb.LIGHT_DIRS)}"
            f"o{cb.N_OBJECTS}sv{cb.SAMPLE_V}rp{cb.RENDER_V}"
        )
        # 结构族自带统计契约版本: st4 单物体 / sl8 遮挡层 / cp1 组合物
        stereo_tag = cb.STEREO_V
        return f"mix_{cb.H}x{cb.W}_{feat_tag}_{lvl_tag}_{stereo_tag}"

    def _block_feats(self, split: str, r: int) -> tuple[mx.array, ...]:
        """一个复制块的 (参数, 特征, 统计), 逐块缓存: 缺哪块渲哪块
        (增量训练的数据侧)。split: tr 训练 / ti 插值测试 / te 外推测试。"""
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / f"{self.cache_tag()}_{split}{r}.safetensors"
        if self.cfg.use_cache and path.exists():
            d = mx.load(str(path))
            return d["P"], d["F"], d["S"]
        seed = {"tr": 42, "ti": 99, "te": 7}[split]
        p = self.codebook.sample(1, seed + r, extrap=split == "te")
        f, s = self.feats_of(p)
        mx.save_safetensors(str(path), {"P": p, "F": f, "S": s})
        print(f"  块缓存 → {path.name}")
        return p, f, s

    def build(self, n_rep: int) -> tuple[mx.array, ...]:
        """→ (Ftr, Ptr, Fi, Pi, Fe, Pe, Str, Si, Se)。训练集 = n_rep 个
        复制块拼接 (逐块缓存, R 增长纯追加); 插值/外推测试集固定 2 块。"""
        tr = [self._block_feats("tr", r) for r in range(n_rep)]
        n_test = 1 if self.codebook.N_OBJECTS > 1 else 2
        ti = [self._block_feats("ti", r) for r in range(n_test)]
        te = [self._block_feats("te", r) for r in range(n_test)]

        def cat(blocks: list, i: int) -> mx.array:
            return mx.concatenate([b[i] for b in blocks])

        return (
            cat(tr, 1), cat(tr, 0), cat(ti, 1), cat(ti, 0), cat(te, 1),
            cat(te, 0), cat(tr, 2), cat(ti, 2), cat(te, 2),
        )

    def feats_of(self, params: mx.array) -> tuple[mx.array, mx.array]:
        """参数行 → 渲染 → (特征 (n,n_feat), 立体统计)。
        单物体拼接 [ẑ,area]; 双层拼接逐层 [u,v,z,area]×2。"""
        cb = self.codebook
        renderer, cam_l, cam_r = Codebook.make_renderer()
        sd = StereoDepth()
        sl = StereoLayers()
        rw = None
        out, stats = [], []
        for p in params.tolist():
            scene = cb.to_scene(p)
            fl = renderer.render(scene, cam_l)
            fr = renderer.render(scene, cam_r)
            vec, rw = self.extractor.of_frame(fl, rw)
            if cb.USES_LAYER_STATS:
                stat = sl.estimate(fl, fr)
                vec = mx.concatenate(
                    [vec, StereoLayers.scaled(mx.array([stat]))[0]]
                )
                stats.append(stat)
            else:
                z_hat, d, area = sd.estimate(fl, fr)
                # 拼接维须缩放到特征方差量级: 裸 area (σ≈600) 会主导 λ 谱,
                # 白化截断阈值 λmax·1e-6 随之抬到 0.36 → 大部分特征方向
                # 被误截 (实测 u R² 0.90→0.73, kind 同步掉)
                vec = mx.concatenate([vec, mx.array([z_hat, area / 1000.0])])
                stats.append([z_hat, d, area])
            mx.eval(vec)
            out.append(vec)
        return mx.stack(out), mx.array(stats, dtype=mx.float32)


    @staticmethod
    def targets(p: mx.array) -> mx.array:
        """参数 → 连续目标 (单物体 4 维 / 双层 8 维)。"""
        if p.shape[1] == 14:
            return p[:, [1, 2, 3, 4, 7, 8, 9, 10]]
        return p[:, 1:5]

    @staticmethod
    def scene_classes(p: mx.array) -> mx.array:
        """参数 → 离散场景因子 (单物体 4 维 / 双层 6 维)。"""
        if p.shape[1] == 14:
            return p[:, [0, 6, 5, 11, 12, 13]].astype(mx.int32)
        return p[:, [0, 5, 6, 7]].astype(mx.int32)
