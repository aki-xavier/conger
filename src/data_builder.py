"""DataBuilder: 数据构建 (含缓存)。离散因子全笛卡尔积 × R 复制,
连续因子逐样本随机。无标准化 —— 对角高斯按维学 σ, 天然免尺度。"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from stereo import StereoDepth


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
            f"sv{cb.SAMPLE_V}rp{cb.RENDER_V}"
        )
        # st4 = 立体管线版本号 (4 = 拼接维缩放版), 非模式开关
        return f"mix_{cb.H}x{cb.W}_{feat_tag}_{lvl_tag}_st4"

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
        ti = [self._block_feats("ti", r) for r in range(2)]
        te = [self._block_feats("te", r) for r in range(2)]

        def cat(blocks: list, i: int) -> mx.array:
            return mx.concatenate([b[i] for b in blocks])

        return (
            cat(tr, 1), cat(tr, 0), cat(ti, 1), cat(ti, 0), cat(te, 1),
            cat(te, 0), cat(tr, 2), cat(ti, 2), cat(te, 2),
        )

    def feats_of(self, params: mx.array) -> tuple[mx.array, mx.array]:
        """参数行 (n,8) → 渲染 → (特征 (n, n_feat), 立体统计 (n,3)).
        平行 rig 双渲染, 左帧走 11 通道特征, 视差深度 ẑ 与
        掩码面积拼接为 2 个观测通道 (z 被几何钉死 → s=表观×zc 随解)。"""
        cb = self.codebook
        renderer, cam_l, cam_r = Codebook.make_renderer()
        sd = StereoDepth()
        rw = None
        out, stats = [], []
        for p in params.tolist():
            scene = cb.to_scene(p)
            fl = renderer.render(scene, cam_l)
            fr = renderer.render(scene, cam_r)
            vec, rw = self.extractor.of_frame(fl, rw)
            z_hat, d, area = sd.estimate(fl, fr)
            # 拼接维须缩放到特征方差量级: 裸 area (σ≈600) 会主导 λ 谱,
            # 白化截断阈值 λmax·1e-6 随之抬到 0.36 → 大部分特征方向
            # 被误截 (实测 u R² 0.90→0.73, kind 同步掉)
            vec = mx.concatenate([vec, mx.array([z_hat, area / 1000.0])])
            mx.eval(vec)
            out.append(vec)
            stats.append([z_hat, d, area])
        return mx.stack(out), mx.array(stats, dtype=mx.float32)


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
