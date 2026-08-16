"""texture_pipeline.py — 全分辨率特征能否恢复纹理类型/粗糙度 (端到端)。

回答主线接线前的关键问题: 全分辨率逐像素特征 (V=228K) + PCA 白化 +
实例 SPN 能否估计纹理类型(离散)与粗糙度(连续)? 还是纹理指纹被逐像素
白化打散、必须走前景掩码区域统计头?

本地 TexturedCodebook 渲染带 map/roughness 的场景, 复用真实
FeatureExtractor + DataBuilder + MixtureSPN 链路, 不改主线。

两个实验 (各自 train/test 分裂, 实例级 SPN = 核回归/核分类):
  P1 纹理类型 (box 平面, checker/stripes/noise, roughness 固定) → 分类
  P2 粗糙度   (sphere 球面, 无贴图, roughness∈[0.2,0.9] 连续)    → 回归

运行: python src/texture_pipeline.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from cga.engine import (
    AmbientLight,
    Color,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    Scene,
)

from codebook import Codebook
from data_builder import DataBuilder
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from texture_probe import _checker, _gray_noise, _stripes


class TexturedCodebook(Codebook):
    """几何/外观由 params 给定; 材质 = map(tex_id) + roughness。tex=-1 → 无贴图。"""

    N_TEX = 3

    def __init__(self, cfg: InverseConfig):
        super().__init__(cfg)
        w, g = (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)
        self.textures = (
            _checker(16, w, g, 4),
            _stripes(16, w, g, 3),
            _gray_noise(16, 0, 0.3, 0.7),
        )

    def to_scene(self, params) -> Scene:
        kind = int(params[0])
        u, v, s, z = (float(x) for x in params[1:5])
        hue, lcol, ldir = (int(p) for p in params[5:8])
        tex = int(params[8])
        rough = float(params[9])
        x, y = self.unproject(u, v, z)
        geom = self.geometry(kind, s)
        scene = Scene(background=Color(self.cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(DirectionalLight(
            Color(self.LIGHT_COLORS[lcol]), 0.7,
            direction=self.LIGHT_DIRS[ldir],
        ))
        map_ = None if tex < 0 else self.textures[tex]
        scene.add(Mesh(
            geom,
            MeshStandardMaterial(Color(0xFFFFFF), roughness=rough, map=map_),
            position=(x, y, z),
        ))
        return scene


def _params_tex(n: int, seed: int) -> mx.array:
    """box 平面, 固定几何/外观/roughness, 只变 tex_id (3 类)。"""
    rows = [
        [2.0, 72.0, 72.0, 0.6, 2.8, 0.0, 0.0, 0.0, float(i % 3), 0.55]
        for i in range(n)
    ]
    return mx.array(rows, dtype=mx.float32)


def _params_rough(n: int, seed: int) -> mx.array:
    """sphere 球面, 无贴图, roughness 连续随机 ∈[0.2,0.9]。"""
    rng = np.random.default_rng(seed)
    rows = [
        [0.0, 72.0, 72.0, 0.6, 2.8, 0.0, 0.0, 0.0, -1.0, float(rng.uniform(0.2, 0.9))]
        for _ in range(n)
    ]
    return mx.array(rows, dtype=mx.float32)


def _fit_predict(f_tr, f_te, t_tr, cat_tr, cat_sizes, rel_floor):
    """实例级 MixtureSPN: 分层 = 离散首因子; → (连续预测, 离散后验)。"""
    net = MixtureSPN.fit(
        f_tr, t_tr, cat_tr[:, 0], rel_floor=rel_floor,
        scene_classes=cat_tr, cat_sizes=cat_sizes,
    )
    t_pred, cat_p, _ = net.predict(f_te)
    return t_pred, cat_p


def run_tex(data: DataBuilder, cfg: InverseConfig) -> None:
    """P1: 纹理类型 (离散 3 类) 分类。"""
    print("\n== P1 纹理类型 (box 平面, 3 类, 全分辨率特征) ==")
    p_tr = _params_tex(60, seed=0)
    p_te = _params_tex(30, seed=99)
    f_tr, _ = data.feats_of(p_tr)
    f_te, _ = data.feats_of(p_te)
    t_tr = p_tr[:, 9:10]                      # 连续目标占位 (roughness 固定)
    cat_tr = p_tr[:, 8:9].astype(mx.int32)
    cat_te = p_te[:, 8:9].astype(mx.int32)
    _, cat_p = _fit_predict(f_tr, f_te, t_tr, cat_tr, (3,), cfg.sigma_rel_floor)
    acc = float(mx.mean((mx.argmax(cat_p, axis=1) == cat_te[:, 0]).astype(mx.float32)))
    print(f"  tex_id 准确率 = {acc:.3f} (chance 0.333)")


def run_rough(data: DataBuilder, cfg: InverseConfig) -> None:
    """P2: 粗糙度 (连续) 回归。"""
    print("\n== P2 粗糙度 (sphere 球面, 连续, 全分辨率特征) ==")
    p_tr = _params_rough(60, seed=0)
    p_te = _params_rough(30, seed=99)
    f_tr, _ = data.feats_of(p_tr)
    f_te, _ = data.feats_of(p_te)
    t_tr = p_tr[:, 9:10]
    t_te = p_te[:, 9:10]
    cat_tr = mx.zeros((p_tr.shape[0], 1), dtype=mx.int32)  # 无离散因子 → 单层
    t_pred, _ = _fit_predict(f_tr, f_te, t_tr, cat_tr, (1,), cfg.sigma_rel_floor)
    ss_base = float(mx.sum((t_te - mx.mean(t_tr)) ** 2))
    ss_res = float(mx.sum((t_te - t_pred) ** 2))
    r2 = 1.0 - ss_res / max(ss_base, 1e-12)
    rmse = float(mx.sqrt(mx.mean((t_te - t_pred) ** 2)))
    print(f"  roughness R² = {r2:.3f}  RMSE = {rmse:.3f}  (基线 R²=0)")


if __name__ == "__main__":
    cfg = InverseConfig()
    cb = TexturedCodebook(cfg)
    data = DataBuilder(cfg, cb, FeatureExtractor(cfg))
    run_tex(data, cfg)
    run_rough(data, cfg)
