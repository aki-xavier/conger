"""texture_roughness_paths.py — 路径2/3: 让 roughness 可估的两种修法实测。

主线 `--n-textures 3` 实测 roughness 负 R² (被几何/外观/纹理共同变化
淹没 + R=1 无复制密度)。本脚本独立对照两条修法:

  路径2  限定 specular 瓣空间可见的 kind: sphere (空间高光瓣) vs box
         (正面均匀着色, 瓣不可见), 各自在 held-out 几何上测 roughness R²。
  路径3  shape 轴专用头: 前景掩码内 8 张谱形图 (slope/residual/bump/
         centroid/spread/skew/kurt/mean_ori) 的 (mean,std) 16 维描述子
         (gc=False, 避免 Wiener floor 随 roughness 泄漏), 对比全分辨率
         特征 (gc=True 主线约定)。

几何 (u,v,s,z) 随机、粗糙度连续 ∈[0.2,0.9]、图元色/光色/光向固定。
运行: python src/texture_roughness_paths.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from codebook import Codebook
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from riesz import RieszWavelet
from stereo import StereoDepth

STAT_MAPS = (
    "slope", "residual", "bump", "centroid", "spread", "skew", "kurt", "mean_ori",
)


def make_params(kind: int, n: int, seed: int) -> mx.array:
    """(n,10) [kind,u,v,s,z,hue=0,lcol=0,ldir=0,tex=-1,roughness]。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        u = rng.uniform(60.0, 84.0)
        v = rng.uniform(60.0, 84.0)
        s = rng.uniform(0.4, 0.7)
        z = rng.uniform(2.6, 3.6)
        rough = rng.uniform(*Codebook.ROUGHNESS)
        rows.append([float(kind), u, v, s, z, 0.0, 0.0, 0.0, -1.0, rough])
    return mx.array(rows, dtype=mx.float32)


def render_feats(cb: Codebook, extractor: FeatureExtractor, params: mx.array):
    """→ (full (N,V), shape (N,16), rough (N,))。full 含立体统计 [ẑ,area/1000]。"""
    renderer, cam_l, cam_r = Codebook.make_renderer()
    sd = StereoDepth()
    rw = None
    rw2 = None
    full, shapes, roughs = [], [], []
    for p in params.tolist():
        scene = cb.to_scene(tuple(p))
        fl = renderer.render(scene, cam_l)
        fr = renderer.render(scene, cam_r)
        # full-res 特征 (主线约定: gc=True 亮度, 拼 [ẑ, area/1000])
        vec, rw = extractor.of_frame(fl, rw)
        z_hat, _, area = sd.estimate(fl, fr)
        vec = mx.concatenate([vec, mx.array([z_hat, area / 1000.0])])
        full.append(vec)
        # shape 轴描述子 (gc=False, 谱形判别避免 floor 泄漏)
        lum = FeatureExtractor.frame_lum(fl)
        if rw2 is None:
            rw2 = RieszWavelet(lum)
        else:
            rw2.update(lum)
        f = rw2.features(gain_control=False)
        m = StereoDepth.foreground_weights(fl) > 0.01
        w = m.astype(mx.float32)
        tot = float(mx.sum(w))
        desc = []
        for name in STAT_MAPS:
            a = getattr(f, name)
            mean = float(mx.sum(a * w)) / tot
            d = a - mean
            desc += [mean, float(mx.sqrt(mx.sum(d * d * w) / tot))]
        shapes.append(desc)
        roughs.append(p[9])
        mx.eval(vec)
    return mx.stack(full), np.array(shapes, dtype=np.float64), np.array(roughs)


def r2(pred: np.ndarray, gt: np.ndarray, base: float) -> float:
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_base = float(np.sum((gt - base) ** 2))
    return 1.0 - ss_res / max(ss_base, 1e-12)


def nn_r2(Xtr, ytr, Xte, yte) -> float:
    """z-score 1-NN 核回归的 held-out R²。"""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    idx = np.argmin(((A[None, :, :] - B[:, None, :]) ** 2).sum(axis=2), axis=1)
    return r2(ytr[idx], yte, float(ytr.mean()))


def spn_r2(full_tr, rough_tr, full_te, rough_te, rel_floor: float) -> float:
    """MixtureSPN (白化 + 实例核回归) 的 held-out roughness R²。"""
    zeros_tr = mx.zeros((full_tr.shape[0], 1), dtype=mx.int32)
    t_tr = mx.array(rough_tr, dtype=mx.float32)[:, None]
    net = MixtureSPN.fit(
        full_tr, t_tr, zeros_tr[:, 0], rel_floor=rel_floor,
        scene_classes=zeros_tr, cat_sizes=(1,),
    )
    t_pred, _, _ = net.predict(full_te)
    return r2(np.asarray(t_pred[:, 0]), rough_te, float(rough_tr.mean()))


def run_kind(cb: Codebook, extractor: FeatureExtractor, kind: int) -> None:
    name = {0: "sphere", 2: "box"}[kind]
    p_tr = make_params(kind, 160, seed=1000 + kind)
    p_te = make_params(kind, 40, seed=2000 + kind)
    f_tr, sh_tr, r_tr = render_feats(cb, extractor, p_tr)
    f_te, sh_te, r_te = render_feats(cb, extractor, p_te)
    r_full = spn_r2(f_tr, r_tr, f_te, r_te, 1e-2)
    r_shape = nn_r2(sh_tr, r_tr, sh_te, r_te)
    print(f"\n== {name} (kind={kind}) roughness R² (held-out 几何) ==")
    print(f"  全分辨率特征 (V={f_tr.shape[1]}): R² = {r_full:+.3f}")
    print(f"  shape 轴描述子 (16d)      : R² = {r_shape:+.3f}")


if __name__ == "__main__":
    cfg = InverseConfig()
    cb = Codebook(cfg)
    ext = FeatureExtractor(cfg)
    run_kind(cb, ext, 0)  # sphere: 空间 specular 瓣
    run_kind(cb, ext, 2)  # box: 正面均匀着色
