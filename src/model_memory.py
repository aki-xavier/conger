"""ModelMemory: 模型内存/硬盘按需加载 + 动态遗忘 (逃生通道 §7.8)。

实测内存画像 (单物体 N=1296): 模型 459.6MB, 其中白化基 basis
(V=228098 × D=497) 占 453.5MB = 98.7%; 分量表 (f_mu/f_var/t_mu/
cat_logp/log_w) 合计 ~6MB。全专家注册表 ~6.3GB。

两个机制:

- **按需加载**: 白化变换 (f_mean+basis) 只在 `_z` 白化时需要; 分量表是
  门控评分/类别契约检查的常驻部分。split 序列化把它们分文件, 分级加载。
- **动态遗忘**: 基内在维截断 (D↓, 内存 + 速度杠杆) 与分量 coreset 驱逐
  (K↓, 推理速度 O(K·D) 杠杆 + 分量内存上界)。与 §2.2 的质心压缩不同:
  驱逐保留原始样本 (coreset 选点不平均), 只在 N≫k_max 时才有意义。

注意 basis 列按升序特征值排列 (`_whiten` 的 eigh 升序), 截断保留尾部
(最高方差) 维。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import mlx.core as mx

from mixture_spn import MixtureSPN
from utils import Utils

_COMPONENT_KEYS = ("log_w", "f_mu", "f_var", "t_mu", "cat_logp")


# ── 按需加载 (split 序列化 + 分级加载) ────────────────────────────────

def split_save(model: MixtureSPN, path: str | Path) -> tuple[Path, Path]:
    """split 序列化: 白化变换 (f_mean+basis) 与分量表分文件。

    返回 (transform_path, components_path)。元数据 (rel_floor/cat_sizes/
    n_stratum) 写入 components 文件头, 供只加载分量表的轻查询。
    """
    path = Path(path)
    t_path = path.with_name(path.name + ".transform.safetensors")
    c_path = path.with_name(path.name + ".components.safetensors")
    mx.save_safetensors(
        str(t_path), {"f_mean": model.f_mean, "basis": model.basis}
    )
    meta = {
        "rel_floor": json.dumps(model.rel_floor),
        "cat_sizes": json.dumps(model.cat_sizes_tuple),
        "n_stratum": json.dumps(model.n_stratum),
    }
    mx.save_safetensors(
        str(c_path),
        {k: getattr(model, k) for k in _COMPONENT_KEYS},
        meta,
    )
    return t_path, c_path


def load_transform(path: str | Path) -> tuple[mx.array, mx.array]:
    """只加载白化变换 (f_mean, basis) —— 模型内存大头, 仅白化需要。"""
    d = mx.load(str(Path(path).with_name(Path(path).name + ".transform.safetensors")))
    return d["f_mean"], d["basis"]


def load_components(path: str | Path) -> tuple[dict, dict]:
    """只加载分量表 + 元数据 (不含 basis), ~6MB, 供门控/契约检查。"""
    c_path = Path(path).with_name(Path(path).name + ".components.safetensors")
    d = cast(dict, mx.load(str(c_path)))
    hd = Utils.st_metadata(str(c_path)).get("__metadata__", {})
    meta = {k: json.loads(v) for k, v in hd.items()}
    if "cat_sizes" in meta:
        meta["cat_sizes"] = tuple(meta["cat_sizes"])
    return d, meta


def assemble(path: str | Path) -> MixtureSPN:
    """从 split 文件完整装配模型 (= 全量加载, 但保留分级边界)。"""
    f_mean, basis = load_transform(path)
    comp, meta = load_components(path)
    return MixtureSPN(
        comp["log_w"],
        comp["f_mu"],
        comp["f_var"],
        comp["t_mu"],
        comp["cat_logp"],
        meta["rel_floor"],
        f_mean,
        basis,
        meta["cat_sizes"],
        meta["n_stratum"],
    )


# ── 动态遗忘 ──────────────────────────────────────────────────────────

def truncate_basis(model: MixtureSPN, d_max: int) -> MixtureSPN:
    """白化基内在维截断: 保留最高方差的 d_max 维。

    基是内存大头 (V×D), D↓ 线性收缩内存 + 加速白化 (V×D matmul) 与
    似然 (K×D)。basis 列升序特征值 → 取尾部 (最高方差) 维。
    """
    d = model.f_mu.shape[1]
    d_max = min(max(1, d_max), d)
    if d_max == d:
        return model
    assert model.basis is not None
    return MixtureSPN(
        model.log_w,
        model.f_mu[:, -d_max:],
        model.f_var[:, -d_max:],
        model.t_mu,
        model.cat_logp,
        model.rel_floor,
        model.f_mean,
        model.basis[:, -d_max:],
        model.cat_sizes_tuple,
        model.n_stratum,
    )


def _coreset(Z: mx.array, k: int, rng: mx.array) -> mx.array:
    """贪婪最远点 (farthest-point): 返回 Z 的 k 个覆盖最广的行下标。

    保留流形覆盖、避免近重复分量; 是 §2.2 质心压缩在「不平均、只选点」
    意义上的替代 —— 小数据不触发, 大数据驱逐近重复才安全。
    """
    n = Z.shape[0]
    if k >= n:
        return mx.arange(n, dtype=mx.int32)
    start = int(mx.random.randint(0, n, shape=(1,), key=rng)[0])
    sel = [start]
    d2 = mx.sum((Z - Z[start][None, :]) ** 2, axis=1)
    while len(sel) < k:
        nxt = int(mx.argmax(d2))
        sel.append(nxt)
        d2 = mx.minimum(d2, mx.sum((Z - Z[nxt][None, :]) ** 2, axis=1))
    return mx.array(sel, dtype=mx.int32)


def forget_components(
    model: MixtureSPN,
    k_max: int,
    policy: str = "coreset",
    seed: int = 0,
) -> MixtureSPN:
    """分量驱逐: 把 K 压到 k_max (coreset/random), 逐 stratum 保持比例。

    推理 O(K·D) → K↓ 加速; 分量内存上界。驱逐后按保留集重估逐层 tied
    方差与均匀权重 (与全量 fit 同估计量, 不引入质心偏差)。
    """
    k = model.f_mu.shape[0]
    k_max = max(model.n_stratum, min(k_max, k))
    if k_max >= k:
        return model
    stratum = mx.argmax(model.cat_logp[:, : model.n_stratum], axis=1)
    rng = mx.random.key(seed)
    kept: list[int] = []
    for j in range(model.n_stratum):
        sel = Utils.nonzero(stratum == j)
        nj = sel.shape[0]
        kj = max(1, round(k_max * nj / k))
        kj = min(kj, nj)
        if policy == "random":
            pick = mx.random.permutation(sel, key=rng)[:kj]
        else:
            pick = sel[_coreset(model.f_mu[sel], kj, rng)]
        kept.extend(int(i) for i in cast(list, pick.tolist()))
    kept.sort()
    idx = mx.array(kept, dtype=mx.int32)
    s_kept = stratum[idx]
    gvar = MixtureSPN._tied_vars(
        model.f_mu[idx], s_kept, model.rel_floor, model.n_stratum
    )
    f_var = gvar[s_kept]
    log_w = mx.full((len(kept),), -math.log(len(kept)))
    return MixtureSPN(
        log_w,
        model.f_mu[idx],
        f_var,
        model.t_mu[idx],
        model.cat_logp[idx],
        model.rel_floor,
        model.f_mean,
        model.basis,
        model.cat_sizes_tuple,
        model.n_stratum,
    )


def model_size_mb(model: MixtureSPN) -> float:
    """模型张量字节数 (MB), 用于内存画像/基准。"""
    tot = 0
    for name in ("log_w", "f_mu", "f_var", "t_mu", "cat_logp", "f_mean", "basis"):
        v = getattr(model, name)
        tot += v.size * (v.itemsize if hasattr(v, "itemsize") else 4)
    return tot / 1e6
