"""SPN 逆渲染 demo: cga engine 渲染合成场景 → Riesz 特征 → SPN 反推 3D 场景码。

场景: 暗背景 + 单个浅色图元 (sphere / cylinder / box), 中心投影在 8×6
网格上、尺寸两档 —— 场景码 (kind, gx, gy, size) 即 cga 三维建模的
离散编码 (code → cga Scene 对象可逆)。

训练数据: 均匀随机采样场景码 → cga engine 渲染 144×144 → Riesz 特征
(深度通道改走亮度: engine 无深度输出) 块池化 8×6×3 通道 → 联合矩阵
[特征(144) | 码(4)] → learn_spn。
推理: 枚举 288 个场景码, SPN 后验 argmax → 重建 cga 场景 (三维建模)。

评估: 码准确率 / 逐变量准确率 / 多数类与最近模板基线 / GT vs 重建渲染。

运行: cd src && python demo_inverse.py [--quick] [--no-cache]
自检: --quick 内置断言 (小数据集 + 阈值按全量运行标定)。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
from cga.engine import (
    AmbientLight,
    BoxGeometry,
    Color,
    CylinderGeometry,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    PerspectiveCamera,
    Renderer,
    Scene,
    SphereGeometry,
)

from riesz import RieszWavelet
from spn import SPN, learn_spn

# ── 场景目录 ──────────────────────────────────────────────────────
H = W = 144
FX = FY = 90.0  # 引擎 fy = H/(2·tan(fov/2)) → 反解 fov
FOV = 2.0 * math.degrees(math.atan((H / 2.0) / FY))
CAM_Z = 5.5  # 相机位置 z (世界), 看向原点
Z0 = 3.0  # 图元中心 z → 相机空间深度 2.5
GRID = (8, 6)
SIZES = (0.35, 0.6)  # 半径/半边长 两档
KINDS = ("sphere", "cylinder", "box")
N_KIND, N_GX, N_GY, N_SIZE = 3, 8, 6, 2
Z0S = (2.5, 3.0, 3.5, 4.0)  # 图元中心世界 z, 4 档 → 单目深度线索 (近大远小)
N_Z = len(Z0S)
N_CODES = N_KIND * N_GX * N_GY * N_SIZE * N_Z  # 1152
FEAT_CH = ("log_mag", "phase_coh", "ori_R")  # 判别力前三 (经验标定见旧 vbgmm)
N_FEAT = len(FEAT_CH) * N_GX * N_GY  # 144
CODE_COLS = tuple(range(N_FEAT, N_FEAT + 5))
CARD = dict(zip(CODE_COLS, (N_KIND, N_GX, N_GY, N_SIZE, N_Z)))

OBJ_COLOR = 0xE8E8E8  # 浅色物体 / 深色背景 → 亮度高对比, Riesz 边缘强
BG_COLOR = 0x141414


# ── 场景码 ⇄ cga Scene ───────────────────────────────────────────


def idx_to_code(i: int) -> tuple[int, int, int, int, int]:
    """码下标 → (kind, gx, gy, size, z); 枚举序字典序, z 在最低位。"""
    z = i % N_Z
    i //= N_Z
    size = i % N_SIZE
    i //= N_SIZE
    gy = i % N_GY
    i //= N_GY
    gx = i % N_GX
    return (i // N_GX, gx, gy, size, z)


def code_to_idx(code: tuple[int, int, int, int, int]) -> int:
    """(kind, gx, gy, size, z) → 码下标 (idx_to_code 逆)。"""
    kind, gx, gy, size, z = code
    return ((((kind * N_GX + gx) * N_GY + gy) * N_SIZE + size) * N_Z + z)


def all_codes() -> mx.array:
    return mx.array([list(idx_to_code(i)) for i in range(N_CODES)], dtype=mx.float32)


def code_to_scene(code: tuple[int, int, int, int, int]) -> Scene:
    """场景码 → cga Scene (三维建模; 逆映射的落点)。

    深度线索: 图元中心始终投影到网格中心 (X/Y 按 z 反投影),
    投影大小 = size·f/zc 随 z 变 —— 近大远小, 与人类单目深度
    线索一致 (熟悉尺寸歧义: size 与 z 的乘积混淆, 见 demo 输出)。
    """
    kind, gx, gy, size, z = code
    u = (gx + 0.5) * W / N_GX
    v = (gy + 0.5) * H / N_GY
    zc = CAM_Z - Z0S[z]
    x = (u - (W - 1) / 2.0) * zc / FX
    y = ((H - 1) / 2.0 - v) * zc / FY  # 相机 Y 向下 → 世界 Y 向上
    s = SIZES[size]
    if kind == 0:
        geom = SphereGeometry(s)
    elif kind == 1:
        geom = CylinderGeometry(s, length=2.2 * s)  # 有限柱: 竖向可观测
    else:
        geom = BoxGeometry(2 * s, 2 * s, 2 * s)
    scene = Scene(background=Color(BG_COLOR))
    scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
    scene.add(DirectionalLight(Color(0xFFFFFF), 0.7, direction=(0.3, -0.7, 0.4)))
    scene.add(
        Mesh(
            geom,
            MeshStandardMaterial(Color(OBJ_COLOR), roughness=0.55),
            position=(x, y, Z0S[z]),
        )
    )
    return scene


def make_renderer() -> tuple[Renderer, PerspectiveCamera]:
    renderer = Renderer(H, W, aa=1)
    cam = PerspectiveCamera(
        fov=FOV, aspect=1.0, near=0.1, far=50.0,
        position=(0.0, 0.0, CAM_Z), target=(0.0, 0.0, 0.0),
    )
    cam.look_at((0.0, 0.0, 0.0))
    return renderer, cam


# ── 特征: 亮度 → Riesz → 块池化 ──────────────────────────────────


def frame_lum(frame: mx.array) -> mx.array:
    """(H,W,4) uint8 → (H,W) float32 亮度 [0,1]。"""
    rgb = frame[..., :3].astype(mx.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def block_pool(fm: mx.array) -> mx.array:
    """(H,W) → (N_GY, N_GX) 块均值 (与场景网格对齐)。"""
    return fm.reshape(N_GY, H // N_GY, N_GX, W // N_GX).mean(axis=(1, 3))


def feature_labels() -> list[str]:
    """特征列语义名: 通道@(gx,gy), 与 block_pool 列序一致 (通道主序)。"""
    return [
        f"{ch}@({gx},{gy})"
        for ch in FEAT_CH
        for gy in range(N_GY)
        for gx in range(N_GX)
    ]


def features_of_lum(
    lum: mx.array, rw: RieszWavelet | None
) -> tuple[mx.array, RieszWavelet]:
    """亮度图 → 特征向量 (N_FEAT,)。复用 RieszWavelet 实例 (核只建一次)。"""
    if rw is None:
        rw = RieszWavelet(lum)
    else:
        rw.update(lum)
    f = rw.features()
    vec = mx.concatenate([block_pool(getattr(f, ch)).reshape(-1) for ch in FEAT_CH])
    return vec, rw


# ── 数据构建 (含缓存) ────────────────────────────────────────────


def build_data(
    n_train: int, n_test: int, use_cache: bool
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """→ (Xtr, Ctr, Xte, Cte): 特征 (n, N_FEAT) + 码 (n, 4), 均 float32。"""
    cache = Path(__file__).resolve().parent.parent / "artifacts"
    cache.mkdir(exist_ok=True)
    tag = f"inv_{H}x{W}_g{N_GX}x{N_GY}_{n_train}_{n_test}.npz"
    path = cache / tag
    if use_cache and path.exists():
        d = mx.load(str(path))
        return d["Xtr"], d["Ctr"], d["Xte"], d["Cte"]

    tr = mx.random.randint(0, N_CODES, shape=(n_train,), key=mx.random.key(42)).tolist()
    te = mx.random.randint(0, N_CODES, shape=(n_test,), key=mx.random.key(99)).tolist()
    renderer, cam = make_renderer()
    rw: RieszWavelet | None = None

    def feats_of(idxs: list[int]) -> mx.array:
        nonlocal rw
        out = []
        for i in idxs:
            scene = code_to_scene(idx_to_code(i))
            frame = renderer.render(scene, cam)
            vec, rw = features_of_lum(frame_lum(frame), rw)
            out.append(vec)
        return mx.stack(out)

    x_tr = feats_of(tr)
    x_te = feats_of(te)
    c_tr = mx.array([list(idx_to_code(i)) for i in tr], dtype=mx.float32)
    c_te = mx.array([list(idx_to_code(i)) for i in te], dtype=mx.float32)
    mx.savez(str(path), Xtr=x_tr, Ctr=c_tr, Xte=x_te, Cte=c_te)
    print(f"数据缓存 → {path.name}")
    return x_tr, c_tr, x_te, c_te


def standardize(
    x_tr: mx.array, x_te: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """逐特征 z-score (训练集统计) → (z_tr, z_te, mu, sd)。

    mu/sd 随模型保存: 加载模型推理必须用同一统计。"""
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = mx.maximum(x_tr.std(axis=0, keepdims=True), 1e-6)
    return (x_tr - mu) / sd, (x_te - mu) / sd, mu, sd


# ── 评估 ──────────────────────────────────────────────────────────


def evaluate(pred_i: list[int], gt_i: list[int]) -> dict[str, float]:
    pred_codes = [idx_to_code(p) for p in pred_i]
    gt_codes = [idx_to_code(g) for g in gt_i]
    n = len(gt_i)
    acc = {
        "code": sum(p == g for p, g in zip(pred_i, gt_i, strict=True)) / n,
        "kind": sum(
            p[0] == g[0] for p, g in zip(pred_codes, gt_codes, strict=True)
        ) / n,
        "gx": sum(p[1] == g[1] for p, g in zip(pred_codes, gt_codes, strict=True)) / n,
        "gy": sum(p[2] == g[2] for p, g in zip(pred_codes, gt_codes, strict=True)) / n,
        "size": sum(
            p[3] == g[3] for p, g in zip(pred_codes, gt_codes, strict=True)
        ) / n,
        "z": sum(
            p[4] == g[4] for p, g in zip(pred_codes, gt_codes, strict=True)
        ) / n,
    }
    return acc


def build_prior(name: str) -> mx.array | None:
    """码先验 log P(c) (外部知识注入, 对应 docs/prior.md 先验体系)。

    flat: 均匀先验 (None, 纯数据似然); edge: 一般视角先验 ——
    图元中心不该贴图像边缘 (gx∈{0,7} 或 gy∈{0,5} 权重压低);
    familiar: 熟悉尺寸先验 —— 大尺寸更常见 (size 偏态 0.7/0.3)。
    返回 (N_CODES,) log 权重; softmax 归一在后验内部完成。
    """
    if name == "flat":
        return None
    w = mx.ones(N_CODES)
    for i in range(N_CODES):
        _, gx, gy, size, _ = idx_to_code(i)
        if name == "edge":
            if gx in (0, N_GX - 1) or gy in (0, N_GY - 1):
                w[i] = 0.3
        elif name == "familiar":
            w[i] = 0.7 if size == 1 else 0.3
        else:
            raise ValueError(f"未知先验: {name}")
    return mx.log(w)


def baseline_majority(tr: list[int], te: list[int]) -> float:
    """多数类: 全测样本押训练集最常见的码。"""
    most = max(set(tr), key=tr.count)
    return sum(m == most for m in te) / len(te)


def baseline_template(
    x_tr: mx.array, c_tr: mx.array, x_te: mx.array, te: list[int]
) -> float:
    """最近模板: 每码取训练特征均值, 测试特征 L2 最近邻 (未见码无法命中)。"""
    code_i = [
        c_tr[:, 0].astype(mx.int32),
        c_tr[:, 1].astype(mx.int32),
        c_tr[:, 2].astype(mx.int32),
        c_tr[:, 3].astype(mx.int32),
        c_tr[:, 4].astype(mx.int32),
    ]
    templates: list[mx.array] = []
    present: list[int] = []
    for i in range(N_CODES):
        kind, gx, gy, size, z = idx_to_code(i)
        sel = (
            (code_i[0] == kind)
            & (code_i[1] == gx)
            & (code_i[2] == gy)
            & (code_i[3] == size)
            & (code_i[4] == z)
        )
        cnt = int(mx.sum(sel))
        if cnt == 0:
            continue
        idx = Utils_nonzero(sel)
        templates.append(mx.sum(x_tr[idx], axis=0) / cnt)
        present.append(i)
    tm = mx.stack(templates)  # (P, V)
    dd = mx.sum((x_te[:, None, :] - tm[None, :, :]) ** 2, axis=2)
    pred = [present[int(mx.argmin(d))] for d in dd]
    return sum(p == g for p, g in zip(pred, te, strict=True)) / len(te)


def Utils_nonzero(sel: mx.array) -> mx.array:
    """布尔掩码 → 索引 (MLX 无布尔索引; 选中位按下标, 未选中按 N)。"""
    flat = sel.reshape(-1)
    k = int(mx.sum(flat))
    key = mx.where(flat, mx.arange(flat.shape[0]), flat.shape[0])
    return mx.argsort(key)[:k]


# ── 可视化 ────────────────────────────────────────────────────────


def plot_panel(
    x_te: mx.array,
    post: mx.array,
    gt_i: list[int],
    pred_i: list[int],
    out: Path,
) -> None:
    """3 个测试样本: GT/Pred 渲染 + log_mag 块图 + P(gx,gy) 热图。"""
    renderer, cam = make_renderer()
    n_show = min(3, len(gt_i))
    picks = (
        [0, len(gt_i) // 2, len(gt_i) - 1] if len(gt_i) >= 3 else list(range(n_show))
    )
    fig, axes = plt.subplots(n_show, 5, figsize=(17, 3.4 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    cols = [
        "GT render", "GT log_mag blocks", "Pred render",
        "Pred log_mag blocks", "P(gx,gy|img)",
    ]
    for row, i in enumerate(picks):
        gt_scene = code_to_scene(idx_to_code(gt_i[i]))
        pd_scene = code_to_scene(idx_to_code(pred_i[i]))
        f_gt = renderer.render(gt_scene, cam)
        f_pd = renderer.render(pd_scene, cam)
        axes[row, 0].imshow(f_gt[..., :3].astype(mx.int32))
        axes[row, 2].imshow(f_pd[..., :3].astype(mx.int32))
        lg = x_te[i, :N_GX * N_GY].reshape(N_GY, N_GX)
        axes[row, 1].imshow(lg, cmap="viridis")
        lg_p = x_te[i, :N_GX * N_GY].reshape(N_GY, N_GX)
        axes[row, 3].imshow(lg_p, cmap="viridis")
        pg = post[i].reshape(N_KIND, N_GX, N_GY, N_SIZE, N_Z)
        pgy = mx.exp(
            mx.logsumexp(pg, axis=(0, 3, 4)) - mx.logsumexp(pg)
        )
        axes[row, 4].imshow(pgy.T, cmap="hot", origin="lower")
        for c in range(5):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
        ok = "✓" if pred_i[i] == gt_i[i] else "✗"
        axes[row, 0].set_title(f"GT  code {idx_to_code(gt_i[i])}")
        axes[row, 2].set_title(f"Pred code {idx_to_code(pred_i[i])} {ok}")
    for c, name in enumerate(cols):
        if n_show == 1:
            axes[0, c].set_xlabel(name, fontsize=9)
        else:
            axes[0, c].set_title(name, fontsize=9)
    fig.suptitle(
        "SPN inverse rendering: GT (cga 3D model) vs single-image reconstruction",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_metrics(acc: dict[str, float], base: dict[str, float], out: Path) -> None:
    names = ["code", "kind", "gx", "gy", "size", "z"]
    vals = [acc[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(range(len(names)), vals, color="#4C72B0")
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
            ha="center", fontsize=9,
        )
    for j, (name, v) in enumerate(base.items(), start=len(names)):
        ax.bar(j, v, color="#DD8452")
        ax.text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names) + len(base)))
    ax.set_xticklabels(names + list(base.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.axhline(1 / N_CODES, color="gray", ls=":", lw=1)
    ax.text(
        len(names) + len(base) - 0.6, 1 / N_CODES + 0.01,
        "chance", fontsize=8, color="gray",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ── 主流程 ────────────────────────────────────────────────────────


def main(
    quick: bool,
    use_cache: bool,
    model_path: Path | None,
    tree: bool,
    prior_name: str,
) -> None:
    # 全量: 4000 训练 (码空间 1152, ≈3.5 样本/码); quick: 600 功能自检
    n_train = 600 if quick else 4000
    n_test = 80 if quick else 200
    min_n = 8 if quick else 3  # 叶最小行数: 小 = 叶码纯 (后验锐)
    print(
        f"[1/5] 数据: train {n_train} / test {n_test} "
        f"(cache={'on' if use_cache else 'off'})"
    )
    x_tr, c_tr, x_te, c_te = build_data(n_train, n_test, use_cache)

    # 模型: 存在 → 加载 (跳过学习, 用模型内 mu/sd); 否则训练并保存
    if model_path is not None and model_path.exists():
        print(f"[2/5] 加载模型 {model_path}")
        spn, extra = SPN.load(model_path)
        mu, sd = extra["mu"], extra["sd"]
        # 评估基线仍需要标准化的 x_tr; 用模型内统计, 与训练时一致
        x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    else:
        x_tr, x_te, mu, sd = standardize(x_tr, x_te)
        assert mx.all(mx.isfinite(x_tr)), "特征含 NaN/inf"
        print("[2/5] learn_spn 结构学习 ...")
        xj = mx.concatenate([x_tr, c_tr], axis=1)
        spn = learn_spn(
            xj, disc_cols=set(CODE_COLS), card=CARD, min_n=min_n, max_depth=14
        )
        print(f"      根节点: {type(spn.root).__name__}")
        if model_path is not None:
            spn.save(model_path, {"mu": mu, "sd": sd})
            print(f"      模型已保存 → {model_path}")

    print("[3/5] 推理: 枚举场景码后验")
    post = spn.posterior(x_te, all_codes())  # (n_test, N_CODES) log 后验
    assert mx.all(mx.isfinite(post)), "后验含 NaN/inf"
    pred_i = mx.argmax(post, axis=1).tolist()
    gt_i = [code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()]

    print("[4/5] 评估 + 基线")
    acc = evaluate(pred_i, gt_i)
    tr_codes = [
        code_to_idx(tuple(int(v) for v in row)) for row in c_tr.tolist()
    ]
    base_maj = baseline_majority(tr_codes, gt_i)
    base_tpl = baseline_template(x_tr, c_tr, x_te, gt_i)
    base = {"majority": base_maj, "template": base_tpl}
    print(
        f"      码: {acc['code']:.3f}  kind: {acc['kind']:.3f}  "
        f"gx: {acc['gx']:.3f}  gy: {acc['gy']:.3f}  "
        f"size: {acc['size']:.3f}  z: {acc['z']:.3f}"
    )
    print(f"      基线: majority {base_maj:.3f} / template {base_tpl:.3f}")

    prior = build_prior(prior_name)
    if prior is not None:
        post_p = spn.posterior(x_te, all_codes(), log_prior=prior)
        pred_p = mx.argmax(post_p, axis=1).tolist()
        acc_p = evaluate(pred_p, gt_i)
        print(
            f"      注入先验[{prior_name}]: 码 {acc_p['code']:.3f}  "
            f"kind {acc_p['kind']:.3f}  gx {acc_p['gx']:.3f}  "
            f"gy {acc_p['gy']:.3f}  size {acc_p['size']:.3f}  z {acc_p['z']:.3f}"
        )

    print("[5/5] 图 → artifacts/")
    artifacts = Path(__file__).resolve().parent.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    plot_panel(x_te, post, gt_i, pred_i, artifacts / "inverse_panel.png")
    plot_metrics(acc, base, artifacts / "inverse_metrics.png")

    if tree:
        labels = dict(enumerate(feature_labels()))
        labels.update(dict(zip(CODE_COLS, ("kind", "gx", "gy", "size", "z"))))
        code_names = {
            CODE_COLS[0]: dict(enumerate(KINDS)),
            CODE_COLS[1]: {i: f"gx={i}" for i in range(N_GX)},
            CODE_COLS[2]: {i: f"gy={i}" for i in range(N_GY)},
            CODE_COLS[3]: {i: f"s={SIZES[i]}" for i in range(N_SIZE)},
            CODE_COLS[4]: {i: f"z={Z0S[i]}" for i in range(N_Z)},
        }
        txt = spn.tree_str(labels, code_names)
        print(txt)
        (artifacts / "spn_tree.txt").write_text(txt)
        # 功能分工: 统计各分裂轴 (哪个码维度被哪些 Sum 节点负责)
        import re
        from collections import Counter

        axes = Counter(re.findall(r"分裂轴 (\w+):", txt))
        axes.pop("码分布相近", None)
        func_names = {
            "kind": "形状辨识 (sphere/cylinder/box)",
            "z": "深度估计 (近大远小, 单目线索)",
            "gx": "横向定位",
            "gy": "纵向定位",
            "size": "尺寸估计",
        }
        print("\n── 功能分工 (Sum 节点数 × 职责) ──")
        for ax, cnt in axes.most_common():
            print(f"  {ax:<5} ×{cnt:>3}  → {func_names.get(ax, ax)}")
        print("树结构 → artifacts/spn_tree.txt")

    # ── 自检断言 (阈值按 2026-08-11 实测标定, 留安全余量) ────────────
    # 全量 N=4000/min_n=3 (码空间 1152): 见当日全量运行记录
    # quick  N=600/min_n=8: 见当日 quick 运行记录
    if quick:
        # quick N=600/min_n=8 实测: code 0.025 kind 0.40 gx 0.55
        # gy 0.64 size 0.55 z 0.35
        assert acc["code"] > 0.02, f"quick: 码准确率过低 {acc['code']:.3f}"
        assert acc["kind"] > 0.30, f"quick: kind 过低 {acc['kind']:.3f}"
        assert acc["gx"] > 0.35, f"quick: gx 过低 {acc['gx']:.3f}"
        assert acc["gy"] > 0.50, f"quick: gy 过低 {acc['gy']:.3f}"
        assert acc["size"] > 0.45, f"quick: size 过低 {acc['size']:.3f}"
        assert acc["z"] > 0.30, f"quick: z 过低 {acc['z']:.3f}"
    else:
        # 全量 N=4000/min_n=3 实测 (码空间 1152): code 0.470 kind 0.835
        # gx 0.895 gy 0.855 size 0.885 z 0.735; template 0.965
        assert acc["code"] > 0.40, f"码准确率过低 {acc['code']:.3f}"
        assert acc["kind"] > 0.78, f"kind 过低 {acc['kind']:.3f}"
        assert acc["gx"] > 0.85, f"gx 过低 {acc['gx']:.3f}"
        assert acc["gy"] > 0.80, f"gy 过低 {acc['gy']:.3f}"
        assert acc["size"] > 0.83, f"size 过低 {acc['size']:.3f}"
        assert acc["z"] > 0.68, f"z 过低 {acc['z']:.3f}"
    print("demo_inverse: 自检 ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小数据集自检模式")
    ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
    ap.add_argument(
        "--model",
        default=None,
        help="SPN 模型路径 (pickle); 存在则加载跳过学习, 否则训练后保存",
    )
    ap.add_argument(
        "--tree",
        action="store_true",
        help="打印 SPN 树结构 (带语义列名) 并存 artifacts/spn_tree.txt",
    )
    ap.add_argument(
        "--prior",
        default="flat",
        choices=("flat", "edge", "familiar"),
        help="推理时注入的码先验 (贝叶斯 P(S)): flat=均匀, "
        "edge=一般视角(图元不贴边), familiar=熟悉尺寸(size 偏态)",
    )
    args = ap.parse_args()
    model = Path(args.model) if args.model else None
    main(
        quick=args.quick,
        use_cache=not args.no_cache,
        model_path=model,
        tree=args.tree,
        prior_name=args.prior,
    )
