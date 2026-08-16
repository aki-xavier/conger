"""Evaluator: 连续量回归指标 (物理单位) + 完整场景离散因子分类。

基线 = 训练均值预测器 (无证据预测): R² 相对它计算 —— 第一性原理的
"什么都不会"参照系。单物体与双层遮挡场景共用同一指标契约。
"""

from __future__ import annotations

import mlx.core as mx

TARGETS = ("u", "v", "s", "z")
LAYERED_TARGETS = ("u0", "v0", "s0", "z0", "u1", "v1", "s1", "z1")
SCENE_FACTORS = (
    ("kind", 0),
    ("hue", 5),
    ("lcol", 6),
    ("ldir", 7),
)
LAYERED_FACTORS = (
    ("kind0", 0),
    ("kind1", 6),
    ("hue0", 5),
    ("hue1", 11),
    ("lcol", 12),
    ("ldir", 13),
)
LAYERED_TARGET_COLS = (1, 2, 3, 4, 7, 8, 9, 10)
# 纹理接线 (10 列参数): 连续目标同单物体 u,v,s,z (roughness 走独立谱形
# 头 RoughnessHead, 由 inverse_app 单独按球面报告); 离散因子 + tex_id
TEXTURED_TARGETS = TARGETS
TEXTURED_FACTORS = SCENE_FACTORS + (("tex", 8),)
TEXTURED_TARGET_COLS = (1, 2, 3, 4)


class Evaluator:
    """回归 + 完整场景因子分类指标。"""

    @staticmethod
    def target_names(p: mx.array) -> tuple[str, ...]:
        """参数宽度 → 连续目标名。"""
        if p.shape[1] == 14:
            return LAYERED_TARGETS
        return TEXTURED_TARGETS if p.shape[1] == 10 else TARGETS

    @staticmethod
    def report(
        name: str,
        p_gt: mx.array,
        t_pred: mx.array,
        scene_pred: tuple[tuple[float, ...], ...],
        p_train: mx.array,
    ) -> dict[str, float]:
        """打印并返回连续目标 RMSE/R² + 离散场景因子分类准确率。"""
        layered = p_gt.shape[1] == 14
        textured = p_gt.shape[1] == 10
        cols = LAYERED_TARGET_COLS if layered else TEXTURED_TARGET_COLS if textured else (1, 2, 3, 4)
        targets = LAYERED_TARGETS if layered else TEXTURED_TARGETS if textured else TARGETS
        factors = LAYERED_FACTORS if layered else TEXTURED_FACTORS if textured else SCENE_FACTORS
        gt = p_gt[:, list(cols)]
        base = mx.mean(p_train[:, list(cols)], axis=0, keepdims=True)
        ss_base = mx.sum((gt - base) ** 2, axis=0)
        ss_res = mx.sum((gt - t_pred[:, : len(cols)]) ** 2, axis=0)
        rmse = mx.sqrt(mx.mean((gt - t_pred[:, : len(cols)]) ** 2, axis=0))
        r2 = 1.0 - ss_res / mx.maximum(ss_base, 1e-12)
        pred = mx.array(scene_pred, dtype=mx.float32)
        out: dict[str, float] = {}
        line = f"  {name}:"
        for nm, j in factors:
            acc = float(mx.mean((pred[:, j] == p_gt[:, j]).astype(mx.float32)))
            out[nm] = acc
            line += f" {nm} {acc:.3f} |"
        for j, nm in enumerate(targets):
            un = "px" if nm.startswith(("u", "v")) else "world"
            out[f"{nm}_rmse"] = float(rmse[j])
            out[f"{nm}_r2"] = float(r2[j])
            line += f" {nm} {float(rmse[j]):.3f}{un} R² {float(r2[j]):.3f} |"
        print(line.rstrip(" |"))
        return out
