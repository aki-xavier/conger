"""Evaluator: 连续量回归指标 (物理单位) + 完整场景离散因子分类。

基线 = 训练均值预测器 (无证据预测): R² 相对它计算 —— 第一性原理的
"什么都不会"参照系。离散场景因子 (kind/hue/lcol/ldir) 以分类
准确率报告, 完整 cga.Scene 重建的全部目标都进入指标。
"""

from __future__ import annotations

import mlx.core as mx

TARGETS = ("u", "v", "s", "z")  # 连续目标列 0..3
UNITS = ("px", "px", "world", "world")
SCENE_FACTORS = (
    ("kind", 0),
    ("hue", 5),
    ("lcol", 6),
    ("ldir", 7),
)


class Evaluator:
    """回归 + 完整场景因子分类指标。"""

    @staticmethod
    def report(
        name: str,
        p_gt: mx.array,  # (M,8) [kind,u,v,s,z,hue,lcol,ldir]
        t_pred: mx.array,  # (M,4) 物理连续目标: u,v,s,z
        scene_pred: tuple[tuple[float, ...], ...],  # (M,8) 预测完整场景参数
        p_train: mx.array,  # (N,8) 训练参数 (基线均值来源)
    ) -> dict[str, float]:
        """打印并返回指标: 逐连续目标 RMSE/R² + 4 个场景因子分类准确率。"""
        gt = p_gt[:, 1:5]
        base = mx.mean(p_train[:, 1:5], axis=0, keepdims=True)  # 均值基线
        ss_base = mx.sum((gt - base) ** 2, axis=0)
        ss_res = mx.sum((gt - t_pred[:, :4]) ** 2, axis=0)
        rmse = mx.sqrt(mx.mean((gt - t_pred[:, :4]) ** 2, axis=0))
        r2 = 1.0 - ss_res / mx.maximum(ss_base, 1e-12)
        pred = mx.array(scene_pred, dtype=mx.float32)
        out: dict[str, float] = {}
        line = f"  {name}:"
        for nm, j in SCENE_FACTORS:
            acc = float(
                mx.mean((pred[:, j] == p_gt[:, j]).astype(mx.float32))
            )
            out[nm] = acc
            line += f" {nm} {acc:.3f} |"
        for j, (nm, un) in enumerate(zip(TARGETS, UNITS, strict=True)):
            out[f"{nm}_rmse"] = float(rmse[j])
            out[f"{nm}_r2"] = float(r2[j])
            line += f" {nm} {float(rmse[j]):.3f}{un} R² {float(r2[j]):.3f} |"
        print(line.rstrip(" |"))
        return out
