"""Evaluator: 连续量回归指标 (物理单位) + kind 分类准确率。

基线 = 训练均值预测器 (无证据预测): R² 相对它计算 —— 第一性原理的
"什么都不会"参照系, 比离散时代的多数类/模板基线更基本。"""

from __future__ import annotations

import mlx.core as mx

TARGETS = ("u", "v", "s", "z")  # 参数列 1..4 (列 0 是 kind)
UNITS = ("px", "px", "world", "world")


class Evaluator:
    """回归 + 分类指标。"""

    @staticmethod
    def report(
        name: str,
        p_gt: mx.array,  # (M,5) [kind,u,v,s,z]
        t_pred: mx.array,  # (M,4) E[t|x]
        kind_pred: mx.array,  # (M,) int
        p_train: mx.array,  # (N,5) 训练参数 (基线均值来源)
    ) -> dict[str, float]:
        """打印并返回指标: 逐目标 RMSE/R² + kind 准确率。"""
        gt = p_gt[:, 1:]
        base = mx.mean(p_train[:, 1:], axis=0, keepdims=True)  # 均值基线
        ss_base = mx.sum((gt - base) ** 2, axis=0)
        ss_res = mx.sum((gt - t_pred) ** 2, axis=0)
        rmse = mx.sqrt(mx.mean((gt - t_pred) ** 2, axis=0))
        r2 = 1.0 - ss_res / mx.maximum(ss_base, 1e-12)
        kind_acc = float(
            mx.mean((kind_pred == p_gt[:, 0].astype(mx.int32)).astype(mx.float32))
        )
        out: dict[str, float] = {"kind": kind_acc}
        line = f"  {name}: kind {kind_acc:.3f}"
        for j, (nm, un) in enumerate(zip(TARGETS, UNITS, strict=True)):
            out[f"{nm}_rmse"] = float(rmse[j])
            out[f"{nm}_r2"] = float(r2[j])
            line += f" | {nm} RMSE {float(rmse[j]):.3f}{un} R² {float(r2[j]):.3f}"
        print(line)
        return out
