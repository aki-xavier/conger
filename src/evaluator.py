"""Evaluator: 连续量回归指标 (物理单位) + kind 分类 + 色相环形误差。

基线 = 训练均值预测器 (无证据预测): R² 相对它计算 —— 第一性原理的
"什么都不会"参照系。色相只在白光子集上评估 (彩光下无观测依据,
色恒常歧义对)。"""

from __future__ import annotations

import math

import mlx.core as mx

from codebook import Codebook
from utils import Utils

TARGETS = ("u", "v", "s", "z")  # 目标列 0..3 (列 4,5 是色相 cos/sin)
UNITS = ("px", "px", "world", "world")


class Evaluator:
    """回归 + 分类 + 环形色相指标。"""

    @staticmethod
    def report(
        name: str,
        p_gt: mx.array,  # (M,8) [kind,u,v,s,z,hue,lcol,ldir]
        t_pred: mx.array,  # (M,6) E[t|x]: u,v,s,z,cosH,sinH
        kind_pred: mx.array,  # (M,) int
        p_train: mx.array,  # (N,8) 训练参数 (基线均值来源)
    ) -> dict[str, float]:
        """打印并返回指标: 逐目标 RMSE/R² + kind 准确率 + 色相环形误差
        (白光子集) 与色相分档准确率。"""
        gt = p_gt[:, 1:5]
        base = mx.mean(p_train[:, 1:5], axis=0, keepdims=True)  # 均值基线
        ss_base = mx.sum((gt - base) ** 2, axis=0)
        ss_res = mx.sum((gt - t_pred[:, :4]) ** 2, axis=0)
        rmse = mx.sqrt(mx.mean((gt - t_pred[:, :4]) ** 2, axis=0))
        r2 = 1.0 - ss_res / mx.maximum(ss_base, 1e-12)
        kind_acc = float(
            mx.mean((kind_pred == p_gt[:, 0].astype(mx.int32)).astype(mx.float32))
        )
        out: dict[str, float] = {"kind": kind_acc}
        line = f"  {name}: kind {kind_acc:.3f}"
        for j, (nm, un) in enumerate(zip(TARGETS, UNITS, strict=True)):
            out[f"{nm}_rmse"] = float(rmse[j])
            out[f"{nm}_r2"] = float(r2[j])
            line += f" | {nm} {float(rmse[j]):.3f}{un} R² {float(r2[j]):.3f}"
        # 色相: 白光子集 (环形误差, atan2 差值 wrap 到 [0,180))
        white = Utils.nonzero(p_gt[:, 6] == Codebook.WHITE)
        if white.shape[0] > 0:
            pred_ang = mx.arctan2(t_pred[white, 5], t_pred[white, 4])
            gt_ang = p_gt[white, 5] * (2.0 * math.pi / Codebook.N_HUE)
            d = mx.abs(pred_ang - gt_ang) % (2.0 * math.pi)
            d = mx.minimum(d, 2.0 * math.pi - d)
            deg = float(mx.mean(d) * 180.0 / math.pi)
            pred_bin = mx.round(
                (pred_ang % (2.0 * math.pi)) / (2.0 * math.pi / Codebook.N_HUE)
            ).astype(mx.int32) % Codebook.N_HUE
            bin_acc = float(
                mx.mean(
                    (pred_bin == p_gt[white, 5].astype(mx.int32)).astype(mx.float32)
                )
            )
            out["hue_deg"] = deg
            out["hue_bin"] = bin_acc
            line += f" | hue Δ{deg:.1f}° bin {bin_acc:.3f} (白光)"
        print(line)
        return out
