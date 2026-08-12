"""Evaluator: 评估 (码/逐变量准确率) 与基线 (多数类/最近模板)。"""

from __future__ import annotations

import mlx.core as mx

from codebook import Codebook
from utils import Utils


class Evaluator:
    """评估与基线。"""

    @staticmethod
    def evaluate(pred_i: list[int], gt_i: list[int]) -> dict[str, float]:
        """码全对准确率 + 逐变量 (kind/gx/gy/size/z) 准确率。"""
        cb = Codebook
        pred_codes = [cb.idx_to_code(p) for p in pred_i]
        gt_codes = [cb.idx_to_code(g) for g in gt_i]
        n = len(gt_i)
        return {
            "code": sum(p == g for p, g in zip(pred_i, gt_i, strict=True)) / n,
            "kind": sum(
                p[0] == g[0] for p, g in zip(pred_codes, gt_codes, strict=True)
            )
            / n,
            "gx": sum(p[1] == g[1] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
            "gy": sum(p[2] == g[2] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
            "size": sum(
                p[3] == g[3] for p, g in zip(pred_codes, gt_codes, strict=True)
            )
            / n,
            "z": sum(p[4] == g[4] for p, g in zip(pred_codes, gt_codes, strict=True))
            / n,
        }

    @staticmethod
    def baseline_majority(tr: list[int], te: list[int]) -> float:
        """多数类: 全测样本押训练集最常见的码。"""
        most = max(set(tr), key=tr.count)
        return sum(m == most for m in te) / len(te)

    @staticmethod
    def baseline_template(
        x_tr: mx.array, c_tr: mx.array, x_te: mx.array, te: list[int]
    ) -> float:
        """最近模板: 每码取训练特征均值, 测试特征 L2 最近邻 (未见码无法命中)。"""
        cb = Codebook
        code_i = [c_tr[:, j].astype(mx.int32) for j in range(5)]
        templates: list[mx.array] = []
        present: list[int] = []
        for i in range(cb.N_CODES):
            sel = mx.ones(x_tr.shape[0], dtype=mx.bool_)
            for j in range(5):
                sel = sel & (code_i[j] == cb.idx_to_code(i)[j])
            cnt = int(mx.sum(sel))
            if cnt == 0:
                continue
            idx = Utils.nonzero(sel)
            templates.append(mx.sum(x_tr[idx], axis=0) / cnt)
            present.append(i)
        tm = mx.stack(templates)  # (P, V)
        # 距离矩阵分块且逐块立即求值: 惰性图全量累积会超 Metal 显存上限
        dd_parts = []
        chunk = 20
        for i in range(0, x_te.shape[0], chunk):
            d = mx.sum((x_te[i : i + chunk, None, :] - tm[None, :, :]) ** 2, axis=2)
            mx.eval(d)
            dd_parts.append(d)
        dd = mx.concatenate(dd_parts)
        pred = [present[int(mx.argmin(d))] for d in dd]
        return sum(p == g for p, g in zip(pred, te, strict=True)) / len(te)
