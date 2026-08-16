"""RoughnessHead: shape 轴区域描述子 → roughness 回归 (球面限定)。

§2.3/texture_roughness_paths 实测: 全分辨率特征对 roughness 负 R²
(sphere/box 均负), 但 8 张谱形图的前景 mean/std 16 维描述子在球面达
R² 0.916、box 仍负 —— roughness 是空间 specular 瓣的谱形信号。本头
= z-score 1-NN 核回归 (与 MixtureSPN 同实例级哲学), 只拟合/预测球面
kind; 非球面由调用方回落默认 0.55 (无信号)。
"""

from __future__ import annotations

import numpy as np


class RoughnessHead:
    """谱形 16d → roughness 的 1-NN 核回归头。"""

    DEFAULT = 0.55

    def __init__(self):
        self.X: np.ndarray | None = None  # (N,16)
        self.y: np.ndarray | None = None  # (N,)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """球面样本的谱形描述子 + roughness 标签。"""
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = self.X.mean(axis=0)
        self.sd = self.X.std(axis=0) + 1e-9

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N,16) → (N,) roughness 估计 (最近邻)。"""
        assert self.X is not None and self.y is not None, "RoughnessHead 未 fit"
        a = (self.X - self.mu) / self.sd
        b = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        idx = np.argmin(((a[None, :, :] - b[:, None, :]) ** 2).sum(axis=2), axis=1)
        return self.y[idx]

    @staticmethod
    def r2(pred: np.ndarray, gt: np.ndarray) -> float:
        """1 − SS_res/SS_base (基线 = gt 均值)。"""
        ss_res = float(np.sum((gt - pred) ** 2))
        ss_base = float(np.sum((gt - gt.mean()) ** 2))
        return 1.0 - ss_res / max(ss_base, 1e-12)
