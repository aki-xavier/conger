"""ToySeriesFamily: 非视觉时间序列机制族 (linear / sine)。

观测是固定网格上的 1D 序列; 参数是机制潜变量; simulate 是该领域的
正向模型。用于验证结构学习框架不依赖视觉/cga。
"""

from __future__ import annotations

import math

import mlx.core as mx


class ToySeriesFamily:
    """机制名 + 参数采样 + 序列编码 + 正向模拟。"""

    X = mx.linspace(-1.0, 1.0, 32)

    def __init__(self, mechanism: str):
        if mechanism not in ("linear", "sine"):
            raise ValueError(f"未知玩具机制 {mechanism}")
        self.mechanism = mechanism

    @property
    def n_params(self) -> int:
        return 2 if self.mechanism == "linear" else 3

    def sample(self, n: int, seed: int) -> mx.array:
        """→ (n,n_params) 均匀参数。"""
        k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
        if self.mechanism == "linear":
            a = -2.0 + 4.0 * mx.random.uniform(shape=(n,), key=k1)
            b = -1.0 + 2.0 * mx.random.uniform(shape=(n,), key=k2)
            return mx.stack([a, b], axis=1)
        amp = 0.5 + 1.5 * mx.random.uniform(shape=(n,), key=k1)
        freq = 2.0 + 3.0 * mx.random.uniform(shape=(n,), key=k2)
        phase = -math.pi + 2 * math.pi * mx.random.uniform(shape=(n,), key=k3)
        return mx.stack([amp, freq, phase], axis=1)

    def simulate(self, params: mx.array) -> mx.array:
        """(n,P) 参数 → (n,T) 观测序列。"""
        x = self.X[None, :]
        if self.mechanism == "linear":
            return params[:, 0:1] * x + params[:, 1:2]
        return params[:, 0:1] * mx.sin(params[:, 1:2] * x + params[:, 2:3])

    def residual(self, observation: mx.array, params: tuple[float, ...]) -> float:
        """RMSE(observed, simulate(params))。"""
        pred = self.simulate(mx.array(params, dtype=mx.float32)[None, :])[0]
        return float(mx.sqrt(mx.mean((observation - pred) ** 2)))

    def encode(self, y: mx.array) -> mx.array:
        """1D 序列 → 摘要特征 (机制族内使用)。"""
        y = y.astype(mx.float32)
        d = y[1:] - y[:-1]
        dd = y[2:] - 2 * y[1:-1] + y[:-2]
        x = self.X
        feats = [
            mx.mean(y), mx.std(y), mx.min(y), mx.max(y), y[-1] - y[0],
            mx.mean(mx.abs(d)), mx.std(d), mx.std(dd),
        ]
        for freq in (2.0, 3.0, 4.0, 5.0):
            feats.append(mx.mean(y * mx.sin(freq * x)))
            feats.append(mx.mean(y * mx.cos(freq * x)))
        return mx.stack(feats).astype(mx.float32)
