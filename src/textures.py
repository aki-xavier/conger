"""textures.py — 程序化 albedo 贴图库 (cga Texture, sRGB 编码)。

主线 Codebook 的纹理自由度来源; 灰度贴图 (零色度) 让纹理信号纯净地
落在 lum 通道, 与图元色相/光色正交。生成器语义与 texture_probe.py /
texture_pipeline.py 的本地副本一致。
"""

from __future__ import annotations

import mlx.core as mx
from cga.engine import Texture


def checker(size: int = 16, c1=(0.9, 0.9, 0.9), c2=(0.5, 0.5, 0.5), tile: int = 4) -> Texture:
    """棋盘 (sRGB 两色, tile = 每格 texel 数)。"""
    px = [
        [
            [*((c1 if ((i // tile) + (j // tile)) % 2 == 0 else c2)), 1.0]
            for j in range(size)
        ]
        for i in range(size)
    ]
    return Texture.from_rgba(px)


def stripes(size: int = 16, c1=(0.9, 0.9, 0.9), c2=(0.5, 0.5, 0.5), period: int = 3) -> Texture:
    """竖条纹 (period = 每带 texel 数)。"""
    px = [
        [
            [*((c1 if (j // period) % 2 == 0 else c2)), 1.0]
            for j in range(size)
        ]
        for i in range(size)
    ]
    return Texture.from_rgba(px)


def gray_noise(size: int = 16, seed: int = 0, lo: float = 0.3, hi: float = 0.7) -> Texture:
    """宽带灰度噪声 (对比度限幅 [lo,hi], 与背景/前景掩码阈值相容)。"""
    arr = mx.random.normal((size, size), key=mx.random.key(seed))
    arr = (arr - mx.min(arr)) / (mx.max(arr) - mx.min(arr))
    arr = (arr * (hi - lo) + lo).tolist()
    return Texture.from_rgba([[[v, v, v, 1.0] for v in row] for row in arr])


def default_library(n: int = 3) -> tuple[Texture, ...]:
    """默认 3 纹理库: checker / stripes / noise。n 截断到 1..3。"""
    lib = (checker(), stripes(), gray_noise())
    return lib[: max(1, min(n, 3))]
