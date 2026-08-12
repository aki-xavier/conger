"""FeatureMaps: RieszWavelet.features() 的输出记录 (跨尺度谱统计)。"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


class FeatureMaps(NamedTuple):
    """RieszWavelet.features() 的输出: 跨尺度谱统计特征, 逐像素。
    11 张 (H,W) float32 特征图 + log_e (H,W,S)。不可变记录,
    不预组特征矩阵 —— 选列组装是下游的事 (见 demo_inverse)。"""

    log_mag: mx.array  # log Σe_s 减邻域均值 —— 局部对比度
    slope: mx.array  # log e_s 对 octave 的最小二乘斜率 —— 幂律衰减
    residual: mx.array  # 拟合 RMS 残差 —— 偏离幂律 = 有峰
    bump: mx.array  # argmax_s e_s, 归一化到 [0,1]
    centroid: mx.array  # 能量分布 p_s 的一阶矩 (octave)
    spread: mx.array  # 二阶矩 (标准差)
    skew: mx.array  # 三阶矩
    kurt: mx.array  # 四阶矩
    ori_R: mx.array  # 跨尺度方向一致性 (2θ 圆均值 resultant)
    mean_ori: mx.array  # 跨尺度平均法向 (−π/2, π/2]
    phase_coh: mx.array  # 跨尺度相位一致性
    log_e: mx.array  # log 逐尺度能量 (H,W,S)
