"""ConstrainedLayeredReconstructor: 受限 layer 子模板的全残差解码。

父 LayeredReconstructor 为避免低密度 SPN 野性残差, 把后层 s/z 锚定到
StereoLayers。动态 layer 子模板已显式约束 scale/lateral/depth gap,
因此允许全部 8 个几何量学习有界残差。
"""

from __future__ import annotations

from layered_reconstructor import LayeredReconstructor


class ConstrainedLayeredReconstructor(LayeredReconstructor):
    """受限 layer 子模板: base/part 全部几何残差可学习。"""

    RESIDUAL_SCALE = (1.0,) * 8
