"""InverseConfig: 逆渲染运行配置 (一切开关的唯一家, 派生量全 property)。

场景/pipeline 说明见 inverse.py 与 docs/architecture.md。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codebook import Codebook
from feature_extractor import FeatureExtractor


@dataclass(frozen=True)
class InverseConfig:
    """运行配置 (一切开关的唯一家); 派生量全是 property, 无游离全局。"""

    quick: bool = False
    use_cache: bool = True
    model_path: Path | None = None
    # σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, 插值平滑度的
    # 原理旋钮 —— 小 = 分量间硬切换 (趋最近邻), 大 = 糊向全局均值
    sigma_rel_floor: float = 1e-2
    equal_luma: bool = False  # 等亮度消融: L 失效 / 色度补位
    occlusion: bool = False  # 遮挡场景: 固定黄柱
    stereo: bool = False  # 双眼视差: 平行 rig 立体帧对 + 视差深度通道

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (9 Riesz + 2 原始拮抗), 全分辨率。"""
        return FeatureExtractor.FEAT

    @property
    def n_feat(self) -> int:
        n = len(self.feat_spec) * Codebook.H * Codebook.W
        return n + 2 if self.stereo else n  # +[ẑ, 掩码面积]

    @property
    def bg_color(self) -> int:
        return 0x1A1A1A if self.equal_luma else 0x141414
