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
    k_components: int = 64  # 混合分量数 (自动定 K = DP-SVI, 未做)
    em_iters: int = 20
    # σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, 插值平滑度的
    # 原理旋钮 —— 小 = 分量间硬切换 (趋最近邻), 大 = 糊向全局均值
    sigma_rel_floor: float = 1e-2
    equal_luma: bool = False  # 等亮度消融: L 失效 / 色度补位
    occlusion: bool = False  # 遮挡场景: 固定黄柱
    test_light: bool = False  # 池外顶光评估 (渲染侧)
    multi_light: bool = False  # 多光照训练 (5 方向池轮流)

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (3 源 × 3 通道 = 9), 全分辨率。"""
        return FeatureExtractor.FEAT

    @property
    def n_feat(self) -> int:
        return len(self.feat_spec) * Codebook.H * Codebook.W

    @property
    def kind_colors(self) -> tuple[int, int, int]:
        # 等亮度: 三色与背景同为亮度 0.10 (L 通路失效, 轮廓只剩色度可辨)
        if self.equal_luma:
            return (0x550000, 0x002B00, 0x0000E0)
        return (0xC0392B, 0x27AE60, 0x2980B9)

    @property
    def bg_color(self) -> int:
        return 0x1A1A1A if self.equal_luma else 0x141414
