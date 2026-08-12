"""InverseConfig: 逆渲染运行配置 (一切开关的唯一家, 派生量全 property)。

场景/码簿/pipeline 说明见 inverse.py 与 docs/architecture.md。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codebook import Codebook
from feature_extractor import FeatureExtractor


@dataclass(frozen=True)
class InverseConfig:
    """运行配置 (一切开关的唯一家); 派生量全是 property, 无游离全局。"""

    model: str = "nb"  # nb=全分辨率逐码贝叶斯; spn=池化+结构学习
    quick: bool = False
    use_cache: bool = True
    model_path: Path | None = None
    tree: bool = False
    prior_name: str = "flat"
    min_n: int | None = None  # spn 叶最小行数 (缺省 quick=8 / 全量=3)
    sigma_floor: float = 1e-6
    equal_luma: bool = False  # 等亮度消融: L 失效 / HS 补位
    occlusion: bool = False  # 遮挡场景: 固定黄柱 + 序数先验
    sequence: int = 0  # >0: 多帧运动先验 (每序列帧数)
    test_light: bool = False  # 光照鲁棒性评估 (需 --model-path)
    multi_light: bool = False  # 多光照训练 (5 方向池轮流)

    @property
    def full_res(self) -> bool:
        """逐码贝叶斯不池化 (SPN 结构学习需要低维)。"""
        return self.model == "nb"

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (3 源 × 3 通道 = 9)。"""
        return FeatureExtractor.FEAT

    @property
    def n_feat(self) -> int:
        n = Codebook.N_GX * Codebook.N_GY
        return len(self.feat_spec) * (Codebook.H * Codebook.W if self.full_res else n)

    @property
    def code_cols(self) -> tuple[int, ...]:
        return tuple(range(self.n_feat, self.n_feat + 5))

    @property
    def card(self) -> dict[int, int]:
        return dict(
            zip(
                self.code_cols,
                (
                    Codebook.N_KIND,
                    Codebook.N_GX,
                    Codebook.N_GY,
                    Codebook.N_SIZE,
                    Codebook.N_Z,
                ),
            )
        )

    @property
    def kind_colors(self) -> tuple[int, int, int]:
        # 等亮度: 三色与背景同为亮度 0.10 (L 通路失效, 轮廓只剩色度可辨)
        if self.equal_luma:
            return (0x550000, 0x002B00, 0x0000E0)
        return (0xC0392B, 0x27AE60, 0x2980B9)

    @property
    def bg_color(self) -> int:
        return 0x1A1A1A if self.equal_luma else 0x141414
