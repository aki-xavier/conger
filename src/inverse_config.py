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

    use_cache: bool = True
    model_path: Path | None = None
    # 训练集复制数 (162 组合×R 帧对): 逐块缓存 + 模型增量追加,
    # 调大只渲染/学习新块, 已训部分不动
    replicates: int = 8
    # σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, 插值平滑度的
    # 原理旋钮 —— 小 = 分量间硬切换 (趋最近邻), 大 = 糊向全局均值
    sigma_rel_floor: float = 1e-2
    # 渲染残差精炼: 结构候选使用逐 kind 尺寸代理, 外观枚举
    # hue×lcol×ldir, 并用左右图残差联合裁决 (完整 Scene 输出级)
    refine_appearance: bool = True
    # 结构候选数: 3 = 覆盖全部 kind 支持集; 1/2 是低成本截断调试
    kind_topk: int = 3
    # 场景结构支持集: 1 = 单图元; 2 = 双图元遮挡/前后层 (实验路径)
    n_objects: int = 1

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (9 Riesz + 2 原始拮抗), 全分辨率。"""
        return FeatureExtractor.FEAT

    @property
    def n_feat(self) -> int:
        n = len(self.feat_spec) * Codebook.H * Codebook.W
        return n + 2  # +[ẑ, 掩码面积] (立体观测通道)

    @property
    def bg_color(self) -> int:
        return 0x141414
