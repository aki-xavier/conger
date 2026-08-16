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
    # composite 候选渲染残差精炼默认关闭: top-k kind/hue × 光照候选
    # 成本高于单物体; 部分几何锚点应先承担主要结构约束
    refine_composite: bool = False
    # 结构候选数: 3 = 覆盖全部 kind 支持集; 1/2 是低成本截断调试
    kind_topk: int = 3
    # 场景结构支持集: 1 = 单图元; 2 = 双图元遮挡/前后层 (实验路径)
    n_objects: int = 1
    # 显式结构族: None 时由 n_objects 兼容推导; composite 是双图元
    # 附着组合模板 (区别于 layered 的独立前后层)
    scene_family: str | None = None
    # 推理期几何↔光照 ECM 精炼 (§7.1): 默认关闭, 单物体验证稳定后再开。
    # 每轮 E 步 54×2 渲染 + M 步坐标搜索 (约 150 渲染/轮)。
    em_refine: bool = False
    em_max_iters: int = 2
    em_appearance_topk: int = 3
    em_tolerance: float = 1.0

    def __post_init__(self) -> None:
        """显式结构族自动同步旧 n_objects 兼容字段。"""
        if self.scene_family in {"layered", "composite"}:
            object.__setattr__(self, "n_objects", 2)
        elif self.scene_family == "single":
            object.__setattr__(self, "n_objects", 1)

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (9 Riesz + 2 原始拮抗), 全分辨率。"""
        return FeatureExtractor.FEAT

    @property
    def family(self) -> str:
        """当前结构族 (scene_family 优先; n_objects 仅作旧配置兼容)。"""
        if self.scene_family is not None:
            return self.scene_family
        return "layered" if self.n_objects == 2 else "single"

    @property
    def n_feat(self) -> int:
        n = len(self.feat_spec) * Codebook.H * Codebook.W
        # single 拼全局 [ẑ,area]; layered/composite 拼两部分 [u,v,z,area]×2
        return n + (2 if self.family == "single" else 8)

    @property
    def bg_color(self) -> int:
        return 0x141414
