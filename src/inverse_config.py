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
    # 解耦边缘 MAP (因果不变估计): True 时 refine_scene 用各因子边缘
    # argmax (反照率对光照、光照对反照率/几何分别边缘化) 替代联合
    # argmax。见 docs §9.3 路线 ①。
    appearance_marginalize: bool = False
    # 白化基内在维截断 (docs §10.3): 默认 48 (全量验收全面优于基线);
    # None 或 <=0 = 全维。设 N 只保留最高方差的 N 维, 模型 ~459MB→(N/497)
    # ·459MB, 且精度实测反升 (截掉白化放大的低方差尾维噪声)。仅影响模型
    # (入模型路径指纹 _d{N}), 不影响数据缓存。
    basis_dim: int | None = 48
    # 显式结构族: single 单图元 / layered 独立前后层 / composite 双图元
    # 附着组合模板 (区别于 layered 的独立前后层)。None 时按 single。
    scene_family: str | None = None
    # 纹理自由度: 0 = 关 (现行单物体管线, 不回归); >0 = 给单物体图元
    # 加 albedo map 纹理类型 (离散, cat_logp) + roughness (连续, t_mu)。
    # 组合数 162 → 162×n_textures, 数据/模型指纹随 n_textures 变化。
    n_textures: int = 0
    # 推理期几何↔光照 ECM 精炼 (§7.1): 默认关闭, 单物体验证稳定后再开。
    # 每轮 E 步 54×2 渲染 + M 步坐标搜索 (约 150 渲染/轮)。
    em_refine: bool = False
    em_max_iters: int = 2
    em_appearance_topk: int = 3
    em_tolerance: float = 1.0
    # §7.1 下一步 ①: s/z 不参与 ECM 坐标搜索 (只精炼 u/v)。s/z 有
    # 投影歧义 (大而远 ≡ 小而近), 贪心搜索把 s 拖坏 (全量验收 s R²
    # 0.508→-0.376)。True = 冻结 s/z (默认); False = 四维全搜 (旧行为)。
    em_freeze_sz: bool = True

    def __post_init__(self) -> None:
        """派生量后处理: basis_dim<=0 = 全维哨兵 → 归一化为 None。"""
        if self.basis_dim is not None and self.basis_dim < 1:
            object.__setattr__(self, "basis_dim", None)

    @property
    def feat_spec(self) -> tuple[tuple[str, str], ...]:
        """唯一特征集: L + 复数色相 (9 Riesz + 2 原始拮抗), 全分辨率。"""
        return FeatureExtractor.FEAT

    @property
    def family(self) -> str:
        """当前结构族 (scene_family; None → single)。"""
        return self.scene_family or "single"

    @property
    def textured(self) -> bool:
        """纹理自由度是否启用 (仅 single 族有意义)。"""
        return self.n_textures > 0

    @property
    def n_feat(self) -> int:
        n = len(self.feat_spec) * Codebook.H * Codebook.W
        # single 拼全局 [ẑ,area]; layered/composite 拼两部分 [u,v,z,area]×2
        return n + (2 if self.family == "single" else 8)

    @property
    def bg_color(self) -> int:
        return 0x141414
