"""逆渲染 demo: cga engine 渲染合成场景 → Riesz 特征 → 反推 3D 场景码。

双模型 (DemoConfig.model):
  nb  (默认) 全分辨率逐码对角高斯贝叶斯 (code_bayes.CodeBayes) —— 不池化,
      精确可增量, 码簿任务最优 (实测 0.965 vs spn 0.470, 秒级 vs 分钟级);
  spn 池化 (8×6) + SPNLearner 结构学习 —— 组合泛化/消融研究对照。

场景: 暗背景 + 单个浅色图元 (sphere / cylinder / box), 中心投影在 8×6
网格上、尺寸两档、深度四档 —— 场景码 (kind, gx, gy, size, z) 即 cga
三维建模的离散编码 (code → cga Scene 对象可逆)。

训练数据: 均匀随机采样场景码 → cga engine 渲染 144×144 → Riesz 特征
(深度通道改走亮度: engine 无深度输出) → 特征矩阵 → 模型。
推理: 枚举 1152 个场景码, 后验 argmax → 重建 cga 场景 (三维建模)。

评估: 码准确率 / 逐变量准确率 / 多数类与最近模板基线 / GT vs 重建渲染。

结构 (无游离状态: 配置集中 DemoConfig, 机制分属各类):
  Codebook        码 ⇄ cga 场景 (领域常量 + 投影)
  DemoConfig      运行配置 (feat/model/消融开关, 派生量全是 property)
  FeatureExtractor 帧 → 特征向量 (池化或全分辨率)
  DataBuilder     数据构建 (缓存) 与标准化
  Priors          码先验工厂 (edge/familiar/occlusion)
  Evaluator       评估与基线
  SequenceRunner  多帧运动先验 (贝叶斯滤波)
  DemoApp         主流程 (训练/推理/评估/可视化/自检)

运行: cd src && python demo_inverse.py [--model nb|spn] [--quick] [--no-cache]
自检: --quick 内置断言 (小数据集 + 阈值按全量运行标定)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codebook import Codebook
from feature_extractor import FeatureExtractor


@dataclass(frozen=True)
class DemoConfig:
    """运行配置 (一切开关的唯一家); 派生量全是 property, 无游离全局。"""

    model: str = "nb"  # nb=全分辨率逐码贝叶斯; spn=池化+结构学习
    feat: str = "l"  # l=亮度 Riesz 3 通道; lhs=+色度; hs=仅色度; rgb=原始
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
        return {
            "l": FeatureExtractor.FEAT_L,
            "lhs": FeatureExtractor.FEAT_LHS,
            "hs": FeatureExtractor.FEAT_HS,
            "rgb": FeatureExtractor.FEAT_RGB,
        }[self.feat]

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
