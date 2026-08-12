"""逆渲染: cga engine 渲染合成场景 → Riesz 全分辨率特征 → MixtureSPN
连续反演 3D 场景参数 (kind, u, v, s, z, 图元色相), 重建 cga 场景。

模型: MixtureSPN —— 全分辨率实例级浅混合 SPN (PCA 白化 + 逐 kind
分层, 每样本一个对角高斯块, 类内 tied 方差, 条件期望推理 ≡ 分层
核回归; 无 EM, 确定性组装)。

场景: 暗背景 + 单图元 (sphere/cylinder/box), 位置/尺寸/深度连续,
图元色 6 色相 (与 kind 解耦 —— kind 只剩形状线索), 光色 3 / 光向 3
nuisance。离散因子全笛卡尔积覆盖 (3×6×3×3 = 162 组合 × R 复制),
数据量最小化设计 (全量 1296 帧 vs 旧 4600)。

数据: 参数采样 → cga 渲染 144×144 → Riesz 特征 (L + 复数色相 + 带
符号拮抗, 11 通道全分辨率) → 实例级组装。
推理: 责任度 (特征证据) → E[t|特征] + P(kind|特征)。
评估: 物理单位 RMSE/R² (基线 = 训练均值), 色相环形误差 (白光子集),
插值 vs 外推分裂。kind 形状线索密度封顶 ≈0.52; s/z 乘积歧义 ×2
(尺寸×深度 + 反照率×光色) 报告制。

结构 (无游离状态: 配置集中 InverseConfig, 机制分属各类):
  Codebook          连续场景参数 ⇄ cga 场景 (领域常量 + 组合采样)
  InverseConfig     运行配置 (开关唯一家, 派生量全 property)
  FeatureExtractor  帧 → 全分辨率特征向量
  DataBuilder       数据构建 (缓存) + 目标组装
  MixtureSPN        实例级浅混合 + 条件期望 (mixture_spn.py)
  Evaluator         回归/分类/色相指标
  InverseApp        主流程 (训练/推理/评估/可视化/自检)

运行: cd src && python inverse.py [--quick] [--equal-luma] [--occlusion]
自检: mixture_spn.py 内嵌 (公理性质/实例回归/白化相关病理/序列化);
      inverse.py --quick 内置断言 (阈值依据见 inverse_app.self_check)。
"""

from inverse_app import InverseApp

if __name__ == "__main__":
    InverseApp(InverseApp.parse_args()).run()
