"""逆渲染: 左右二维图像 → MixtureSPN → 完整 cga.Scene 重建。

输出覆盖当前场景族的全部可控参数: kind, u, v, s, z, 图元色相,
光色, 光向; 相机/背景/环境光/材质由 Codebook 固定配置提供。

模型: MixtureSPN —— 全分辨率实例级浅混合 SPN (PCA 白化 + 逐 kind
分层, 每样本一个对角高斯块, 类内 tied 方差; 连续条件期望 ≡ 分层
核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性组装)。外观/结构
精炼: 覆盖全部 kind, 逐 kind 尺寸代理 × hue×光色×光向候选, 用左右图
渲染残差做联合裁决。

场景: 默认暗背景 + 单图元 (sphere/cylinder/box), 位置/尺寸/深度连续,
图元色 6 色相 (与 kind 解耦 —— kind 只剩形状线索), 光色 3 / 光向 3
均为监督目标; `--n-objects 2` 启用双图元遮挡/前后层实验族 (2916 组合)。
离散因子全笛卡尔积覆盖 × R 复制。

数据: 参数采样 → cga 渲染 144×144 立体帧对 (平行 rig, B=0.2) →
左帧 Riesz 特征 (L + 复数色相 + 带符号拮抗, 11 通道全分辨率) +
视差几何观测 (ẑ, 掩码面积) → 实例级组装。
推理: 责任度 (特征证据) → E[u,v,s−ŝ,z−ẑ|特征] +
P(kind,hue,lcol,ldir|特征) → 全 kind 结构候选 (共享几何评分 + 逐 kind
s 重校准) × 外观候选渲染残差后验 → SceneEstimate (MAP cga.Scene +
候选不确定性)。
评估: 物理单位 RMSE/R² (基线 = 训练均值) + 4 个场景因子分类准确率,
插值 vs 外推分裂。

结构 (无游离状态: 配置集中 InverseConfig, 机制分属各类):
  Codebook          单物体场景参数 ⇄ cga 场景 (领域常量 + 组合采样)
  LayeredCodebook   双物体遮挡/前后层采样与 cga 场景
  LayeredReconstructor 双层 SPN 输出 → SceneEstimate
  InverseConfig     运行配置 (开关唯一家, 派生量全 property)
  FeatureExtractor  帧 → 全分辨率特征向量
  DataBuilder       数据构建 (缓存) + 目标组装
  MixtureSPN        实例级浅混合 + 连续/离散条件推理
  SceneReconstructor 帧对/模型输出 → 候选渲染后验 → 完整 cga.Scene
  SceneEstimate     MAP Scene + SPN/渲染候选后验 + top 假设
  Evaluator         回归 + 完整场景因子分类指标
  InverseApp        主流程 (训练/推理/评估/可视化/自检)

运行: cd src && python inverse.py [--replicates R] [--no-cache] [...]
自检: pytest tests/ (mixture_spn 公理/回归/白化病理/序列化 + color +
      stereo + scene_reconstructor); 集成自检 pytest -m slow ≡
      inverse.py 内置断言 (阈值依据见 inverse_app.self_check)。
"""

from inverse_app import InverseApp

if __name__ == "__main__":
    InverseApp(InverseApp.parse_args()).run()
