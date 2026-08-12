"""逆渲染: cga engine 渲染合成场景 → Riesz 全分辨率特征 → MixtureSPN
连续反演 3D 场景参数 (kind, u, v, s, z), 重建 cga 场景。

模型: MixtureSPN —— 全分辨率浅混合 SPN (K 个对角高斯块的 Sum, GMR 式
联合 EM, 条件期望推理 ≡ 核回归)。离散场景码体系 (逐码贝叶斯/池化
SPN/码网格) 已整体退役: 连续物理量 (位置/尺寸/深度) 的离散化只是
后验求积, 连续列 + 插值/外推探针才是逆渲染的诚实形态。

场景: 暗背景 + 单个浅色图元 (sphere/cylinder/box), 位置/尺寸/深度
连续采样 (训练范围内均匀; 外推测试采样支撑集外区间)。图元色绑定
kind。

数据: 参数采样 → cga 渲染 144×144 → Riesz 特征 (L + 复数色相
9 通道全分辨率, V=186624) → 联合 EM。
推理: 责任度 (特征证据) → E[t|特征] + P(kind|特征)。
评估: 物理单位 RMSE/R² (基线 = 训练均值预测器), 插值 vs 外推分裂。

结构 (无游离状态: 配置集中 InverseConfig, 机制分属各类):
  Codebook          连续场景参数 ⇄ cga 场景 (领域常量 + 投影/采样)
  InverseConfig     运行配置 (开关唯一家, 派生量全 property)
  FeatureExtractor  帧 → 全分辨率特征向量
  DataBuilder       数据构建 (缓存)
  MixtureSPN        浅混合 SPN + 联合 EM + 条件期望 (mixture_spn.py)
  Evaluator         回归/分类指标
  InverseApp        主流程 (训练/推理/评估/可视化/自检)

运行: cd src && python inverse.py [--quick] [--no-cache] [--components K]
自检: mixture_spn.py 内嵌 (公理性质/EM 恢复/序列化); inverse.py --quick
      内置断言 (阈值依据见 inverse_app.self_check 注释)。
薄入口: 全部实现见 inverse_app.py (InverseApp) 及其协作类。
"""

from inverse_app import InverseApp

if __name__ == "__main__":
    InverseApp(InverseApp.parse_args()).run()
