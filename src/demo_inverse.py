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
薄入口: 全部实现见 demo_app.py (DemoApp) 及其协作类
(demo_config/codebook/feature_extractor/data_builder/priors/evaluator/
sequence_runner)。

运行: cd src && python demo_inverse.py [--model nb|spn] [--quick]
"""


from demo_app import DemoApp

if __name__ == "__main__":
    DemoApp(DemoApp.parse_args()).run()
