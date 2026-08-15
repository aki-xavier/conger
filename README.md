# conger

SPN 逆渲染研究: 左右两张二维立体图像 → Riesz 全分辨率特征 → MixtureSPN → **完整 `cga.Scene` 重建** (kind, u, v, s, z, 图元色相, 光色, 光向)。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块; 连续条件期望 ≡ 分层核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性)。SPN 初估后, `SceneReconstructor` 覆盖全部 kind, 结构评分沿用共享几何，候选返回前再按各自 kind 的面积→尺寸代理重校准 s，并与 hue×光色×光向候选一起做左右图渲染残差联合精炼。完整机制决策见 `docs/architecture.md`。

训练数据全因子覆盖设计: 单物体离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制; `--n-objects 2` 时双图元前后层为 kind0×kind1×hue0×hue1×光色×光向 = 2916 组合, 约 70% 样本强制投影重叠。图元色与 kind 解耦; 光色/光向不再丢弃, 而是作为完整场景输出显式监督。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (单物体组合采样+投影) / `layered_codebook.py` (双物体遮挡/前后层) / `composite_codebook.py` (双图元附着组合模板) / `feature_extractor.py` (11 通道) / `data_builder.py` / `scene_reconstructor.py` (帧对/参数 → 完整 Scene) / `layered_reconstructor.py` (双层 SPN 解码) / `composite_reconstructor.py` (组合模板 SPN 解码) / `expert_registry.py` (结构专家注册/加载) / `structure_gate.py` (结构专家门控) / `structure_birth.py` (未知结构出生队列) / `template_grammar.py` + `composite_template_proposer.py` (有界文法与残差驱动组合模板提案) / `structured_hypothesis.py` (统一结构化假设/后验对象) / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`, `src/stereo.py` (单物体视差), `src/stereo_layers.py` + `src/contour_completion.py` + `src/joint_layer_optimizer.py` (逐层视差、轮廓补全与遮挡联合优化)
- 测试: `tests/` (pytest; 单元黑盒 + slow 集成自检) / `src/riesz_selftest.py` (可视化脚本)
- `docs/architecture.md` — 架构与机制决策录

## 运行

```bash
pytest                       # 单元黑盒测试 (mixture_spn/color/stereo/scene)
pytest -m slow               # 集成自检 (默认全 kind 结构精炼, 约二十分钟)
cd src
python riesz_selftest.py     # Riesz 自检 + 自然图特征可视化
python inverse.py            # 全量立体 (1296 帧对): 完整 Scene 重建
```

选项: `--sigma-rel-floor` (核带宽下限)、`--replicates R` (训练集复制数, 调大触发增量训练)、`--no-cache` (跳过数据缓存)、`--no-refine-appearance` (跳过候选渲染残差精炼)、`--kind-topk {1,2,3}` (结构候选数, 默认 3 = 覆盖全部 kind)、`--scene-family {single,layered,composite}` (单图元 / 独立前后层 / 附着组合模板)、`--n-objects {1,2}` (旧配置兼容)、`--model-path` (模型 safetensors 存取, 默认 `artifacts/spn_kindgeo_<数据指纹>`、`spn_layered_anchor_<数据指纹>` 或 `spn_composite_<数据指纹>`; 存在即加载, K 不足则增量追加)。

- 通用结构学习: `src/structured_hypothesis.py` / `forward_model.py` / `generic_structure_gate.py` / `generic_expert_registry.py`; 非视觉验证域为 `src/toy_series_family.py` + `src/toy_series_expert.py`

## 推理接口

```python
from inverse_app import InverseApp
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from scene_reconstructor import SceneReconstructor

app = InverseApp(InverseConfig())
net = MixtureSPN.load(f"artifacts/spn_kindgeo_{app.data.cache_tag()}.safetensors")
renderer, cam_l, cam_r = SceneReconstructor.rig()
# fl/fr 为同一 cga.Scene 在训练 rig 下的左/右二维渲染帧
estimate = app.reconstruct_scene(net, fl, fr)
scene = estimate.scene                 # MAP cga.Scene
params = estimate.params               # MAP 完整场景参数
kind_p, hue_p, lcol_p, ldir_p = estimate.factor_marginals()
for h in estimate.hypotheses:          # top 候选完整场景
    print(h.probability, h.residual, h.params)
```

`estimate.scene` 是包含预测 DirectionalLight 的 `cga.Scene`; `params` 是
渲染残差精炼后的 `(kind,u,v,s,z,hue,lcol,ldir)`; `spn_posterior` 是
SPN 初估的 4 因子后验; `candidate_posterior` 是渲染残差联合后验,
避免把逆渲染歧义过早压成单个点估计。

## 增量学习

- 同一结构内增加样本: `MixtureSPN.add()` 追加实例分量并冻结旧 PCA 基;
- 新类别: `expand_categories()` 对旧分量 padding, `cat_sizes/n_stratum`
  随模型序列化;
- 新颖性: `StructuredHypothesis` 返回责任度、后验熵、渲染残差与综合诊断分;
- 新结构: `StructureGate` 按各专家重建残差与模板复杂度计算
  `p(structure|images)`, `StructureBirthController` 聚合连续不兼容样本并生成
  `StructureBirthRequest`; 请求可携带 `CompositeTemplateProposer` 由
  `TemplateGrammar` 的有界 attach/layer/mirror/repeat 空间生成、再由左右图
  残差筛出的组合模板提案。调用方提供新结构配置后可用
  `train_and_register()` 显式训练并注册新专家。

## 结构专家门控

```python
from expert_registry import ExpertRegistry

registry = ExpertRegistry.default()  # 已训练的单物体 + 双层专家
decision = registry.decide(fl, fr)
estimate = decision.estimate
print(decision.posterior, decision.needs_new_structure)
```

`ExpertRegistry.default()` 要求对应结构模型已存在; 缺模型时默认
fail closed，可用 `missing_ok=True` 显式跳过。组合模板模型训练完成后
可用 `include_composite=True` 加入默认门控。若要让未知结构请求自动
携带组合候选:

```python
from composite_template_proposer import CompositeTemplateProposer
from structure_birth import StructureBirthController

birth = StructureBirthController(
    proposer=CompositeTemplateProposer(
        operations=("attach", "layer", "mirror", "repeat")
    )
)
registry = ExpertRegistry(experts, birth_controller=birth)
```

## 通用结构学习框架 (非视觉验证)

`StructuredHypothesis`/`ForwardModel`/`GenericStructureGate`/
`GenericExpertRegistry` 把视觉管线抽象为: 观测编码 → 结构内参数估计 →
正向模拟残差 → 结构后验 → 出生请求。视觉路径直接使用
`StructuredHypothesis`, 其中 `scene` 字段承载 `cga.Scene`; 视觉
`StructureGate` 继承通用门控。
`ToySeriesFamily` 提供线性/振荡两个非视觉专家; 测试验证线性/振荡输入
被正确门控, 二次机制触发 `StructureBirthRequest`。这说明 MixtureSPN、
结构门控和出生控制不依赖图像或 cga。

实测 (结构-外观联合精炼版, 全量立体 N=1296, `kind_topk=3`): 插值 u,v R² 0.930/0.945 / s R² 0.508 / z R² 0.831 / kind 0.753 / hue 1.000 / lcol 0.994 / ldir 0.895; 外推 u,v R² 0.949/0.953 / s,z R² 0.922/0.956 / kind 0.617 / hue 0.981 / lcol 0.880 / ldir 0.772。结构评分沿用共享几何避免尺寸代理偏差，MAP 后按 kind 重校准 s; 色相与光照由候选重渲染联合裁决。

消融: 旧固定几何 top-3 为 kind 0.753 / s R² 0.332; 纯解析逐 kind 几何会使插值 s R² 降至 0.160 (掩码观测偏差不可忽略); 共享评分 + kind 后校准得到上述最优平衡。

双层遮挡实验族 (`--n-objects 2 --replicates 1`, N=2916, sl8): StereoLayers 逐层视差 + JointLayerOptimizer 遮挡联合优化后, 插值 kind0/kind1 0.398/0.357、hue0/hue1 0.421/0.171、lcol/ldir 0.390/0.370; u0/v0/u1/v1 R² 0.537/0.711/0.466/0.432。联合模板负责中心/深度, 面积由可见区+轮廓补全 soft fusion 提供; 后层 s/z 仍为负 R², 遮挡几何仍未达到正式阈值。

显式组合模板族 (`--scene-family composite`): `CompositeCodebook` 把两个已有图元组成 base + attached part; 附着件不再是独立前后层, 而是由底座按尺度比例、横向偏移、接触重叠和轻微深度差导出。首阶段使用全局立体锚点与 8 维几何直读目标, 用来验证“组合模板可训练、可渲染、可注册为结构专家”; 自动模板提案已由有界文法生成, 训练仍保持显式。

首版全量实测 (R=1, N=2916, cp1, 无渲染残差精炼): 插值 kind0/kind1 0.513/0.360、hue0/hue1 0.498/0.364、lcol/ldir 0.405/0.372; u0/v0/u1/v1 R² 0.907/0.516/0.891/0.794。外推 u/v R² 0.929/0.733/0.915/0.833。s/z 仍为负或弱 R² (插值 s0/z0 -0.528/-0.692), 说明全局锚点不足以恢复组合内部尺寸/深度; 下一阶段应把组合模板的几何锚点升级为部分感知残差。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
