# conger

SPN 逆渲染研究: 左右两张二维立体图像 → Riesz 全分辨率特征 → MixtureSPN → **完整 `cga.Scene` 重建** (kind, u, v, s, z, 图元色相, 光色, 光向)。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块; 连续条件期望 ≡ 分层核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性)。SPN 初估后, `SceneReconstructor` 覆盖全部 kind, 结构评分沿用共享几何，候选返回前再按各自 kind 的面积→尺寸代理重校准 s，并与 hue×光色×光向候选一起做左右图渲染残差联合精炼。完整机制决策见 `docs/architecture.md`。

训练数据全因子覆盖设计: 单物体离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制; `--n-objects 2` 时双图元前后层为 kind0×kind1×hue0×hue1×光色×光向 = 2916 组合, 约 70% 样本强制投影重叠。图元色与 kind 解耦; 光色/光向不再丢弃, 而是作为完整场景输出显式监督。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (单物体组合采样+投影) / `layered_codebook.py` (双物体遮挡/前后层) / `composite_codebook.py` (双图元附着组合模板) / `composite_geometry.py` (base/part 部分感知几何锚点) / `structure_geometry.py` (观测级结构几何证据) / `feature_extractor.py` (11 通道) / `data_builder.py` / `scene_reconstructor.py` (帧对/参数 → 完整 Scene) / `layered_reconstructor.py` (双层 SPN 解码) / `composite_reconstructor.py` (组合模板 SPN 解码) / `expert_registry.py` (结构专家注册/加载/血缘树) / `template_lineage.py` (parent/delta 模板继承契约) / `template_delta_learner.py` + `child_codebook_factory.py` + `child_template_workflow.py` (提案约束学习、子 Codebook 物化与注册编排) / `child_template_benchmark.py` (真实样本子模板闭环基准) / `structure_gate.py` (结构专家门控) / `structure_benchmark.py` (跨结构门控基准) / `structure_birth.py` (未知结构出生队列) / `template_grammar.py` + `composite_template_proposer.py` (有界文法与残差驱动组合模板提案) / `structured_hypothesis.py` (统一结构化假设/后验对象) / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
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

选项: `--sigma-rel-floor` (核带宽下限)、`--replicates R` (训练集复制数, 调大触发增量训练)、`--no-cache` (跳过数据缓存)、`--no-refine-appearance` (跳过单物体候选渲染残差精炼)、`--refine-composite` (组合模板启用 top-k kind/hue/light 渲染残差精炼, 默认关闭)、`--kind-topk {1,2,3}` (结构候选数, 默认 3 = 覆盖全部 kind)、`--scene-family {single,layered,composite}` (单图元 / 独立前后层 / 附着组合模板)、`--n-objects {1,2}` (旧配置兼容)、`--model-path` (模型 safetensors 存取, 默认 `artifacts/spn_kindgeo_<数据指纹>`、`spn_layered_anchor_<数据指纹>` 或 `spn_composite_<数据指纹>`; 存在即加载, K 不足则增量追加)。

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

registry = ExpertRegistry.default()  # 已训练的 single + layered + composite
decision = registry.decide(fl, fr)
estimate = decision.estimate
print(decision.posterior, decision.needs_new_structure)
```

`ExpertRegistry.default()` 要求对应结构模型已存在; 缺模型时默认
fail closed，可用 `missing_ok=True` 显式跳过。结构门控分数由渲染残差、
模板复杂度和 `StructureGeometry` 的观测级几何证据组成: 单模板紧致性、
前后层视差/空间分离、组合接触线分别约束 single/layered/composite。

联合门控基准:

```bash
uv run python src/structure_benchmark.py \
  --samples-per-family 3 --seed 1234 --no-single-refine
```

两个随机种子 (各 3×3=9 样本) 均为 9/9, 合计 18/18; 默认含单物体
渲染精炼的注册表在每族 1 样本探针上也为 3/3。注意这是小样本
结构门控验收, 不是场景内参数精度指标。

若要让未知结构请求自动携带组合候选:

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

## 模板继承血缘

`TemplateLineage` 显式记录每个结构模板的 `parent_family` 与 `delta`:
`single` 是根模板, `layered` 通过独立前后层 delta 继承 single,
`composite` 通过 `attached_on_top` 关系 delta 继承 layered。
`ExpertRegistry.lineages()` 返回当前专家血缘表, `children_of(parent)`
查询直接子模板。残差提案也携带 `parent_family/delta`, 因此后续可以把
多个出生请求中的相似 delta 聚类成数据驱动的子模板, 而不是平铺注册
无血缘的新结构。

### 数据驱动子模板学习 (第一阶段)

`TemplateDeltaLearner` 聚合多个 `StructureBirthRequest.proposals`, 按
`parent_family + operation` 分组; 达到证据阈值后, 对 ratio、横向偏移、
part kind/hue 等 delta 字段估计带边距的约束范围, 生成
`ChildTemplateSpec`。`ChildCodebookFactory` 当前可把 attach 子模板规格
物化为受限 `CompositeCodebook` 子类, 并通过 `TEMPLATE_VARIANT` 隔离数据
缓存; `ExpertRegistry.train_and_register(..., codebook_cls=...)`
可把该子模板显式训练并注册。自动学习只生成规格和类, 训练仍保持显式。

端到端基准:

```bash
uv run python src/child_template_benchmark.py --seed 12345
```

该基准用真实渲染样本产生 attach 提案, 学习得到
`composite → composite_attach_5025cfd1` 子模板 (scale_ratio 0.43–0.62,
lateral_ratio -0.02–0.02, part kind/hue 固定), 以 R=4/648 样本训练;
在同子分布 3 个 held-out 样本上, 子模板对父 composite 门控 3/3
(posterior 0.815/0.699/0.599)。这验证的是子模板出生闭环, 不是开放世界
模板发明。

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

显式组合模板族 (`--scene-family composite`): `CompositeCodebook` 把两个已有图元组成 base + attached part; 附着件不再是独立前后层, 而是由底座按尺度比例、横向偏移、接触重叠和轻微深度差导出。`CompositeGeometry` 在前景掩码上搜索接触线并分别拟合 base/part 圆/方模板, 再在右图模板窗口内估计部件视差; MixtureSPN 只学习 8 个几何量相对这些锚点的有界残差。自动模板提案由有界文法生成, 训练仍保持显式。

全局锚点基线 (cp1) 的插值 s0/z0 R² 为 -0.528/-0.692; 部分感知锚点 (cp2, R=1, N=2916, 无精炼) 提升到 s0/z0 0.089/0.823, part s1/z1 -0.474/0.875; u0/v0/u1/v1 R² 0.982/0.972/0.990/0.970。外推 u/v R² 0.978/0.985/0.985/0.985, s/z R² 0.732/0.921/0.429/0.949。类别指标仍接近 cp1 (kind0/kind1 0.511/0.355, hue0/hue1 0.504/0.365), 说明下一步瓶颈主要是部分外观/结构辨识, 可再用 `--refine-composite` 做候选渲染裁决。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
