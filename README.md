# conger

SPN 逆渲染研究: 左右两张二维立体图像 → Riesz 全分辨率特征 → MixtureSPN → **完整 `cga.Scene` 重建** (kind, u, v, s, z, 图元色相, 光色, 光向)。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块; 连续条件期望 ≡ 分层核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性)。SPN 初估后, `SceneReconstructor` 覆盖全部 kind, 结构评分沿用共享几何，候选返回前再按各自 kind 的面积→尺寸代理重校准 s，并与 hue×光色×光向候选一起做左右图渲染残差联合精炼。完整机制决策见 `docs/architecture.md`。

本仓库是 **V 语言移植版**: 模块 `conger` 平铺在仓库根目录 (每个 Python `src/*.py` 对应一个 `*.v`), 依赖经默认 V 模块目录 `~/.vmodules` 解析 (符号链接到本地 [`cga`](../cga) 渲染引擎与 [`mlx-v`](../mlx-v) 绑定)。

训练数据全因子覆盖设计: 单物体离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制; `--scene-family layered` 时双图元前后层为 kind0×kind1×hue0×hue1×光色×光向 = 2916 组合, 约 70% 样本强制投影重叠。图元色与 kind 解耦; 光色/光向不再丢弃, 而是作为完整场景输出显式监督。

## 构建与测试

```bash
make test        # v -no-memory-limit test .
make fmt         # v fmt -w .
```

（等价直接命令：`v -no-memory-limit test .`，依赖经 `~/.vmodules` 解析，无需 `VMODULES` 环境变量。）

38 个 V 测试文件全部通过 (单元黑盒 + Riesz 自检 + 结构几何/门控/模板闭环)。`pytest -m slow` 的完整立体集成自检 (`test_inverse.py`, 冷缓存渲染 1296×2 帧, 分钟级) 未移植为 V 测试: 其训练/推理主链路已移植为 `InverseApp.run` / `self_check`, 但阈值按 Python 引擎标定, 未在 V 引擎上重标。

## 模块 (一文件一类)

核心链路: `mixture_spn.v` (MixtureSPN) / `inverse_config.v` (配置唯一家) / `codebook.v` + `codebook_const.v` (单物体组合采样+投影+常量) / `layered_codebook.v` (双物体遮挡/前后层) / `composite_codebook.v` (双图元附着组合模板) / `lateral_codebook.v` (mirror/repeat 横向组合模板) / `composite_geometry.v` (上下 base/part 几何锚点) / `lateral_composite_geometry.v` (横向组合几何锚点) / `structure_geometry.v` (观测级结构几何证据) / `feature_extractor.v` (11 通道) / `data_builder.v` / `scene_reconstructor.v` (帧对/参数 → 完整 Scene) / `layered_reconstructor.v` (双层 SPN 解码) / `composite_reconstructor.v` (组合模板 SPN 解码) / `expert_registry.v` (结构专家注册/加载/血缘树) / `registry_manifest.v` (动态子模板与 pending spec 持久化) / `template_lineage.v` (parent/delta 模板继承契约) / `template_delta_learner.v` + `child_codebook_factory.v` + `child_template_workflow.v` (提案约束学习、子 Codebook 物化与注册编排) / `structure_gate.v` (结构专家门控) / `structure_benchmark.v` (跨结构门控基准汇总) / `structure_birth.v` (未知结构出生队列) / `template_grammar.v` + `composite_template_proposer.v` (有界文法与残差驱动组合模板提案) / `structured_hypothesis.v` (统一结构化假设/后验对象) / `evaluator.v` / `inverse_app.v` (训练/推理主流程)。

前端: `riesz.v` + `riesz_scale.v` + `feature_maps.v` (Riesz 小波, mlx 复数 FFT), `color.v`, `vecmath.v` + `mlxutil.v` (纯 f64 向量原语与 MLX 辅助), `stereo.v` (单物体视差), `stereo_layers.v` + `contour_completion.v` + `joint_layer_optimizer.v` (逐层视差、轮廓补全与遮挡联合优化), `types.v` (异构 MetaValue)。

通用结构学习: `structured_hypothesis.v` / `forward_model.v` / `generic_structure_gate.v` / `generic_expert_registry.v` / `generic_em.v`; 非视觉验证域为 `toy_series_family.v` + `toy_series_expert.v`。

测试: 根目录 `*_test.v` (38 个文件)。`docs/architecture.md` — 架构与机制决策录。

## 推理接口

主流程入口 `inverse_app.v`: `new_inverse_app(cfg)` / `new_inverse_app_cb(cfg, codebook)` 构造应用, `app.reconstruct_scene(net, fl, fr)` 把左右帧对解码为 `StructuredHypothesis` (MAP `cga.Scene` + 候选联合后验); 单物体 `sr_from_frames`, 组合 `cr_from_frames`, 双层 `lrc_from_frames`。

`estimate.scene` 是包含预测 DirectionalLight 的 `cga.Scene`; `params` 是渲染残差精炼后的 `(kind,u,v,s,z,hue,lcol,ldir)`; `spn_posterior` 是 SPN 初估的 4 因子后验; `candidate_posterior` 是渲染残差联合后验, 避免把逆渲染歧义过早压成单个点估计。

## 增量学习

- 同一结构内增加样本: `MixtureSPN.add()` 追加实例分量并冻结旧 PCA 基;
- 新类别: `expand_categories()` 对旧分量 padding, `cat_sizes/n_stratum` 随模型序列化;
- 新颖性: `StructuredHypothesis` 返回责任度、后验熵、渲染残差与综合诊断分;
- 新结构: `StructureGate` 按各专家重建残差与模板复杂度计算 `p(structure|images)`, `StructureBirthController` 聚合连续不兼容样本并生成 `StructureBirthRequest`; 请求可携带 `CompositeTemplateProposer` 由 `TemplateGrammar` 的有界 attach/layer/mirror/repeat 空间生成、再由左右图残差筛出的组合模板提案。调用方提供新结构配置后可用 `train_and_register()` 显式训练并注册新专家。

## 结构专家门控

`default_expert_registry()` 构建 single + layered + composite 三专家; `registry.decide(fl, fr)` 返回 `GenericStructureDecision` (best 估计 + 结构后验 + `needs_new_structure` 出生信号)。缺模型时 `scene_expert_from_config` / `load_manifest(missing_ok=false)` 默认 fail closed。

结构门控分数由渲染残差、模板复杂度和 `StructureGeometry` 的观测级几何证据组成: 单模板紧致性、前后层视差/空间分离、组合接触线分别约束 single/layered/composite。`children_of(parent)` 查询直接子模板, `lineages()` 返回血缘表。

## 模板继承血缘

`TemplateLineage` 显式记录每个结构模板的 `parent_family` 与 `delta`: `single` 是根模板, `layered` 通过独立前后层 delta 继承 single, `composite` 通过 `attached_on_top` 关系 delta 继承 layered。`ExpertRegistry.lineages()` 返回当前专家血缘表, `children_of(parent)` 查询直接子模板。残差提案也携带 `parent_family/delta`, 因此后续可以把多个出生请求中的相似 delta 聚类成数据驱动的子模板, 而不是平铺注册无血缘的新结构。

### 数据驱动子模板学习 (第一阶段)

`TemplateDeltaLearner` 聚合多个 `StructureBirthRequest.proposals`, 按 `parent_family + operation` 分组; 达到证据阈值后, 对 ratio、横向偏移、part kind/hue 等 delta 字段估计带边距的约束范围, 生成 `ChildTemplateSpec`。`ccf_build` (ChildCodebookFactory) 可把 attach/layer/mirror/repeat 子模板规格物化为受限子 Codebook, 并通过 `template_variant()` 隔离数据缓存; `train_and_register(..., codebook)` 可把该子模板显式注册。`enable_child_template_learning()` 后, 出生请求会自动转换为 `pending_child_specs`, 但只有 `confirm_child_template(name)` 才会物化并注册; 自动学习只生成规格, 训练仍保持显式。

`registry_manifest.json` (经 `rm_save`/`rm_load`) 保存 ChildTemplateSpec、parent/delta、约束、pending 列表和模型路径; 动态 Codebook 重启时由 `ccf_build` 重新物化, safetensors 仍只保存 MixtureSPN 参数。

## 通用结构学习框架 (非视觉验证)

`StructuredHypothesis`/`ForwardModel`/`GenericStructureGate`/`GenericExpertRegistry` 把视觉管线抽象为: 观测编码 → 结构内参数估计 → 正向模拟残差 → 结构后验 → 出生请求。视觉路径直接使用 `StructuredHypothesis`, 其中 `scene` 字段承载 `cga.Scene`。`ToySeriesFamily` 提供线性/振荡两个非视觉专家; 测试验证线性/振荡输入被正确门控, 二次机制触发 `StructureBirthRequest`。这说明 MixtureSPN、结构门控和出生控制不依赖图像或 cga。

## 实验指标 (Python 实测, 未在 V 引擎重标)

以下数字为 Python (mlx 0.32.0) 实测结果; V 移植保留了同一算法与阈值 (`InverseApp.self_check`), 但底层 mlx-v 链接 mlx-c 0.6.0, 数值精度与 Python 略有差异 (如某统计后验 0.793 vs 0.8), 因此这些数字仅作量级参照。

- 结构-外观联合精炼版, 全量立体 N=1296, `kind_topk=3`: 插值 u,v R² 0.930/0.945 / s R² 0.508 / z R² 0.831 / kind 0.753 / hue 1.000 / lcol 0.994 / ldir 0.895; 外推 u,v R² 0.949/0.953 / s,z R² 0.922/0.956 / kind 0.617 / hue 0.981 / lcol 0.880 / ldir 0.772。
- 消融: 旧固定几何 top-3 为 kind 0.753 / s R² 0.332; 纯解析逐 kind 几何会使插值 s R² 降至 0.160 (掩码观测偏差不可忽略); 共享评分 + kind 后校准得到上述最优平衡。
- 双层遮挡实验族 (`--scene-family layered --replicates 1`, N=2916, sl8): StereoLayers 逐层视差 + JointLayerOptimizer 遮挡联合优化后, 插值 kind0/kind1 0.398/0.357、hue0/hue1 0.421/0.171、lcol/ldir 0.390/0.370; u0/v0/u1/v1 R² 0.537/0.711/0.466/0.432。后层 s/z 仍为负 R², 遮挡几何仍未达到正式阈值。
- 显式组合模板族 (`--scene-family composite`): 全局锚点基线 (cp1) 的插值 s0/z0 R² 为 -0.528/-0.692; 部分感知锚点 (cp2, R=1, N=2916, 无精炼) 提升到 s0/z0 0.089/0.823, part s1/z1 -0.474/0.875; u0/v0/u1/v1 R² 0.982/0.972/0.990/0.970。

## 未移植部分

- 薄 CLI 入口 `inverse.py` 与 argparse 基准/探针脚本 (`*_benchmark.py` / `*_probe.py` / `basis_sweep.py` / `riesz_selftest.py` 的可视化部分): 其算法核心均已移植并有测试覆盖, 仅 `main()`+argparse 壳未移植 (需 `cmd/` 主模块结构)。
- 纹理/粗糙度实验 (`texture_pipeline.py` / `texture_probe.py` / `texture_roughness_paths.py`): numpy 依赖 (含 numpy FFT), `n_textures > 0` 特性默认关闭、无测试覆盖; `textures.v` (贴图库) 与 `roughness_head.v` (纯 V 1-NN 回归头) 已移植。
- `test_inverse.py` (slow 全链路集成自检): 见「构建与测试」。

## 依赖

- [`cga`](../cga) — 渲染引擎 (V 端口)。
- [`mlx-v`](../mlx-v) — MLX C API 的 V 绑定 (复数 FFT、safetensors、随机数等)。

两者经默认 V 模块目录 `~/.vmodules` 解析。首次在机器上配置一次符号链接即可：

```bash
ln -sfn /path/to/cga    ~/.vmodules/cga
ln -sfn /path/to/mlx-v  ~/.vmodules/mlx
```

之后 `v test .` / `make test` 均无需 `VMODULES` 环境变量。
