# conger 架构与流程图（内核）

> **V 移植 + 拆分说明**: 本仓库已从 Python 移植到 V, 并拆分为两个平行项目 —— **`conger`(本仓库) 为通用 SPN / 结构学习内核** (模块 `conger`: MixtureSPN + 通用结构学习框架 + 模板学习 + 模型内存 / 持久化 + 纯数学 / MLX 工具 + 非视觉验证域), **`conger-vision`(平行项目) 为视觉层** (模块 `conger_vision`: cga 渲染、Codebook、Riesz 前端、立体几何、场景重建、纹理、外观 / ECM、`InverseApp`, `import conger` 依赖内核)。
>
> 视觉层的机制决策（Riesz 前端、渲染、立体几何、双层/组合/lateral 场景族、ECM、外观/因果、模型内存验证）已迁移至 [`conger-vision/docs/architecture.md`](../conger-vision/docs/architecture.md)。本文档聚焦内核自身的分层架构、数据流、控制流与模块边界。

内核不依赖图像、`cga` 渲染或任何视觉概念；视觉侧经 `import conger` 复用本内核。

## 内核架构与主管线（本仓库 `conger` · V 内核，2026-08-18）

> 本节梳理**当前仓库**（通用 SPN / 结构学习内核，模块 `conger`）的核心架构
> 设计与主管线流程，与 `conger-vision` 的「全系统（内核 + 视觉）机制决策录」互补：
> 视觉主线与历史机制演进（原 §0–§11）已迁移至 [`conger-vision/docs/architecture.md`](../conger-vision/docs/architecture.md)，本节聚焦内核自身的分层架构、数据流、
> 控制流与模块边界。视觉侧 `conger-vision` 经 `import conger` 复用本内核；
> 内核不依赖图像、`cga` 渲染或任何视觉概念。

### 内核分层架构（一文件一类，模块平铺在仓库根目录）

```mermaid
flowchart TD
    subgraph L1["① 核心 SPN"]
        A["mixture_spn.v — MixtureSPN<br/>PCA 白化 · 实例级组装(无 EM) · 条件期望 · 增量/类别扩展 · safetensors 序列化"]
    end
    subgraph L2["② 通用结构学习框架"]
        B["structured_hypothesis.v<br/>StructuredHypothesis[T] (scene=泛型载荷 T)"]
        C["generic_structure_gate.v<br/>GenericStructureGate[T]: decide / decide_hierarchical"]
        D["generic_expert_registry.v<br/>GenericExpertRegistry[T] + GenericExpert[T] 接口"]
        E["structure_birth.v<br/>StructureBirthController / StructureBirthRequest"]
        F["generic_em.v · forward_model.v<br/>EMLoop[M,O,R] / ForwardModel 协议"]
    end
    subgraph L3["③ 模板学习"]
        G["template_proposal.v · template_grammar.v · template_lineage.v<br/>提案 / 有界文法 / 血缘与 ChildTemplateSpec"]
        H["template_delta_learner.v · causal_edge.v<br/>提案→约束 · delta→因果边"]
    end
    subgraph L4["④ 模型内存与持久化"]
        I["model_memory.v · registry_manifest.v<br/>split/load/assemble/truncate/forget · RegistryManifest(JSON)"]
    end
    subgraph L5["⑤ 工具"]
        J["types.v (TemplateDelta/Metadata/Constraints) · vecmath.v (f64 原语+RNG) · mlxutil.v (MLX 辅助)"]
    end
    subgraph L6["⑥ 非视觉验证域"]
        K["toy_series_family.v · toy_series_expert.v · structure_benchmark.v<br/>线性/振荡专家 · accuracy/confusion/ECE"]
    end
    L1 --> L2 --> L3 --> L4
    L5 -.-> L1 & L2 & L3 & L4 & L6
    L6 -.-> L1 & L2 & E
```

- **依赖方向单向**：核心 SPN → 结构框架 → 模板学习 → 持久化；工具层（⑤）被其余各层复用；
  验证域（⑥）以 `ToySeries` 时间序列实例驱动核心 SPN、结构门控与出生控制，**证明整套
  通用结构学习框架可脱离视觉独立运行**（视觉路径只是它的一个领域适配器）。
- **泛型载荷**：`StructuredHypothesis[T].scene` 为泛型载荷 `T`（视觉层用 `cga.Scene`、非视觉验证域用 `voidptr`），内核不解箱 —— 这是「内核不依赖视觉」的关键边界。`ForwardModel` 的观测参数仍为 `voidptr`（协议占位）。
- **接口协议**：`GenericExpert[T].estimate(observation mlx.Array) → StructuredHypothesis[T]`、
  `ForwardModel.residual(observation voidptr, params []f64) f64`、
  `TemplateProposer.propose(cases []StructureCase) []TemplateProposal`。

### 主管线流程

```mermaid
flowchart LR
    subgraph TRAIN["A. 训练（fit）"]
        T0["f, t, stratum, scene_classes<br/>cat_sizes, basis_dim"] --> T1["whiten()<br/>Gram eigh(CPU) → f_mean·basis·z"]
        T1 --> T2["tied_vars()<br/>逐 stratum tied 对角方差"]
        T2 --> T3["实例级组装<br/>f_mu=z · t_mu=t · cat_logp · 均匀 log_w"]
        T3 --> TM["MixtureSPN (+init_norm)"]
    end
    subgraph INF["B. 推理（predict）"]
        I0["f"] --> I1["z = (f − f_mean)·basis"] --> I2["logq_feat(z) 分块"] --> I3["r = softmax(logq)"]
        I3 --> I4["E[t|x] = r·t_mu ≡ 分层核回归<br/>P(scene|x) = r·exp(cat_logp)"]
        I4 --> IH["StructuredHypothesis"]
    end
    subgraph GAT["C. 结构门控 + 出生"]
        G0["各专家 estimate"] --> G1["scores = ℓ + λ·C + η·G"] --> G2["softmax → posterior"]
        G2 --> G3{"needs_new_structure?"}
        G3 -->|否| G4["MAP 专家 (with_structure)"]
        G3 -->|是| G5["StructureBirthController 累积 cases"] --> G6["StructureBirthRequest (min_cases 达阈)"]
    end
    subgraph TPL["D. 模板学习"]
        G6 --> D1["TemplateDeltaLearner.tdl_learn<br/>按 parent|operation 分组 → 约束范围 → ChildTemplateSpec"]
        D1 --> D2["ChildTemplateSpec.lineage → TemplateLineage"]
        D2 --> D3["CausalDeltaLearner.learn<br/>按 env 分组 → agreement → CausalEdge"]
    end
    subgraph PER["E. 持久化"]
        TM --> P1["save / split_save<br/>safetensors + 明文 meta"]
        P1 --> P2["load_mixture_spn / load_components / assemble_model"]
        P1 --> P3["truncate_basis(D↓) / forget_components(K↓)"]
        D2 --> P4["RegistryManifest → registry_manifest.json"]
    end
    IH -.-> GAT
```

**A. 训练（确定性，无 EM）** —— `fit_mixture_spn(f, t, stratum, rel_floor, scene_classes,
cat_sizes, basis_dim)`：先 `whiten`（Gram 特征分解得到 `f_mean`/`basis`/白化坐标 `z`，可选
`basis_dim` 截断到最高方差内在维），再 `tied_vars` 求逐 stratum 的类内 tied 方差，最后把
**每个样本装配成一个对角高斯分量**（`f_mu=z`、`t_mu=t`、`cat_logp` 为 one-hot 对数、
`log_w` 均匀），`init_norm` 预计算特征侧归一化常数。`add` 为冻结白化基的增量追加
（tied 方差全量重估）；`expand_categories` 为新类别 padding `-inf` 扩展类别契约。

**B. 推理** —— `predict(f)` 返回三元组 `(E[t|x], P(scene factors|x), 责任度 r)`：白化后
`logq_feat` 按 `nc×kc` 分块算未归一化对数联合（避免物化 N×K 大矩阵），`r = softmax(logq)`
是逐分量的特征证据（责任度）；`r·t_mu` 即连续条件期望（≡ 分层核回归），`r·exp(cat_logp)`
即离散场景因子的条件后验分类。

**C. 结构门控 + 出生** —— `GenericExpertRegistry.decide(observation)` 把同一观测路由到
所有注册专家取 `estimate`，`GenericStructureGate` 以 `score = ℓ + λ·complexity + η·geometry_cost`
（`ℓ` 渲染/模拟残差、`C` 模板复杂度、`G` 观测级几何证据）做 softmax 成结构后验；两级
`decide_hierarchical` 先父族后族内。当原始残差超 `birth_residual` 且（可选）后验低于
`posterior_floor` 时判定「需要新结构」，`StructureBirthController.observe` 累积证据，达
`min_cases` 后产出一个携带候选提案的 `StructureBirthRequest`（并清空队列）。

**D. 模板学习** —— `TemplateDeltaLearner.tdl_learn` 把多个出生请求的 `TemplateProposal`
按 `parent_family|operation` 分组，估计数值约束范围（ratio/lateral/depth_gap…）与离散支持集
（part_kind/part_hue），产出可序列化的 `ChildTemplateSpec`（哈希命名、含 evidence_count /
residual_mean / score_mean）；`lineage()` 转成 `TemplateLineage` 血缘对象。`CausalDeltaLearner`
再把 delta 边按环境（env/seed/case_index）分组，用跨环境一致度 `agreement` 区分稳定机制
（因果边）与漂移伪相关。

**E. 持久化** —— `MixtureSPN.save` / `split_save` 写 safetensors（参数张量 + 明文 meta
`rel_floor`/`cat_sizes`/`n_stratum`）；`load_transform`/`load_components` 分级按需加载
（分量表常驻、白化基按需）；`assemble_model` 全量装配；`truncate_basis` 截内在维、`forget_components`
按 stratum 比例驱逐分量；`RegistryManifest` 把子模板血缘、约束、pending specs 与模型路径
JSON 持久化（`rm_save`/`rm_load`），供跨进程恢复。

### 内核模块 → 文件速查

| 层 | 文件 | 关键公开符号 |
|---|---|---|
| 核心 SPN | `mixture_spn.v` | `MixtureSPN` · `fit_mixture_spn` · `fit_simple` · `predict` · `whiten` · `add` · `expand_categories` · `save`/`load_mixture_spn` |
| 结构框架 | `structured_hypothesis.v` | `StructuredHypothesis[T]` · `HypothesisCandidate` · `new_hypothesis[T]` · `factor_marginals` |
| | `generic_structure_gate.v` | `GenericStructureGate[T]` · `GenericStructureDecision[T]` · `softmax_map` · `decide` · `decide_hierarchical` |
| | `generic_expert_registry.v` | `GenericExpert[T]` · `GenericExpertRegistry[T]` |
| | `structure_birth.v` | `StructureCase` · `StructureBirthRequest` · `StructureBirthController` |
| | `generic_em.v` · `forward_model.v` | `EMLoop[M,O,R]` · `EMResult[T]` · `ForwardModel` |
| 模板学习 | `template_proposal.v` | `TemplateProposal` · `TemplateProposer` |
| | `template_grammar.v` | `TemplateGrammar` · `TemplateRule` · `primitives`/`composites`/`rules` |
| | `template_lineage.v` | `TemplateLineage` · `ChildTemplateSpec` · `single/layered/composite/lateral_lineage` |
| | `template_delta_learner.v` | `TemplateDeltaLearner` · `tdl_learn` · `tdl_spec` · `tdl_hash` |
| | `causal_edge.v` | `CausalEdge` · `CausalDeltaLearner` · `is_causal` |
| 内存/持久化 | `model_memory.v` | `split_save` · `load_transform` · `load_components` · `assemble_model` · `truncate_basis` · `forget_components` · `coreset` · `model_size_mb` |
| | `registry_manifest.v` | `RegistryManifest` · `RegisteredChildTemplate` · `rm_save`/`rm_load` |
| 工具 | `types.v` | `TemplateDelta` · `TemplateMetadata` · `TemplateConstraints` |
| | `vecmath.v` | `Rng` · `linspace` · `solve_2x2` · `solve_n` · `lstsq_2` · `kabsch_2d` |
| | `mlxutil.v` | `nonzero_indices` · `axis_var`/`axis_std` · `axis_logsumexp` · `eigh_cpu` · `split_keys` · `fft2`/`ifft2` |
| 验证域 | `toy_series_family.v` · `toy_series_expert.v` | `ToySeriesFamily`(linear/sine) · `ToySeriesExpert` · `train_toy_expert` |
| | `structure_benchmark.v` | `StructureCaseResult` · `StructureBenchmarkSummary` · `sb_summarize` · `sb_ece` |
