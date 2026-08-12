# conger 架构与流程图

SPN 逆渲染研究: cga engine 渲染合成场景 → Riesz 特征 → 反演 3D 场景码。
本文档是全部流程的总图; 各模块 docstring 有机制细节, `prior.md` 有先验体系。

## 1. 主链路: 逆渲染 demo (inverse.py)

```mermaid
flowchart LR
    subgraph DATA["数据 (DataBuilder)"]
        CODE["场景码 (kind,gx,gy,size,z)<br/>1152 组合均匀采样"] --> SCENE["Codebook.to_scene<br/>cga Scene"]
        SCENE --> RENDER["Renderer 渲染<br/>144×144 帧"]
        RENDER --> FEAT["FeatureExtractor<br/>Riesz log_mag/phase_coh/ori_R"]
    end
    FEAT -->|"nb (默认): 全分辨率 62208 维"| CB["CodeBayes.fit<br/>逐码对角高斯<br/>充分统计量精确可增量"]
    FEAT -->|"spn: 池化 8×6×3 = 144 维"| LEARN["SPNLearner.learn<br/>G 检验 Product /<br/>k-means Sum"]
    LEARN --> SPN["SPN 树"]
    CB --> POST
    SPN --> POST["posterior: 枚举 1152 码<br/>log 后验 + 先验注入<br/>(edge/familiar/occlusion)"]
    POST --> EVAL["Evaluator<br/>码 + 逐变量准确率"]
    POST --> RECON["重建: argmax 码<br/>→ to_scene 再渲染"]
    POST -->|"--sequence"| SEQ["SequenceRunner<br/>贝叶斯前向滤波"]
```

实测: nb 码准确率 0.965 (秒级训练) / spn 0.470 (分钟级, 组合泛化研究对照)。

## 2. 开放集: 门控双轨联合系统 (experiment_joint.py)

```mermaid
flowchart TD
    F["帧流 (已知码 + 未见码 + 新类别)"] --> G{"CodeBayes.gate<br/>等先验似然比:<br/>全局兜底分量 vs 最佳已知分量"}
    G -->|"已知 (≈99%)"| FAST["快轨 CodeBayes<br/>posterior_all argmax 回答<br/>+ 自标注吸收 (精确统计)"]
    G -->|未见| SLOW["慢轨 SPN (池化)<br/>变量级回答 kind/gx/gy<br/>(组合泛化)"]
    SLOW --> PROMOTE["提升: grow 临时分量<br/>+ absorb 首帧统计<br/>(交接格式 = 充分统计量)"]
    PROMOTE -->|"码簿 +1, 后续帧自动转快轨"| FAST
    FAST --> OUT["统一输出: 码 + 后验"]
    SLOW --> OUT2["未知标记 + 变量边缘"]
```

实测: 提升覆盖 86.5% 后码 acc 0.861, 分量纯度 1.000; 码簿外新类别 (圆盘)
判新 100% 且 SPN 位置泛化 gx 0.41 / gy 0.74。

## 3. 在线学习: OnlineSPN 吸收-生长-修订 (online_spn.py)

```mermaid
flowchart LR
    B["新批样本"] --> E["E 步: 软路由<br/>叶后验 = 路径先验 × 叶似然"]
    E --> M["M 步: 统计常驻累加<br/>叶 (n,μ,M2) / 码联合计数表 / Sum 计数"]
    M --> R["refresh: 参数重建<br/>(Chan 合并防 float32 抵消)"]
    R --> G{"叶码计数<br/>≥ 下限?"}
    G -->|是| SPLIT["生长: 码空间加权 k-means 分裂<br/>子叶继承 5% 伪计数先验<br/>+ 当批行按码分组播种"]
    G -->|否| KEEP["叶保持, 下批再查"]
    SPLIT --> B
    KEEP --> B
    M -.->|"稀疏调度 (--rev-at)"| REV["⑤ 修订: reservoir (Vitter R)<br/>SPNLearner 重构 + 重吸收"]
```

实测 (N=4000, 5 批): 纯在线 = 全量 × 0.82; 加中途单次修订 (cap=2048)
= ×0.98 (0.460 vs 0.470), 时间省 30%+。高频修订反而 ×0.61
(证据丢失税) —— 修订要稀疏。

## 4. 模块结构 (一文件一类)

```mermaid
flowchart LR
    subgraph SPNF["SPN 族"]
        NODE["node.py: Node<br/>(多态契约)"] --> LEAF["leaf.py: Leaf"]
        LEAF --> GL["gauss_leaf.py"] & CL["cat_leaf.py"]
        NODE --> PR["product.py"] & SM["sum_node.py"]
        GL & CL & PR & SM --> SPNM["spn.py: SPN<br/>推理 + safetensors 序列化"]
        SPNM --> LR["spn_learner.py"] & ON["online_spn.py"]
    end
    subgraph DEMO["inverse 族"]
        CBK["codebook.py"] --> FEX["feature_extractor.py"] --> DCFG["inverse_config.py"]
        CBK & FEX & DCFG --> DB["data_builder.py"] & PRI["priors.py"] & EV["evaluator.py"] & SR["sequence_runner.py"]
        DB & PRI & EV & SR --> APP["inverse_app.py: InverseApp"]
        APP --> ENTRY["inverse.py<br/>薄 CLI 入口"]
    end
    subgraph FRONT["前端 / 模型"]
        RS["riesz_scale.py"] & FM["feature_maps.py"] --> RW["riesz.py: RieszWavelet"]
        CBM["code_bayes.py: CodeBayes"]
    end
    EXP["experiment_incremental / fullres / joint<br/>(各一个实验类)"]
    RW --> FEX
    SPNM & CBM --> APP
    LR & ON --> EXP
```

依赖方向全部单向; codebook/feature_extractor 仅 TYPE_CHECKING 引
InverseConfig 防环; 反序列化工厂 (node_from_records) 在依赖顶点 SPN。

## 5. 持久化 (safetensors)

```mermaid
flowchart LR
    SAVE["model.save() / 缓存"] --> ST[".safetensors<br/>8B 头长 + JSON 明文头<br/>+ 张量二进制体"]
    ST --> READ["mx.load (张量)<br/>Utils.st_metadata (头)"]
    ST -.->|"config: cards/dim/floor/n_vars<br/>人读可检查"| HUMAN["明文元数据"]
```

- 数据缓存: `artifacts/*.safetensors` (gitignore);
- 模型: SPN 树扁平化 (DFS 先序 + CSR 子索引 + 分类型负载表),
  CodeBayes 平铺统计量; `.pkl` 后缀走旧 pickle 格式向后兼容。
