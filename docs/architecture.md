# conger 架构与流程图

SPN 逆渲染研究: cga engine 渲染合成场景 → Riesz 全分辨率特征 → 连续反演
3D 场景参数。本文档是全部流程的总图; 各模块 docstring 有机制细节。

## 0. 主线切换 (2026-08-12)

离散场景码体系 (逐码贝叶斯 CodeBayes / 池化 SPN / 码网格 / 码先验 /
在线 SPN) 已整体退役删除。理由: 位置/尺寸/深度是连续物理量, 离散码
只是后验求积节点; 逐码机制的索引结构就是码本身, 连续任务下不适用。
git 历史保留全部旧实现 (最后一次提交 decbb89..7ae963f 区间)。

现行唯一主线: 连续采样 + 全分辨率浅混合 SPN (MixtureSPN)。

## 1. 主链路 (inverse.py)

```mermaid
flowchart LR
    subgraph DATA["数据 (DataBuilder)"]
        SAMPLE["连续参数采样 (kind,u,v,s,z)<br/>训练范围内均匀; 外推探针范围外"]
        --> SCENE["Codebook.to_scene<br/>cga Scene"]
        SCENE --> RENDER["Renderer 渲染<br/>144×144 帧"]
        RENDER --> FEAT["FeatureExtractor<br/>11 通道全分辨率 (V=228K):<br/>L×3 Riesz (gain control)<br/>色度×3 Riesz (无 gc, 保色相幅度)<br/>色度×2 原始 (带符号拮抗)"]
    end
    FEAT --> W["PCA 白化 (Gram eigh, CPU)<br/>186K→D≤N−1 无损降维<br/>对角高斯≡原空间全协方差"]
    W --> EM["逐 kind 分层联合 EM<br/>P(kind)·P(f,t|kind), 各 K/3 分量<br/>方差逆伽马收缩 (Ledoit-Wolf)"]
    EM --> PRED["predict: 责任度 (特征证据)<br/>E[t|x]=r@t_mu, P(kind|x)=r@onehot"]
    PRED --> EVAL["Evaluator: 物理单位 RMSE/R²<br/>(基线=训练均值) + kind 准确率<br/>插值 vs 外推分裂"]
    PRED --> RECON["重建: 预测参数 → to_scene 再渲染"]
```

实测 (N=4000, K=64): 插值 kind 0.897 / u,v RMSE 6.6px (R²≈0.90) /
z R² 0.44; s/z 弱是物理 (单目单帧仅乘积可观测 = 熟悉尺寸歧义,
R²>0 部分来自边界线索)。外推报告制: 核回归边界饱和不完美
(s/z R² 可为负) —— 核机器上限, 升级路径 mixture of linear experts。

## 2. 关键机制决策 (全是实测驱动的判决)

```mermaid
flowchart TD
    Q1["kind 曾 0.47"] --> A1["病理 1: kind 与连续因子独立采样,<br/>无约束 EM 按位置聚类 → 分量结构性混色"]
    A1 --> F1["修复 1: 逐 kind 分层拟合<br/>(生成结构 P(kind)·P(f,t|kind))"]
    F1 --> A2["病理 2: 能量特征符号盲 + 对比度归一化<br/>→ 色相幅度比 (kind 主线索) 被前端抹掉"]
    A2 --> F2["修复 2: 色度关 gain_control +<br/>2 个带符号原始色度通道"]
    F2 --> A3["病理 3: 相邻像素强相关, 对角高斯<br/>把相关维当独立选票 → 色度被亮度淹没"]
    A3 --> F3["修复 3: PCA 白化 (白化对角 ≡ 原始全协方差)"]
    F3 --> A4["病理 4: 高维小样本, 分量方差在零空间<br/>维撞地板 → 责任度被零空间抖动主导"]
    A4 --> F4["修复 4: 方差逆伽马收缩<br/>(等效 20 虚样本, nk≫20 纯数据)"]
```

## 3. 模块结构 (一文件一类)

```mermaid
flowchart LR
    subgraph CORE["逆渲染族"]
        CBK["codebook.py<br/>连续采样+投影"] --> FEX["feature_extractor.py<br/>11 通道"] --> DCFG["inverse_config.py"]
        CBK & FEX & DCFG --> DB["data_builder.py"] & EV["evaluator.py"]
        DB & EV --> APP["inverse_app.py: InverseApp"]
        APP --> ENTRY["inverse.py 薄入口"]
    end
    subgraph FRONT["前端"]
        RS["riesz_scale.py"] & FM["feature_maps.py"] --> RW["riesz.py: RieszWavelet"]
        RW --> FEX
    end
    MSP["mixture_spn.py: MixtureSPN<br/>白化+分层 EM+条件期望+序列化<br/>内嵌 4 组黑盒自检"] --> APP
    RW --> ST["riesz_selftest.py"]
```

依赖方向单向; codebook/feature_extractor 仅 TYPE_CHECKING 引
InverseConfig 防环。

## 4. 持久化 (safetensors)

- 数据缓存: `artifacts/mix_*.safetensors` (配置指纹文件名, gitignore);
- 模型: MixtureSPN.save/load —— 参数张量 (含白化基 basis (V,D),
  全量模式约 3GB) + rel_floor 入 JSON 明文头; Utils.st_metadata 可查。
- 注意: 旧 `inv_*`/`fullres_*`/`joint_*` 缓存属已删除的离散体系, 可清。

## 5. 待办 (研究升级路径, 均未做)

- mixture of linear experts: 块内放开特征↔目标交叉协方差
  (治外推边界饱和; 低秩 + SVI)
- DP-SVI 自动定 K (分量数从超参变推断量, 接 structure learning 遗产)
- online EM (数据量超内存时平移; 旧 OnlineSPN 已是其骨肉)
- 训练数据全因子重设计 (光位/光色/图元色组合覆盖; 色恒常歧义对
  拆条件监督 —— 讨论已定调, 未实现)
