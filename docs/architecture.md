# conger 架构与流程图

SPN 逆渲染研究: cga engine 渲染合成场景 → Riesz 全分辨率特征 → 连续反演
3D 场景参数。本文档是全部流程的总图 + 机制决策录; 各模块 docstring 有
机制细节。

## 0. 主线切换 (2026-08-12/13)

两次退役: ① 离散场景码体系 (逐码贝叶斯/池化 SPN/码网格, 最后提交
7ae963f) —— 连续物理量的离散化只是后验求积; ② EM/质心压缩层
(提交 523b97a 的 MixtureSPN 初版) —— 小数据 + 弯曲流形下质心把点
平均到流形外, 实例级 (每样本一分量) 才是最小数据设计的正确形态。
git 历史保留全部旧实现。

现行唯一主线: 全因子覆盖连续采样 + 全分辨率实例级浅混合 SPN。

## 1. 主链路 (inverse.py)

```mermaid
flowchart LR
    subgraph DATA["数据 (DataBuilder)"]
        SAMPLE["全因子组合采样:<br/>kind3×图元色6×光色3×光向3=162 组合<br/>全笛卡尔 × R 连续复制 (最小数据)"]
        --> SCENE["Codebook.to_scene<br/>cga Scene"]
        SCENE --> RENDER["Renderer 渲染<br/>144×144 帧"]
        RENDER --> FEAT["FeatureExtractor<br/>11 通道全分辨率 (V=228K):<br/>L×3 Riesz (gain control)<br/>色度×3 Riesz (无 gc, 保色相幅度)<br/>色度×2 原始 (带符号拮抗)"]
    end
    FEAT --> W["PCA 白化 (Gram eigh, CPU)<br/>228K→D≤N−1 无损降维<br/>对角高斯≡原空间全协方差"]
    W --> ASM["实例级组装 (无 EM):<br/>逐 kind 分层, 分量=全部样本,<br/>类内 tied 方差, 均匀权重"]
    ASM --> PRED["predict: 责任度 (特征证据)<br/>E[t|x]=r@t_mu ≡ 分层核回归<br/>P(kind|x)=类条件似然比"]
    PRED --> EVAL["Evaluator: 物理单位 RMSE/R²<br/>+ kind + 色相环形误差 (白光子集)<br/>插值 vs 外推分裂"]
    PRED --> RECON["重建: 预测参数 → to_scene 再渲染"]
```

实测 (N=1296): 插值 u,v RMSE 5.4/5.2px (R²≈0.93) / 白光色相 bin 0.68
(Δ28°) / kind 0.52 (形状线索密度封顶 —— 颜色解耦的有意代价) /
s,z 报告制 (乘积歧义 ×2: 尺寸×深度 + 反照率×光色, 1-NN 同值 = 物理)。
外推报告制: 核回归边界饱和上限 (升级路径 linear experts)。

## 2. 机制决策录 (全部实测驱动, 按时间序)

### 2.1 特征与分层 (2026-08-12)

```mermaid
flowchart TD
    A1["kind 0.47: 无约束 EM 按位置聚类<br/>(kind 与连续因子独立采样 → 结构性混色)"]
    --> F1["逐 kind 分层拟合 P(kind)·P(f,t|kind)"]
    F1 --> A2["仍 0.47: 对比度归一化 + 能量符号盲<br/>→ 色相幅度比 (kind 主线索) 被前端抹掉"]
    A2 --> F2["色度关 gain_control + 2 个带符号原始拮抗通道"]
    F2 --> A3["仍 0.47: 相邻像素强相关, 对角高斯<br/>把相关维当独立选票淹没色度 (1-NN 0.95)"]
    A3 --> F3["PCA 白化 (白化对角 ≡ 原空间全协方差)"]
    F3 --> A4["0.68: 分量方差在零空间维撞地板,<br/>责任度被零空间抖动主导"]
    A4 --> F4["方差逆伽马收缩 (Ledoit-Wolf)"]
```

### 2.2 EM 的四条退化通道 → 实例级 (2026-08-13, 全因子数据重设计时)

图元色与 kind 解耦 + 光色/光向 nuisance 后, 逐版本实测定位:

```mermaid
flowchart TD
    B0["目标: 组合覆盖 + 数据最小 + 速度快<br/>初版 162 组合 R=1: 全崩 (u R²≈0)"]
    B0 --> B1["① 稀疏平铺? R→4 无效<br/>→ 判别实验: 自由 1-NN u R²0.94<br/>数据/度量无罪, 病在模型"]
    B1 --> B2["② 质心压缩? K→216 无效<br/>③ 目标 razor 门控 E 步 (winner-take-all,<br/>死分量均值爆炸 90× 流形距) → 删目标项"]
    B2 --> B3["④ 方差无上限 (大方差吃一切, 活 19/216)<br/>→ 上限=类全局 + 权重均匀化 → 仍无效"]
    B3 --> B4["⑤ 真根: nk≈3 时每分量方差噪声,<br/>−½Σlog(var) 项 ±115 nats 淹没距离选择<br/>→ 类内 tied 方差: u R² 0.50"]
    B4 --> B5["⑥ 终审: EM 40 轮逐位不变 = 质心表示力<br/>上限; 实例级 (K=N, 无 EM) → u R² 0.90"]
```

教训沉淀: EM/质心压缩是大数据优化; 小数据 + 弯曲流形时质心把点
平均到流形外。数据均匀采样 ⟹ 均匀权重是正确先验 (学权重反而
引入 log_w≈−20 死亡螺旋)。

## 3. 模块结构 (一文件一类)

```mermaid
flowchart LR
    subgraph CORE["逆渲染族"]
        CBK["codebook.py<br/>组合采样+投影+调色板"] --> FEX["feature_extractor.py<br/>11 通道"] --> DCFG["inverse_config.py"]
        CBK & FEX & DCFG --> DB["data_builder.py"] & EV["evaluator.py"]
        DB & EV --> APP["inverse_app.py: InverseApp"]
        APP --> ENTRY["inverse.py 薄入口"]
    end
    subgraph FRONT["前端"]
        RS["riesz_scale.py"] & FM["feature_maps.py"] --> RW["riesz.py: RieszWavelet"]
        RW --> FEX
    end
    MSP["mixture_spn.py: MixtureSPN<br/>白化+实例级组装+条件期望+序列化"] --> APP
    RW --> ST["riesz_selftest.py"]
```

依赖方向单向; codebook/feature_extractor 仅 TYPE_CHECKING 引
InverseConfig 防环。

## 4. 持久化 (safetensors)

- 数据缓存: `artifacts/mix_*.safetensors` (配置指纹文件名, gitignore);
- 模型: MixtureSPN.save/load —— 参数张量 (含白化基 basis (V,D),
  全量约 1.5GB) + rel_floor 入 JSON 明文头; Utils.st_metadata 可查。

## 5. 待办 (按价值排序; 已对照实例级架构审判, 过时项已删)

1. **逐 kind PPCA 似然比**: kind 形状线索的度量升级 (各类自己的
   白化子空间 + log|det| 修正, 跨类密度可比化)
2. **池外光照探针**: held-out 光向/光色, 验证"覆盖 → 不变性"
   设计主张
3. **大数据逃生通道** (N~10⁴ 触发): PCA 基按内在维度截断 /
   子样本估基全量套用 / ANN 索引加速推理 / 压缩蒸馏 —— EM 若
   回归只能作实例模型的对照验证压缩件 (退化通道病历见 §2.2)

已删过时项 (2026-08-13 审判): DP-SVI 自动定 K (实例模型 K=N, 无
分量数可定) / online EM (无 EM 可 online, 需求并入逃生通道) /
熟悉尺寸先验重接 + 局部线性核回归 (单目 s/z 歧义的解法, 单目
模式已删, 立体下 s/z 由几何解决) / 参考物破解色恒常 (随遮挡
模式一起删)。

## 6. 双眼视差 (stereo.py, 2026-08-13)

```mermaid
flowchart LR
    S["场景参数"] --> LR["平行 rig 双渲染<br/>(±B/2=±0.1, 光轴平行)"]
    LR --> M["色度能量+亮度对比掩码<br/>(背景恒 S=0)"]
    M --> D["软质心视差 d=cx_L−cx_R<br/>(亚像素; 单凸物体唯一无歧义量)"]
    D --> Z["ẑ = CAM_Z − FX·B/d<br/>(物理截断防野值)"]
    Z --> RES["残差融合: 模型学 z−ẑ, s−ŝ<br/>(ŝ=√(area/π)·zc/FX)<br/>—— 拼接稀释教训见下"]
```

实测 (全量): 视差管线 ẑ RMSE 0.35 (bias +0.30 = 可见面≠中心的系统
偏差, 残差学习标定); 融合后 z R² 0.85 / s R² 0.41
/ **外推 z R² 0.96** (几何量不饱和, 核回归边界问题在 z/s
上消解)。

踩坑记录 (全部实测): ① 角部样本 (大 s×近 z) 边距反演出画 → 空掩码
1/d 爆炸 → 采样器加取景约束拒绝采样; ② 裸 area (σ≈600) 拼特征
主导 λ 谱 → 白化截断阈值抬升误截特征方向 → 拼接维先缩放到特征
方差量级; ③ 强几何观测直接拼特征被白化稀释 (1/647 维) → 残差
参数化融合。

**增量训练** (2026-08-13): 采样逐复制块独立种子 (SAMPLE_V=3) → R
增长纯追加; 数据逐块缓存 (缺哪块渲哪块); 模型 `MixtureSPN.add`
= 冻结白化基追加分量 + tied 方差全量重估 (实例级 f_mu 即样本,
与全量 fit 同估计量)。唯一近似 = 基冻结: 新样本跑出旧主子空间
的方向不可表示, 分布漂移大时重新 fit。
