# conger

SPN 逆渲染研究: cga engine 渲染合成场景 → Riesz 全分辨率特征 → **连续**反演 3D 场景参数 (kind, u, v, s, z, 图元色相), 重建 cga 场景。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块, 条件期望 ≡ 分层核回归; 无 EM, 确定性)。完整机制决策见 `docs/architecture.md`。

训练数据全因子覆盖设计: 离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制 —— 组合覆盖保证且数据量最小 (全量 1296 帧 vs 旧 4600)。图元色与 kind 解耦 (kind 只剩形状线索, 任务升级); 光色/光向为 nuisance。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN + 4 组黑盒自检)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (组合采样+投影) / `feature_extractor.py` (11 通道) / `data_builder.py` / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`
- 自检: `src/mixture_spn.py` (内嵌) / `src/riesz_selftest.py`
- `docs/architecture.md` — 架构与机制决策录

## 运行

```bash
cd src
python mixture_spn.py        # 模型自检 (公理性质/实例回归/白化相关病理/序列化)
python riesz_selftest.py     # Riesz 自检 + 自然图特征可视化
python inverse.py --quick    # 小数据集自检 (648 样本)
python inverse.py            # 全量 (1296 样本): 插值 u/v R² 0.93, 白光色相 bin 0.68
```

选项: `--sigma-rel-floor` (核带宽下限)、`--equal-luma` (等亮度消融)、`--occlusion` (遮挡场景)、`--model-path` (safetensors 存取; 全量白化基约 1.5GB)。

实测 (全量): 插值 u,v RMSE 5.4/5.2px (R²≈0.93) / 白光色相 bin 0.68 (Δ28°) / kind 0.52 (形状线索密度封顶, 颜色解耦的有意代价) / s,z 报告制 (乘积歧义 ×2 的物理上限)。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
