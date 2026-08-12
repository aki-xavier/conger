# conger

SPN 逆渲染研究: cga engine 渲染合成场景 → Riesz 全分辨率特征 → **连续**反演 3D 场景参数 (kind, u, v, s, z), 重建 cga 场景。模型为 MixtureSPN (全分辨率浅混合 SPN: PCA 白化 + 逐 kind 分层联合 EM + 条件期望推理)。离散场景码体系 (逐码贝叶斯/池化 SPN/码网格) 已于 2026-08-12 整体退役 —— 连续物理量的离散化只是后验求积, 连续列 + 插值/外推探针才是逆渲染的诚实形态。完整机制决策见 `docs/architecture.md`。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN + 4 组黑盒自检)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (连续采样+投影) / `feature_extractor.py` (11 通道) / `data_builder.py` / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`
- 自检: `src/mixture_spn.py` (内嵌) / `src/riesz_selftest.py`
- `docs/prior.md` — 先验体系 (离散时代遗产, 待重接); `docs/architecture.md` — 架构与机制决策

## 运行

```bash
cd src
python mixture_spn.py        # 模型自检 (公理性质/EM 恢复/白化相关病理/序列化)
python riesz_selftest.py     # Riesz 自检 + 自然图特征可视化
python inverse.py --quick    # 小数据集自检 (800 样本)
python inverse.py            # 全量 (4000 样本): 插值 kind 0.897, u/v RMSE 6.6px
```

选项: `--components K` (混合分量数, 默认 64)、`--em-iters`、`--sigma-rel-floor` (核带宽下限)、`--equal-luma` (等亮度消融)、`--occlusion` (遮挡场景)、`--multi-light`/`--test-light` (光照泛化)、`--model-path` (safetensors 存取; 注意全量白化基约 3GB)。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
