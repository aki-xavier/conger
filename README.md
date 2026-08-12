# conger

SPN (Sum-Product Network) 逆渲染研究: cga engine 渲染合成场景 → Riesz 单演小波特征 → 反演 3D 场景码 (kind/gx/gy/size/z), 重建 cga 场景。主模型为全分辨率逐码贝叶斯 (CodeBayes, 码 0.965); SPN 为组合泛化/消融对照。完整流程图见 `docs/architecture.md`。

## 模块 (一文件一类)

- 模型: `src/code_bayes.py` (CodeBayes 逐码贝叶斯, 精确可增量)
- SPN 族: `src/node.py` / `leaf.py` / `gauss_leaf.py` / `cat_leaf.py` / `product.py` / `sum_node.py` (节点多态体系), `src/spn.py` (SPN 推理+序列化), `src/spn_learner.py` (SPNLearner 结构学习), `src/online_spn.py` (OnlineSPN 在线学习)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` / `feature_extractor.py` / `data_builder.py` / `priors.py` / `evaluator.py` / `sequence_runner.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`
- 实验: `src/experiment_incremental.py` (增量+修订 vs 全量), `src/experiment_fullres.py` (全分辨率三臂+未见码探针), `src/experiment_joint.py` (开放集门控双轨)
- 自检: `src/spn_selftest.py` / `riesz_selftest.py` / `code_bayes.py` (内嵌)
- `docs/prior.md` — 先验体系; `docs/architecture.md` — 架构与完整流程图

## 运行

```bash
cd src
python code_bayes.py           # CodeBayes 自检
python spn_selftest.py         # SPN + OnlineSPN 自检 (7 组)
python riesz_selftest.py       # Riesz 自检 + 自然图特征可视化
python inverse.py --quick # 逆渲染小数据集自检 (默认 nb 模型)
python inverse.py         # 全量 nb (4000 样本, 码 0.965)
python inverse.py --model spn  # 池化 SPN 对照 (0.470)
python experiment_incremental.py --rev-cap 2048 --rev-at 3  # 增量+修订 (×0.98)
python experiment_fullres.py   # 全分辨率三臂 + 未见码泛化探针
python experiment_joint.py     # 开放集门控双轨 + 提升
```

demo 选项: `--feat l|lhs|hs|rgb` (特征通路)、`--equal-luma` (等亮度消融)、`--occlusion` (遮挡+序数先验)、`--sequence N` (运动先验)、`--multi-light`/`--test-light` (光照泛化)、`--prior edge,familiar` (码先验)、`--tree` (树语义可视化)。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
