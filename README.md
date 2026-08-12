# conger

SPN (Sum-Product Network) 逆渲染研究: cga engine 渲染合成场景 → Riesz 单演小波特征 → SPN 结构学习 → 贝叶斯反演 3D 场景码 (kind/gx/gy/size/z), 重建 cga 场景。

## 模块

- `src/code_bayes.py` — 逐码对角高斯贝叶斯 (全分辨率, 精确可增量; 码簿任务主模型)
- `src/riesz.py` — Riesz 小波前端 (跨尺度谱统计特征, 增益控制)
- `src/color.py` — 颜色空间转换 (LAB/HSL/对数色度/白平衡, MLX)
- `src/spn.py` — SPN 结构学习 + 后验推理 + OnlineSPN 在线学习器 (软路由/统计常驻/延迟生长)
- `src/demo_inverse.py` — 逆渲染 demo (--model nb|spn, 先验注入/消融/可视化)
- `src/experiment_incremental.py` — 增量学习实验 (纯在线/周期修订 vs 全量)
- `src/experiment_fullres.py` — 全分辨率三臂对照 (nb/flat-EM/spn + 未见码探针)
- `src/experiment_joint.py` — 开放集联合系统 (门控双轨 + 提升, 探针: 未见码族/新类别)
- `docs/prior.md` — 先验体系 (一般视角/熟悉尺寸/遮挡序数/运动连续性等)

## 运行

```bash
cd src
python code_bayes.py           # CodeBayes 自检
python spn.py                  # SPN + OnlineSPN 自检
python riesz.py                # Riesz 自检 + 自然图特征可视化
python demo_inverse.py --quick # 逆渲染小数据集自检 (默认 nb 模型)
python demo_inverse.py         # 全量 nb (4000 样本, 码 0.965)
python demo_inverse.py --model spn  # 池化 SPN 对照 (0.470)
python experiment_incremental.py --rev-cap 2048 --rev-at 3  # 增量+修订 (×0.98)
python experiment_fullres.py   # 全分辨率三臂 + 未见码泛化探针
```

demo 选项: `--feat l|lhs|hs|rgb` (特征通路)、`--equal-luma` (等亮度消融)、`--occlusion` (遮挡+序数先验)、`--sequence N` (运动先验)、`--multi-light`/`--test-light` (光照泛化)、`--prior edge,familiar` (码先验)、`--tree` (树语义可视化)。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
