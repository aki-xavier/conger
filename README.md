# conger

SPN (Sum-Product Network) 逆渲染研究: cga engine 渲染合成场景 → Riesz 单演小波特征 → SPN 结构学习 → 贝叶斯反演 3D 场景码 (kind/gx/gy/size/z), 重建 cga 场景。

## 模块

- `src/riesz.py` — Riesz 小波前端 (跨尺度谱统计特征, 增益控制)
- `src/color.py` — 颜色空间转换 (LAB/HSL/对数色度/白平衡, MLX)
- `src/spn.py` — SPN 结构学习 (G 检验 Product / k-means Sum) + 后验推理 + 码先验注入
- `src/demo_inverse.py` — 逆渲染 demo (数据构建/评估/可视化/消融实验)
- `src/experiment_incremental.py` — 结构增量学习 vs 全量重训对比实验
- `docs/prior.md` — 先验体系 (一般视角/熟悉尺寸/遮挡序数/运动连续性等)

## 运行

```bash
cd src
python spn.py                  # SPN 自检
python riesz.py                # Riesz 自检 + 自然图特征可视化
python demo_inverse.py --quick # 逆渲染小数据集自检
python demo_inverse.py         # 全量 (4000 训练样本, 码空间 1152)
python experiment_incremental.py  # 增量学习实验
```

demo 选项: `--feat l|lhs|hs|rgb` (特征通路)、`--equal-luma` (等亮度消融)、`--occlusion` (遮挡+序数先验)、`--sequence N` (运动先验)、`--multi-light`/`--test-light` (光照泛化)、`--prior edge,familiar` (码先验)、`--tree` (树语义可视化)。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
