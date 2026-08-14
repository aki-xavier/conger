# conger

SPN 逆渲染研究: 左右两张二维立体图像 → Riesz 全分辨率特征 → MixtureSPN → **完整 `cga.Scene` 重建** (kind, u, v, s, z, 图元色相, 光色, 光向)。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块; 连续条件期望 ≡ 分层核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性)。SPN 初估后, `SceneReconstructor` 覆盖全部 kind, 结构评分沿用共享几何，候选返回前再按各自 kind 的面积→尺寸代理重校准 s，并与 hue×光色×光向候选一起做左右图渲染残差联合精炼。完整机制决策见 `docs/architecture.md`。

训练数据全因子覆盖设计: 单物体离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制; `--n-objects 2` 时双图元前后层为 kind0×kind1×hue0×hue1×光色×光向 = 2916 组合, 约 70% 样本强制投影重叠。图元色与 kind 解耦; 光色/光向不再丢弃, 而是作为完整场景输出显式监督。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (单物体组合采样+投影) / `layered_codebook.py` (双物体遮挡/前后层) / `feature_extractor.py` (11 通道) / `data_builder.py` / `scene_reconstructor.py` (帧对/参数 → 完整 Scene) / `layered_reconstructor.py` (双层 SPN 解码) / `scene_estimate.py` (Scene 后验返回对象) / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`, `src/stereo.py` (单物体视差), `src/stereo_layers.py` + `src/contour_completion.py` (逐层视差与后层轮廓补全)
- 测试: `tests/` (pytest; 单元黑盒 + slow 集成自检) / `src/riesz_selftest.py` (可视化脚本)
- `docs/architecture.md` — 架构与机制决策录

## 运行

```bash
pytest                       # 单元黑盒测试 (mixture_spn/color/stereo/scene)
pytest -m slow               # 集成自检 (默认全 kind 结构精炼, 约二十分钟)
cd src
python riesz_selftest.py     # Riesz 自检 + 自然图特征可视化
python inverse.py            # 全量立体 (1296 帧对): 完整 Scene 重建
```

选项: `--sigma-rel-floor` (核带宽下限)、`--replicates R` (训练集复制数, 调大触发增量训练)、`--no-cache` (跳过数据缓存)、`--no-refine-appearance` (跳过候选渲染残差精炼)、`--kind-topk {1,2,3}` (结构候选数, 默认 3 = 覆盖全部 kind)、`--n-objects {1,2}` (1 单图元 / 2 双图元遮挡前后层)、`--model-path` (模型 safetensors 存取, 默认 `artifacts/spn_kindgeo_<数据指纹>` 或 `spn_layered_<数据指纹>`; 存在即加载, K 不足则增量追加)。

## 推理接口

```python
from inverse_app import InverseApp
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from scene_reconstructor import SceneReconstructor

app = InverseApp(InverseConfig())
net = MixtureSPN.load(f"artifacts/spn_kindgeo_{app.data.cache_tag()}.safetensors")
renderer, cam_l, cam_r = SceneReconstructor.rig()
# fl/fr 为同一 cga.Scene 在训练 rig 下的左/右二维渲染帧
estimate = app.reconstruct_scene(net, fl, fr)
scene = estimate.scene                 # MAP cga.Scene
params = estimate.params               # MAP 完整场景参数
kind_p, hue_p, lcol_p, ldir_p = estimate.factor_marginals()
for h in estimate.hypotheses:          # top 候选完整场景
    print(h.probability, h.residual, h.params)
```

`estimate.scene` 是包含预测 DirectionalLight 的 `cga.Scene`; `params` 是
渲染残差精炼后的 `(kind,u,v,s,z,hue,lcol,ldir)`; `spn_posterior` 是
SPN 初估的 4 因子后验; `candidate_posterior` 是渲染残差联合后验,
避免把逆渲染歧义过早压成单个点估计。

实测 (结构-外观联合精炼版, 全量立体 N=1296, `kind_topk=3`): 插值 u,v R² 0.930/0.945 / s R² 0.508 / z R² 0.831 / kind 0.753 / hue 1.000 / lcol 0.994 / ldir 0.895; 外推 u,v R² 0.949/0.953 / s,z R² 0.922/0.956 / kind 0.617 / hue 0.981 / lcol 0.880 / ldir 0.772。结构评分沿用共享几何避免尺寸代理偏差，MAP 后按 kind 重校准 s; 色相与光照由候选重渲染联合裁决。

消融: 旧固定几何 top-3 为 kind 0.753 / s R² 0.332; 纯解析逐 kind 几何会使插值 s R² 降至 0.160 (掩码观测偏差不可忽略); 共享评分 + kind 后校准得到上述最优平衡。

双层遮挡实验族 (`--n-objects 2 --replicates 1`, N=2916, sl4): StereoLayers 逐层视差 + soft-fusion 轮廓补全后, 插值 kind0/kind1 0.397/0.364、hue0/hue1 0.415/0.166、lcol/ldir 0.397/0.372; v0/u1/v1 R² 0.723/0.395/0.356, 前层 z0 R² 0.223。后层 s/z 仍为负 R²: 补全可改善部分可见轮廓，但当前错误聚类下面积先验仍不稳定。当前策略为前层全残差、后层 u/v 残差 + s/z 锚点; 下一步需逐层轮廓联合优化或更高复制密度。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
