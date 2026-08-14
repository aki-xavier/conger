# conger

SPN 逆渲染研究: 左右两张二维立体图像 → Riesz 全分辨率特征 → MixtureSPN → **完整 `cga.Scene` 重建** (kind, u, v, s, z, 图元色相, 光色, 光向)。模型为 MixtureSPN (全分辨率实例级浅混合 SPN: PCA 白化 + 逐 kind 分层, 每样本一个对角高斯块; 连续条件期望 ≡ 分层核回归, 离散场景因子 ≡ 条件后验分类; 无 EM, 确定性)。SPN 初估几何/kind 后, `SceneReconstructor` 再枚举 hue×光色×光向候选并用左右图渲染残差做联合精炼。完整机制决策见 `docs/architecture.md`。

训练数据全因子覆盖设计: 离散因子 (kind 3 × 图元色 6 色相 × 光色 3 × 光向 3 = 162 组合) 全笛卡尔积 × R 连续复制 —— 组合覆盖保证且数据量最小 (全量 1296 帧对 vs 旧 4600)。图元色与 kind 解耦 (kind 只剩形状线索, 任务升级); 光色/光向不再丢弃, 而是作为完整场景输出显式监督。

## 模块 (一文件一类)

- 模型: `src/mixture_spn.py` (MixtureSPN)
- demo 族: `src/inverse_config.py` (配置唯一家) / `codebook.py` (组合采样+投影) / `feature_extractor.py` (11 通道) / `data_builder.py` / `scene_reconstructor.py` (帧对/参数 → 完整 Scene) / `evaluator.py` / `inverse_app.py`, `src/inverse.py` 为薄 CLI 入口
- 前端: `src/riesz.py` + `riesz_scale.py` + `feature_maps.py` (Riesz 小波), `src/color.py`, `src/utils.py`
- 测试: `tests/` (pytest; 单元黑盒 + slow 集成自检) / `src/riesz_selftest.py` (可视化脚本)
- `docs/architecture.md` — 架构与机制决策录

## 运行

```bash
pytest                       # 单元黑盒测试 (mixture_spn/color/stereo/scene)
pytest -m slow               # 集成自检 (含 54 候选渲染残差精炼, 分钟级)
cd src
python riesz_selftest.py     # Riesz 自检 + 自然图特征可视化
python inverse.py            # 全量立体 (1296 帧对): 完整 Scene 重建
```

选项: `--sigma-rel-floor` (核带宽下限)、`--replicates R` (训练集复制数, 调大触发增量训练)、`--no-cache` (跳过数据缓存)、`--no-refine-appearance` (跳过 54 组色相/光照候选的渲染残差精炼)、`--model-path` (模型 safetensors 存取, 默认 `artifacts/spn_full_<数据指纹>`; 存在即加载, K 不足则增量追加)。

## 推理接口

```python
from inverse_app import InverseApp
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from scene_reconstructor import SceneReconstructor

app = InverseApp(InverseConfig())
net = MixtureSPN.load(f"artifacts/spn_full_{app.data.cache_tag()}.safetensors")
renderer, cam_l, cam_r = SceneReconstructor.rig()
# fl/fr 为同一 cga.Scene 在训练 rig 下的左/右二维渲染帧
scene, params, posterior = app.reconstruct_scene(net, fl, fr)
```

`scene` 是包含预测 DirectionalLight 的 `cga.Scene`; `params` 是
渲染残差精炼后的 `(kind,u,v,s,z,hue,lcol,ldir)`; `posterior` 是 SPN
初估的 4 因子后验, 用于检查共享责任度下的不确定性。

实测 (渲染残差精炼版, 全量立体 N=1296): 插值 u,v R² 0.930/0.945 / s R² 0.332 / z R² 0.831 / kind 0.577 / hue 0.994 / lcol 0.972 / ldir 0.830; 外推 u,v R² 0.949/0.953 / s,z R² 0.909/0.956 / kind 0.515 / hue 0.981 / lcol 0.861 / ldir 0.731。几何量由视差钉死且外推不饱和; 色相与光照由候选重渲染联合裁决。

依赖: mlx / matplotlib / numpy / pillow + 本地 path 依赖 [cga](../cga) (渲染引擎)。
