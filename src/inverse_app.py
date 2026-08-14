"""InverseApp: 逆渲染主流程 —— 左右二维图像 → 完整 cga.Scene 重建。

训练数据 → MixtureSPN → 连续/离散场景参数后验 → 物理单位指标
(插值/外推分裂) → 含光照的场景重建可视化 → 自检 + CLI。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx

from codebook import Codebook
from data_builder import DataBuilder
from evaluator import TARGETS, Evaluator
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from mixture_spn import MixtureSPN
from scene_reconstructor import SceneReconstructor


class InverseApp:
    """主流程: 立体图像数据 → MixtureSPN → 完整场景参数 → 重建评估。"""

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg
        self.codebook = Codebook(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.data = DataBuilder(cfg, self.codebook, self.extractor)

    def run(self) -> None:
        cfg = self.cfg
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        n_tr = 162 * cfg.replicates
        print(
            f"[1/4] 数据: train {n_tr} / 插值 324 / 外推 324 "
            f"(162 组合×R={cfg.replicates}, 逐块缓存)"
        )
        f_tr, p_tr, f_ti, p_ti, f_te, p_te, s_tr, s_ti, s_te = (
            self.data.build(cfg.replicates)
        )
        assert mx.all(mx.isfinite(f_tr)), "特征含 NaN/inf"
        # 视差管线独立评估 (几何 ẑ vs GT z; 外推同样几何有效)
        for nm, st, pp in (("插值", s_ti, p_ti), ("外推", s_te, p_te)):
            err = st[:, 0] - pp[:, 4]
            print(
                f"  视差管线 {nm}: ẑ RMSE "
                f"{float(mx.sqrt(mx.mean(err**2))):.4f} "
                f"bias {float(mx.mean(err)):+.4f} "
                f"(d 中位 {float(mx.median(st[:, 1])):.2f}px)"
            )
        t_tr = DataBuilder.targets(p_tr)
        c_tr = DataBuilder.scene_classes(p_tr)

        # 强几何观测 + 残差学习: ẑ/ŝ 直接拼特征会被白化稀释
        # (1/647 维, 实测 z R² 0.02); 模型改为学 z−ẑ 与 s−ŝ 的
        # 标定残差 (可见面≠中心偏差的形状相关部分), 推理后加回
        t_tr = mx.concatenate(
            [
                t_tr[:, :2],
                (t_tr[:, 2] - SceneReconstructor.s_proxy(s_tr))[:, None],
                (t_tr[:, 3] - s_tr[:, 0])[:, None],
            ],
            axis=1,
        )

        # 模型默认持久化: 路径随数据指纹 + 输出契约 (旧 spn_ 模型缺
        # 完整场景类目头, 不复用)。加载后 K < 当前数据量 → 增量追加新块
        # (白化基冻结, 见 MixtureSPN.add); K ≥ 数据量 → 直接用 (模型
        # 训练集是当前请求的超集)
        model_path = cfg.model_path or artifacts / (
            f"spn_full_{self.data.cache_tag()}.safetensors"
        )
        if model_path.exists():
            net = MixtureSPN.load(model_path)
            k_have = net.f_mu.shape[0]
            if k_have < n_tr:
                print(f"[2/4] 增量训练: K {k_have} → {n_tr} (追加新块)")
                net.add(
                    f_tr[k_have:], t_tr[k_have:], c_tr[k_have:, 0],
                    c_tr[k_have:],
                )
                net.save(model_path)
            else:
                print(f"[2/4] 加载模型 {model_path.name} (K={k_have})")
        else:
            print(f"[2/4] MixtureSPN 实例级组装 (K=N={n_tr}, V={f_tr.shape[1]}) ...")
            net = MixtureSPN.fit(
                f_tr,
                t_tr,
                c_tr[:, 0],
                rel_floor=cfg.sigma_rel_floor,
                scene_classes=c_tr,
                cat_sizes=SceneReconstructor.CAT_SIZES,
            )
            net.save(model_path)
            print(f"      模型已保存 → {model_path.name}")

        print("[3/4] 推理: 连续目标条件期望 + 场景因子条件后验")
        ti_raw, ci_p, _ = net.predict(f_ti)
        te_raw, ce_p, _ = net.predict(f_te)
        # 残差加回代理 → 物理量 (s,z)
        ti_pred = SceneReconstructor.physical_targets(ti_raw, s_ti)
        te_pred = SceneReconstructor.physical_targets(te_raw, s_te)
        ci_pred = SceneReconstructor.params(ti_raw, ci_p, s_ti)
        ce_pred = SceneReconstructor.params(te_raw, ce_p, s_te)
        if cfg.refine_appearance:
            print(
                "  渲染残差精炼: hue×lcol×ldir = 54 组合 × 左右视图 "
                "(固定 SPN 几何/kind)"
            )
            ci_pred = self.refine_scenes(ci_pred, p_ti, "插值")
            ce_pred = self.refine_scenes(ce_pred, p_te, "外推")

        print("[4/4] 评估 (物理单位 + 完整场景离散因子; 基线 = 训练均值)")
        mi = Evaluator.report("插值", p_ti, ti_pred, ci_pred, p_tr)
        me = Evaluator.report("外推", p_te, te_pred, ce_pred, p_tr)

        artifacts.mkdir(exist_ok=True)
        self.plot_scatter(p_tr, p_ti, ti_pred, p_te, te_pred,
                          artifacts / "inverse_scatter.png")
        self.plot_recon(p_ti, ci_pred, artifacts / "inverse_recon.png")
        print(f"      图 → {artifacts.name}/ (scatter + recon)")
        self.self_check(mi, me)

    def reconstruct_scene(
        self,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
    ):
        """左/右二维图像 → 完整 cga.Scene (含渲染残差精炼光照)。

        公开推理接口: 帧必须是 Codebook.make_renderer 训练 rig 的渲染
        输出; 返回 (scene, 场景参数, 场景因子后验)。"""
        return SceneReconstructor.from_frames(
            self, net, fl, fr, refine=self.cfg.refine_appearance
        )

    def refine_scenes(
        self,
        scene_pred: tuple[tuple[float, ...], ...],
        p_gt: mx.array,
        name: str,
    ) -> tuple[tuple[float, ...], ...]:
        """对预测场景逐个做候选渲染残差精炼。

        数据缓存只存特征/统计, 不存原图; 这里用 GT 场景重新渲染的像素
        作为模型输入。GT 参数本身不进入精炼, 只用于生成观测帧。"""
        renderer, cam_l, cam_r = SceneReconstructor.rig()
        out = []
        for i, (prm, gt) in enumerate(zip(scene_pred, p_gt.tolist())):
            scene_gt = self.codebook.to_scene(tuple(float(x) for x in gt))
            fl = renderer.render(scene_gt, cam_l)
            fr = renderer.render(scene_gt, cam_r)
            out.append(
                SceneReconstructor.refine_appearance(
                    self.codebook, prm, fl, fr, renderer, cam_l, cam_r
                )[0]
            )
            if (i + 1) % 100 == 0:
                print(f"    {name}: {i + 1}/{len(scene_pred)}")
        return tuple(out)

    # ── 可视化 ──────────────────────────────────────────────────────

    @staticmethod
    def plot_scatter(
        p_tr: mx.array,
        p_ti: mx.array, ti_pred: mx.array,
        p_te: mx.array, te_pred: mx.array,
        out: Path,
    ) -> None:
        """4 目标 GT vs Pred 散点 (插值蓝/外推红) + 训练支撑集边界。"""
        cb = Codebook
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        rng = {"s": cb.S_RANGE + cb.S_EXTRA, "z": cb.Z_RANGE + cb.Z_EXTRA}
        for j, ax in enumerate(axes.flat):
            nm = TARGETS[j]
            gi, pi = p_ti[:, j + 1], ti_pred[:, j]
            ge, pe = p_te[:, j + 1], te_pred[:, j]
            ax.scatter(gi, pi, s=8, alpha=0.6, label="interp")
            ax.scatter(ge, pe, s=8, alpha=0.6, c="r", label="extrap")
            lo = float(mx.min(mx.concatenate([gi, ge])))
            hi = float(mx.max(mx.concatenate([gi, ge])))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            if nm in rng:
                r = rng[nm]
                for b in (r[0], r[1]):  # 训练支撑集边界 (横纵两向)
                    ax.axvline(b, c="gray", ls=":", lw=0.8)
                    ax.axhline(b, c="gray", ls=":", lw=0.8)
            ax.set_xlabel(f"GT {nm}")
            ax.set_ylabel(f"pred {nm}")
            ax.legend(fontsize=8)
        fig.suptitle("inverse rendering: continuous regression, interp vs extrap")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    def plot_recon(
        self, p_gt: mx.array, scene_pred: tuple[tuple[float, ...], ...], out: Path
    ) -> None:
        """3 个插值样本: GT 渲染 vs 完整预测场景重建渲染 (闭环 sanity)。"""
        renderer, cam, _ = Codebook.make_renderer()  # 重建对比用左视图
        n = p_gt.shape[0]
        picks = [0, n // 2, n - 1]
        fig, axes = plt.subplots(len(picks), 2, figsize=(5, 2.6 * len(picks)))
        for row, i in enumerate(picks):
            gt = p_gt[i].tolist()
            pd = scene_pred[i]
            for col, prm in enumerate((gt, pd)):
                img = renderer.render(self.codebook.to_scene(prm), cam)
                axes[row, col].imshow(img[..., :3].astype(mx.int32))
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            axes[row, 0].set_ylabel(
                f"k{gt[0]:.0f} h{gt[5]:.0f} l{gt[6]:.0f}/{gt[7]:.0f}", fontsize=8
            )
        axes[0, 0].set_title("GT render")
        axes[0, 1].set_title("Pred full scene")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    # ── 自检断言 (阈值依据见各注释; 2026-08-12 全量运行标定) ────────

    def self_check(self, mi: dict[str, float], me: dict[str, float]) -> None:
        # kind 颜色解耦后只剩形状线索 (色度泄漏捷径拆除, 这是任务升级
        # 的有意代价)。实测 (实例级模型, rp2 线性光照管线): 全量 0.55
        # —— 密度封顶 (同密度 1-NN 同值)。阈值 0.45 只防
        # 机制崩溃 (随机 0.33); 逐 kind 白化 (PPCA 似然比) 是升级候选
        assert mi["kind"] > 0.45, f"kind 准确率过低 {mi['kind']:.3f}"
        # 插值位置回归: 实测全量 5.9/5.1 (旧网格半档 9px
        # 以下 = 连续模型优于量化误差的及格线)
        assert mi["u_rmse"] < 9.0, f"插值 u RMSE {mi['u_rmse']:.2f}px"
        assert mi["v_rmse"] < 9.0, f"插值 v RMSE {mi['v_rmse']:.2f}px"
        if self.cfg.refine_appearance:
            # 渲染残差精炼后的外观契约 (2026-08-13 全量实测): hue 0.994 /
            # lcol 0.972 / ldir 0.830。阈值明显高于随机 (6 档 0.167,
            # 3 档 0.333), 防精炼级失效而非追求当前上限
            assert mi["hue"] > 0.9, f"图元色相准确率 {mi['hue']:.3f}"
            assert mi["lcol"] > 0.85, f"光色准确率过低 {mi['lcol']:.3f}"
            assert mi["ldir"] > 0.7, f"光向准确率过低 {mi['ldir']:.3f}"
        else:
            # 无精炼调试路径: SPN 共享责任度只要求显著超随机
            # (首版实测 hue/lcol/ldir = 0.577/0.457/0.367)
            assert mi["hue"] > 0.5, f"SPN 色相准确率 {mi['hue']:.3f}"
            assert mi["lcol"] > 0.4, f"SPN 光色准确率 {mi['lcol']:.3f}"
            assert mi["ldir"] > 0.34, f"SPN 光向准确率 {mi['ldir']:.3f}"
        # 视差把 z 几何钉死 → s=表观×zc 随解 (乘积歧义破解)。
        # 实测 (2026-08-13, rp2 管线): z R² 全量 0.85,
        # s R² 全量 0.41; 外推 z R² 0.96 (几何不饱和)
        assert mi["z_r2"] > 0.6, f"插值 z R² {mi['z_r2']:.3f}"
        assert mi["s_r2"] > 0.2, f"插值 s R² {mi['s_r2']:.3f}"
        print("inverse: 自检 ✓ (完整 Scene: 几何 + 颜色 + 光照)")

    # ── CLI ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_args() -> InverseConfig:
        """CLI → InverseConfig (一切开关的唯一家)。"""
        ap = argparse.ArgumentParser()
        ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
        ap.add_argument(
            "--model-path",
            default=None,
            help="模型存取路径 (safetensors, 默认 artifacts/spn_full_<数据指纹>); "
            "存在则加载跳过组装, 否则组装后保存",
        )
        ap.add_argument(
            "--sigma-rel-floor",
            type=float,
            default=1e-2,
            help="σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, "
            "插值平滑度旋钮",
        )
        ap.add_argument(
            "--replicates",
            type=int,
            default=8,
            help="训练集复制数 R (162 组合×R 帧对): 逐块缓存, 调大触发增量训练",
        )
        ap.add_argument(
            "--no-refine-appearance",
            action="store_true",
            help="跳过 hue×lcol×ldir 候选渲染残差精炼 (快, 但光照输出退化)",
        )
        a = ap.parse_args()
        return InverseConfig(
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            sigma_rel_floor=a.sigma_rel_floor,
            replicates=a.replicates,
            refine_appearance=not a.no_refine_appearance,
        )
