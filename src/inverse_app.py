"""InverseApp: 逆渲染主流程 —— 左右二维图像 → 完整 cga.Scene 重建。

训练数据 → MixtureSPN → 连续/离散场景参数后验 → 物理单位指标
(插值/外推分裂) → 含光照的场景重建可视化 → 自检 + CLI。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx

from codebook import Codebook
from composite_codebook import CompositeCodebook
from composite_reconstructor import CompositeReconstructor
from data_builder import DataBuilder
from evaluator import LAYERED_TARGET_COLS, TEXTURED_TARGET_COLS, Evaluator
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from layered_child_reconstructor import ConstrainedLayeredReconstructor
from layered_codebook import LayeredCodebook
from layered_reconstructor import LayeredReconstructor
from mixture_spn import MixtureSPN
from scene_reconstructor import SceneReconstructor


class InverseApp:
    """主流程: 立体图像数据 → MixtureSPN → 完整场景参数 → 重建评估。"""

    def __init__(self, cfg: InverseConfig, codebook: Codebook | None = None):
        self.cfg = cfg
        if codebook is not None:
            self.codebook = codebook
        elif cfg.family == "single":
            self.codebook = Codebook(cfg)
        elif cfg.family == "layered":
            self.codebook = LayeredCodebook(cfg)
        elif cfg.family == "composite":
            self.codebook = CompositeCodebook(cfg)
        else:
            raise ValueError(f"未知 scene_family: {cfg.family}")
        self.extractor = FeatureExtractor(cfg)
        self.data = DataBuilder(cfg, self.codebook, self.extractor)

    def layered_reconstructor(self):
        """父 layered 后层锚定; 受限子模板允许全残差学习。"""
        if self.codebook.TEMPLATE_VARIANT:
            return ConstrainedLayeredReconstructor
        return LayeredReconstructor

    def default_model_path(self, artifacts: Path | None = None) -> Path:
        """当前结构专家配置的默认模型路径 (注册表/训练共用契约)。"""
        root = artifacts or Path(__file__).resolve().parent.parent / "artifacts"
        prefix = {
            "single": "spn_kindgeo",
            "layered": "spn_layered_anchor",
            "composite": "spn_composite",
        }[self.cfg.family]
        dim_tag = f"_d{self.cfg.basis_dim}" if self.cfg.basis_dim else ""
        return root / f"{prefix}_{self.data.cache_tag()}{dim_tag}.safetensors"

    def run(self, artifacts: Path | None = None) -> None:
        cfg = self.cfg
        artifacts = artifacts or Path(__file__).resolve().parent.parent / "artifacts"
        n_tr = self.codebook.N_COMBO * cfg.replicates
        n_test = self.codebook.N_COMBO * (1 if cfg.family != "single" else 2)
        print(
            f"[1/4] 数据: train {n_tr} / 插值 {n_test} / 外推 {n_test} "
            f"({self.codebook.N_COMBO} 组合×R={cfg.replicates}, "
            f"family={cfg.family}, 逐块缓存)"
        )
        f_tr, p_tr, f_ti, p_ti, f_te, p_te, s_tr, s_ti, s_te = self.data.build(
            cfg.replicates
        )
        assert mx.all(mx.isfinite(f_tr)), "特征含 NaN/inf"
        # 视差管线独立评估 (单物体几何 ẑ vs GT z; 双层逐层统计另有专题)
        if cfg.family == "single":
            pairs = (("插值", s_ti, p_ti), ("外推", s_te, p_te))
            for nm, st, pp in pairs:
                err = st[:, 0] - pp[:, 4]
                print(
                    f"  视差管线 {nm}: ẑ RMSE "
                    f"{float(mx.sqrt(mx.mean(err**2))):.4f} "
                    f"bias {float(mx.mean(err)):+.4f} "
                    f"(d 中位 {float(cast(Any, mx).median(st[:, 1])):.2f}px)"
                )
        elif cfg.family == "layered":
            print("  视差管线: 双层遮挡使用 StereoLayers 逐层统计")
        else:
            if str(self.codebook.GEOMETRY_FAMILY) == "lateral":
                print("  视差管线: 横向组合使用 LateralCompositeGeometry 统计")
            else:
                print("  视差管线: 附着组合使用 CompositeGeometry base/part 统计")
        t_tr = DataBuilder.targets(p_tr)
        c_tr = DataBuilder.scene_classes(p_tr)

        if cfg.family == "single":
            # 强几何观测 + 残差学习: ẑ/ŝ 直接拼特征会被白化稀释
            # (1/647 维, 实测 z R² 0.02); 模型改为学 z−ẑ 与 s−ŝ 的
            # 标定残差 (可见面≠中心偏差的形状相关部分), 推理后加回
            t_tr = mx.concatenate(
                [
                    t_tr[:, :2],
                    (
                        t_tr[:, 2]
                        - SceneReconstructor.s_proxy(c_tr[:, 0], s_tr)
                    )[:, None],
                    (t_tr[:, 3] - s_tr[:, 0])[:, None],
                ],
                axis=1,
            )

        elif cfg.family == "composite":
            t_tr = CompositeReconstructor.residual_targets(t_tr, c_tr, s_tr)
        else:
            t_tr = self.layered_reconstructor().residual_targets(t_tr, c_tr, s_tr)

        # 模型默认持久化: 路径随数据指纹 + 输出/几何残差契约 (旧
        # spn_full_ 模型的 s 残差仍基于统一球代理, 不复用)。加载后
        # K < 当前数据量 → 增量追加新块
        # (白化基冻结, 见 MixtureSPN.add); K ≥ 数据量 → 直接用 (模型
        # 训练集是当前请求的超集)
        model_path = cfg.model_path or self.default_model_path(artifacts)
        if model_path.exists():
            net = MixtureSPN.load(model_path)
            k_have = net.f_mu.shape[0]
            if k_have < n_tr:
                print(f"[2/4] 增量训练: K {k_have} → {n_tr} (追加新块)")
                net.add(
                    f_tr[k_have:],
                    t_tr[k_have:],
                    c_tr[k_have:, 0],
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
                cat_sizes=(
                    SceneReconstructor.cat_sizes(cfg.n_textures)
                    if cfg.family == "single"
                    else LayeredReconstructor.CAT_SIZES
                ),
                basis_dim=cfg.basis_dim,
            )
            net.save(model_path)
            print(f"      模型已保存 → {model_path.name}")

        print("[3/4] 推理: 连续目标条件期望 + 场景因子条件后验")
        ti_raw, ci_p, _ = net.predict(f_ti)
        te_raw, ce_p, _ = net.predict(f_te)
        if cfg.family == "single":
            # 残差加回 kind-conditioned 代理 → 物理量 (s,z)
            ki0 = mx.argmax(ci_p[:, : Codebook.N_KIND], axis=1)
            ke0 = mx.argmax(ce_p[:, : Codebook.N_KIND], axis=1)
            ti_pred = SceneReconstructor.physical_targets(ti_raw, s_ti, ki0)
            te_pred = SceneReconstructor.physical_targets(te_raw, s_te, ke0)
            sz = SceneReconstructor.cat_sizes(cfg.n_textures)
            ci_pred = SceneReconstructor.params(ti_raw, ci_p, s_ti, sz)
            ce_pred = SceneReconstructor.params(te_raw, ce_p, s_te, sz)
            if cfg.refine_appearance and not cfg.textured:
                print(
                    f"  渲染残差精炼: top{cfg.kind_topk} kind × "
                    "hue×lcol×ldir 候选 × 左右视图"
                )
                ci_pred = self.refine_scenes(ci_pred, ci_p, s_ti, p_ti, "插值")
                ce_pred = self.refine_scenes(ce_pred, ce_p, s_te, p_te, "外推")
                ti_pred = SceneReconstructor.targets_from_params(ci_pred)
                te_pred = SceneReconstructor.targets_from_params(ce_pred)
        elif cfg.family == "composite":
            ci_pred = CompositeReconstructor.params(ti_raw, ci_p, s_ti)
            ce_pred = CompositeReconstructor.params(te_raw, ce_p, s_te)
            ti_pred = CompositeReconstructor.targets_from_params(ci_pred)
            te_pred = CompositeReconstructor.targets_from_params(ce_pred)
            if cfg.refine_composite:
                print("  组合渲染残差精炼: top-k kind/hue/light 候选")
                ci_pred = self.refine_composite_scenes(
                    ci_pred, ci_p, p_ti, "插值"
                )
                ce_pred = self.refine_composite_scenes(
                    ce_pred, ce_p, p_te, "外推"
                )
                ti_pred = CompositeReconstructor.targets_from_params(ci_pred)
                te_pred = CompositeReconstructor.targets_from_params(ce_pred)
            if str(self.codebook.GEOMETRY_FAMILY) == "lateral":
                print("  横向组合: 左右 part 模板锚点 + SPN 有界残差")
            else:
                print("  附着组合: base/part 模板锚点 + SPN 有界残差")
        else:
            reconstructor = self.layered_reconstructor()
            ci_pred = reconstructor.params(ti_raw, ci_p, s_ti)
            ce_pred = reconstructor.params(te_raw, ce_p, s_te)
            ti_pred = reconstructor.targets_from_params(ci_pred)
            te_pred = reconstructor.targets_from_params(ce_pred)
            print("  双层遮挡: SPN 后验报告模式 (渲染残差精炼待分层几何)")

        print("[4/4] 评估 (物理单位 + 完整场景离散因子; 基线 = 训练均值)")
        mi = Evaluator.report("插值", p_ti, ti_pred, ci_pred, p_tr)
        me = Evaluator.report("外推", p_te, te_pred, ce_pred, p_tr)

        artifacts.mkdir(exist_ok=True)
        self.plot_scatter(
            p_tr, p_ti, ti_pred, p_te, te_pred, artifacts / "inverse_scatter.png"
        )
        self.plot_recon(p_ti, ci_pred, artifacts / "inverse_recon.png")
        print(f"      图 → {artifacts.name}/ (scatter + recon)")
        self.self_check(mi, me)

    def reconstruct_scene(
        self,
        net: MixtureSPN,
        fl: mx.array,
        fr: mx.array,
    ):
        """左/右二维图像 → StructuredHypothesis (MAP Scene + 结构化候选后验)。

        公开推理接口: 帧必须是 Codebook.make_renderer 训练 rig 的渲染
        输出; 返回值包含 SPN 后验、渲染候选后验和 top 场景假设。"""
        if self.cfg.family == "layered":
            return self.layered_reconstructor().from_frames(self, net, fl, fr)
        if self.cfg.family == "composite":
            return CompositeReconstructor.from_frames(
                self, net, fl, fr, refine=self.cfg.refine_composite
            )
        return SceneReconstructor.from_frames(
            self,
            net,
            fl,
            fr,
            refine=self.cfg.refine_appearance,
            kind_topk=self.cfg.kind_topk,
        )

    def refine_scenes(
        self,
        scene_pred: tuple[tuple[float, ...], ...],
        cat_p: mx.array,
        stats: mx.array,
        p_gt: mx.array,
        name: str,
    ) -> tuple[tuple[float, ...], ...]:
        """对预测场景逐个做候选渲染残差精炼。

        数据缓存只存特征/统计, 不存原图; 这里用 GT 场景重新渲染的像素
        作为模型输入。GT 参数本身不进入精炼, 只用于生成观测帧。"""
        renderer, cam_l, cam_r = SceneReconstructor.rig()
        out = []
        for i, (prm, kp, st, gt) in enumerate(
            zip(scene_pred, cat_p, stats, cast(list, p_gt.tolist()), strict=True)
        ):
            scene_gt = self.codebook.to_scene(tuple(float(x) for x in gt))
            fl = renderer.render(scene_gt, cam_l)
            fr = renderer.render(scene_gt, cam_r)
            refined = SceneReconstructor.refine_scene(
                self.codebook,
                prm,
                kp[: Codebook.N_KIND],
                st,
                fl,
                fr,
                kind_topk=self.cfg.kind_topk,
                renderer=renderer,
                cam_l=cam_l,
                cam_r=cam_r,
                marginalize=self.cfg.appearance_marginalize,
            )[0]
            # 几何↔光照 ECM 精炼 (§7.1), 与推理链路共用同一 helper
            refined, _ = SceneReconstructor.em_refine(self, refined, fl, fr)
            out.append(refined)
            if (i + 1) % 100 == 0:
                print(f"    {name}: {i + 1}/{len(scene_pred)}")
        return tuple(out)

    def refine_composite_scenes(
        self,
        scene_pred: tuple[tuple[float, ...], ...],
        cat_p: mx.array,
        p_gt: mx.array,
        name: str,
    ) -> tuple[tuple[float, ...], ...]:
        """对组合预测逐个做 top-k 结构/外观候选渲染残差精炼。"""
        renderer, cam_l, cam_r = SceneReconstructor.rig()
        out = []
        for i, (prm, kp, gt) in enumerate(
            zip(scene_pred, cat_p, cast(list, p_gt.tolist()), strict=True)
        ):
            scene_gt = self.codebook.to_scene(tuple(float(x) for x in gt))
            fl = renderer.render(scene_gt, cam_l)
            fr = renderer.render(scene_gt, cam_r)
            out.append(
                CompositeReconstructor.refine_scene(
                    self.codebook,
                    prm,
                    kp,
                    fl,
                    fr,
                    renderer=renderer,
                    cam_l=cam_l,
                    cam_r=cam_r,
                )[0]
            )
            if (i + 1) % 100 == 0:
                print(f"    {name}: {i + 1}/{len(scene_pred)}")
        return tuple(out)

    # ── 可视化 ──────────────────────────────────────────────────────

    @staticmethod
    def plot_scatter(
        p_tr: mx.array,
        p_ti: mx.array,
        ti_pred: mx.array,
        p_te: mx.array,
        te_pred: mx.array,
        out: Path,
    ) -> None:
        """连续目标 GT vs Pred 散点 (插值蓝/外推红)。"""
        cb = Codebook
        names = Evaluator.target_names(p_ti)
        if p_ti.shape[1] == 14:
            cols = LAYERED_TARGET_COLS
        elif p_ti.shape[1] == 10:
            cols = TEXTURED_TARGET_COLS
        else:
            cols = (1, 2, 3, 4)
        n = len(names)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.5 * ((n + 1) // 2), 8))
        rng = {"s": cb.S_RANGE + cb.S_EXTRA, "z": cb.Z_RANGE + cb.Z_EXTRA}
        for j, ax in enumerate(axes.flat):
            if j >= n:
                ax.axis("off")
                continue
            nm = names[j]
            gi, pi = p_ti[:, cols[j]], ti_pred[:, j]
            ge, pe = p_te[:, cols[j]], te_pred[:, j]
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
            gt = cast(list[float], p_gt[i].tolist())
            pd = scene_pred[i]
            for col, prm in enumerate((gt, pd)):
                img = renderer.render(self.codebook.to_scene(prm), cam)
                axes[row, col].imshow(img[..., :3].astype(mx.int32))
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            if len(gt) == 14:
                label = (
                    f"k{gt[0]:.0f}/{gt[6]:.0f} h{gt[5]:.0f}/{gt[11]:.0f} "
                    f"l{gt[12]:.0f}/{gt[13]:.0f}"
                )
            else:
                label = f"k{gt[0]:.0f} h{gt[5]:.0f} l{gt[6]:.0f}/{gt[7]:.0f}"
            axes[row, 0].set_ylabel(label, fontsize=8)
        axes[0, 0].set_title("GT render")
        axes[0, 1].set_title("Pred full scene")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    # ── 自检断言 (阈值依据见各注释; 2026-08-12 全量运行标定) ────────

    def self_check(self, mi: dict[str, float], me: dict[str, float]) -> None:
        if self.cfg.family in {"layered", "composite"} or self.cfg.textured:
            vals = list(mi.values()) + list(me.values())
            assert all(math.isfinite(v) for v in vals), "多图元/纹理指标含 NaN/inf"
            tag = "textured" if self.cfg.textured else self.cfg.family
            print(f"{tag}: 报告模式 ✓ (结构/纹理族自检; 阈值待标定)")
            return
        # 全 kind 结构精炼实测 (kindgeo 契约, 2026-08-13): 插值 0.753;
        # 阈值防结构候选机制崩溃 (随机 0.33), 不把当前上限硬编码过紧
        kind_floor = 0.65 if self.cfg.refine_appearance else 0.45
        assert mi["kind"] > kind_floor, (
            f"kind 准确率过低 {mi['kind']:.3f} (阈值 {kind_floor})"
        )
        # 插值位置回归: 实测全量 5.9/5.1 (旧网格半档 9px
        # 以下 = 连续模型优于量化误差的及格线)
        assert mi["u_rmse"] < 9.0, f"插值 u RMSE {mi['u_rmse']:.2f}px"
        assert mi["v_rmse"] < 9.0, f"插值 v RMSE {mi['v_rmse']:.2f}px"
        if self.cfg.refine_appearance:
            # 渲染残差精炼后的外观契约 (kindgeo 全量实测): hue 1.000 /
            # lcol 0.994 / ldir 0.895。阈值明显高于随机 (6 档 0.167,
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
        # 视差把 z 几何钉死 → s=表观×zc 随解; kindgeo 契约用
        # kind-conditioned 面积代理重校准 s。全量实测 z R² 0.831,
        # s R² 0.508; 外推 s/z R² 0.922/0.956 (几何不饱和)
        assert mi["z_r2"] > 0.6, f"插值 z R² {mi['z_r2']:.3f}"
        s_floor = 0.4 if self.cfg.refine_appearance else 0.2
        assert mi["s_r2"] > s_floor, (
            f"插值 s R² {mi['s_r2']:.3f} (阈值 {s_floor})"
        )
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
            help="模型存取路径 (safetensors, 默认 artifacts/spn_kindgeo_<数据指纹>); "
            "存在则加载跳过组装, 否则组装后保存",
        )
        ap.add_argument(
            "--sigma-rel-floor",
            type=float,
            default=1e-2,
            help="σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, 插值平滑度旋钮",
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
        ap.add_argument(
            "--refine-composite",
            action="store_true",
            help="组合模板启用 top-k kind/hue/light 渲染残差精炼 (默认关闭)",
        )
        ap.add_argument(
            "--kind-topk",
            type=int,
            default=3,
            choices=(1, 2, 3),
            help="结构候选数: top-k kind 进入渲染残差联合后验 (默认全覆盖 3)",
        )
        ap.add_argument(
            "--scene-family",
            default=None,
            choices=("single", "layered", "composite"),
            help="结构族: single 单图元 / layered 独立前后层 / composite 附着组合模板",
        )
        ap.add_argument(
            "--em-refine",
            action="store_true",
            help="推理期几何↔光照 ECM 精炼 (§7.1, 默认关闭, 仅单物体)",
        )
        ap.add_argument(
            "--em-no-freeze-sz",
            action="store_true",
            help="ECM 几何坐标搜索恢复四维全搜 (u,v,s,z; 默认冻结 s/z 只搜 u/v)",
        )
        ap.add_argument(
            "--appearance-marginalize",
            action="store_true",
            help="解耦边缘 MAP (因果不变估计): 反照率对光照、光照对反照率/几何"
            "分别边缘化后 argmax (默认关闭, 走联合 argmax)",
        )
        ap.add_argument(
            "--n-textures",
            type=int,
            default=0,
            help="纹理自由度: 0=关(默认); >0=单物体加 n 种 albedo map 纹理(离散) "
            "+ roughness(连续), 组合数 162→162×n",
        )
        ap.add_argument(
            "--basis-dim",
            type=int,
            default=48,
            help="白化基内在维截断 (docs §10.3): 默认 48 (全面优于基线); "
            "设 0 或负数回全维; 模型路径带 _dN 后缀",
        )
        a = ap.parse_args()
        return InverseConfig(
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            sigma_rel_floor=a.sigma_rel_floor,
            replicates=a.replicates,
            refine_appearance=not a.no_refine_appearance,
            refine_composite=a.refine_composite,
            kind_topk=a.kind_topk,
            scene_family=a.scene_family,
            em_refine=a.em_refine,
            em_freeze_sz=not a.em_no_freeze_sz,
            n_textures=a.n_textures,
            appearance_marginalize=a.appearance_marginalize,
            basis_dim=a.basis_dim,
        )
