"""InverseApp: 逆渲染主流程 (连续版) —— 数据 → EM → 条件期望 →
物理单位指标 (插值/外推分裂) → 可视化 → 自检 + CLI。"""

from __future__ import annotations

import argparse
import dataclasses
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


class InverseApp:
    """主流程: 连续采样数据 → MixtureSPN(EM) → 条件期望 → 插值/外推评估。"""

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg
        self.codebook = Codebook(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.data = DataBuilder(cfg, self.codebook, self.extractor)

    def run(self) -> None:
        cfg = self.cfg
        n_tr, n_i, n_e = (800, 100, 100) if cfg.quick else (4000, 300, 300)
        print(
            f"[1/4] 数据: train {n_tr} / 插值 {n_i} / 外推 {n_e} "
            f"(cache={'on' if cfg.use_cache else 'off'}, K={cfg.k_components})"
        )
        f_tr, p_tr, f_ti, p_ti, f_te, p_te = self.data.build(
            n_tr, n_i, n_e, cfg.use_cache
        )
        assert mx.all(mx.isfinite(f_tr)), "特征含 NaN/inf"
        t_tr = p_tr[:, 1:]
        k_tr = p_tr[:, 0].astype(mx.int32)

        if cfg.model_path is not None and cfg.model_path.exists():
            print(f"[2/4] 加载模型 {cfg.model_path}")
            net = MixtureSPN.load(cfg.model_path)
        else:
            print(
                f"[2/4] MixtureSPN 联合 EM (K={cfg.k_components}, "
                f"≤{cfg.em_iters} 轮, V={f_tr.shape[1]}) ..."
            )
            net = MixtureSPN.fit(
                f_tr, t_tr, k_tr,
                k=cfg.k_components,
                iters=cfg.em_iters,
                rel_floor=cfg.sigma_rel_floor,
                key=mx.random.key(0),
            )
            if cfg.model_path is not None:
                net.save(cfg.model_path)
                print(f"      模型已保存 → {cfg.model_path}")

        print("[3/4] 推理: 条件期望 E[t|特征]")
        ti_pred, ki_p, _ = net.predict(f_ti)
        te_pred, ke_p, _ = net.predict(f_te)
        ki_pred = mx.argmax(ki_p, axis=1)
        ke_pred = mx.argmax(ke_p, axis=1)

        print("[4/4] 评估 (物理单位; 基线 = 训练均值预测器)")
        mi = Evaluator.report("插值", p_ti, ti_pred, ki_pred, p_tr)
        me = Evaluator.report("外推", p_te, te_pred, ke_pred, p_tr)

        if cfg.test_light:
            print("  池外光照 (顶光) 重渲染评估:")
            cfg2 = dataclasses.replace(cfg, test_light=True)
            db = DataBuilder(cfg2, Codebook(cfg2), FeatureExtractor(cfg2))
            # 同一组插值参数, 仅换光照重渲染 (cache tag 含 tl 指纹)
            f_tl = db.feats_of(p_ti)
            tt_pred, kt_p, _ = net.predict(f_tl)
            Evaluator.report(
                "池外光", p_ti, tt_pred, mx.argmax(kt_p, axis=1), p_tr
            )

        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        artifacts.mkdir(exist_ok=True)
        self.plot_scatter(p_tr, p_ti, ti_pred, p_te, te_pred,
                          artifacts / "inverse_scatter.png")
        self.plot_recon(p_ti, ti_pred, ki_pred, artifacts / "inverse_recon.png")
        print(f"      图 → {artifacts.name}/ (scatter + recon)")
        self.self_check(mi, me)

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
        self, p_gt: mx.array, t_pred: mx.array, kind_pred: mx.array, out: Path
    ) -> None:
        """3 个插值样本: GT 渲染 vs 预测参数重建渲染 (闭环 sanity)。"""
        renderer, cam = Codebook.make_renderer()
        n = p_gt.shape[0]
        picks = [0, n // 2, n - 1]
        fig, axes = plt.subplots(len(picks), 2, figsize=(5, 2.6 * len(picks)))
        for row, i in enumerate(picks):
            gt = p_gt[i].tolist()
            pd = [float(kind_pred[i])] + t_pred[i].tolist()
            for col, prm in enumerate((gt, pd)):
                img = renderer.render(self.codebook.to_scene(prm), cam)
                axes[row, col].imshow(img[..., :3].astype(mx.int32))
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            axes[row, 0].set_ylabel(
                f"u{gt[1]:.0f} v{gt[2]:.0f} s{gt[3]:.2f} z{gt[4]:.2f}", fontsize=8
            )
        axes[0, 0].set_title("GT render")
        axes[0, 1].set_title("Pred recon")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    # ── 自检断言 (阈值依据见各注释; 2026-08-12 全量运行标定) ────────

    def self_check(self, mi: dict[str, float], me: dict[str, float]) -> None:
        cfg = self.cfg
        if cfg.equal_luma:
            # 等亮度: lum 通道变纯噪声, 白化把噪声维放大到单位方差 →
            # 信号被稀释。实测 quick 上限: 原始色度 1-NN 0.91 / 白化全维
            # 0.81 / 模型 0.73 (平铺稀疏再损)。阈值 0.65 只防机制崩溃
            assert mi["kind"] > 0.65, f"等亮度 kind 过低 {mi['kind']:.3f}"
            print("inverse: 等亮度消融自检 ✓ (kind 色度补位, 回归报告制)")
            return
        # 色度绑定 kind → 强线索 (白化 1-NN 上限 0.94); 混合平铺有容量
        # 损失。阈值 0.75: 2026-08-12 实测 quick 0.87, 留余量;
        # 低于此 = 机制破坏 (历史病理值 0.47, 见 mixture_spn 白化注释)
        assert mi["kind"] > 0.75, f"kind 准确率过低 {mi['kind']:.3f}"
        # 插值位置回归须优于旧离散网格的半档宽 9px (旧网格曾以高准确率
        # 识别位置; 连续模型连量化误差都打不过则机制失效)。
        # 实测: quick 8.0/6.7, 全量 6.5/6.3
        assert mi["u_rmse"] < 9.0, f"插值 u RMSE {mi['u_rmse']:.2f}px"
        assert mi["v_rmse"] < 9.0, f"插值 v RMSE {mi['v_rmse']:.2f}px"
        # s/z: 单目单帧仅乘积可观测 (熟悉尺寸歧义), R²>0 的部分来自
        # 边界线索 (大 s 压缩位置边距)。阈值只防机制崩溃:
        # 实测 s R² quick 0.17/全量 0.19, z R² quick 0.29/全量 0.41
        assert mi["s_r2"] > 0.1, f"插值 s R² {mi['s_r2']:.3f}"
        assert mi["z_r2"] > 0.2, f"插值 z R² {mi['z_r2']:.3f}"
        # 外推报告制: 核回归边界饱和不完美 (预测漂移, s/z R² 可为负,
        # 2026-08-12 实测) —— 核机器边界行为的已知上限; 升级路径
        # mixture of linear experts, 见架构文档
        print("inverse: 自检 ✓ (外推为报告制, 见 self_check 注释)")

    # ── CLI ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_args() -> InverseConfig:
        """CLI → InverseConfig (一切开关的唯一家)。"""
        ap = argparse.ArgumentParser()
        ap.add_argument("--quick", action="store_true", help="小数据集自检模式")
        ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
        ap.add_argument(
            "--model-path",
            default=None,
            help="模型存取路径 (safetensors); 存在则加载跳过 EM, 否则训练后保存",
        )
        ap.add_argument(
            "--components", type=int, default=64, help="混合分量数 K (默认 64)"
        )
        ap.add_argument("--em-iters", type=int, default=20, help="EM 最大轮数")
        ap.add_argument(
            "--sigma-rel-floor",
            type=float,
            default=1e-2,
            help="σ 带宽下限 (各维全局 std 的相对比例): 核回归带宽, "
            "插值平滑度旋钮",
        )
        ap.add_argument(
            "--equal-luma",
            action="store_true",
            help="等亮度消融: 三色与背景同亮度且无明暗 → L 通路失效, 色度补位",
        )
        ap.add_argument(
            "--occlusion", action="store_true", help="遮挡场景: 固定黄色竖柱"
        )
        ap.add_argument(
            "--multi-light",
            action="store_true",
            help="多光照训练: 5 方向池轮流渲染 (数据增广 → 光照不变)",
        )
        ap.add_argument(
            "--test-light",
            action="store_true",
            help="追加池外顶光评估 (同一组插值参数换光照重渲染)",
        )
        a = ap.parse_args()
        return InverseConfig(
            quick=a.quick,
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            k_components=a.components,
            em_iters=a.em_iters,
            sigma_rel_floor=a.sigma_rel_floor,
            equal_luma=a.equal_luma,
            occlusion=a.occlusion,
            multi_light=a.multi_light,
            test_light=a.test_light,
        )
