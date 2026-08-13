"""InverseApp: 逆渲染主流程 (连续版) —— 数据 → EM → 条件期望 →
物理单位指标 (插值/外推分裂) → 可视化 → 自检 + CLI。"""

from __future__ import annotations

import argparse
import math
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
        k_tr = p_tr[:, 0].astype(mx.int32)

        def s_proxy(stats: mx.array) -> mx.array:
            """表观尺寸代理: √(area/π)·zc/FX (形状系数留给模型残差学)。"""
            return mx.sqrt(stats[:, 2] / math.pi) * (
                Codebook.CAM_Z - stats[:, 0]
            ) / Codebook.FX

        # 强几何观测 + 残差学习: ẑ/ŝ 直接拼特征会被白化稀释
        # (1/647 维, 实测 z R² 0.02); 模型改为学 z−ẑ 与 s−ŝ 的
        # 标定残差 (可见面≠中心偏差的形状相关部分), 推理后加回
        t_tr = mx.concatenate(
            [
                t_tr[:, :2],
                (t_tr[:, 2] - s_proxy(s_tr))[:, None],
                (t_tr[:, 3] - s_tr[:, 0])[:, None],
                t_tr[:, 4:],
            ],
            axis=1,
        )

        # 模型默认持久化: 路径随数据指纹 (与块缓存同标签, 数据配置
        # 变 → 旧模型自动失效)。加载后 K < 当前数据量 → 增量追加新块
        # (白化基冻结, 见 MixtureSPN.add); K ≥ 数据量 → 直接用 (模型
        # 训练集是当前请求的超集)
        model_path = cfg.model_path or artifacts / (
            f"spn_{self.data.cache_tag()}.safetensors"
        )
        if model_path.exists():
            net = MixtureSPN.load(model_path)
            k_have = net.f_mu.shape[0]
            if k_have < n_tr:
                print(f"[2/4] 增量训练: K {k_have} → {n_tr} (追加新块)")
                net.add(
                    f_tr[k_have:], t_tr[k_have:], k_tr[k_have:]
                )
                net.save(model_path)
            else:
                print(f"[2/4] 加载模型 {model_path.name} (K={k_have})")
        else:
            print(f"[2/4] MixtureSPN 实例级组装 (K=N={n_tr}, V={f_tr.shape[1]}) ...")
            net = MixtureSPN.fit(
                f_tr, t_tr, k_tr, rel_floor=cfg.sigma_rel_floor
            )
            net.save(model_path)
            print(f"      模型已保存 → {model_path.name}")

        print("[3/4] 推理: 条件期望 E[t|特征]")
        ti_pred, ki_p, _ = net.predict(f_ti)
        te_pred, ke_p, _ = net.predict(f_te)
        # 残差加回代理 → 物理量 (s,z)
        ti_pred = mx.concatenate(
            [
                ti_pred[:, :2],
                (ti_pred[:, 2] + s_proxy(s_ti))[:, None],
                (ti_pred[:, 3] + s_ti[:, 0])[:, None],
                ti_pred[:, 4:],
            ],
            axis=1,
        )
        te_pred = mx.concatenate(
            [
                te_pred[:, :2],
                (te_pred[:, 2] + s_proxy(s_te))[:, None],
                (te_pred[:, 3] + s_te[:, 0])[:, None],
                te_pred[:, 4:],
            ],
            axis=1,
        )
        ki_pred = mx.argmax(ki_p, axis=1)
        ke_pred = mx.argmax(ke_p, axis=1)

        print("[4/4] 评估 (物理单位; 基线 = 训练均值预测器; 色相评白光子集)")
        mi = Evaluator.report("插值", p_ti, ti_pred, ki_pred, p_tr)
        me = Evaluator.report("外推", p_te, te_pred, ke_pred, p_tr)

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
        """3 个插值样本: GT 渲染 vs 预测参数重建渲染 (闭环 sanity)。
        预测的 nuisance (光色/光向) 不可观测 → 重建用 GT 的 (场景参数
        的角色是内容量, 见 docs/architecture.md)。"""
        renderer, cam, _ = Codebook.make_renderer()  # 重建对比用左视图
        n = p_gt.shape[0]
        picks = [0, n // 2, n - 1]
        fig, axes = plt.subplots(len(picks), 2, figsize=(5, 2.6 * len(picks)))
        for row, i in enumerate(picks):
            gt = p_gt[i].tolist()
            # 预测: kind argmax + u,v,s,z + 色相 atan2 (nuisance 沿用 GT)
            import math

            hue_pred = (
                math.atan2(float(t_pred[i, 5]), float(t_pred[i, 4]))
                % (2 * math.pi)
            ) / (2 * math.pi / Codebook.N_HUE)
            pd = (
                [float(kind_pred[i])] + t_pred[i, :4].tolist()
                + [hue_pred] + gt[6:8]
            )
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
        # kind 颜色解耦后只剩形状线索 (色度泄漏捷径拆除, 这是任务升级
        # 的有意代价)。实测 (实例级模型, rp2 线性光照管线): 全量 0.55
        # —— 密度封顶 (同密度 1-NN 同值)。阈值 0.45 只防
        # 机制崩溃 (随机 0.33); 逐 kind 白化 (PPCA 似然比) 是升级候选
        assert mi["kind"] > 0.45, f"kind 准确率过低 {mi['kind']:.3f}"
        # 插值位置回归: 实测全量 5.9/5.1 (旧网格半档 9px
        # 以下 = 连续模型优于量化误差的及格线)
        assert mi["u_rmse"] < 9.0, f"插值 u RMSE {mi['u_rmse']:.2f}px"
        assert mi["v_rmse"] < 9.0, f"插值 v RMSE {mi['v_rmse']:.2f}px"
        # 色相 (白光子集, 6 档随机 0.167): 实测 bin 全量 0.67,
        # Δ 全量 33.2° (档位间距 60°, Δ<30° ≈ 命中档)
        assert mi["hue_bin"] > 0.5, f"白光色相 bin 准确率 {mi['hue_bin']:.3f}"
        # 视差把 z 几何钉死 → s=表观×zc 随解 (乘积歧义破解)。
        # 实测 (2026-08-13, rp2 管线): z R² 全量 0.85,
        # s R² 全量 0.41; 外推 z R² 0.96 (几何不饱和)
        assert mi["z_r2"] > 0.6, f"插值 z R² {mi['z_r2']:.3f}"
        assert mi["s_r2"] > 0.2, f"插值 s R² {mi['s_r2']:.3f}"
        print("inverse: 自检 ✓ (立体: z 几何钉死, s 随解)")

    # ── CLI ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_args() -> InverseConfig:
        """CLI → InverseConfig (一切开关的唯一家)。"""
        ap = argparse.ArgumentParser()
        ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
        ap.add_argument(
            "--model-path",
            default=None,
            help="模型存取路径 (safetensors, 默认 artifacts/spn_<数据指纹>); "
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
        a = ap.parse_args()
        return InverseConfig(
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            sigma_rel_floor=a.sigma_rel_floor,
            replicates=a.replicates,
        )
