"""InverseApp: 逆渲染 demo 主流程 (训练/推理/评估/可视化/自检) + CLI。"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx

from code_bayes import CodeBayes
from codebook import Codebook
from data_builder import DataBuilder
from evaluator import Evaluator
from feature_extractor import FeatureExtractor
from inverse_config import InverseConfig
from priors import Priors
from riesz import RieszWavelet
from sequence_runner import SequenceRunner
from spn import SPN
from spn_learner import SPNLearner


class InverseApp:
    """主流程: 数据 → 训练/加载 → 推理 → 评估 → 可视化 → 自检。"""

    def __init__(self, cfg: InverseConfig):
        self.cfg = cfg
        self.codebook = Codebook(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.data = DataBuilder(cfg, self.codebook, self.extractor)
        self.priors = Priors(cfg, self.codebook)
        self.sequences = SequenceRunner(cfg, self.codebook, self.extractor)

    def run(self) -> None:
        cfg = self.cfg
        cb = self.codebook
        n_train = 600 if cfg.quick else 4000
        n_test = 80 if cfg.quick else 200
        min_n = cfg.min_n
        if min_n is None:
            min_n = 8 if cfg.quick else 3  # 叶最小行数: 小 = 叶码纯 (后验锐)
        print(
            f"[1/5] 数据: train {n_train} / test {n_test} "
            f"(cache={'on' if cfg.use_cache else 'off'}, "
            f"model={cfg.model}, min_n={min_n})"
        )
        x_tr, c_tr, x_te, c_te = self.data.build(n_train, n_test, cfg.use_cache)
        tr_codes = [
            cb.code_to_idx(tuple(int(v) for v in row)) for row in c_tr.tolist()
        ]

        # 模型: 存在 → 加载; 否则训练并保存。nb 用原始特征 (无预处理),
        # spn 用 z-score (mu/sd 随模型保存, 加载时复用)
        net: SPN | CodeBayes
        mu: mx.array | None
        sd: mx.array | None
        if cfg.model_path is not None and cfg.model_path.exists():
            print(f"[2/5] 加载模型 {cfg.model_path}")
            if cfg.model == "nb":
                net, extra = CodeBayes.load(cfg.model_path)
            else:
                net, extra = SPN.load(cfg.model_path)
            mu, sd = extra.get("mu"), extra.get("sd")
            if mu is not None:
                x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
        elif cfg.model == "nb":
            assert mx.all(mx.isfinite(x_tr)), "特征含 NaN/inf"
            print("[2/5] CodeBayes 逐码充分统计 (全分辨率, 精确可增量) ...")
            net = CodeBayes.fit(
                x_tr,
                mx.array(tr_codes, dtype=mx.int32),
                cards=(cb.N_KIND, cb.N_GX, cb.N_GY, cb.N_SIZE, cb.N_Z),
            )
            mu = sd = None
            if cfg.model_path is not None:
                net.save(cfg.model_path)
                print(f"      模型已保存 → {cfg.model_path}")
        else:
            x_tr, x_te, mu, sd = self.data.standardize(x_tr, x_te)
            assert mx.all(mx.isfinite(x_tr)), "特征含 NaN/inf"
            print("[2/5] SPNLearner 结构学习 ...")
            xj = mx.concatenate([x_tr, c_tr], axis=1)
            net = SPNLearner(
                disc_cols=set(cfg.code_cols),
                card=cfg.card,
                min_n=min_n,
                max_depth=14,
                sigma_floor=cfg.sigma_floor,
            ).learn(xj)
            print(f"      根节点: {type(net.root).__name__}")
            if cfg.model_path is not None:
                net.save(cfg.model_path, {"mu": mu, "sd": sd})
                print(f"      模型已保存 → {cfg.model_path}")
        if mu is None:  # nb 无预处理: 恒等占位 (序列/光照评估复用)
            mu = mx.zeros((1, cfg.n_feat))
            sd = mx.ones((1, cfg.n_feat))

        print("[3/5] 推理: 枚举场景码后验")
        # 分块: 全批输入矩阵 × 多棵 eval 图同时构建会超 Metal 显存上限;
        # 逐块 mx.eval (立即求值, 图小) 再拼接, 结果同全批
        codes = cb.all_codes()
        parts = []
        for i in range(0, n_test, 8):
            p = net.posterior(x_te[i : i + 8], codes)
            mx.eval(p)  # 立即求值, 释放该块 eval 图
            parts.append(p)
        post = mx.concatenate(parts)  # (n_test, N_CODES) log 后验
        assert mx.all(mx.isfinite(post)), "后验含 NaN/inf"
        pred_i = mx.argmax(post, axis=1).tolist()
        gt_i = [cb.code_to_idx(tuple(int(v) for v in row)) for row in c_te.tolist()]

        print("[4/5] 评估 + 基线")
        acc = Evaluator.evaluate(pred_i, gt_i)
        base_maj = Evaluator.baseline_majority(tr_codes, gt_i)
        base_tpl = Evaluator.baseline_template(x_tr, c_tr, x_te, gt_i)
        base = {"majority": base_maj, "template": base_tpl}
        print(
            f"      码: {acc['code']:.3f}  kind: {acc['kind']:.3f}  "
            f"gx: {acc['gx']:.3f}  gy: {acc['gy']:.3f}  "
            f"size: {acc['size']:.3f}  z: {acc['z']:.3f}"
        )
        print(f"      基线: majority {base_maj:.3f} / template {base_tpl:.3f}")

        prior = self.priors.build(cfg.prior_name)
        if cfg.occlusion and cfg.prior_name == "occlusion":
            # 遮挡序数先验是 per-sample 的: 重渲染测试帧检测黄柱面积缺失
            renderer, cam = Codebook.make_renderer()
            frames = []
            for row in c_te.tolist():
                code = cb.idx_to_code(cb.code_to_idx(tuple(int(v) for v in row)))
                scene = cb.to_scene(code)
                frames.append(renderer.render(scene, cam))
            prior = self.priors.occlusion(frames)  # (n_test, N_CODES)
        if prior is not None:
            # occlusion 是 (M,K) 逐样本; 其余是 (K,) 广播
            post_p = net.posterior(x_te, cb.all_codes(), log_prior=prior)
            pred_p = mx.argmax(post_p, axis=1).tolist()
            acc_p = Evaluator.evaluate(pred_p, gt_i)
            print(
                f"      注入先验[{cfg.prior_name}]: 码 {acc_p['code']:.3f}  "
                f"kind {acc_p['kind']:.3f}  gx {acc_p['gx']:.3f}  "
                f"gy {acc_p['gy']:.3f}  size {acc_p['size']:.3f}  z {acc_p['z']:.3f}"
            )

        print("[5/5] 图 → artifacts/")
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        artifacts.mkdir(exist_ok=True)
        self.plot_panel(x_te, post, gt_i, pred_i, artifacts / "inverse_panel.png")
        self.plot_metrics(acc, base, artifacts / "inverse_metrics.png")

        if cfg.sequence > 0:
            print("\n[6/5] 多帧运动先验 (prior.md 运动与时间先验)")
            self.sequences.run(
                net, mu, sd, n_seqs=10, n_frames=cfg.sequence, seq_seed=0
            )
            return
        if cfg.test_light:
            print("\n[6/5] 光照鲁棒性评估 (训练右上光, 测试池外顶光)")
            self.run_test_light(net, mu, sd, n_test)
            return

        if cfg.tree:
            if cfg.model != "spn":
                print("--tree 仅 spn 模型 (nb 无结构可视化)")
            else:
                self.print_tree(net, artifacts)

        self.self_check(acc)

    # ── 可视化 ──────────────────────────────────────────────────────

    def plot_panel(
        self,
        x_te: mx.array,
        post: mx.array,
        gt_i: list[int],
        pred_i: list[int],
        out: Path,
    ) -> None:
        """3 个测试样本: GT/Pred 渲染 + 特征图 + P(gx,gy) 热图。"""
        cfg = self.cfg
        cb = self.codebook
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        n_show = min(3, len(gt_i))
        picks = (
            [0, len(gt_i) // 2, len(gt_i) - 1]
            if len(gt_i) >= 3
            else list(range(n_show))
        )
        fig, axes = plt.subplots(n_show, 5, figsize=(17, 3.4 * n_show))
        if n_show == 1:
            axes = axes[None, :]
        ch = cfg.n_feat // len(cfg.feat_spec)  # 每通道尺寸
        fshape = (
            (cb.N_GY, cb.N_GX) if ch == cb.N_GX * cb.N_GY else (cb.H, cb.W)
        )
        unit = "blocks" if ch == cb.N_GX * cb.N_GY else "map"
        cols = [
            "GT render",
            f"GT {cfg.feat_spec[0][1]} {unit}",
            "Pred render",
            f"Pred {cfg.feat_spec[0][1]} {unit}",
            "P(gx,gy|img)",
        ]
        for row, i in enumerate(picks):
            gt_scene = cb.to_scene(cb.idx_to_code(gt_i[i]))
            pd_scene = cb.to_scene(cb.idx_to_code(pred_i[i]))
            f_gt = renderer.render(gt_scene, cam)
            f_pd = renderer.render(pd_scene, cam)
            axes[row, 0].imshow(f_gt[..., :3].astype(mx.int32))
            axes[row, 2].imshow(f_pd[..., :3].astype(mx.int32))
            lg = x_te[i, :ch].reshape(fshape)
            axes[row, 1].imshow(lg, cmap="viridis")
            # Pred 特征图: 从重建渲染重算 (与 GT 同管线, 首通道)
            vec_pd, rw = self.extractor.of_frame(f_pd, rw)
            mx.eval(vec_pd)
            lg_p = vec_pd[:ch].reshape(fshape)
            axes[row, 3].imshow(lg_p, cmap="viridis")
            pg = post[i].reshape(cb.N_KIND, cb.N_GX, cb.N_GY, cb.N_SIZE, cb.N_Z)
            pgy = mx.exp(mx.logsumexp(pg, axis=(0, 3, 4)) - mx.logsumexp(pg))
            axes[row, 4].imshow(pgy.T, cmap="hot", origin="lower")
            for c in range(5):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            ok = "✓" if pred_i[i] == gt_i[i] else "✗"
            axes[row, 0].set_title(f"GT  code {cb.idx_to_code(gt_i[i])}")
            axes[row, 2].set_title(f"Pred code {cb.idx_to_code(pred_i[i])} {ok}")
        for c, name in enumerate(cols):
            if n_show == 1:
                axes[0, c].set_xlabel(name, fontsize=9)
            else:
                axes[0, c].set_title(name, fontsize=9)
        fig.suptitle(
            "inverse rendering: GT (cga 3D model) vs single-image reconstruction",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    @staticmethod
    def plot_metrics(acc: dict[str, float], base: dict[str, float], out: Path) -> None:
        names = ["code", "kind", "gx", "gy", "size", "z"]
        vals = [acc[n] for n in names]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        bars = ax.bar(range(len(names)), vals, color="#4C72B0")
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9,
            )
        for j, (name, v) in enumerate(base.items(), start=len(names)):
            ax.bar(j, v, color="#DD8452")
            ax.text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names) + len(base)))
        ax.set_xticklabels(names + list(base.keys()))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.axhline(1 / Codebook.N_CODES, color="gray", ls=":", lw=1)
        ax.text(
            len(names) + len(base) - 0.6, 1 / Codebook.N_CODES + 0.01,
            "chance", fontsize=8, color="gray",
        )
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

    def print_tree(self, net: SPN | CodeBayes, artifacts: Path) -> None:
        """SPN 树结构文本可视化 (带语义列名) + 功能分工统计。"""
        assert isinstance(net, SPN)
        cb = self.codebook
        labels = dict(enumerate(self.extractor.labels()))
        labels.update(
            dict(zip(self.cfg.code_cols, ("kind", "gx", "gy", "size", "z")))
        )
        code_names = {
            self.cfg.code_cols[0]: dict(enumerate(cb.KINDS)),
            self.cfg.code_cols[1]: {i: f"gx={i}" for i in range(cb.N_GX)},
            self.cfg.code_cols[2]: {i: f"gy={i}" for i in range(cb.N_GY)},
            self.cfg.code_cols[3]: {i: f"s={cb.SIZES[i]}" for i in range(cb.N_SIZE)},
            self.cfg.code_cols[4]: {i: f"z={cb.Z0S[i]}" for i in range(cb.N_Z)},
        }
        txt = net.tree_str(labels, code_names)
        print(txt)
        (artifacts / "spn_tree.txt").write_text(txt)
        # 功能分工: 统计各分裂轴 (哪个码维度被哪些 Sum 节点负责)
        import re
        from collections import Counter

        axes = Counter(re.findall(r"分裂轴 (\w+):", txt))
        axes.pop("码分布相近", None)
        func_names = {
            "kind": "形状辨识 (sphere/cylinder/box)",
            "z": "深度估计 (近大远小, 单目线索)",
            "gx": "横向定位",
            "gy": "纵向定位",
            "size": "尺寸估计",
        }
        print("\n── 功能分工 (Sum 节点数 × 职责) ──")
        for ax, cnt in axes.most_common():
            print(f"  {ax:<5} ×{cnt:>3}  → {func_names.get(ax, ax)}")
        print("树结构 → artifacts/spn_tree.txt")

    def run_test_light(
        self, net: SPN | CodeBayes, mu: mx.array, sd: mx.array, n_test: int
    ) -> None:
        """光照变化评估: 池外顶光重渲染测试码 → 特征 → 后验。
        对比同一模型在正常光照下的准确率, 检验鲁棒性。"""
        cfg = dataclasses.replace(self.cfg, test_light=True)
        cb2 = Codebook(cfg)
        extractor2 = FeatureExtractor(cfg)
        cb = self.codebook
        te = mx.random.randint(
            0, cb.N_CODES, shape=(n_test,), key=mx.random.key(99)
        ).tolist()
        renderer, cam = Codebook.make_renderer()
        rw: RieszWavelet | None = None
        feats = []
        for i in te:
            scene = cb2.to_scene(cb.idx_to_code(i))
            vec, rw = extractor2.of_frame(renderer.render(scene, cam), rw)
            mx.eval(vec)
            feats.append(vec)
        x_te = (mx.stack(feats) - mu) / sd
        codes = cb.all_codes()
        parts = []
        for i in range(0, n_test, 8):
            p = net.posterior(x_te[i : i + 8], codes)
            mx.eval(p)
            parts.append(p)
        post = mx.concatenate(parts)
        pred = mx.argmax(post, axis=1).tolist()
        gt = [cb.code_to_idx(cb.idx_to_code(i)) for i in te]
        acc = Evaluator.evaluate(pred, gt)
        print(
            f"  光照变化测试: 码 {acc['code']:.3f}  kind {acc['kind']:.3f}  "
            f"gx {acc['gx']:.3f}  gy {acc['gy']:.3f}  size {acc['size']:.3f}  "
            f"z {acc['z']:.3f}"
        )
        if self.cfg.multi_light:
            # 多光照增广应显著优于单光照的池外泛化 (单光照实测 0.080)
            assert acc["code"] > 0.15, f"多光照池外泛化不足 {acc['code']:.3f}"
        print("inverse: 光照鲁棒性评估 ✓")

    # ── 自检断言 (阈值按 2026-08-11/12 实测标定, 留安全余量) ─────────

    def self_check(self, acc: dict[str, float]) -> None:
        cfg = self.cfg
        if cfg.model == "nb":
            if not cfg.equal_luma and not cfg.multi_light:
                # nb 标定 (2026-08-12): 全量 ≈0.96 (模板上限, fullres 实测);
                # quick N=600 实测 0.287 (码覆盖率上限 1−e^{−0.52}≈0.41 打头)
                if cfg.quick:
                    assert acc["code"] > 0.25, (
                        f"quick nb: 码准确率过低 {acc['code']:.3f}"
                    )
                else:
                    assert acc["code"] > 0.90, f"nb: 码准确率过低 {acc['code']:.3f}"
                    assert acc["kind"] > 0.93, f"nb: kind 过低 {acc['kind']:.3f}"
                print("inverse: nb 自检 ✓")
            else:
                print("inverse: nb 消融模式 (断言按 spn 标定, 跳过)")
            return
        if cfg.equal_luma:
            # 等亮度: L 通路失效 (噪声淹没), 复数色相通路补位 (对照实验)
            assert acc["code"] > 0.30, (
                f"等亮度下色度通路应补位, 实测 {acc['code']:.3f}"
            )
            print("inverse: 等亮度消融自检 ✓ (色度补位)")
            return
        if cfg.multi_light:
            # 多光照模式实测: 正常 0.360 (5 光照分摊样本) / 池外 0.265
            assert acc["code"] > 0.30, f"多光照: 码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.70, f"多光照: kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.85, f"多光照: gx 过低 {acc['gx']:.3f}"
            assert acc["z"] > 0.50, f"多光照: z 过低 {acc['z']:.3f}"
        elif cfg.quick:
            # quick N=600/min_n=8 实测: code 0.025 kind 0.40 gx 0.55
            # gy 0.64 size 0.55 z 0.35
            assert acc["code"] > 0.02, f"quick: 码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.30, f"quick: kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.35, f"quick: gx 过低 {acc['gx']:.3f}"
            assert acc["gy"] > 0.50, f"quick: gy 过低 {acc['gy']:.3f}"
            assert acc["size"] > 0.45, f"quick: size 过低 {acc['size']:.3f}"
            assert acc["z"] > 0.30, f"quick: z 过低 {acc['z']:.3f}"
        else:
            # 全量 N=4000/min_n=3 实测 (码空间 1152): code 0.470 kind 0.835
            # gx 0.895 gy 0.855 size 0.885 z 0.735; template 0.965
            assert acc["code"] > 0.40, f"码准确率过低 {acc['code']:.3f}"
            assert acc["kind"] > 0.78, f"kind 过低 {acc['kind']:.3f}"
            assert acc["gx"] > 0.85, f"gx 过低 {acc['gx']:.3f}"
            assert acc["gy"] > 0.80, f"gy 过低 {acc['gy']:.3f}"
            assert acc["size"] > 0.83, f"size 过低 {acc['size']:.3f}"
            assert acc["z"] > 0.68, f"z 过低 {acc['z']:.3f}"
        print("inverse: 自检 ✓")


    @staticmethod
    def parse_args() -> InverseConfig:
        """CLI → InverseConfig (一切开关的唯一家)。"""
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--model",
            default="nb",
            choices=("nb", "spn"),
            help="模型: nb=全分辨率逐码贝叶斯 (默认, 精确可增量, 码簿任务最优); "
            "spn=池化+结构学习 (组合泛化/消融研究对照)",
        )
        ap.add_argument("--quick", action="store_true", help="小数据集自检模式")
        ap.add_argument("--no-cache", action="store_true", help="跳过数据缓存读写")
        ap.add_argument(
            "--model-path",
            default=None,
            help="模型存取路径 (safetensors); 存在则加载跳过学习, 否则训练后保存",
        )
        ap.add_argument(
            "--tree",
            action="store_true",
            help="打印 SPN 树结构 (带语义列名) 并存 artifacts/spn_tree.txt",
        )
        ap.add_argument(
            "--min-n",
            type=int,
            default=None,
            help="叶最小行数 (spn 结构复杂度先验); 缺省 quick=8 / 全量=3",
        )
        ap.add_argument(
            "--equal-luma",
            action="store_true",
            help="等亮度模式: 三色与背景同为亮度 0.10 且无明暗 → L 通路失效, "
            "展示 HS 补位 (断言: l 应失效, lhs 应补位)",
        )
        ap.add_argument(
            "--sigma-floor",
            type=float,
            default=1e-6,
            help="高斯叶 σ 下限 (spn, 平滑性先验 prior.md)",
        )
        ap.add_argument(
            "--occlusion",
            action="store_true",
            help="遮挡场景: 固定黄色竖柱 + 序数先验 (--prior occlusion 注入,"
            "黄柱被遮 ⟹ 主图元在前)",
        )
        ap.add_argument(
            "--sequence",
            type=int,
            default=0,
            help="多帧运动先验: 每序列帧数 (>0 启用), gx/gy 随机游走,"
            "时序平滑注入上一帧转移先验",
        )
        ap.add_argument(
            "--test-light",
            action="store_true",
            help="光照鲁棒性评估: 测试集换光照方向 (需 --model-path), 检验 "
            "Riesz gain_control 归一化 vs 原始 RGB",
        )
        ap.add_argument(
            "--multi-light",
            action="store_true",
            help="多光照训练: 5 方向池轮流渲染 (数据增广 → 光照不变); "
            "配合 --test-light 用池外顶光验证泛化",
        )
        ap.add_argument(
            "--prior",
            default="flat",
            help="推理时注入的码先验 (贝叶斯 P(S)), 逗号组合如 'edge,familiar': "
            "flat=均匀, edge=一般视角(不贴边), familiar=熟悉尺寸(size 偏态), "
            "occlusion=遮挡序数 (需 --occlusion)",
        )
        a = ap.parse_args()
        return InverseConfig(
            model=a.model,
            quick=a.quick,
            use_cache=not a.no_cache,
            model_path=Path(a.model_path) if a.model_path else None,
            tree=a.tree,
            prior_name=a.prior,
            min_n=a.min_n,
            sigma_floor=a.sigma_floor,
            equal_luma=a.equal_luma,
            occlusion=a.occlusion,
            sequence=a.sequence,
            test_light=a.test_light,
            multi_light=a.multi_light,
        )
