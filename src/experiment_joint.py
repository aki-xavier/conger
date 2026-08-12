"""开放集联合系统实验: CodeBayes 快轨 + SPN 慢轨 + 新颖度门控 + 提升。

架构 (两模型联合工作方式):
  帧 → 门控 (CodeBayes.gate, 等先验似然比)
    ├─ 已知 → 快轨: posterior_all argmax 回答 + 自标注吸收
    │        (流式无真值, 预测即标签 —— 门控近完美时无污染)
    └─ 未见 → 慢轨: SPN (池化, 组合结构) 变量级回答 (kind/gx/gy)
             + 立即提升: grow 临时分量 + 吸收 (确定性渲染首帧即足;
             噪声世界改复发计数, 机制相同)
  提升后该码族帧自动转快轨 (门控把临时分量当已知)。

探针:
  A 未见码族 (码簿内未训组合, 10% 码): 应被门控捕获 → 提升 →
    后续帧码级准确率追上 nb 水平 (预言机解析临时分量身份, 仅评估用);
  B 新类别 (圆盘, 码簿外): 门控应全体判新, SPN 给位置 (gx/gy)
    —— kind 无真值可选, 这是慢轨不可替代的角色。

运行: python experiment_joint.py
"""

import time
from pathlib import Path

import mlx.core as mx
from cga.engine import (
    AmbientLight,
    CircleGeometry,
    Color,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    Scene,
)

from code_bayes import CodeBayes
from codebook import Codebook
from demo_config import DemoConfig
from feature_extractor import FeatureExtractor
from riesz import RieszWavelet
from spn_learner import SPNLearner


class JointExperiment:
    """门控双轨联合系统 (阶段驱动: phase1 监督批 → phase2 混合流 →
    phase3 新测 + 探针 B)。"""

    N_TRAIN, N_STREAM, N_EVAL, N_PROBE = 2000, 600, 200, 100
    NOVEL_RATE = 0.3  # 流中未见码帧比例
    DISC_COLOR = 0x8E44AD  # 探针 B: 紫色圆盘
    SPLIT_SEED = 123  # 码族划分种子 (与 experiment_fullres 一致)

    def __init__(self):
        self.cfg = DemoConfig()
        self.codebook = Codebook(self.cfg)
        cb = Codebook
        # 码族划分与 experiment_fullres 同种子 (90% 训 / 10% 保留)
        perm = mx.random.permutation(
            cb.N_CODES, key=mx.random.key(self.SPLIT_SEED)
        ).tolist()
        self.unseen = set(perm[: cb.N_CODES // 10])
        self.train_codes = [i for i in range(cb.N_CODES) if i not in self.unseen]
        self.unseen_codes = perm[: cb.N_CODES // 10]

    @staticmethod
    def sample_codes(pool: list[int], n: int, key: int) -> list[int]:
        idx = mx.random.randint(0, len(pool), shape=(n,), key=mx.random.key(key))
        return [pool[int(i)] for i in idx.tolist()]

    def feats_of(
        self, scenes: list[Scene], renderer, cam, rw: RieszWavelet
    ) -> tuple[mx.array, mx.array]:
        """场景序列 → (全分辨率 (n,62208), 池化 (n,144))。"""
        fe = FeatureExtractor
        full, pooled = [], []
        for scene in scenes:
            frame = renderer.render(scene, cam)
            rw.update(fe.frame_lum(frame))
            f = rw.features()
            v = mx.concatenate(
                [f.log_mag.reshape(-1), f.phase_coh.reshape(-1), f.ori_R.reshape(-1)]
            )
            p = mx.concatenate(
                [
                    fe.block_pool(f.log_mag).reshape(-1),
                    fe.block_pool(f.phase_coh).reshape(-1),
                    fe.block_pool(f.ori_R).reshape(-1),
                ]
            )
            mx.eval(v, p)  # 逐帧求值, 防惰性图累积
            full.append(v)
            pooled.append(p)
        return mx.stack(full), mx.stack(pooled)

    def disc_scene(self, gx: int, gy: int) -> Scene:
        """探针 B: 紫色圆盘 (码簿外新类别), 位置投影同 codebook。"""
        cb = self.codebook
        x, y = cb.project(gx, gy, 3.0)
        scene = Scene(background=Color(self.cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(
            DirectionalLight(Color(0xFFFFFF), 0.7, direction=cb.LIGHT_DIRS[0])
        )
        scene.add(
            Mesh(
                CircleGeometry(0.5),
                MeshStandardMaterial(Color(self.DISC_COLOR), roughness=0.55),
                position=(x, y, 3.0),
            )
        )
        return scene

    def build(self) -> dict[str, mx.array]:
        cache = Path(__file__).resolve().parent.parent / "artifacts"
        cache.mkdir(exist_ok=True)
        path = cache / (
            f"joint_{self.N_TRAIN}_{self.N_STREAM}_{self.N_EVAL}_"
            f"{self.N_PROBE}.safetensors"
        )
        if path.exists():
            return mx.load(str(path))
        cb = self.codebook
        renderer, cam = Codebook.make_renderer()
        rw = RieszWavelet(mx.zeros((cb.H, cb.W)))
        t0 = time.monotonic()
        tr = self.sample_codes(self.train_codes, self.N_TRAIN, 42)
        # 流: 70% 已知码 / 30% 未见码, 交错
        u = mx.random.uniform(shape=(self.N_STREAM,), key=mx.random.key(55)).tolist()
        st = [
            self.sample_codes(self.unseen_codes, 1, 55000 + i)[0]
            if r < self.NOVEL_RATE
            else self.sample_codes(self.train_codes, 1, 77000 + i)[0]
            for i, r in enumerate(u)
        ]
        es = self.sample_codes(self.train_codes, self.N_EVAL, 99)
        eu = self.sample_codes(self.unseen_codes, self.N_EVAL, 77)
        cells = [
            (int(row[0]) % cb.N_GX, int(row[1]) % cb.N_GY)
            for row in mx.random.randint(
                0, 8, shape=(self.N_PROBE, 2), key=mx.random.key(66)
            ).tolist()
        ]
        d: dict[str, mx.array] = {}
        d["xf1"], d["xp1"] = self.feats_of(
            [cb.to_scene(cb.idx_to_code(i)) for i in tr], renderer, cam, rw
        )
        d["c1"] = mx.array(tr, dtype=mx.float32)
        d["xfs"], d["xps"] = self.feats_of(
            [cb.to_scene(cb.idx_to_code(i)) for i in st], renderer, cam, rw
        )
        d["cs"] = mx.array(st, dtype=mx.float32)
        d["xes"], d["xpes"] = self.feats_of(
            [cb.to_scene(cb.idx_to_code(i)) for i in es], renderer, cam, rw
        )
        d["ces"] = mx.array(es, dtype=mx.float32)
        d["xeu"], d["xpeu"] = self.feats_of(
            [cb.to_scene(cb.idx_to_code(i)) for i in eu], renderer, cam, rw
        )
        d["ceu"] = mx.array(eu, dtype=mx.float32)
        d["xpb"], d["xppb"] = self.feats_of(
            [self.disc_scene(gx, gy) for gx, gy in cells], renderer, cam, rw
        )
        d["cpb"] = mx.array(cells, dtype=mx.float32)
        print(f"渲染+特征 {time.monotonic()-t0:.0f}s → 缓存 {path.name}")
        mx.save_safetensors(str(path), d)
        return mx.load(str(path))

    def run(self) -> None:
        cb = self.codebook
        d = self.build()
        codes = cb.all_codes()
        c1 = [int(v) for v in d["c1"].tolist()]
        cs = [int(v) for v in d["cs"].tolist()]
        ces = [int(v) for v in d["ces"].tolist()]
        ceu = [int(v) for v in d["ceu"].tolist()]

        # ── phase 1: 监督批 (快轨 fit + 慢轨结构学习) ───────────────
        t0 = time.monotonic()
        model = CodeBayes.fit(d["xf1"], mx.array(c1, dtype=mx.int32), cards=cb.CARDS)
        mu = d["xp1"].mean(axis=0, keepdims=True)
        sd = mx.maximum(d["xp1"].std(axis=0, keepdims=True), 1e-6)
        xz = (d["xp1"] - mu) / sd
        code_arr = mx.array([list(cb.idx_to_code(c)) for c in c1], dtype=mx.float32)
        xj = mx.concatenate([xz, code_arr], axis=1)
        n_feat = d["xp1"].shape[1]
        card = dict(zip(range(n_feat, n_feat + 5), cb.CARDS))
        spn = SPNLearner(disc_cols=set(card), card=card, min_n=3, max_depth=14).learn(
            xj
        )
        print(f"phase1: 快轨 fit + 慢轨 learn_spn ({time.monotonic()-t0:.0f}s)")

        def spn_pred(xp_row: mx.array) -> tuple[int, ...]:
            """慢轨变量级回答: 池化特征 → SPN 后验 argmax 码元组。"""
            p = spn.posterior((xp_row - mu) / sd, codes)
            mx.eval(p)
            return cb.idx_to_code(int(mx.argmax(p[0])))

        # ── phase 2: 混合流 (门控 → 快/慢轨 → 提升) ──────────────────
        prov_true: dict[int, list[int]] = {}  # 临时分量 → 帧真值码 (评估用)
        fast_hit = fast_n = 0
        slow_var = {"kind": 0, "gx": 0, "gy": 0}
        slow_n = 0
        t0 = time.monotonic()
        for i in range(self.N_STREAM):
            true = cs[i]
            xf = d["xfs"][i : i + 1]
            _, novel = model.gate(xf)
            if bool(novel[0]):
                pt = spn_pred(d["xps"][i : i + 1])  # 慢轨语义
                gt = cb.idx_to_code(true)
                slow_var["kind"] += int(pt[0] == gt[0])
                slow_var["gx"] += int(pt[1] == gt[1])
                slow_var["gy"] += int(pt[2] == gt[2])
                slow_n += 1
                idx = model.grow()  # 提升: 新内容进码簿
                model.absorb(xf, mx.array([idx], dtype=mx.int32))
                prov_true.setdefault(idx, []).append(true)
            else:
                post = model.posterior_all(xf)
                mx.eval(post)
                idx = int(mx.argmax(post[0]))
                model.absorb(xf, mx.array([idx], dtype=mx.int32))  # 自标注吸收
                if idx >= cb.N_CODES:
                    prov_true[idx].append(true)
                else:
                    fast_hit += idx == true
                    fast_n += 1
        print(
            f"phase2: 流 {self.N_STREAM} 帧 ({time.monotonic()-t0:.0f}s) | "
            f"快轨 {fast_n} 帧 acc {fast_hit/max(fast_n,1):.3f} | "
            f"慢轨 {slow_n} 帧 kind {slow_var['kind']/max(slow_n,1):.3f} "
            f"gx {slow_var['gx']/max(slow_n,1):.3f} "
            f"gy {slow_var['gy']/max(slow_n,1):.3f} | "
            f"提升 {len(prov_true)} 分量"
        )

        def resolve(idx: int) -> int:
            """分量下标 → 码下标: 码簿内恒等; 临时分量真值多数票 (仅评估)。"""
            if idx < cb.N_CODES:
                return idx
            votes = prov_true.get(idx) or [0]
            return max(set(votes), key=votes.count)

        # ── phase 3: 新测 (门控精度 / seen 码 acc / 提升覆盖) ─────────
        es_novel = es_hit = 0
        for i in range(self.N_EVAL):
            _, novel = model.gate(d["xes"][i : i + 1])
            es_novel += bool(novel[0])
            idx = int(mx.argmax(model.posterior_all(d["xes"][i : i + 1])[0]))
            es_hit += resolve(idx) == ces[i]

        eu_novel = eu_hit = eu_cov = 0
        eu_var = {"kind": 0, "gx": 0, "gy": 0}
        for i in range(self.N_EVAL):
            _, novel = model.gate(d["xeu"][i : i + 1])
            if bool(novel[0]):
                eu_novel += 1
                pt = spn_pred(d["xpeu"][i : i + 1])
                gt = cb.idx_to_code(ceu[i])
                eu_var["kind"] += int(pt[0] == gt[0])
                eu_var["gx"] += int(pt[1] == gt[1])
                eu_var["gy"] += int(pt[2] == gt[2])
            else:
                eu_cov += 1
                idx = int(mx.argmax(model.posterior_all(d["xeu"][i : i + 1])[0]))
                eu_hit += resolve(idx) == ceu[i]

        # 探针 B: 新类别 (圆盘)
        pb_novel = 0
        pb_var = {"gx": 0, "gy": 0}
        cpb = [tuple(int(v) for v in row) for row in d["cpb"].tolist()]
        for i in range(self.N_PROBE):
            _, novel = model.gate(d["xpb"][i : i + 1])
            pb_novel += bool(novel[0])
            pt = spn_pred(d["xppb"][i : i + 1])
            pb_var["gx"] += int(pt[1] == cpb[i][0])
            pb_var["gy"] += int(pt[2] == cpb[i][1])

        print(
            f"\nphase3 seen : 误判新 {es_novel/self.N_EVAL:.3f} | "
            f"码 acc {es_hit/self.N_EVAL:.3f}"
        )
        print(
            f"phase3 unseen: 仍判新 {eu_novel/self.N_EVAL:.3f} (SPN 变量级: "
            f"kind {eu_var['kind']/max(eu_novel,1):.3f} "
            f"gx {eu_var['gx']/max(eu_novel,1):.3f} "
            f"gy {eu_var['gy']/max(eu_novel,1):.3f}) | "
            f"已覆盖 {eu_cov/self.N_EVAL:.3f} 码 acc {eu_hit/max(eu_cov,1):.3f}"
        )
        pur = []
        for k, votes in prov_true.items():
            maj = max(set(votes), key=votes.count)
            pur.append(votes.count(maj) / len(votes))
        print(
            f"提升分量: {len(prov_true)} 个, 平均纯度 "
            f"{sum(pur)/max(len(pur),1):.3f} (多数真值码占比)"
        )
        print(
            f"probe B (圆盘): 判新 {pb_novel/self.N_PROBE:.3f} | "
            f"SPN 位置 gx {pb_var['gx']/self.N_PROBE:.3f} "
            f"gy {pb_var['gy']/self.N_PROBE:.3f}"
        )

        # ── 标定断言 (2026-08-12 实测标定, 留余量) ───────────────────
        # seen 误判新 0.060 不是门控失误: 1037 个 seen 码在 2000+420 帧里
        # 覆盖率 ≈1−e^{−2.3}≈0.90, 未采到的码被判新是正确行为
        assert es_novel / self.N_EVAL < 0.10, "已知码误判新超过未覆盖码份额"
        assert es_hit / self.N_EVAL > 0.90, f"seen 码 acc 过低 {es_hit/self.N_EVAL:.3f}"
        assert eu_hit / max(eu_cov, 1) > 0.80, "已提升码 acc 过低"
        assert pb_novel / self.N_PROBE > 0.95, "新类别未被门控捕获"
        print("experiment_joint: 完成 ✓")


if __name__ == "__main__":
    JointExperiment().run()
