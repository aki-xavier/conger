import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx

from color import Color
from riesz import FeatureMaps, RieszWavelet
from utils import Utils
from vbgmm import VBGMM

# ── 方向场采样与传播原语 (见 EdgePrior 静态方法) ───────────────────
#
# 第一性原理位置: 边缘是局部一维结构 —— 沿切向 (垂直法向 ori) 信号
# 不变 → 相邻像素互相提供证据 (吸引耦合); 沿法向响应呈单峰脊线 →
# 候选互相竞争 (排斥耦合, NMS)。VB-GMM 给出逐像素似然 (证据项),
# 本模块把这个各向异性空域先验作用在似然图上, 得到完整后验。
# 顺序必须先在特征空间聚类出似然, 再做空间传播 —— 反过来先平滑
# 特征会稀释弱边缘证据, 污染似然本身。
#
# 模块流程:
#
#   like (VBGMM.edge_likelihood) + feat (mean_ori/ori_R) + rw (能量/λ)
#        │
#        ├─ scale(): dir_smooth/hops 按 λ_min/3 绑定最细波长
#        ▼
#   enhance():
#        smooth_normal(): 2θ 向量场按 ori_R 加权各向同性扩散
#           (缺口继承方向 → 允许桥接; 交叉处抵消 → 不混方向)
#        propagate(): 切向 ±多跳距 (1/2/4/8px·尺度) 双线性采样
#           (precomp_gather 索引/权重只算一次)
#           支撑 = max(双侧几何均值, 单侧半价), 方向门控 cos²(2Δθ)
#           更新只升不降 (L+gain·S·(1−L)), max|Δ|<eps 定点早停
#        ▼   enh ─(虚线反馈)→ VBGMM.feedback_round (见 vbgmm.py)
#   nms(): 法向 ±1px 软 NMS, 门控信号 = 总能量 Σe_s
#        (似然是簇级平台, 脊线位置只在原始能量剖面里;
#         先抬线再瘦线, 必须在 enhance 之后)
#        ▼
#   细化边缘图 (软 NMS 输出仅相对有意义, 显示时需归一)
#
#   旁路: enhance_per_scale() 逐尺度通道 (H,W,S) —— ori_R 低处
#   多结构竞争的诊断通路, 不含 GMM 似然, 非逐帧管线


@dataclass(slots=True)
class EdgePrior:
    """无状态各向异性先验。输入逐像素似然图 + Riesz 特征, 输出
    增强/细化后的边缘图。

    跨尺度分工 (fuse statistically, separate geometrically):
    似然由跨尺度融合特征聚类给出; 方向几何按 ori_R 分治 ——
    ori_R 高 = 单一主导结构, mean_ori 可信, 走融合快速路;
    ori_R 低 = 多结构竞争, conf 门控自动关闭传播 (不混方向),
    需要分辨时走 per-scale 通道。
    """

    n_iter: int = 3  # 传播轮数 (多跳采样下缺口按跳距直接桥接)
    lam: float = 0.5  # 每轮向支撑移动的步长
    gain: float = 0.3  # 超加性: 长程一致对似然的额外抬升
    beta: float = 8.0  # NMS 软硬度
    dir_smooth: int = 6  # 方向场内绘轮数 ≈ 可桥接缺口半径 (λ_min/3 单位)
    hops: tuple[int, ...] = (1, 2, 4, 8)  # 传播跳距: 最远桥接半径 (同单位)
    eps: float = 1e-3  # 传播定点早停阈值 (max|Δout|)

    # ── 方向场采样与传播原语 ──────────────────────────────────────

    @staticmethod
    def grid(shape: tuple[int, ...]) -> tuple[mx.array, mx.array]:
        """(row, col) 坐标网格, 尾部维度广播。"""
        yy, xx = mx.meshgrid(
            mx.arange(shape[0], dtype=mx.float32),
            mx.arange(shape[1], dtype=mx.float32),
            indexing="ij",
        )
        while yy.ndim < len(shape):
            yy = yy[..., None]
            xx = xx[..., None]
        return yy, xx

    @staticmethod
    def precomp_gather(
        shape: tuple[int, ...],
        dy: mx.array,
        dx: mx.array,
        yy: mx.array,
        xx: mx.array,
    ):
        """预计算固定偏移场 (dy, dx) 下的双线性采样闭包。

        方向场在一个 like 图的传播过程中不变, 索引/插值权重只需算
        一次; 每轮迭代只剩 2 次 gather + 算术, 这是实时化的关键。
        m 为 (H,W) 或 (H,W,S)。"""
        h, w = shape[:2]
        y = mx.clip(yy + dy, 0.0, h - 1.001)
        x = mx.clip(xx + dx, 0.0, w - 1.001)
        y0, x0 = y.astype(mx.int32), x.astype(mx.int32)
        fy, fx = y - y0, x - x0
        i00 = y0 * w + x0
        i01, i10, i11 = i00 + 1, i00 + w, i00 + w + 1

        def sample(m: mx.array) -> mx.array:
            """双线性采样闭包: m 为 (H,W) 或 (H,W,S)。"""
            if m.ndim == 2:
                flat = m.reshape(-1)
                v00, v01 = flat[i00], flat[i01]
                v10, v11 = flat[i10], flat[i11]
            else:
                flat = m.reshape(h * w, m.shape[-1])

                def g(idx: mx.array) -> mx.array:
                    """按扁平索引 gather 并还原 (H,W,S) 形状。"""
                    r = mx.take_along_axis(flat, idx.reshape(h * w, -1), axis=0)
                    return r.reshape(m.shape)

                v00, v01, v10, v11 = g(i00), g(i01), g(i10), g(i11)
            top = v00 * (1 - fx) + v01 * fx
            bot = v10 * (1 - fx) + v11 * fx
            return top * (1 - fy) + bot * fy

        return sample

    @staticmethod
    def blur3(m: mx.array) -> mx.array:
        """3x3 五点均值 (edge 填充), 只作用在前两维。"""
        pad_width = [(1, 1), (1, 1)] + [(0, 0)] * (m.ndim - 2)
        p = mx.pad(m, pad_width, mode="edge")
        acc = p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1]
        acc = acc + p[1:-1, :-2] + p[1:-1, 2:]
        return acc / 5.0

    @staticmethod
    def smooth_normal(
        normal: mx.array, conf: mx.array, n_iter: int = 6
    ) -> tuple[mx.array, mx.array]:
        """方向场内绘: 2θ 向量场按 conf 加权各向同性扩散。

        ori_R 低混着两种情形 —— "无局部结构" (缺口/平坦) 与 "多结构
        竞争" (交叉)。不内绘则缺口处方向无定义, 传播进不去, 桥接
        无从谈起。扩散后低置信区从邻域继承方向; 返回的 |v| 作为新
        置信度: 交叉处邻域方向互相抵消 → |v| 仍低 → 不混方向;
        缺口处邻域方向一致 → |v| 中等 → 允许桥接。
        """
        v_re = conf * mx.cos(2 * normal)
        v_im = conf * mx.sin(2 * normal)
        for _ in range(n_iter):
            v_re = EdgePrior.blur3(v_re)
            v_im = EdgePrior.blur3(v_im)
        norm = mx.sqrt(v_re**2 + v_im**2)
        return 0.5 * mx.arctan2(v_im, v_re), mx.clip(norm, 0.0, 1.0)

    @staticmethod
    def propagate(
        like: mx.array,
        normal: mx.array,
        conf: mx.array,
        n_iter: int = 3,
        lam: float = 0.3,
        gain: float = 0.3,
        hops: tuple[int, ...] = (1, 2, 4),
        eps: float = 1e-3,
    ) -> mx.array:
        """切向连续性传播 (吸引耦合)。

        沿 ±切向在多个跳距 (1/2/4px) 采样邻居似然, 取各跳支撑的最
        大值 —— 长程连续一步到位, 缺口按最近可及跳距桥接, 不靠逐
        轮稀释。方向门控 w = max(0, cos 2Δθ)² (2θ 统计处理法向 ±π
        模糊), 再乘邻居置信 conf。每跳支撑 = max(双侧几何均值, 单
        侧半价): 双侧最强, 端点单侧半价, 孤立点无支撑。
        更新 target = max(S, L + gain·S·(1−L)), L ← (1−λ)L + λ·target:
        S > L 时直接吸引到支撑 (桥接有效); S ≈ L 时 gain 项提供温和
        超加性 (长程一致抬升弱边缘, 增幅随轮数累积, 不宜多轮);
        只升不降 —— 孤立高响应的抑制交给聚类 (GMM 对孤立亮斑
        本来就判非边缘), 传播不重复发明惩罚机制。
        max|Δout| < eps 时提前退出 (定点收敛, 通常末轮才触发)。
        """
        yy, xx = EdgePrior.grid(like.shape)
        ang = normal + math.pi / 2.0  # 切向
        dy, dx = mx.sin(ang), mx.cos(ang)
        sides = []
        for hop in hops:
            gp = EdgePrior.precomp_gather(like.shape, dy * hop, dx * hop, yy, xx)
            gm = EdgePrior.precomp_gather(like.shape, -dy * hop, -dx * hop, yy, xx)
            wp = mx.maximum(mx.cos(2 * (gp(normal) - normal)), 0.0) ** 2 * gp(conf)
            wm = mx.maximum(mx.cos(2 * (gm(normal) - normal)), 0.0) ** 2 * gm(conf)
            sides.append((gp, gm, wp, wm))

        out = like
        for _ in range(n_iter):
            support = mx.zeros_like(out)
            for gp, gm, wp, wm in sides:
                a = gp(out) * wp
                b = gm(out) * wm
                s = mx.maximum(mx.sqrt(a * b), 0.5 * mx.maximum(a, b))
                support = mx.maximum(support, s)
            target = mx.maximum(support, out + gain * support * (1.0 - out))
            new = mx.clip((1.0 - lam) * out + lam * target, 0.0, 1.0)
            if float(mx.max(mx.abs(new - out))) < eps:  # 定点早停: 不再变化即收敛
                out = new
                break
            out = new
        return out

    @staticmethod
    def soft_nms(
        like: mx.array, loc: mx.array, normal: mx.array, beta: float = 8.0
    ) -> mx.array:
        """法向软 NMS (排斥耦合)。

        似然图是簇级量 —— 同一分量的像素共享同一似然, 形成量化平
        台, 亚平台的脊线位置信息不在其中, 只在原始响应里。所以门
        控用连续定位信号 loc (如总能量 Σe_s): 沿法向 ±1px 比较,
        相对差 sigmoid —— 上坡侧被压, 只留局部峰。输出 = 似然 ×
        门控: 聚类回答 "是不是边缘", 能量剖面回答 "精确在哪"。
        必须在切向增强之后做 (先抬线再瘦线, 不会误杀弱边缘)。
        """
        yy, xx = EdgePrior.grid(like.shape)
        gp = EdgePrior.precomp_gather(
            like.shape, mx.sin(normal), mx.cos(normal), yy, xx
        )
        gm = EdgePrior.precomp_gather(
            like.shape, -mx.sin(normal), -mx.cos(normal), yy, xx
        )

        def gate(other: mx.array) -> mx.array:
            """相对差的 sigmoid 门控 (loc 高于对侧 → ≈1)。"""
            z = beta * (loc - other) / mx.maximum(loc, 1e-6)
            return 1.0 / (1.0 + mx.exp(-z))

        return like * gate(gp(loc)) * gate(gm(loc))

    def scale(self, rw: RieszWavelet) -> tuple[int, tuple[int, ...]]:
        """把像素量纲的 dir_smooth/hops 绑定到滤波器组最细波长:
        空间作用半径的物理含义是"相对于最细结构尺度的多远",
        λ_min 变化 (分辨率/核参数调整) 时半径随之缩放。"""
        s = rw.lam_min / 3.0
        ds = max(3, round(self.dir_smooth * s))
        hp = tuple(max(1, round(h * s)) for h in self.hops)
        return ds, hp

    def enhance(self, like: mx.array, feat: FeatureMaps, rw: RieszWavelet) -> mx.array:
        """融合快速路: 方向场先内绘, 再沿切向传播。"""
        ds, hp = self.scale(rw)
        normal, conf = self.smooth_normal(feat.mean_ori, feat.ori_R, ds)
        return self.propagate(
            like, normal, conf, self.n_iter, self.lam, self.gain, hp, self.eps
        )

    def enhance_per_scale(self, rw: RieszWavelet) -> mx.array:
        """逐尺度通道 (H,W,S): 每个尺度用各自 ori 传播各自的能量
        份额 p_s = e_s/Σe, 跨尺度不混 —— ori_R 低处的多结构竞争
        在这里被分开。诊断用: 不做方向内绘, 也不含 GMM 似然。"""
        e = mx.stack([s.energy for s in rw.scales], axis=-1)
        p = e / mx.maximum(mx.sum(e, axis=-1, keepdims=True), 1e-12)
        ori = mx.stack([s.ori for s in rw.scales], axis=-1)
        conf = mx.broadcast_to(mx.sqrt(p), p.shape)  # 能量弱的方向不可信
        _, hp = self.scale(rw)
        return self.propagate(
            p, ori, conf, self.n_iter, self.lam, self.gain, hp, self.eps
        )

    def nms(self, like: mx.array, feat: FeatureMaps, rw: RieszWavelet) -> mx.array:
        """法向软 NMS: 定位信号 = 总能量 Σe_s (连续, 有真实脊线)。"""
        loc = mx.stack([s.energy for s in rw.scales], axis=-1)
        loc = mx.sum(loc, axis=-1)
        return self.soft_nms(like, loc, feat.mean_ori, self.beta)

    @staticmethod
    def visualize(
        img: mx.array,
        like: mx.array,
        enh: mx.array,
        thin: mx.array,
        out_path: str | Path,
    ):
        """边缘图管线各阶段可视化 (似然/增强/NMS/脊线叠加)。"""
        disp = thin / mx.maximum(mx.max(thin), 1e-12)  # 软 NMS 输出仅相对有意义
        ridge = mx.stack([img, img, img], axis=-1)
        ridge = ridge.at[:, :, 0].add(disp * 0.8)  # 脊线染红
        ridge = mx.clip(ridge, 0.0, 1.0)
        plots = [
            ("original", "gray", img),
            ("likelihood", "gray", like),
            ("enhanced", "gray", enh),
            ("nms (normalized)", "gray", disp),
            ("ridge overlay", None, ridge),
        ]
        fig = Utils.visualize(plots)
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    from PIL import Image

    # ── synthetic validation ─────────────────────────────────────────
    # 与 vbgmm 同款条带 + 两个空域先验专属探针:
    #   gap  —— 弱边缘在 rows 56:66 抹成 16px 渐变 (阶跃消失)
    #   blob —— 平坦区里一个 3x3 孤立亮斑
    H, W = 128, 256
    img = mx.full((H, W), 0.2)
    img[:, 64:128] = 0.25
    img[:, 128:192] = 0.8
    img[:, 192:] = Utils.make_grating((H, 64), 8.0, 0.0)[:, :64]
    ramp = mx.linspace(0.2, 0.25, 16)[None, :] * mx.ones((10, 1))
    img[56:66, 56:72] = ramp
    img = img + mx.random.normal((H, W), key=mx.random.key(3)) * 0.01
    img[100:103, 90:93] = img[100:103, 90:93] + 0.3

    rw = RieszWavelet(img)
    feat = rw.features()
    gm = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    like = gm.edge_likelihood((H, W))
    prior = EdgePrior()
    enh = prior.enhance(like, feat, rw)
    thin = prior.nms(enh, feat, rw)

    regions = [
        ("weak edge @64   ", slice(20, 50), slice(62, 66)),
        ("weak gap @64    ", slice(57, 65), slice(62, 66)),
        ("strong edge @128", slice(20, 120), slice(126, 130)),
        ("blob @100,90    ", slice(99, 104), slice(89, 94)),
        ("flat interior   ", slice(20, 50), slice(90, 120)),
        ("tex interior    ", slice(20, 120), slice(200, 250)),
    ]
    print(f"{'region':<17s} {'like':>5s} {'enh':>5s} {'nms峰值':>6s}")
    for name, rs, cs in regions:
        print(
            f"{name}: {float(like[rs, cs].mean()):.2f}  "
            f"{float(enh[rs, cs].mean()):.2f}  "
            f"{float(thin[rs, cs].max()):.2f}"
        )
    # 脊线宽度与定位: 强边缘处 thin > 半峰 的列数与峰位
    row = thin[40, 118:138]
    width = int(mx.sum(row > 0.5 * mx.max(row)))
    peak = int(mx.argmax(row)) + 118
    print(f"strong edge 脊线宽={width}px, 峰位@{peak} (期望 128±1)")

    # ── EdgePrior → VBGMM 阻尼反馈 (虚线边, flow.md 迭代协议) ──────
    # 分歧回注软聚类做不确定度门控的微调; frac 锚定在前馈 r 上
    frac = gm.class_fraction("edge")
    r_fb = gm.feedback_round(enh)
    s = mx.sum(r_fb, axis=1)
    assert float(mx.max(mx.abs(s - 1.0))) < 1e-3, "反馈后责任仍须归一"
    like_fb = (r_fb @ frac).reshape(H, W)
    enh_fb = prior.enhance(like_fb, feat, rw)
    margin = 1.0 - mx.max(gm.r, axis=1)
    print(
        f"反馈衬底: max margin = {float(mx.max(margin)):.4f} "
        f"(VB 后验近 one-hot → 反馈调整量≈0, 见 feedback_round docstring)"
    )
    print("反馈后 (like_fb / enh_fb):")
    for name, rs, cs in regions:
        print(
            f"{name}: {float(like_fb[rs, cs].mean()):.4f}  "
            f"{float(enh_fb[rs, cs].mean()):.4f}"
        )
    # 机制断言: 反馈自限有界; gap 桥接留在 enh, 不伪造进似然;
    # 孤立亮斑不被放大 —— 这是本轮探索的正面结论 (分歧保留), 不是失败
    gap = (slice(57, 65), slice(62, 66))
    blob = (slice(99, 104), slice(89, 94))
    assert float(mx.max(mx.abs(like_fb - like))) < 0.1, "反馈应自限有界"
    assert float(like_fb[gap].mean()) < 0.05, "gap 桥接在 enh, 不进似然"
    assert float(like_fb[blob].mean()) < 0.05, "孤立亮斑不应被反馈放大"

    path = Utils.project_root() / "artifacts/edgemap_synth.png"
    EdgePrior.visualize(img, like, enh, thin, path)
    print(path)

    # ── natural images ───────────────────────────────────────────────
    for img_name in [
        "12.png",
        "nat10.jpg",
        "nat1015.jpg",
        "nat1016.jpg",
        "nat1018.jpg",
        "nat1035.jpg",
    ]:
        im = Image.open(Utils.project_root() / f"images/{img_name}").convert("L")
        arr = Color.image_to_mlx(im)
        rw2 = RieszWavelet(arr)
        feat2 = rw2.features()
        gm2 = VBGMM(VBGMM.feature_matrix(feat2), k_max=48)
        like2 = gm2.edge_likelihood(arr.shape)
        t0 = time.perf_counter()
        enh2 = prior.enhance(like2, feat2, rw2)
        thin2 = prior.nms(enh2, feat2, rw2)
        mx.eval(thin2)
        t1 = time.perf_counter()
        path2 = Utils.project_root() / f"artifacts/edgemap_{img_name}"
        EdgePrior.visualize(arr, like2, enh2, thin2, path2)
        print(
            f"{img_name} {arr.shape}: K_eff={gm2.k_eff()}, "
            f"enhance+nms {1000 * (t1 - t0):.0f}ms → {path2}"
        )

    # ── 逐尺度通道 (以 12.png 为例) ──────────────────────────────────
    t0 = time.perf_counter()
    ps = prior.enhance_per_scale(rw2)
    mx.eval(ps)
    t1 = time.perf_counter()
    print(f"per-scale enhance {ps.shape}: {1000 * (t1 - t0):.0f}ms (诊断用, 非逐帧)")

    # ── 逐帧全链路计时 (实时管线形态, 以 12.png 为例) ───────────────
    # 后台慢速暖启动刷新 VBGMM 后验; 逐帧 = update + 特征刷新 + infer + 先验
    for rep in range(3):
        t0 = time.perf_counter()
        rw2.update(arr)
        feat2 = rw2.features()
        x_ = VBGMM.feature_matrix(feat2)
        r_ = gm2.infer(x_)
        l_ = gm2.edge_likelihood(arr.shape, x_, r_)
        e_ = prior.enhance(l_, feat2, rw2)
        t_ = prior.nms(e_, feat2, rw2)
        mx.eval(t_)
        t1 = time.perf_counter()
        print(f"逐帧全链路 rep{rep}: {1000 * (t1 - t0):.0f}ms")
