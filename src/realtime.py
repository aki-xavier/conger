"""逐帧全链路: 冷启动 → 逐帧 online VB + EdgePrior + 后台 grouping。

模块流程:

  frame 0: RieszWavelet + VBGMM 冷启动 (全量拟合, 一次性开销)
           │  frame t (t≥1):
           ├─ rw.update(img) → features()         (特征刷新, ~10ms)
           ├─ gm.online_update(X, ρ)              (一次 E 步 + EWMA, ~百ms)
           ├─ class_likelihood → EdgePrior.enhance (空间先验, ~10ms)
           └─ GroupingTracker.submit(enh, ori)    (非阻塞, ~0.4ms)
                └─ 后台线程: 全量 grouping + 链 id 对应 (数秒, 只留最新帧)

  消费: tracker.latest() / wait_next() → TrackedResult (稳定 tid)

这就是 flow.md 的层间节奏落地: 逐帧路径 = 特征+在线后验+空间先验
(亚秒), 组织层结果以后台延迟换取。
"""

import time
from dataclasses import dataclass

import mlx.core as mx

from edgemap import EdgePrior
from grouping import GroupingTracker
from riesz import RieszWavelet
from vbgmm import VBGMM


@dataclass(slots=True)
class FrameOut:
    """一帧的逐帧路径输出。"""

    like: mx.array  # 边缘似然 (H,W)
    enh: mx.array  # 增强边缘图 (H,W)
    ms: float  # 逐帧路径耗时 (不含后台 grouping)


class RealtimePipeline:
    """逐帧管线门面: riesz 增量 → online VB → EdgePrior →
    后台 grouping → (可选) 分割。"""

    def __init__(
        self,
        img0: mx.array,
        k_max: int = 48,
        rho: float = 0.2,
        tracker: GroupingTracker | None = None,
        with_segment: bool = True,
    ):
        """首帧冷启动 (全量拟合) 并提交后台; ρ = online 遗忘因子。
        with_segment: 后台链路是否延伸到分割层 (SceneSegmenter)。"""
        self.rw = RieszWavelet(img0)
        feat = self.rw.features()
        self.gm = VBGMM(VBGMM.feature_matrix(feat), k_max=k_max)
        self.prior = EdgePrior()
        self.rho = rho
        if tracker is not None:
            self.tracker = tracker
        else:
            seg = None
            if with_segment:
                from segment import SceneSegmenter

                seg = SceneSegmenter(tau=0.5)
            self.tracker = GroupingTracker(segmenter=seg)
        like0 = self.gm.edge_likelihood(img0.shape)
        self._submit(like0, feat)

    def _submit(self, like: mx.array, feat) -> mx.array:
        """增强 + 提交后台 (带两路似然供分割层); 返回增强图。"""
        enh = self.prior.enhance(like, feat, self.rw)
        tex = self.gm.class_likelihood("texture").reshape(like.shape)
        self.tracker.submit(enh, feat.mean_ori, like, tex)
        return enh

    def step(self, img: mx.array) -> FrameOut:
        """一帧: 特征刷新 → online_update → 边缘似然 → 增强 → 提交后台。"""
        t0 = time.perf_counter()
        self.rw.update(img)
        feat = self.rw.features()
        x = VBGMM.feature_matrix(feat)
        r = self.gm.online_update(x, rho=self.rho)
        like = self.gm.class_likelihood("edge", x=x, r=r).reshape(img.shape)
        enh = self._submit(like, feat)
        mx.eval(enh)
        return FrameOut(like, enh, 1000 * (time.perf_counter() - t0))

    def close(self) -> None:
        """停止后台线程。"""
        self.tracker.close()


if __name__ == "__main__":
    # ── 移动边界序列: 逐帧跟踪 + 链 id 稳定 ─────────────────────────
    # 与 vbgmm 自检同族: 三条边界每帧右移 2px, 共 6 帧
    from utils import Utils

    H, W = 128, 256

    def frame(f: int) -> mx.array:
        """第 f 帧: 弱边缘/强边缘/纹理边界整体右移 2f px。"""
        im = mx.full((H, W), 0.2)
        im[:, 64 + 2 * f : 128 + 2 * f] = 0.25
        im[:, 128 + 2 * f : 192 + 2 * f] = 0.8
        im[:, 192 + 2 * f :] = Utils.make_grating((H, 64 - 2 * f), 8.0, 0.0)
        return im + mx.random.normal((H, W), key=mx.random.key(10 + f)) * 0.01

    t0 = time.perf_counter()
    pipe = RealtimePipeline(frame(0), rho=0.3)
    t1 = time.perf_counter()
    print(f"冷启动: {t1 - t0:.1f}s")

    tid_seq = []
    out = FrameOut(mx.zeros((H, W)), mx.zeros((H, W)), 0.0)
    for f in range(1, 6):
        out = pipe.step(frame(f))
        tr = pipe.tracker.wait_next(timeout=120.0)
        assert tr is not None and tr.tids, f"帧 {f} 后台无结果"
        tid_seq.append(tr.tids)
        seg_info = (
            f"子区域 {int(mx.max(tr.segment.subregions))}"
            if tr.segment is not None
            else "无分割"
        )
        print(
            f"帧 {f}: 逐帧路径 {out.ms:.0f}ms | "
            f"后台 v{tr.version} 链 {len(tr.tids)} 条 "
            f"(最老 age={max(tr.ages)}) | {seg_info}"
        )
    pipe.close()

    # 跟踪有效性: 末帧弱边缘 (≈74) 似然显著高于旧位置 (≈64)
    weak_new = float(out.like[:, 72:77].mean())
    weak_old = float(out.like[:, 62:66].mean())
    assert weak_new > 0.25 and weak_new > weak_old + 0.1, (
        f"跟踪失效: 新 {weak_new:.2f} 旧 {weak_old:.2f}"
    )
    # 链 id 稳定: 首帧的链在后续各帧保持同 tid
    first = set(tid_seq[0])
    stable = [t for t in first if all(t in ts for ts in tid_seq[1:])]
    assert stable, f"无跨帧稳定链: {tid_seq}"
    print(f"跟踪: 弱边缘新位置 {weak_new:.2f} > 旧位置 {weak_old:.2f} ✓")
    print(f"链 id: 稳定 tid {stable}, tid 序列 {tid_seq}")
