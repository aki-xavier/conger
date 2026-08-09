"""场景分割层 (Scene Segmentation, flow.md §2): 边界强度 → 区域层级 → 软标签。

模块流程:

  enh (EdgePrior 增强图 = 边界强度 E∈[0,1])
       │  Watershed: 极小值平台为种子, heapq 优先洪泛 (4 邻域)
       ▼  → 盆地标签 (过分割) + 相邻区域弧 (逐接触点 E)
  RegionHierarchy: 弧强度 = 全弧 E 均值 (防 single-link 污染),
       按强度升序合并 (加权均值重算新弧), 被吞弧的像素记合并高度
       ▼  → UCM (超度量等高线图); cut(τ): τ 以下合并全部执行
  (S 层) ContourCut: 组织层轮廓折线栅格化为掩码, 硬切区域
  (Y 层) PixelLabelLayer: 温度软化像素后验 × 子区域先验 → 软标签

数值纪律: 洪泛/合并是顺序算法, 用 Python heapq + 标量列表 (一次
tolist 后纯 Python 标量, 比逐元素数组访问快); 数组统计与渲染留在
MLX。本层是离线层, 不做逐帧承诺 (逐帧管线止于 edgemap)。

已知近似: 脊线像素在洪泛中被单侧盆地吸收, 同侧碎片盆地在脊线
像素上的接触会把碎片弧的均值抬高 (上界 = 脊高) —— 碎片比理论
鞍点高度活得久一些, 但合并的全序保持正确 (碎片 < 真墙)。接触
值取 max(对像素 E), 弧强度取全弧均值 (防 single-link 污染)。
"""

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import mlx.core as mx

from utils import Utils


class RegionMap(NamedTuple):
    """R 层输出: 分水岭盆地 + 合并层级 (UCM 经 hierarchy.ucm 取)。"""

    labels: mx.array  # (H,W) int32 盆地标签 (1..K)
    hierarchy: RegionHierarchy  # 合并层级 (cut/ucm)


# ── R 层: 分水岭 ──────────────────────────────────────────────────


class Watershed:
    """优先洪泛分水岭 (Meyer 洪泛, 4 邻域)。

    种子 = 极小值平台 (无严格更低邻居的像素的连通分量); 洪泛按水位
    单调上升, 像素归属先到的洪泛; 两洪泛相遇处记录区域间弧。
    """

    @staticmethod
    def run(E: mx.array) -> tuple[list[list[int]], dict[tuple[int, int], list[float]]]:
        """E (H,W) 边界强度 → (盆地标签 (H,W) int 嵌套列表, 弧表)。

        弧表键为区域 id 对 (a<b), 值为每次接触的边界强度 (取接触
        两像素 E 的较大者, 近似脊线高度)。
        """
        h, w = E.shape

        # 极小值平台: 没有严格更低邻居 (4 邻域) 的像素 (MLX 批量判)
        p = mx.pad(E, [(1, 1), (1, 1)], constant_values=float("inf"))
        has_lower = (
            (p[1:-1, :-2] < E)
            | (p[1:-1, 2:] < E)
            | (p[:-2, 1:-1] < E)
            | (p[2:, 1:-1] < E)
        )
        is_min = (~has_lower).reshape(-1).tolist()
        e = E.reshape(-1).tolist()

        # 平台连通分量 (BFS, 扁平索引) → 种子
        labels = [0] * (h * w)
        n_seeds = 0
        for i0 in range(h * w):
            if not is_min[i0] or labels[i0] != 0:
                continue
            n_seeds += 1
            stack = [i0]
            labels[i0] = n_seeds
            while stack:
                i = stack.pop()
                y, x = divmod(i, w)
                if x > 0 and is_min[i - 1] and labels[i - 1] == 0:
                    labels[i - 1] = n_seeds
                    stack.append(i - 1)
                if x < w - 1 and is_min[i + 1] and labels[i + 1] == 0:
                    labels[i + 1] = n_seeds
                    stack.append(i + 1)
                if y > 0 and is_min[i - w] and labels[i - w] == 0:
                    labels[i - w] = n_seeds
                    stack.append(i - w)
                if y < h - 1 and is_min[i + w] and labels[i + w] == 0:
                    labels[i + w] = n_seeds
                    stack.append(i + w)

        # 优先洪泛: 水位 = max(当前像素 E, 来路水位), 单调上升
        # (扁平索引 + 收窄 heap 元组, 局部绑定微优化)
        heap = [(e[i], i) for i in range(h * w) if labels[i] > 0]
        heapq.heapify(heap)
        arcs: dict[tuple[int, int], list[float]] = {}
        while heap:
            lvl, i = heapq.heappop(heap)
            y, x = divmod(i, w)
            lab = labels[i]
            # 邻域展开顺序必须与旧实现一致 (上/下/左/右):
            # 接触事件序列敏感 (arcs 是逐接触点的 E 值列表)
            if y > 0:
                nl = labels[i - w]
                if nl == 0:
                    labels[i - w] = lab
                    heapq.heappush(heap, (max(e[i - w], lvl), i - w))
                elif nl != lab:
                    key = (lab, nl) if lab < nl else (nl, lab)
                    arcs.setdefault(key, []).append(max(e[i], e[i - w]))
            if y < h - 1:
                nl = labels[i + w]
                if nl == 0:
                    labels[i + w] = lab
                    heapq.heappush(heap, (max(e[i + w], lvl), i + w))
                elif nl != lab:
                    key = (lab, nl) if lab < nl else (nl, lab)
                    arcs.setdefault(key, []).append(max(e[i], e[i + w]))
            if x > 0:
                nl = labels[i - 1]
                if nl == 0:
                    labels[i - 1] = lab
                    heapq.heappush(heap, (max(e[i - 1], lvl), i - 1))
                elif nl != lab:
                    key = (lab, nl) if lab < nl else (nl, lab)
                    arcs.setdefault(key, []).append(max(e[i], e[i - 1]))
            if x < w - 1:
                nl = labels[i + 1]
                if nl == 0:
                    labels[i + 1] = lab
                    heapq.heappush(heap, (max(e[i + 1], lvl), i + 1))
                elif nl != lab:
                    key = (lab, nl) if lab < nl else (nl, lab)
                    arcs.setdefault(key, []).append(max(e[i], e[i + 1]))
        return [labels[y * w:(y + 1) * w] for y in range(h)], arcs


# ── R 层: 区域合并层级 (UCM) ──────────────────────────────────────


class RegionHierarchy:
    """区域合并层级: 弧强度 = 全弧 E 均值, 升序合并 → 超度量等高线图。

    逐像素升序 union-find 是 single-link 陷阱 (一条强弧上的单个弱
    像素会污染整条弧); 这里以弧为单位: 强度取全弧均值, 合并时按
    接触点数加权重算新弧强度。边界像素的 UCM 值 = 其两侧区域首次
    同属一个分量时的合并高度 (合并树的 LCA 高度)。
    """

    def __init__(
        self, labels: list[list[int]], arcs: dict[tuple[int, int], list[float]]
    ):
        """由分水岭盆地与弧表构建合并树, 并渲染 UCM 图。"""
        h, w = len(labels), len(labels[0])
        self._labels = labels
        self.n_basins = max(max(row) for row in labels)

        # 每条弧的边界像素 (UCM 渲染用): 两侧标签不同的像素
        # 批量化: MLX diff 判不等 (水平+垂直), key 编码 (a·(K+1)+b)
        # 排序分组 —— Python pass 只处理边界像素 (自然图 ~5-10%
        # 像素), 不再是全图 22 万双层循环
        lab = mx.array(labels)
        kb = self.n_basins + 1
        eq_h = lab[:, :-1] != lab[:, 1:]  # 像素 (y,x) vs (y,x+1)
        k_h = int(mx.sum(eq_h))
        key_h = mx.zeros((max(k_h, 1), 3), dtype=mx.int32)
        if k_h:
            fi_h = Utils.nonzero(eq_h)
            yy_h = fi_h // (w - 1)
            xx_h = fi_h % (w - 1)
            a_h = lab[yy_h, xx_h]
            b_h = lab[yy_h, xx_h + 1]
            key_h = mx.stack(
                [mx.minimum(a_h, b_h) * kb + mx.maximum(a_h, b_h),
                 yy_h, xx_h], axis=-1)
        eq_v = lab[:-1, :] != lab[1:, :]  # 像素 (y,x) vs (y+1,x)
        k_v = int(mx.sum(eq_v))
        key_v = mx.zeros((max(k_v, 1), 3), dtype=mx.int32)
        if k_v:
            fi_v = Utils.nonzero(eq_v)
            yy_v = fi_v // w
            xx_v = fi_v % w
            a_v = lab[yy_v, xx_v]
            b_v = lab[yy_v + 1, xx_v]
            key_v = mx.stack(
                [mx.minimum(a_v, b_v) * kb + mx.maximum(a_v, b_v),
                 yy_v, xx_v], axis=-1)
        all_k = mx.concatenate([key_h[:k_h], key_v[:k_v]])
        order = mx.argsort(all_k[:, 0])
        arc_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
        # 一次 tolist 全同步 —— 逐元素索引是 7s/11 万次的同步地狱
        rows = all_k.tolist()
        prev = -1
        cur: list[tuple[int, int]] = []
        for i in order.tolist():
            kk, yy_i, xx_i = rows[i]
            if kk != prev:
                if cur:
                    arc_pixels[(prev // kb, prev % kb)] = cur
                prev = kk
                cur = []
            cur.append((yy_i, xx_i))
        if cur:
            arc_pixels[(prev // kb, prev % kb)] = cur

        strength = {k: sum(v) / len(v) for k, v in arcs.items()}
        count = {k: len(v) for k, v in arcs.items()}
        adj: dict[int, set[int]] = {}
        for a, b in arcs:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        parent = list(range(self.n_basins + 1))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        # 初始规模 = 盆地真实像素数 (合并优先级需知小侧是谁;
        # 原实现全 1 只是 union-by-size 的性能技巧)
        size = [0] * (self.n_basins + 1)
        for a, c in Counter(x for row in labels for x in row).items():
            size[a] = c
        # 尺度先验 (BSDS 评测暴露的逻辑问题): 纯弧强度竞争下,
        # 纹理碎片凭局部强边活到高 τ → 过度分割 (1577 区 vs GT
        # ~10)。初始堆按 强度×小侧规模折扣 排序, 碎片无论边强
        # 先并 —— 与纹理身份无关, 伪装色安全 (伪装鱼是大区)。
        # 折扣只作用初始堆: 合并重推的弧用原始强度 —— 否则合并
        # 路径被打乱, 缺口桥弧的弱接触被提前内化, 桥均值从
        # 0.220 跳到 0.504 (实测破掉缺口稀释性质/测试 5)
        size_ref = max(16.0, 0.001 * h * w)  # 对象尺度参考 (0.1% 图幅)

        def eff(key: tuple[int, int]) -> float:
            """初始弧有效强度。
            弱弧 (桥/缺口候选) 不折扣: 桥盆地小是常态, 折扣会让它
            先并入一侧, 反而毁掉缺口稀释路径 (测试 5 的教训);
            强弧 (纹理碎片的墙) 才按小侧规模折扣先并。"""
            raw = strength.get(key, 0.0)
            if raw < 0.3:  # 弱三分位语义: 稀释区
                return raw
            ra, rb = find(key[0]), find(key[1])
            small = min(size[ra], size[rb])
            return raw * min(1.0, (small / size_ref) ** 0.5)

        heap = [(eff(k), k[0], k[1]) for k in strength]
        heapq.heapify(heap)
        ucm = [[0.0] * w for _ in range(h)]
        self._merges: list[tuple[float, int, int]] = []  # (高度, 存留, 被吸收)
        while heap:
            s, a, b = heapq.heappop(heap)
            ra, rb = find(a), find(b)
            if ra == rb:
                continue  # 弧已在先前的合并中内化
            key = (min(ra, rb), max(ra, rb))
            # 过期判: 原始强度或有效强度任一匹配 (两阶段入堆)
            if key not in strength or (
                abs(strength.get(key, -1.0) - s) > 1e-9 and abs(eff(key) - s) > 1e-9
            ):
                continue  # 过期堆项 (弧已重连/重算)
            # 执行合并: 该弧边界像素记当前高度
            strength.pop(key)
            count.pop(key)
            for y, x in arc_pixels.pop(key, []):
                ucm[y][x] = s
            # 并大留小 (union by size): 存留侧的弧键/强度不变, 旧堆项
            # 仍有效 —— 只需重连被吸收侧的弧, 堆弹出从 O(Σfrontier)
            # (实测 4M) 降到 O(N log N)
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            self._merges.append((s, ra, rb))
            parent[rb] = ra
            size[ra] += size[rb]
            for x in adj.get(rb, set()):
                rx = find(x)
                if rx in (ra, rb):
                    continue
                old_b = (min(rb, x), max(rb, x))
                if old_b not in strength:
                    continue  # 该邻弧已是 (ra,x) 形态或已内化
                s2 = strength.pop(old_b)
                c2 = count.pop(old_b)
                pix2 = arc_pixels.pop(old_b, [])
                old_a = (min(ra, rx), max(ra, rx))
                if old_a in strength:
                    # 存留侧本就有到 x 的弧: 接触点数加权均值重算,
                    # 旧堆项强度失配自然作废
                    nc = count[old_a] + c2
                    ns = (strength[old_a] * count[old_a] + s2 * c2) / nc
                    strength[old_a] = ns
                    count[old_a] = nc
                    arc_pixels[old_a] = arc_pixels.get(old_a, []) + pix2
                    heapq.heappush(heap, (ns, old_a[0], old_a[1]))
                else:
                    # 重键 (ra, x), 强度不变
                    nkey = (min(ra, rx), max(ra, rx))
                    strength[nkey] = s2
                    count[nkey] = c2
                    arc_pixels[nkey] = pix2
                    heapq.heappush(heap, (s2, nkey[0], nkey[1]))
                adj.setdefault(rx, set()).discard(rb)
                adj[rx].add(ra)
            adj[ra] = (adj.get(ra, set()) | adj.get(rb, set())) - {ra, rb}
            adj.pop(rb, None)
        self._ucm = mx.array(ucm, dtype=mx.float32)

    @property
    def merges(self) -> list[tuple[float, int, int]]:
        """合并事件序列 (高度, 根a, 根b), 按高度升序。"""
        return self._merges

    @property
    def ucm(self) -> mx.array:
        """超度量等高线图 (H,W): 边界像素为合并高度, 区域内 0。"""
        return self._ucm

    def cut(self, tau: float) -> mx.array:
        """在合并高度 τ 处切层级 → 区域标签图 (1..K, mx.int32)。

        重放 τ 以下的合并事件 (与构建同序同规则, 结构一致):
        内边界 (强度 ≤ τ) 消失, 外边界保留。
        """
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            root = a
            while root in parent:
                root = parent[root]
            while a in parent:  # 路径压缩
                parent[a], a = root, parent[a]
            return root

        for s, ra, rb in self._merges:
            if s > tau:
                break  # 合并事件按高度升序
            parent[find(rb)] = find(ra)
        roots = [find(i) for i in range(self.n_basins + 1)]
        remap = {r: k for k, r in enumerate(sorted(set(roots[1:])))}
        out = [[remap[roots[lab]] + 1 for lab in row] for row in self._labels]
        return mx.array(out, dtype=mx.int32)

    def n_regions(self, tau: float) -> int:
        """cut(τ) 的区域数 (不重渲染, 数 τ 以下的净合并)。"""
        return self.n_basins - sum(1 for s, _, _ in self._merges if s <= tau)


# ── R 层: 门面 ────────────────────────────────────────────────────


@dataclass(slots=True)
class RegionLayer:
    """R 层门面: 边界强度图 → 分水岭盆地 + 合并层级。"""

    def run(self, enh: mx.array) -> RegionMap:
        """enh (H,W) ∈[0,1] → RegionMap。"""
        labels, arcs = Watershed.run(enh)
        return RegionMap(mx.array(labels), RegionHierarchy(labels, arcs))


# ── S 层: 轮廓硬切分 ──────────────────────────────────────────────
#
# p(S|R,L): 组织层轮廓图元 L 为硬约束 (轮廓线强制切开区域)。
# 吃折线 (grouping 链的亚像素点列) 不吃 blade —— 避免 blade→像素
# 网格的反投影; blade 仍作为语义句柄由调用方保存。


class ContourCut:
    """轮廓折线/圆弧 → 1px 切割掩码 (膨胀 1px 防走样断缝)。"""

    @staticmethod
    def rasterize(
        shape: tuple[int, int],
        polylines: list[mx.array] = (),
        circles: list[tuple[tuple[float, float], float]] = (),
        min_len: float = 10.0,
    ) -> list[list[bool]]:
        """折线 ((N,2) (row,col) 亚像素) 与圆 ((x,y),ρ) 栅格化为
        (H,W) bool 掩码。短于 min_len 的折线不参与切割 (防碎链过切)。
        """
        h, w = shape
        mask = [[False] * w for _ in range(h)]

        def draw(y: float, x: float) -> None:
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < h and 0 <= ix < w:
                mask[iy][ix] = True

        for pl in polylines:
            pts = pl.tolist()
            if len(pts) < 2:
                continue
            total = 0.0
            for i in range(len(pts) - 1):
                dy = pts[i + 1][0] - pts[i][0]
                dx = pts[i + 1][1] - pts[i][1]
                total += math.sqrt(dy * dy + dx * dx)
            if total < min_len:
                continue
            for i in range(len(pts) - 1):
                dy = pts[i + 1][0] - pts[i][0]
                dx = pts[i + 1][1] - pts[i][1]
                seg = math.sqrt(dy * dy + dx * dx)
                n = max(2, int(seg / 0.5) + 1)  # 0.5px 步长防断缝
                for k in range(n):
                    t = k / (n - 1)
                    draw(
                        pts[i][0] * (1 - t) + pts[i + 1][0] * t,
                        pts[i][1] * (1 - t) + pts[i + 1][1] * t,
                    )
        for (cx, cy), rho in circles:
            n = max(8, int(2 * math.pi * rho / 0.5))
            for k in range(n):
                a = 2 * math.pi * k / n
                draw(cy + rho * math.sin(a), cx + rho * math.cos(a))

        # 膨胀 1px (3x3 结构元)
        out = [[False] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if mask[y][x]:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                out[ny][nx] = True
        return out


class SubregionLayer:
    """S 层: R 区域标签 × 轮廓掩码 → 子区域连通分量。

    BFS 只在同 R 标签且未被掩码的像素间扩展: 轮廓线强制切开区域,
    无轮廓处区域内部保持连通 (软过渡由 R 层级承担)。
    """

    @staticmethod
    def run(labels: mx.array, mask: list[list[bool]]) -> mx.array:
        """labels (H,W) int32 (cut(τ) 结果) + mask (H,W) bool →
        子区域标签 (H,W) int32; 掩码像素为 0。"""
        lab = labels.tolist()
        h, w = len(lab), len(lab[0])
        out = [[0] * w for _ in range(h)]
        nxt = 0
        for y0 in range(h):
            for x0 in range(w):
                if out[y0][x0] != 0 or mask[y0][x0]:
                    continue
                nxt += 1
                region = lab[y0][x0]
                stack = [(y0, x0)]
                out[y0][x0] = nxt
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and out[ny][nx] == 0
                            and not mask[ny][nx]
                            and lab[ny][nx] == region
                        ):
                            out[ny][nx] = nxt
                            stack.append((ny, nx))
        return mx.array(out, dtype=mx.int32)


# ── Y 层: 像素软标签 ──────────────────────────────────────────────


@dataclass(slots=True)
class PixelLabelLayer:
    """Y 层: p(y_n|S,I_n) —— 子区域先验 × 温度软化的像素后验。

    像素分数取自 VBGMM 的类似然 (edge/texture), flat = 1−edge−tex
    截断 ≥0 (ponytail: flat 是剩余类, 无独立证据规则)。温度 T 软化
    是必需的: VB 后验近 one-hot 时区域先验乘上去纹丝不动 (615 纳特
    墙, 见 vbgmm.feedback_round 的实测)。只消费 vbgmm 输出, 不改它。
    """

    temperature: float = 2.5  # 软化温度 (合成验证定)
    lam: float = 0.4  # 像素项权重; 区域先验权重 = 1−λ (偏区域平滑)

    @staticmethod
    def _scatter_mean(vals: mx.array, lab: mx.array, n: int) -> mx.array:
        """按标签分组求通道均值: vals (N,C) + 标签 (N,) → (n,C)。"""
        cols = [
            mx.zeros((n,)).at[lab].add(vals[:, c]) for c in range(int(vals.shape[1]))
        ]
        out = mx.stack(cols, axis=-1)
        cnt = mx.zeros((n,)).at[lab].add(mx.ones((int(lab.shape[0]),)))
        return out / mx.maximum(cnt, 1.0)[:, None]

    def soften(self, p: mx.array) -> mx.array:
        """温度软化: normalize(p^(1/T))。p (...,C)。"""
        q = mx.power(mx.maximum(p, 1e-12), 1.0 / self.temperature)
        return q / q.sum(axis=-1, keepdims=True)

    def run(
        self,
        like_edge: mx.array,
        like_tex: mx.array,
        sublabels: mx.array,
        macro: mx.array | None = None,
        w_macro: float = 0.3,
    ) -> tuple[mx.array, mx.array]:
        """两路类似然 + 子区域标签 → (软标签 (H,W,3), 硬标签 (H,W))。
        通道序: edge / texture / flat。
        macro: 可选宏簇图 (H,W) int (vbgmm.macro_labels) —— 语义先验
        π_macro (簇内 p̃ 均值), q ∝ π_sub^(1−λ) · p̃^λ · π_macro^w_macro;
        S 层先验管空间, 宏簇先验管语义, 乘性独立注入。"""
        flat = mx.maximum(1.0 - like_edge - like_tex, 0.0)
        p = self.soften(mx.stack([like_edge, like_tex, flat], axis=-1))
        h, w = like_edge.shape
        lab = sublabels.reshape(-1)
        n = int(mx.max(sublabels)) + 1
        # 区域先验 π_k: 子区域内 p̃ 的均值 (区域内近似 i.i.d.),
        # 逐通道 scatter-add (同 grouping 的弧长批算)
        pi = self._scatter_mean(p.reshape(-1, 3), lab, n)
        pi_pix = pi[lab].reshape(h, w, 3)
        q = mx.power(pi_pix, 1.0 - self.lam) * mx.power(p, self.lam)
        if macro is not None:
            # 宏簇语义先验: 簇内 p̃ 均值 (同一套 scatter 批算)
            lab_m = macro.reshape(-1)
            m = int(mx.max(macro)) + 1
            pi_m = self._scatter_mean(p.reshape(-1, 3), lab_m, m)
            q = q * mx.power(pi_m[lab_m].reshape(h, w, 3), w_macro)
        q = q / mx.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
        return q, mx.argmax(q, axis=-1).astype(mx.int32)


# ── 子区域跨帧对应 ─────────────────────────────────────────────────


@dataclass(slots=True)
class SubregionTracker:
    """子区域跨帧身份 (flow.md §2 的区域对应, 与 GroupingTracker 的
    链 tid 同族): 质心近邻匹配 → 稳定 rid 图。

    分割层每帧的子区域标签是当帧重排的 (无身份), 场景图节点按
    当帧 id 渲染会随帧错位 (实测); 本器把当帧标签映射为跨帧稳定
    的 rid, 场景图/渲染全部以 rid 为键。"""

    match_radius: float = 12.0  # 对应距离阈值 (px, 应大于帧间位移+松弛)
    app_weight_std: float = 2.0  # 多少 σ 的表观差 ≈ 耗尽几何门半径
    _prev: list[tuple[int, float, float, int, list | None]] = field(
        default_factory=list
    )  # (rid, row, col, area, 表观均值向量|None)
    _next_rid: int = 1

    def run(
        self, sub: mx.array, app: mx.array | None = None
    ) -> tuple[mx.array, dict[int, int], dict[int, int]]:
        """当帧子区域标签 (H,W) → (rid 图 (H,W) int32, {rid: 面积})。
        app: (H,W,C) 表观特征图 (强度/似然通道) —— 给则匹配代价加
        表观项: 几何硬门 (match_radius) 不变, 门内按 几何+表观σ差
        (全局逐像素 σ 归一) 取最小 (prior.md 不变性假设: 表观跨视角
        稳定是"同一物体"判断的根基; 治几何交叉时的身份交换)。"""
        lab = sub.reshape(-1)
        h, w = sub.shape
        n = int(mx.max(sub))
        yy, xx = mx.meshgrid(
            mx.arange(h, dtype=mx.float32),
            mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )

        # 逐区域面积/质心 (scatter 批量)
        def sc(v: mx.array) -> mx.array:
            return mx.zeros((n + 1,)).at[lab].add(v)

        cnt = sc(mx.ones((int(lab.shape[0]),)))
        rs = sc(yy.reshape(-1)) / mx.maximum(cnt, 1.0)
        cs = sc(xx.reshape(-1)) / mx.maximum(cnt, 1.0)
        # 表观签名: 逐区域通道均值, 按通道全局 (逐像素) σ 归一 ——
        # 别用区域间 σ: 表观无区分度时区域间 σ→0 会放大噪声
        # (实测链跟踪退化: 稳定 tid 13→5), 全局 σ 跨帧稳定
        means: list[list[float]] | None = None
        if app is not None:
            c_dim = int(app.shape[2])
            mm = mx.stack(
                [
                    sc(app[:, :, k].reshape(-1)) / mx.maximum(cnt, 1.0)
                    for k in range(c_dim)
                ],
                axis=-1,
            )[1:]  # (n, C)
            sig = mx.std(app.reshape(-1, c_dim), axis=0) + 1e-6
            means = (mm / sig).tolist()
        app_w = (self.match_radius / self.app_weight_std) ** 2
        # 匹配: 当帧区域 → 上帧最近质心 (一对一贪心, 区域数百量级)。
        # 备选身份 (分歧保留): 亚军候选记 alts, 供下游仲裁翻案
        # (不武断硬指派 —— 匹配模糊时下游可用其他证据重审)
        prev = list(self._prev)
        used: set[int] = set()
        remap = [0] * (n + 1)
        alts: dict[int, int] = {}  # 当帧区域标签 → 亚军 rid
        new_prev: list[tuple[int, float, float, int, list | None]] = []
        for r in range(1, n + 1):
            cr, cc, ca = float(rs[r]), float(cs[r]), int(cnt[r])
            am = means[r - 1] if means is not None else None
            best, bi = math.inf, -1
            second, si = math.inf, -1
            for pi, (rid, pr, pc, _, pam) in enumerate(prev):
                d2 = (cr - pr) ** 2 + (cc - pc) ** 2
                # 几何硬门 (match_radius); 剪枝界 = min(second, 门):
                # 几何 ≥ second 的总分必 ≥ second (app≥0)
                if d2 >= min(second, self.match_radius**2) or pi in used:
                    continue
                if am is not None and pam is not None:
                    d2 = d2 + app_w * sum((a - b) ** 2 for a, b in zip(am, pam))
                if d2 < best:
                    second, si = best, bi
                    best, bi = d2, pi
                elif d2 < second:
                    second, si = d2, pi
            if bi >= 0 and best < self.match_radius**2:
                used.add(bi)
                rid = prev[bi][0]
                if si >= 0:
                    alts[r] = prev[si][0]
            else:
                rid = self._next_rid
                self._next_rid += 1
            remap[r] = rid
            new_prev.append((rid, cr, cc, ca, am))
        self._prev = new_prev
        rid_map = mx.array(remap, dtype=mx.int32)[lab].reshape(h, w)
        areas = {rid: a for rid, _, _, a, _ in new_prev}
        return rid_map, areas, alts


# ── 总装门面 ──────────────────────────────────────────────────────


class SegmentResult(NamedTuple):
    """分割层输出 (flow.md §2): 区域层级 + 子区域 + 逐像素软标签。"""

    regions: mx.array  # R 层 cut(τ) 标签 (H,W) int32
    subregions: mx.array  # S 层标签 (H,W) int32 (轮廓硬切后)
    soft: mx.array  # Y 层软标签 (H,W,3): edge/texture/flat
    hard: mx.array  # argmax 硬标签 (H,W) int32
    ucm: mx.array  # 超度量等高线图 (H,W)


def grouping_contours(res) -> tuple[list[mx.array], list]:
    """grouping.GroupingResult → S 层切割输入 (折线列, 圆参数列)。
    blade 不反投影: 折线取 edgel 亚像素点列, 圆取构造时的参数。"""
    polys = [res.edgels.pos[ch] for ch in res.chains]
    return polys, list(res.circle_params)


@dataclass(slots=True)
class SceneSegmenter:
    """分割层门面: enh + 类似然 (+ 可选轮廓/宏簇/深度反馈) → SegmentResult。"""

    tau: float = 0.3  # R 层切割高度
    min_len: float = 10.0  # 轮廓参与切割的最短长度 (px)
    temperature: float = 2.5  # Y 层温度
    lam: float = 0.4  # Y 层像素项权重
    scan_ds: int = 1  # 降采样域分割 (CS 稀疏哲学: 顺序算法对分辨率
    # 平方敏感; 仅 realtime 提速路径用, eval 保持 1 —— 探针先行)

    def run(
        self,
        enh: mx.array,
        like_edge: mx.array,
        like_tex: mx.array,
        polylines: list[mx.array] = (),
        circles: list[tuple[tuple[float, float], float]] = (),
        macro: mx.array | None = None,
        prior_map: mx.array | None = None,
        w_prior: float = 0.5,
        scan_ds: int | None = None,
    ) -> SegmentResult:
        """边界强度 + 类似然 (+ 轮廓/宏簇/深度反馈) → 完整分割结果。

        prior_map: 深度融合层的深度不连续反馈 D (flow.md §2 虚线边,
        第二轮起) —— 以概率或注入边界强度 E' = 1−(1−E)(1−w·D):
        D 提升边界处的高层级概率, 对已有强边界无副作用 (并集语义)。
        """
        if prior_map is not None:
            enh = 1.0 - (1.0 - enh) * (1.0 - w_prior * prior_map)
        if scan_ds is None:
            scan_ds = self.scan_ds
        if scan_ds > 1:
            # 降采样域分割 (CS 测量稀疏哲学: 顺序算法对分辨率平方
            # 敏感, 洪泛/合并/T 结全 ÷ds²; 探针先行, 仅 realtime 路径)
            h, w = enh.shape
            enh_l = enh.reshape(
                h // scan_ds, scan_ds, w // scan_ds, scan_ds
            ).mean(axis=(1, 3))  # 均值池化反混叠 (HS ds 同款)
            rm = RegionLayer().run(enh_l)
            regions_l = rm.hierarchy.cut(self.tau)
            polys_l = [p / float(scan_ds) for p in polylines]
            circ_l = [
                ((cx / scan_ds, cy / scan_ds), rho / scan_ds)
                for (cx, cy), rho in circles
            ]
            mask_l = ContourCut.rasterize(
                enh_l.shape, polys_l, circ_l, self.min_len / scan_ds
            )
            sub_l = SubregionLayer.run(regions_l, mask_l)

            def up(x: mx.array) -> mx.array:
                """最近邻升采样 (区域标签是整数, repeat 保持)。"""
                return mx.repeat(mx.repeat(x, scan_ds, axis=0), scan_ds, axis=1)

            regions = up(regions_l)
            sub = up(sub_l)
            ucm = up(rm.hierarchy.ucm)
            mx.eval(regions, sub, ucm)  # 跨线程物化 (同 scenegraph 教训)
        else:
            rm = RegionLayer().run(enh)
            regions = rm.hierarchy.cut(self.tau)
            mask = ContourCut.rasterize(enh.shape, polylines, circles, self.min_len)
            sub = SubregionLayer.run(regions, mask)
            ucm = rm.hierarchy.ucm
        soft, hard = PixelLabelLayer(self.temperature, self.lam).run(
            like_edge, like_tex, sub, macro
        )
        return SegmentResult(regions, sub, soft, hard, ucm)


if __name__ == "__main__":
    import time

    from utils import Utils

    # ── 合成验证 1: 四象限, 不同强度边界 → 合并顺序与 cut ──────────
    H, W = 96, 128
    E = mx.random.uniform(shape=(H, W), key=mx.random.key(0)) * 0.04  # 区域内部: 低噪声
    E[:, 64] = 0.2  # 左上|右上: 弱边界
    E[48:, 64] = 0.9  # 左下|右下: 强边界
    E[48, :64] = 0.5  # 左上|左下: 中边界
    E[48, 64:] = 0.8  # 右上|右下: 较强边界
    E[48, 64] = 0.9  # 十字点取最强

    t0 = time.perf_counter()
    rm = RegionLayer().run(E)
    t1 = time.perf_counter()
    hier = rm.hierarchy
    print(f"分水岭盆地 {hier.n_basins} 个, 建层级 {1000 * (t1 - t0):.0f}ms")
    # 噪声碎片弧受脊线像素污染被抬高 (上界=脊高, 见模块注释), 故断言
    # 象限归属语义 (对碎片免疫) 而非精确区域数:
    P = {"TL": (24, 32), "TR": (24, 96), "BL": (72, 32), "BR": (72, 96)}

    def rel(tau: float) -> dict[str, int]:
        """τ 处四个象限代表点的区域标签。"""
        lab = hier.cut(tau).tolist()
        return {k: lab[p[0]][p[1]] for k, p in P.items()}

    r = rel(0.19)
    assert len(set(r.values())) == 4, f"τ=0.19 四墙应俱在: {r}"
    r = rel(0.3)
    assert r["TL"] == r["TR"] and len(set(r.values())) == 3, f"τ=0.3: {r}"
    r = rel(0.6)
    ok = r["TL"] == r["TR"] == r["BL"] and r["BR"] != r["TL"]
    assert ok and len(set(r.values())) == 2, f"τ=0.6: {r}"
    r = rel(0.95)
    assert len(set(r.values())) == 1, f"τ=0.95 应全合: {r}"
    print("  合并顺序: 弱墙(0.2) → 中墙(0.5) → 强墙(0.8+) 逐级解除 ✓")
    counts = [hier.n_regions(t) for t in (0.05, 0.15, 0.25, 0.55, 0.95)]
    print(f"  区域数随 τ: {counts} (碎片使绝对数 > 4, 语义不变量为准)")

    # ── 合成验证 2: 边界缺口稀释弧强度 ─────────────────────────────
    E2 = mx.random.uniform(shape=(64, 128), key=mx.random.key(1)) * 0.04
    E2[:, 64] = 0.7  # 完整强边界
    E2[24:40, 64] = 0.02  # 16px 缺口
    hier2 = RegionLayer().run(E2).hierarchy
    E3 = mx.random.uniform(shape=(64, 128), key=mx.random.key(1)) * 0.04
    E3[:, 64] = 0.7  # 对照: 无缺口
    hier3 = RegionLayer().run(E3).hierarchy
    s_gap = max(s for s, _, _ in hier2.merges)
    s_intact = max(s for s, _, _ in hier3.merges)
    assert s_gap < s_intact - 0.03, (
        f"缺口弧强度应被稀释: {s_gap:.2f} vs 完整 {s_intact:.2f}"
    )
    print(f"缺口边界: 弧强度 {s_gap:.2f} < 完整边界 {s_intact:.2f} (稀释生效)")

    # ── S 层验证: 轮廓硬切 + 短链保护 ─────────────────────────────
    regions = hier.cut(0.19)
    cut_line = [mx.array([[0.0, 30.0], [47.0, 30.0]])]  # 纵贯左上象限
    short = [mx.array([[10.0, 10.0], [12.0, 12.0]])]  # 短链 (< min_len)
    mask = ContourCut.rasterize((H, W), cut_line + short, [], min_len=10.0)
    sub = SubregionLayer.run(regions, mask).tolist()
    assert sub[20][10] != sub[20][50], "切割线两侧应分属不同子区域"
    assert sub[20][80] == sub[20][110], "未切割象限内应保持连通"
    assert sub[10][6] == sub[14][14], "短链不应产生切割"
    print("S 层: 轮廓硬切生效, 短链保护生效")

    # ── Y 层验证: 区域先验吸收孤立像素翻转 ─────────────────────────
    HY, WY = 64, 128
    sublab = mx.zeros((HY, WY), dtype=mx.int32)
    sublab[:, :64] = 1
    sublab[:, 64:] = 2
    le = mx.full((HY, WY), 0.1)
    le[:, 64:] = 0.9
    spikes = [(10, 10), (20, 30), (40, 50)]  # 左区孤立高响应
    for y, x in spikes:
        le[y, x] = 0.9
    lt = mx.full((HY, WY), 0.05)
    soft, hard = PixelLabelLayer().run(le, lt, sublab)
    hard_l = hard.tolist()
    assert hard_l[30][30] == 2, "左区多数类应为 flat"
    assert hard_l[30][100] == 0, "右区多数类应为 edge"
    for y, x in spikes:
        assert hard_l[y][x] == 2, f"孤立尖峰 ({y},{x}) 应被区域先验吸收为 flat"
    print("Y 层: 孤立像素翻转被区域先验吸收 (硬标签跟随区域多数)")

    # ── Y 层 × 宏簇先验: 宏簇语义注入与权重 ────────────────────────
    one_region = mx.ones((HY, WY), dtype=mx.int32)  # 单一子区域
    macro_map = mx.zeros((HY, WY), dtype=mx.int32)
    macro_map[:, 64:] = 1  # 左右两个宏簇
    le2 = mx.full((HY, WY), 0.1)
    le2[:, 64:] = 0.9  # 左宏簇多 flat, 右宏簇多 edge
    le2[30, 100] = 0.1  # 右宏簇里的少数派 flat 像素
    le2[30, 30] = 0.9  # 左宏簇里的少数派 edge 像素
    _, hard_m = PixelLabelLayer().run(le2, lt, one_region, macro=macro_map)
    hard_ml = hard_m.tolist()
    assert hard_ml[30][100] == 0, "右宏簇少数派应被宏簇先验判为 edge"
    # 反向 (判回 flat) 需更强权重: 像素证据与 S 先验都偏向 edge 时,
    # w_macro 要 >0.6 才能翻盘 —— 这是先验强度的预期行为, 不是 bug
    _, hard_m2 = PixelLabelLayer().run(
        le2, lt, one_region, macro=macro_map, w_macro=0.8
    )
    assert hard_m2.tolist()[30][30] == 2, "强宏簇先验下左区少数派应判回 flat"
    print("Y 层×宏簇: 语义先验生效 (默认权重翻转弱证据侧, w=0.8 翻转强证据侧)")

    # ── 深度反馈钩子 (prior_map): 填补缺口边界 ─────────────────────
    # E2 的 16px 缺口边界: τ=0.5 时两区已合并; 注入 D 填补缺口后
    # 弧强度恢复, 两区应保持分离
    zero_like = mx.zeros((64, 128))
    seg_no = SceneSegmenter(tau=0.5).run(E2, zero_like, zero_like)
    D = mx.zeros((64, 128))
    D[24:40, 64] = 1.0  # 深度不连续恰好覆盖缺口
    seg_fb = SceneSegmenter(tau=0.5).run(
        E2, zero_like, zero_like, prior_map=D, w_prior=0.8
    )
    pt_l, pt_r = (32, 32), (32, 96)
    same_no = int(seg_no.regions[pt_l]) == int(seg_no.regions[pt_r])
    same_fb = int(seg_fb.regions[pt_l]) == int(seg_fb.regions[pt_r])
    assert same_no, "缺口弧 0.36 < τ=0.5, 无反馈应合并"
    assert not same_fb, "深度反馈填补缺口后两区应保持分离"
    print("深度反馈: prior_map 填补缺口边界, 区域分离保持 (虚线边接通)")

    # ── grouping → segment 接线验证: 轮廓硬切恢复欠分割区域 ──────
    # T 形轮廓场: 分水岭在 τ=1 全并为一个区域, S 层轮廓应切出
    # 横杠/竖杠/背景 三个子区域 (轮廓即 grouping 折线的角色)
    Et = mx.full((128, 128), 0.02)
    Et[40, 16:113] = 0.9  # 横杠上沿
    Et[56, 16:113] = 0.9  # 横杠下沿 (贯穿竖杠 → T 同构)
    Et[40:57, 16] = 0.9
    Et[40:57, 112] = 0.9
    Et[56:97, 60] = 0.9  # 竖杠两侧
    Et[56:97, 76] = 0.9
    Et[96, 60:77] = 0.9
    polys = [
        mx.array([[40.0, 16.0], [40.0, 112.0]]),
        mx.array([[56.0, 16.0], [56.0, 112.0]]),
        mx.array([[40.0, 16.0], [56.0, 16.0]]),
        mx.array([[40.0, 112.0], [56.0, 112.0]]),
        mx.array([[56.0, 60.0], [96.0, 60.0], [96.0, 76.0], [56.0, 76.0]]),
    ]
    hier_t = RegionLayer().run(Et).hierarchy
    one = hier_t.cut(1.0)
    assert int(mx.max(one)) == 1, "τ=1 应并为单区域"
    mask_t = ContourCut.rasterize((128, 128), polys, [], min_len=10.0)
    sub_t = SubregionLayer.run(one, mask_t).tolist()
    pts = {"横杠": (48, 64), "竖杠": (70, 68), "背景": (80, 30)}
    got = {k: sub_t[p[0]][p[1]] for k, p in pts.items()}
    assert len(set(got.values())) == 3, f"轮廓应切出三个子区域: {got}"
    print(f"grouping→segment: τ=1 单区域被轮廓切出 {got} 三个子区域")

    # ── T 图像全管线接线: riesz→vbgmm→edgemap→grouping→segment ─────
    from edgemap import EdgePrior
    from grouping import PerceptualGrouping
    from riesz import RieszWavelet
    from vbgmm import VBGMM

    img_t = mx.full((128, 128), 0.2)
    img_t[40:56, 16:112] = 0.7  # 横杠
    img_t[56:96, 60:76] = 0.7  # 竖杠 (顶边没入横杠 → T)
    img_t = img_t + mx.random.normal((128, 128), key=mx.random.key(5)) * 0.01

    t0 = time.perf_counter()
    rw_t = RieszWavelet(img_t)
    feat_t = rw_t.features()
    gm_t = VBGMM(VBGMM.feature_matrix(feat_t), k_max=48)
    enh_t = EdgePrior().enhance(gm_t.edge_likelihood((128, 128)), feat_t, rw_t)
    gres = PerceptualGrouping().run(enh_t, feat_t.mean_ori)
    polys, circs = grouping_contours(gres)
    seg_t = SceneSegmenter(tau=0.5).run(
        enh_t,
        gm_t.edge_likelihood((128, 128)),
        gm_t.class_likelihood("texture").reshape((128, 128)),
        polys,
        circs,
    )
    t1 = time.perf_counter()
    print(
        f"T 图像接线: grouping 链 {len(gres.chains)} 条, T 结 "
        f"{len(gres.t_junctions)} 个 → 区域 {int(mx.max(seg_t.regions))} → "
        f"子区域 {int(mx.max(seg_t.subregions))}, 全链路 {t1 - t0:.1f}s"
    )

    # ── 12.png 冒烟: 无轮廓 (离线层, 大图轮廓见 T 图像验证) ────────
    from PIL import Image

    from color import Color

    im = Image.open(Utils.project_root() / "images/12.png").convert("L")
    arr = Color.image_to_mlx(im)
    rw = RieszWavelet(arr)
    feat = rw.features()
    gm = VBGMM(VBGMM.feature_matrix(feat), k_max=48)
    enh = EdgePrior().enhance(gm.edge_likelihood(arr.shape), feat, rw)

    t0 = time.perf_counter()
    seg = SceneSegmenter(tau=0.3).run(
        enh,
        gm.edge_likelihood(arr.shape),
        gm.class_likelihood("texture").reshape(arr.shape),
    )
    t1 = time.perf_counter()
    print(f"12.png {arr.shape}: segment (无轮廓) {t1 - t0:.1f}s")

    # 可视化: 原图 / enh / UCM / cut(0.3) 区域 / Y 层软标签 (RGB)
    import matplotlib.pyplot as plt

    fig = Utils.visualize(
        [
            ("original", "gray", arr),
            ("enh (E)", "gray", enh),
            ("UCM", "gray", seg.ucm),
            ("regions cut(0.3)", "tab20", seg.regions.astype(mx.float32)),
            ("soft labels (R=edge G=tex B=flat)", None, seg.soft),
        ]
    )
    out = Utils.project_root() / "artifacts/segment_12.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)

    # ── 表观不变性匹配: 几何交叉时的身份保持 (prior.md 不变性假设) ──
    # 两条窄带对向移动 ±10px (中心距 16, 双方候选都在 12px 门内):
    # 纯几何必然交换身份; 表观 (亮/暗) 应拉回正确对应
    subA = mx.zeros((32, 96), dtype=mx.int32)
    subA[:, 46:50] = 1  # A.r1 亮, 质心 col=48
    subA[:, 62:66] = 2  # A.r2 暗, 质心 col=64
    appA = mx.zeros((32, 96, 1))
    appA = appA.at[:, 46:50, 0].add(0.9)
    appA = appA.at[:, 62:66, 0].add(0.1)
    subB = mx.zeros((32, 96), dtype=mx.int32)
    subB[:, 56:60] = 1  # B.r1 亮, 质心 col=58 (右移 10)
    subB[:, 52:56] = 2  # B.r2 暗, 质心 col=54 (左移 10)
    appB = mx.zeros((32, 96, 1))
    appB = appB.at[:, 56:60, 0].add(0.9)  # B.r1 亮 (=A.r1 的表观)
    appB = appB.at[:, 52:56, 0].add(0.1)  # B.r2 暗 (=A.r2)
    tk = SubregionTracker()
    ridA, _, _ = tk.run(subA, app=appA)
    ridB, _, altsB = tk.run(subB, app=appB)
    # 纯几何: B.r1(col58) 更近 A.r2(col64); 表观应纠正为
    # B.r1(亮) ↔ A.r1(亮), B.r2(暗) ↔ A.r2(暗)
    r1_A, r2_A = int(ridA[16, 48]), int(ridA[16, 64])
    r1_B, r2_B = int(ridB[16, 58]), int(ridB[16, 54])
    assert r1_B == r1_A, f"亮区身份应跨帧保持: {r1_B} vs {r1_A}"
    assert r2_B == r2_A, f"暗区身份应跨帧保持: {r2_B} vs {r2_A}"
    # 备选身份 (分歧保留): B.r1 的亚军应是 A.r2 (几何上更近的那个)
    assert altsB.get(1) == r2_A, f"亚军候选: {altsB}"
    # 对照: 无表观时几何交叉会交换 (验证测试场景本身有鉴别力)
    tk2 = SubregionTracker()
    tk2.run(subA)
    ridB2, _, _ = tk2.run(subB)
    assert int(ridB2[16, 58]) != r1_A or int(ridB2[16, 54]) != r2_A, (
        "无表观对照应发生交换/断链"
    )
    print("表观匹配: 几何交叉下亮/暗区身份保持 (对照组交换) ✓")
