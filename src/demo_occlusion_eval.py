"""遮挡序数指派评估 (iBims-1, 项目最独特机制的实数据检验)。

链路: run_ours → PerceptualGrouping (T 结) → constraints_from_grouping
(序数约束) → 对 GT 深度验证。iBims 深度跳变派生 GT 遮挡边界与
前/后序 (跳变近侧 = front)。

指标 (README 协议"遮挡序数"通道落地):
  a. 约束检出密度: 每场景产生的序数约束数 / GT 遮挡边界像素 (稀疏
     度量化 —— demo_occlusion 单图实测检出率 3%);
  b. 序正确率: 每个约束在其位置, front 区域 GT 深度 < behind 区域
     GT 深度 的比例 (违序率口径, demo_occlusion 同款)。

用法: PYTHONPATH=src .venv/bin/python3 src/demo_occlusion_eval.py [样本数]
"""

import sys
import time

import mlx.core as mx
import numpy as np
import scipy.io


def main(n_scenes: int = 6) -> None:
    import pathlib

    from color import Color
    from edgemap import EdgePrior
    from fusion import OcclusionOrder
    from grouping import PerceptualGrouping
    from riesz import RieszWavelet
    from segment import SceneSegmenter, grouping_contours
    from vbgmm import VBGMM

    mats = sorted(pathlib.Path(
        "/tmp/datasets/ibims1/ibims1_core_mat").glob("*.mat"))
    step = max(1, len(mats) // n_scenes)
    mats = mats[::step]
    print(f"遮挡序数评估: iBims-1 子集 {len(mats)} 场景")

    rows = []
    t0 = time.perf_counter()
    for i, mp in enumerate(mats):
        m = scipy.io.loadmat(str(mp))["data"][0, 0]
        rgb = m["rgb"].astype(np.uint8)
        depth = m["depth"].astype(np.float64)
        valid = (m["mask_transp"].astype(bool)) & \
            (m["mask_invalid"].astype(bool)) & (depth > 0)
        H, W = rgb.shape[:2]
        # 全链路 (demo_occlusion 同款): 真实 like/tex 进分割
        lum, hs = Color.split_dual_path(
            mx.array(rgb.astype(np.float32) / 255.0)
        )
        rw = RieszWavelet(lum)
        feat = rw.features()
        gm_l = VBGMM.fast_fit(
            VBGMM.feature_matrix(feat), (H, W), k_max=48, coreset=8192
        )
        like_l = gm_l.edge_likelihood((H, W))
        tex_l = gm_l.class_likelihood("texture").reshape(H, W)
        like_h = mx.zeros((H, W))
        like = 1 - (1 - like_l) * (1 - like_h)
        tex = tex_l
        enh = EdgePrior().enhance(like, feat, rw)
        res = PerceptualGrouping().run(enh, feat.mean_ori)
        polys, circs = grouping_contours(res)
        seg = SceneSegmenter(tau=0.3).run(enh, like, tex, polys, circs)
        sub = seg.subregions
        # 约束 = (pos, front_rid, behind_rid) —— front/behind 是区域 id
        cons = OcclusionOrder.constraints_from_grouping(res, sub)
        n_tj = len(res.t_junctions)
        # E4: 假 T 结率 —— T 结位置距 GT 深度跳变 ≤5px 的比例
        d = np.log(np.maximum(depth, 1e-3))
        thr = np.log(1.15)
        gtb = np.zeros(depth.shape, dtype=bool)
        gtb[:, 1:] |= np.abs(np.diff(d, axis=1)) > thr
        gtb[1:, :] |= np.abs(np.diff(d, axis=0)) > thr
        gtb &= valid
        from scipy import ndimage

        gtb_d = ndimage.binary_dilation(gtb, iterations=5)
        n_real = sum(
            1 for t in res.t_junctions
            if int(round(t.pos[0])) < H and int(round(t.pos[1])) < W
            and gtb_d[int(round(t.pos[0])), int(round(t.pos[1]))]
        )
        n_tj_real = n_real
        # 序正确率 + 区域大小分档 (E5 判别: 区域碎片假设)
        ok, tot, ok_big, tot_big, ok_sm, tot_sm = 0, 0, 0, 0, 0, 0
        ok_real, tot_real = 0, 0  # E6: 约束位置在深度跳变邻域的序正确率
        ok_loc, tot_loc = 0, 0  # E7: 结点处 ±off 的 GT 局部深度序
        # 局部采样: 复制映射几何 (through 链最近 edgel 法向, 定向到终止链中部落位侧)
        pos, normal = res.edgels.pos, res.edgels.normal
        by_pos = {t.pos: t for t in res.t_junctions}
        for cn in cons:
            t = by_pos.get(cn.pos)
            if t is None:
                continue
            ch_f = res.chains[t.front]
            ptsf = pos[ch_f]
            jr, jc = cn.pos
            ii = int(mx.argmin((ptsf[:, 0] - jr) ** 2 + (ptsf[:, 1] - jc) ** 2))
            nidx = int(ch_f[ii])
            nr, nc2 = float(normal[nidx, 0]), float(normal[nidx, 1])
            ch_b = res.chains[t.behind]
            midx = int(ch_b[len(ch_b) // 2])
            if nr * (float(pos[midx, 0]) - jr) + nc2 * (float(pos[midx, 1]) - jc) < 0:
                nr, nc2 = -nr, -nc2
            pr, pc = float(ptsf[ii, 0]), float(ptsf[ii, 1])
            # 局部 GT 深度: ±3px 小补丁中位 (区域无关)
            def _patch(r0, c0):
                r0, c0 = int(round(r0)), int(round(c0))
                r0 = min(max(r0, 0), H - 1)
                c0 = min(max(c0, 0), W - 1)
                r1, c1 = min(r0 + 3, H), min(c0 + 3, W)
                return np.median(depth[r0:r1, c0:c1])
            z_loc_f = _patch(pr - 3 * nr, pc - 3 * nc2)
            z_loc_b = _patch(pr + 3 * nr, pc + 3 * nc2)
            if np.isfinite(z_loc_f) and np.isfinite(z_loc_b):
                tot_loc += 1
                ok_loc += 1 if z_loc_f < z_loc_b else 0
            fr, be = cn.front, cn.behind
            cu, cv = int(round(cn.pos[1])), int(round(cn.pos[0]))
            at_real = (0 <= cv < H and 0 <= cu < W
                       and gtb_d[cv, cu])
            fr, be = cn.front, cn.behind
            m_f = (sub == fr).reshape(-1) & valid.reshape(-1)
            m_b = (sub == be).reshape(-1) & valid.reshape(-1)
            nf, nb = int(m_f.sum()), int(m_b.sum())
            if nf < 10 or nb < 10:
                continue
            zf = np.median(depth.reshape(-1)[m_f])
            zb = np.median(depth.reshape(-1)[m_b])
            corr = 1 if zf < zb else 0
            tot += 1
            ok += corr
            if at_real:
                tot_real += 1
                ok_real += corr
            if min(nf, nb) >= 500:  # 大区域: 真实表面
                tot_big += 1
                ok_big += corr
            else:
                tot_sm += 1
                ok_sm += corr
        rate = ok / tot if tot else float("nan")
        rows.append((mp.stem, n_tj, len(cons), tot, rate,
                     ok_big, tot_big, ok_sm, tot_sm, n_tj_real,
                     ok_real, tot_real, ok_loc, tot_loc))
        print(f"[{i + 1}/{len(mats)}] {mp.stem}: T 结 {n_tj}, "
              f"约束 {len(cons)}, 可验 {tot}, 序正确率 {rate:.2f} "
              f"({(time.perf_counter() - t0) / 60:.1f}min)")

    tot_c, tot_v, tot_ok = sum(r[2] for r in rows), sum(r[3] for r in rows), \
        sum(int(r[4] * r[3]) for r in rows if r[4] == r[4])
    print(f"\n== 汇总 ({len(rows)} 场景) ==")
    n_tj = sum(r[1] for r in rows)
    print(f"  T 结总数: {n_tj} (均 {n_tj / len(rows):.0f}/场景)")
    print(f"  序数约束总数: {tot_c} (均 {tot_c / len(rows):.1f}/场景, "
          f"映射率 {100 * tot_c / max(sum(r[1] for r in rows), 1):.1f}%)")
    print(f"  可验证约束: {tot_v}, 序正确率: {tot_ok / max(tot_v, 1):.3f}")
    ob, tb = sum(r[5] for r in rows), sum(r[6] for r in rows)
    os_, ts = sum(r[7] for r in rows), sum(r[8] for r in rows)
    print(f"  E5 分档: 大区域 (min≥500px) {tb} 约束, 序正确率 "
          f"{ob / max(tb, 1):.3f}; 小区域 {ts} 约束, "
          f"{os_ / max(ts, 1):.3f}")
    n_real = sum(r[9] for r in rows)
    print(f"  E4 假 T 结率: {n_real}/{n_tj} 落在深度跳变邻域 "
          f"({100 * n_real / max(n_tj, 1):.1f}%)")
    or_, tr_ = sum(r[10] for r in rows), sum(r[11] for r in rows)
    print(f"  E6 真 T 结约束: {tr_} 个在深度跳变邻域, 序正确率 "
          f"{or_ / max(tr_, 1):.3f}")
    ol, tl = sum(r[12] for r in rows), sum(r[13] for r in rows)
    print(f"  E7 局部深度序 (结点±3px, 区域无关): {tl} 个, 正确率 "
          f"{ol / max(tl, 1):.3f}")
    print("  (对照: demo_occlusion 单图映射成功率 ~1-3%; 序正确率是"
          "机制正确性的度量, 检出稀疏是另一问题)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
