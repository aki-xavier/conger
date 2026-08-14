"""StereoLayers: 遮挡感知逐层双目几何。

左图逐像素水平块匹配 → disparity map + 匹配置信度 → (x,y,视差) 2-means
分为前/后两层; 后层再做形状模板轮廓补全 → 每层 (u,v,z,area)。
"""

from __future__ import annotations

import mlx.core as mx

from codebook import Codebook
from contour_completion import ContourCompleter
from feature_extractor import FeatureExtractor
from joint_layer_optimizer import JointLayerOptimizer
from stereo import StereoDepth
from utils import Utils


class StereoLayers:
    """平行双目 RGB 帧对 → 前/后层几何统计。"""

    D_RANGE = (5, 12)  # z∈[2.3,4.3] 的物理视差范围 (d=FX·B/zc)
    PATCH_R = 4  # 9×9: 平滑图元上比 5×5 更抗跨层误配
    MIN_CONF = 1.08  # second-best/best 代价下限, 平面弱纹理抑制

    @staticmethod
    def _centroid(weights: mx.array) -> tuple[float, float, float]:
        """权重图 → (u,v,有效面积)。"""
        ys = mx.arange(weights.shape[0], dtype=mx.float32)[:, None]
        xs = mx.arange(weights.shape[1], dtype=mx.float32)[None, :]
        total = float(mx.sum(weights))
        if total <= 1e-8:
            c = (Codebook.W - 1) / 2.0
            return c, c, 0.0
        u = float(mx.sum(weights * xs) / total)
        v = float(mx.sum(weights * ys) / total)
        return u, v, total

    @classmethod
    def disparity_map(
        cls, fl: mx.array, fr: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """逐像素块匹配 → (视差, 置信度, 有效掩码)。

        只在左右图同时可比较的 x 范围内有效; 代价为 5×5 RGB SSD。"""
        rgb_l = fl[..., :3].astype(mx.float32) / 255.0
        rgb_r = fr[..., :3].astype(mx.float32) / 255.0
        def features(rgb: mx.array, frame: mx.array) -> mx.array:
            lum = FeatureExtractor.frame_lum(frame)
            re, im = FeatureExtractor.frame_chroma(frame)
            gx = mx.roll(lum, -1, axis=1) - lum
            gy = mx.roll(lum, -1, axis=0) - lum
            return mx.concatenate(
                [rgb, re[..., None], im[..., None],
                 (5.0 * gx)[..., None], (5.0 * gy)[..., None]],
                axis=2,
            )
        rgb_l = features(rgb_l, fl)
        rgb_r = features(rgb_r, fr)
        h, w = rgb_l.shape[:2]
        d0, d1 = cls.D_RANGE
        r = cls.PATCH_R
        costs = []
        x0, x1 = d1 + r, w - r
        for d in range(d0, d1 + 1):
            c = mx.zeros((h - 2 * r, x1 - x0), dtype=mx.float32)
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    lp = rgb_l[dy : h - 2 * r + dy, x0 - r + dx : x1 - r + dx]
                    rp = rgb_r[dy : h - 2 * r + dy, x0 - r - d + dx : x1 - r - d + dx]
                    c = c + mx.sum((lp - rp) ** 2, axis=2)
            costs.append(c)
        cost = mx.stack(costs, axis=2)  # (h', w', D)
        order = mx.sort(cost, axis=2)
        best = order[:, :, 0]
        second = order[:, :, 1]
        conf = second / mx.maximum(best, 1e-6)
        disp = mx.arange(d0, d1 + 1, dtype=mx.float32)[mx.argmin(cost, axis=2)]
        full_d = mx.zeros((h, w), dtype=mx.float32)
        full_conf = mx.zeros((h, w), dtype=mx.float32)
        valid = mx.zeros((h, w), dtype=mx.bool_)
        full_d[r : h - r, d1 + r : w - r] = disp
        full_conf[r : h - r, d1 + r : w - r] = conf
        valid[r : h - r, d1 + r : w - r] = conf > cls.MIN_CONF
        return full_d, full_conf, valid

    @classmethod
    def _cluster_layers(
        cls, disp: mx.array, fw: mx.array, valid: mx.array
    ) -> tuple[mx.array, tuple[float, ...], tuple[float, ...]] | None:
        """(x,y,disparity) 加权 2-means → (前层掩码, 前中心, 后中心)。"""
        idx = Utils.nonzero(valid.reshape(-1))
        if idx.shape[0] < 32:
            return None
        h, w = disp.shape
        xs = (idx % w).astype(mx.float32)
        ys = (idx // w).astype(mx.float32)
        ds = disp.reshape(-1)[idx]
        ws = fw.reshape(-1)[idx]
        ds_sorted = mx.sort(ds)
        n = ds_sorted.shape[0]
        d_lo = float(ds_sorted[n // 4])
        d_hi = float(ds_sorted[(3 * n) // 4])
        cx = float(mx.sum(xs * ws) / mx.sum(ws))
        cy = float(mx.sum(ys * ws) / mx.sum(ws))
        c_lo = [cx, cy, d_lo]
        c_hi = [cx, cy, d_hi]
        for _ in range(16):
            dl = (
                ((xs - c_lo[0]) / 40.0) ** 2
                + ((ys - c_lo[1]) / 40.0) ** 2
                + ((ds - c_lo[2]) / 4.0) ** 2
            )
            dh = (
                ((xs - c_hi[0]) / 40.0) ** 2
                + ((ys - c_hi[1]) / 40.0) ** 2
                + ((ds - c_hi[2]) / 4.0) ** 2
            )
            near_hi = dh < dl
            for c, m in ((c_lo, ~near_hi), (c_hi, near_hi)):
                wm = mx.where(m, ws, 0.0)
                tot = float(mx.sum(wm))
                if tot > 1e-8:
                    c[0] = float(mx.sum(wm * xs) / tot)
                    c[1] = float(mx.sum(wm * ys) / tot)
                    c[2] = float(mx.sum(wm * ds) / tot)
        if c_hi[2] - c_lo[2] < 0.5:
            return None
        yy, xx = mx.meshgrid(
            mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )
        dl = (
            ((xx - c_lo[0]) / 40.0) ** 2
            + ((yy - c_lo[1]) / 40.0) ** 2
            + ((disp - c_lo[2]) / 4.0) ** 2
        )
        dh = (
            ((xx - c_hi[0]) / 40.0) ** 2
            + ((yy - c_hi[1]) / 40.0) ** 2
            + ((disp - c_hi[2]) / 4.0) ** 2
        )
        front = valid & (dh < dl)
        return front, tuple(c_hi), tuple(c_lo)

    @classmethod
    def estimate(cls, fl: mx.array, fr: mx.array) -> tuple[float, ...]:
        """→ (u0,v0,z0,area0,u1,v1,z1,area1), 0=前层/大视差。"""
        fw = StereoDepth.foreground_weights(fl)
        fg = fw > 0.01
        disp, _, valid = cls.disparity_map(fl, fr)
        valid = valid & fg
        clustered = cls._cluster_layers(disp, fw, valid)
        if clustered is None:
            return cls._fallback(fl, fr)
        front, (_, _, d_front), (_, _, d_back) = clustered
        back = valid & ~front
        u0, v0, area0 = cls._centroid(fw * front.astype(mx.float32))
        u1, v1, area1 = cls._centroid(fw * back.astype(mx.float32))
        # 后层轮廓补全: 完整模板 − 前层遮挡 ↔ 观测可见区域
        cu, cv, c_area, _, c_score = ContourCompleter.complete(front, back)
        if c_area > 0.0:
            # soft fusion: 轮廓残差越低权重越高; 训练集测得补全面积
            # 中位膨胀约 1.5×, 因此先按 2/3 收缩 (sl4)
            w = min(max((0.30 - c_score) / 0.25, 0.0), 1.0)
            c_area *= 2.0 / 3.0
            u1 = (1.0 - w) * u1 + w * cu
            v1 = (1.0 - w) * v1 + w * cv
            area1 = (1.0 - w) * area1 + w * c_area
        z0 = Codebook.CAM_Z - Codebook.FX * Codebook.STEREO_BASE / d_front
        z1 = Codebook.CAM_Z - Codebook.FX * Codebook.STEREO_BASE / d_back
        joint = JointLayerOptimizer.optimize(
            fg, disp, valid, front, back, d_front, d_back
        )
        if joint is not None:
            # 联合模板负责中心/深度; 面积仍走可见区+补全 soft fusion,
            # 因为模板尺度在错误聚类下会膨胀 (实测 s R² 大降)
            return (joint[0], joint[1], joint[2], area0,
                    joint[4], joint[5], joint[6], area1)
        return (u0, v0, z0, area0, u1, v1, z1, area1)

    @staticmethod
    def _fallback(fl: mx.array, fr: mx.array) -> tuple[float, ...]:
        """不可分/弱纹理时退化为同一全局质心的两层占位。"""
        fw = StereoDepth.foreground_weights(fl)
        u, v, area = StereoLayers._centroid(fw)
        z, _, _ = StereoDepth().estimate(fl, fr)
        return (u, v, z, area, u, v, z, area)

    @staticmethod
    def scaled(stats: mx.array) -> mx.array:
        """(N,8) 原始层统计 → 与全分辨率特征方差量级兼容的拼接维。"""
        out = []
        for off in (0, 4):
            out.extend(
                [
                    stats[:, off] / Codebook.W,
                    stats[:, off + 1] / Codebook.H,
                    stats[:, off + 2],
                    stats[:, off + 3] / 1000.0,
                ]
            )
        return mx.stack(out, axis=1)
