"""Priors: 码先验工厂 (外部知识注入, 对应 docs/prior.md 先验体系)。"""

from __future__ import annotations

import math

import mlx.core as mx
from cga.engine import AmbientLight, Color, DirectionalLight, Scene

from codebook import Codebook
from demo_config import DemoConfig
from feature_extractor import FeatureExtractor


class Priors:
    """码先验工厂 (外部知识注入, 对应 docs/prior.md 先验体系)。"""

    def __init__(self, cfg: DemoConfig, codebook: Codebook):
        self.cfg = cfg
        self.codebook = codebook

    def build(self, name: str) -> mx.array | None:
        """码先验 log P(c)。name 可逗号组合 (如 "edge,familiar"):
        各先验 log 相加 (= 概率相乘)。
          flat: 均匀先验 (None, 纯数据似然);
          edge: 一般视角 —— 图元中心不该贴图像边缘;
          familiar: 熟悉尺寸 —— 大尺寸更常见 (0.7/0.3);
          occlusion: 遮挡序数 (per-sample, 由 occlusion() 逐帧构造)。
        log 权重在 posterior 内 softmax 归一。"""
        names = [n.strip() for n in name.split(",")]
        if "occlusion" in names:
            # per-sample 先验, 与全局 (K,) 先验形状不兼容 → 不可组合
            if len(names) > 1:
                raise ValueError("occlusion 先验不可与其他先验组合 (per-sample)")
            return None
        if names == ["flat"]:
            return None
        cb = self.codebook
        w = mx.ones(cb.N_CODES)
        for n in names:
            if n == "flat":
                continue
            for i in range(cb.N_CODES):
                _, gx, gy, size, _ = cb.idx_to_code(i)
                if n == "edge":
                    if gx in (0, cb.N_GX - 1) or gy in (0, cb.N_GY - 1):
                        w[i] *= 0.3
                elif n == "familiar":
                    w[i] *= 0.7 if size == 1 else 0.3
                else:
                    raise ValueError(f"未知先验: {n}")
        return mx.log(w)

    def occlusion(self, frames: list[mx.array]) -> mx.array:
        """遮挡序数先验 (per-sample, N_CODES 每帧): 黄柱面积缺失
        (< 0.85·F0) ⟹ 主图元遮住黄柱 ⟹ 主不比遮挡物后 ⟹ 排除 z=4.0
        (其余档中性) —— 遮挡逻辑 (prior.md 物理先验): A 遮 B ⟹ A 在前。
        注意: z=3.5 同深时主图元先渲染 (z-buffer 严格 <) 也遮黄, 故不排除。"""
        f0 = self.occluder_f0()
        cb = self.codebook
        lp = mx.zeros((len(frames), cb.N_CODES))
        for i, fr in enumerate(frames):
            if self.yellow_area(fr) < 0.85 * f0:
                for j in range(cb.N_CODES):
                    if cb.idx_to_code(j)[4] == 3:  # z=4.0: 主在后, 不可能遮黄
                        lp[i, j] = math.log(0.1)
        return lp

    @staticmethod
    def yellow_area(frame: mx.array) -> float:
        """黄色遮挡物像素数 (色相阈值: 黄 H≈0.12, S>0.4)。"""
        h, s = FeatureExtractor.frame_hs(frame)
        mask = (s > 0.4) & (h > 0.07) & (h < 0.18)
        return float(mx.sum(mask))

    def occluder_f0(self) -> float:
        """黄柱无遮挡时的黄色像素数 (固定值, 离线预计算)。"""
        renderer, cam = Codebook.make_renderer()
        scene = Scene(background=Color(self.cfg.bg_color))
        scene.add(AmbientLight(Color(0xFFFFFF), 0.5))
        scene.add(
            DirectionalLight(Color(0xFFFFFF), 0.7, direction=Codebook.LIGHT_DIRS[0])
        )
        scene.add(self.codebook.occluder())
        return self.yellow_area(renderer.render(scene, cam))
