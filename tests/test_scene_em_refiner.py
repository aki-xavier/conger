"""SceneEMRefiner: 几何↔光照 ECM 精炼的黑盒测试。"""

import mlx.core as mx

from codebook import Codebook
from generic_em import EMLoop
from inverse_config import InverseConfig
from scene_em_refiner import SceneEMRefiner


def _err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(mx.sqrt(mx.mean((mx.array(a) - mx.array(b)) ** 2)))


def _render_true(
    cb: Codebook, renderer, cam_l, cam_r, true_geom: tuple[float, ...]
):
    """真值 (kind, u, v, s, z, hue, lcol, ldir) → 左右帧。"""
    true_scene_params = (0.0, *true_geom, 2.0, 0.0, 1.0)
    scene = cb.to_scene(true_scene_params)
    return renderer.render(scene, cam_l), renderer.render(scene, cam_r)


def test_scene_em_refines_geometry_toward_truth() -> None:
    """从扰动几何出发, ECM 应把几何拉向真值 (默认 u/v 主导)。"""
    cb = Codebook(InverseConfig())
    renderer, cam_l, cam_r = Codebook.make_renderer()
    true_geom = (72.0, 72.0, 0.45, 3.2)
    fl, fr = _render_true(cb, renderer, cam_l, cam_r, true_geom)

    init = (68.0, 76.0, 0.52, 3.0)  # 扰动初始几何
    refiner = SceneEMRefiner(cb, 0, fl, fr, appearance_topk=3)
    loop = EMLoop(refiner, max_iters=4, tol=0.0)  # 固定 4 轮
    result = loop.run((fl, fr), init)

    got = result.params
    assert _err(got, true_geom) < _err(init, true_geom), (
        f"ECM 未拉近几何: init 误差 {_err(init, true_geom):.3f}, "
        f"got 误差 {_err(got, true_geom):.3f}"
    )


def test_scene_em_freezes_sz_by_default() -> None:
    """默认 freeze=(F,F,T,T): u/v 拉近真值, s/z 保持初值不动。"""
    cb = Codebook(InverseConfig())
    renderer, cam_l, cam_r = Codebook.make_renderer()
    true_geom = (72.0, 72.0, 0.45, 3.2)
    fl, fr = _render_true(cb, renderer, cam_l, cam_r, true_geom)

    init = (68.0, 76.0, 0.52, 3.0)
    refiner = SceneEMRefiner(cb, 0, fl, fr, appearance_topk=3)
    loop = EMLoop(refiner, max_iters=4, tol=0.0)
    result = loop.run((fl, fr), init)

    got = result.params
    assert got[2] == init[2] and got[3] == init[3], (
        f"s/z 应冻结: got={got} init={init}"
    )
    uv_err = float(mx.sqrt((got[0] - true_geom[0]) ** 2 + (got[1] - true_geom[1]) ** 2))
    init_uv_err = float(
        mx.sqrt((init[0] - true_geom[0]) ** 2 + (init[1] - true_geom[1]) ** 2)
    )
    assert uv_err < init_uv_err, f"u/v 应拉近: {uv_err:.3f} vs {init_uv_err:.3f}"


class _QuadraticRefiner(SceneEMRefiner):
    """合成二次残差 + 单一外观: 几何极小在 target, 不渲染、确定性。"""

    def __init__(self, freeze: tuple[bool, bool, bool, bool]):
        super().__init__(
            Codebook(InverseConfig()),
            0,
            mx.zeros((8, 8, 3)),
            mx.zeros((8, 8, 3)),
            appearance_topk=1,
            freeze=freeze,
        )
        self._appearances = [(0, 0, 0)]
        self.target = (10.0, 20.0, 0.5, 1.0)

    def _residual(self, geometry: tuple[float, ...], appearance) -> float:
        return sum((g - t) ** 2 for g, t in zip(geometry, self.target, strict=True))


def test_scene_em_maximize_skips_frozen_dims() -> None:
    """maximize 冻结维不做坐标搜索 (合成二次残差, 确定性)。"""
    init = (0.0, 0.0, 0.3, 0.7)
    refiner = _QuadraticRefiner(freeze=(False, False, True, True))
    q = refiner.responsibilities(init, None)
    got = refiner.maximize(q, None, init)
    assert got[2] == 0.3 and got[3] == 0.7, f"s/z 应冻结: {got}"
    assert got[0] > 0.0 and got[1] > 0.0, f"u/v 应移动: {got}"

    ref_full = _QuadraticRefiner(freeze=(False, False, False, False))
    qf = ref_full.responsibilities(init, None)
    gotf = ref_full.maximize(qf, None, init)
    assert gotf[2] != 0.3, f"四维全搜下 s 应参与搜索: {gotf}"
