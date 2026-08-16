"""SceneEMRefiner: 几何↔光照 ECM 精炼的黑盒测试。"""

import mlx.core as mx

from codebook import Codebook
from generic_em import EMLoop
from inverse_config import InverseConfig
from scene_em_refiner import SceneEMRefiner


def _err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(mx.sqrt(mx.mean((mx.array(a) - mx.array(b)) ** 2)))


def test_scene_em_refines_geometry_toward_truth() -> None:
    """从扰动几何出发, ECM 应把 (u,v,s,z) 拉向真值。"""
    cb = Codebook(InverseConfig())
    renderer, cam_l, cam_r = Codebook.make_renderer()
    # 真值 (kind, u, v, s, z, hue, lcol, ldir)
    true_geom = (72.0, 72.0, 0.45, 3.2)
    true_scene_params = (0.0, *true_geom, 2.0, 0.0, 1.0)
    scene = cb.to_scene(true_scene_params)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)

    init = (68.0, 76.0, 0.52, 3.0)  # 扰动初始几何
    refiner = SceneEMRefiner(cb, 0, fl, fr, appearance_topk=3)
    loop = EMLoop(refiner, max_iters=4, tol=0.0)  # 固定 4 轮
    result = loop.run((fl, fr), init)

    got = result.params
    assert _err(got, true_geom) < _err(init, true_geom), (
        f"ECM 未拉近几何: init 误差 {_err(init, true_geom):.3f}, "
        f"got 误差 {_err(got, true_geom):.3f}"
    )
