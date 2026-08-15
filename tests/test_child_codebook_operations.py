"""ChildCodebookFactory 多操作物化测试。"""

import random

from child_codebook_factory import ChildCodebookFactory
from codebook import Codebook
from inverse_app import InverseApp
from inverse_config import InverseConfig
from lateral_codebook import LateralCompositeCodebook
from lateral_composite_geometry import LateralCompositeGeometry
from layered_codebook import LayeredCodebook
from scene_reconstructor import SceneReconstructor
from template_lineage import ChildTemplateSpec


def _spec(operation: str, name: str) -> ChildTemplateSpec:
    family = "layered" if operation == "layer" else "composite"
    constraints = {
        "relation": operation,
        "scale_ratio": (0.4, 0.6),
        "part_kinds": (1,),
        "part_hues": (2,),
    }
    if operation == "layer":
        constraints["lateral_ratio"] = (-0.1, 0.1)
        constraints["depth_gap"] = (0.7, 0.9)
    else:
        constraints["period_ratio"] = (0.18, 0.22)
    return ChildTemplateSpec(
        name=name,
        family=family,
        parent_family="composite",
        operation=operation,
        constraints=constraints,
        complexity=1.5,
        generation=3,
        evidence_count=2,
        residual_mean=1.0,
        score_mean=2.5,
    )


def test_layer_child_codebook_constraints() -> None:
    """layer 子模板应使用比例/横向/深度约束和受限离散支持集。"""
    cls = ChildCodebookFactory.build(_spec("layer", "layer_child"))
    assert issubclass(cls, LayeredCodebook)
    assert cls.N_COMBO == 3 * 1 * 6 * 1 * 3 * 3
    vals = cls._sample_pair(random.Random(1), False)
    assert 0.4 <= vals[6] / vals[2] <= 0.6
    assert 0.7 <= vals[3] - vals[7] <= 0.9
    assert abs(vals[4] - vals[0]) <= 0.1 * (
        Codebook.EXTENT * vals[2] * Codebook.FX / (Codebook.CAM_Z - vals[3])
        + Codebook.EXTENT * vals[6] * Codebook.FX / (Codebook.CAM_Z - vals[7])
    ) + 1e-6


def test_lateral_child_codebooks_constraints() -> None:
    """mirror/repeat 子模板应同 kind/hue、同深度、part 在 base 右侧。"""
    for op, spacing in (("mirror", 5.0), ("repeat", 7.5)):
        cls = ChildCodebookFactory.build(_spec(op, f"{op}_child"))
        assert issubclass(cls, LateralCompositeCodebook)
        assert cls.SPACING_FACTOR == spacing
        assert cls.N_COMBO == 1 * 1 * 3 * 3
        vals = cls._sample_composite(random.Random(2), False)
        assert vals[4] > vals[0]
        assert vals[7] == vals[3]
        assert 0.4 <= vals[6] / vals[2] <= 0.6


def test_lateral_geometry_and_frame_features() -> None:
    """横向子模板应走垂直分隔几何锚点。"""
    cls = ChildCodebookFactory.build(_spec("mirror", "mirror_geom"))
    cfg = InverseConfig(scene_family="composite")
    cb = cls(cfg)
    app = InverseApp(cfg, codebook=cb)
    prm = tuple(float(x) for x in cb.sample(1, 7)[0].tolist())
    renderer, cam_l, cam_r = Codebook.make_renderer()
    scene = cb.to_scene(prm)
    fl = renderer.render(scene, cam_l)
    fr = renderer.render(scene, cam_r)
    st = LateralCompositeGeometry.estimate(fl, fr)
    assert st[4] > st[0]
    assert abs(st[2] - st[6]) < 0.3
    vec, stats, _ = SceneReconstructor.frame_features(app, fl, fr)
    assert vec.shape[1] == cfg.n_feat
    assert stats.shape == (1, 8)
