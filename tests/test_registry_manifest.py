"""RegistryManifest 测试: 动态子模板规格与专家恢复。"""

import pytest

from child_codebook_factory import ChildCodebookFactory
from expert_registry import ExpertRegistry
from inverse_app import InverseApp
from inverse_config import InverseConfig
from layered_codebook import LayeredCodebook
from mixture_spn import MixtureSPN
from registry_manifest import RegisteredChildTemplate, RegistryManifest
from template_lineage import ChildTemplateSpec


class _ParentExpert:
    def lineage(self):
        return LayeredCodebook.TEMPLATE_LINEAGE


def _spec(name: str = "layered_attach_test") -> ChildTemplateSpec:
    return ChildTemplateSpec(
        name=name,
        family="composite",
        parent_family="layered",
        operation="attach",
        constraints={
            "relation": "attach",
            "scale_ratio": (0.4, 0.6),
            "lateral_ratio": (-0.1, 0.1),
            "part_kinds": (1,),
            "part_hues": (2,),
        },
        complexity=1.5,
        generation=2,
        evidence_count=2,
        residual_mean=10.0,
        score_mean=11.5,
    )


def test_registry_manifest_roundtrip(tmp_path) -> None:
    """registered/pending ChildTemplateSpec 应可 JSON 往返。"""
    registered = _spec("registered_child")
    pending = _spec("pending_child")
    path = tmp_path / "registry_manifest.json"
    RegistryManifest(
        children=(RegisteredChildTemplate(registered, "child.safetensors"),),
        pending=(pending,),
    ).save(path)
    out = RegistryManifest.load(path)
    assert out.children[0].spec.name == "registered_child"
    assert out.children[0].model_path == "child.safetensors"
    assert out.pending[0].constraints["part_kinds"] == [1]


def test_registry_manifest_restores_child_expert(monkeypatch, tmp_path) -> None:
    """重启后应由 spec 重新物化动态 Codebook 并加载对应模型。"""
    spec = _spec()
    child_cls = ChildCodebookFactory.build(spec)
    cfg = InverseConfig(scene_family="composite")
    model_path = InverseApp(
        cfg, codebook=child_cls(cfg)
    ).default_model_path(tmp_path)
    model_path.touch()
    registry = ExpertRegistry({"layered": _ParentExpert()})
    registry.child_specs[spec.name] = spec
    registry.child_model_paths[spec.name] = str(model_path)
    manifest_path = registry.save_manifest(tmp_path / "manifest.json")

    restored = ExpertRegistry({"layered": _ParentExpert()})
    monkeypatch.setattr(
        MixtureSPN, "load", classmethod(lambda cls, path: object())
    )
    restored.load_manifest(manifest_path, artifacts=tmp_path, missing_ok=False)
    assert spec.name in restored.experts
    assert restored.experts[spec.name].lineage().family == spec.name
    assert restored.children_of("layered") == (spec.name,)


def test_registry_manifest_missing_model_policy(tmp_path) -> None:
    """缺模型时 missing_ok 控制跳过或 fail closed。"""
    spec = _spec("missing_child")
    path = tmp_path / "manifest.json"
    RegistryManifest(
        children=(RegisteredChildTemplate(spec, str(tmp_path / "none")),)
    ).save(path)
    registry = ExpertRegistry({"layered": _ParentExpert()})
    registry.load_manifest(path, artifacts=tmp_path, missing_ok=True)
    assert spec.name not in registry.experts
    with pytest.raises(FileNotFoundError):
        registry.load_manifest(path, artifacts=tmp_path, missing_ok=False)
