"""RegistryManifest: 动态子模板与专家注册表的 JSON 持久化。

safetensors 只保存 MixtureSPN 参数; 本 manifest 保存恢复结构专家所需的
ChildTemplateSpec、血缘、pending 规格和模型路径。动态 Codebook 类在加载时
由 ChildCodebookFactory 按 spec 重新物化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from template_lineage import ChildTemplateSpec


@dataclass(frozen=True)
class RegisteredChildTemplate:
    """已训练注册的动态子模板记录。"""

    spec: ChildTemplateSpec
    model_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "model_path": self.model_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegisteredChildTemplate:
        return cls(
            spec=ChildTemplateSpec.from_dict(data["spec"]),
            model_path=data.get("model_path"),
        )


@dataclass(frozen=True)
class RegistryManifest:
    """registry_manifest.json 的强类型表示。"""

    children: tuple[RegisteredChildTemplate, ...] = field(default_factory=tuple)
    pending: tuple[ChildTemplateSpec, ...] = field(default_factory=tuple)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "children": [c.to_dict() for c in self.children],
            "pending": [p.to_dict() for p in self.pending],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryManifest:
        version = int(data.get("version", 0))
        if version != 1:
            raise ValueError(f"不支持的 registry manifest 版本: {version}")
        return cls(
            children=tuple(
                RegisteredChildTemplate.from_dict(x)
                for x in data.get("children", [])
            ),
            pending=tuple(
                ChildTemplateSpec.from_dict(x)
                for x in data.get("pending", [])
            ),
            version=version,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf8",
        )

    @classmethod
    def load(cls, path: Path) -> RegistryManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf8")))
