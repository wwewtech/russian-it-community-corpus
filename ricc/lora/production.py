"""Resolve production and experimental LoRA adapters from the registry.

The release registry is the shared source of truth. Interactive defaults use
only verified full-epoch adapters; explicitly selected historical local
adapters remain available for research with a visible warning. Remote Hub
references always include an immutable revision and a subfolder.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_MODEL_REPO = "wwewtech/russian-it-community-lora"
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class AdapterSpec:
    """A base model and the exact adapter/tokenizer sources to load."""

    slug: str
    base_model: str
    base_revision: str
    adapter_repo: str
    adapter_revision: str
    subfolder: str
    policy_compliant: bool
    source: Literal["local", "hub"]

    @property
    def hub_reference(self) -> str:
        return f"{self.adapter_repo}/{self.subfolder}@{self.adapter_revision}"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def registry_path(root: Path | None = None) -> Path:
    return (root or repository_root()) / "lora_adapters" / "registry.json"


def _entry(slug: str, root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root)
    if not path.is_file():
        raise RuntimeError(f"Adapter registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    matches = [entry for entry in registry.get("adapters", []) if entry.get("slug") == slug]
    if len(matches) != 1:
        raise RuntimeError(f"Adapter '{slug}' is not uniquely declared in {path}")
    return cast(dict[str, Any], matches[0])


def _spec(entry: dict[str, Any], *, source: Literal["local", "hub"]) -> AdapterSpec:
    base_model = str(entry.get("base_model") or "")
    base_revision = str(entry.get("base_model_revision") or "")
    adapter_revision = str(entry.get("hf_model_revision") or "")
    if not base_model or base_model == "unknown":
        raise RuntimeError(f"Adapter '{entry.get('slug')}' has no base model")
    if source == "hub" and (not _COMMIT.fullmatch(base_revision) or not _COMMIT.fullmatch(adapter_revision)):
        raise RuntimeError(
            f"Adapter '{entry.get('slug')}' lacks immutable base/adapter revisions; "
            "historical adapters require an explicit local weights checkout"
        )
    return AdapterSpec(
        slug=str(entry["slug"]),
        base_model=base_model,
        base_revision=base_revision,
        adapter_repo=str(entry.get("hf_model_repo") or DEFAULT_MODEL_REPO),
        adapter_revision=adapter_revision,
        subfolder=str(entry["slug"]),
        policy_compliant=bool(entry.get("policy_compliant")),
        source=source,
    )


def default_adapter(root: Path | None = None) -> AdapterSpec:
    """Return the sole verified production adapter from the registry."""
    path = registry_path(root)
    if not path.is_file():
        raise RuntimeError(f"Adapter registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in registry.get("adapters", []) if entry.get("policy_compliant")]
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one policy-compliant production adapter, found {len(entries)}")
    spec = _spec(entries[0], source="hub")
    local_dir = (root or repository_root()) / "lora_adapters" / spec.slug
    if (local_dir / "adapter_model.safetensors").is_file() and (local_dir / "adapter_config.json").is_file():
        return AdapterSpec(**{**spec.__dict__, "source": "local"})
    return spec


def resolve_adapter(
    slug: str | None = None,
    *,
    root: Path | None = None,
    source: Literal["local", "hub", "auto"] = "auto",
) -> AdapterSpec:
    """Resolve a registry adapter, rejecting unsafe local-path traversal.

    ``None`` selects the verified production default. An explicitly selected
    non-compliant adapter is experimental, not a production recommendation.
    """
    if slug is None or slug in {"", "default"}:
        return default_adapter(root)
    if slug in {"none", "base"}:
        raise ValueError("Base-only inference must be selected without an AdapterSpec")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug):
        raise ValueError(f"Invalid adapter id: {slug!r}")
    entry = _entry(slug, root)
    compliant = bool(entry.get("policy_compliant"))
    if not compliant:
        warnings.warn(
            f"Adapter '{slug}' is historical/experimental, not a verified full-epoch production release",
            UserWarning,
            stacklevel=2,
        )
    local_dir = (root or repository_root()) / "lora_adapters" / slug
    has_weights = (local_dir / "adapter_model.safetensors").is_file()
    if source == "auto":
        actual_source: Literal["local", "hub"] = "local" if has_weights else "hub"
        if actual_source == "hub" and not compliant:
            raise RuntimeError(
                f"Adapter '{slug}' is not production-compliant and lacks a pinned remote revision; "
                "use an explicit local weights checkout for experimental inference"
            )
    else:
        actual_source = source
        if actual_source == "hub" and not compliant:
            raise RuntimeError(
                f"Adapter '{slug}' is not production-compliant and cannot be selected as a pinned Hub release; "
                "use an explicit local weights checkout for experimental inference"
            )
    return _spec(entry, source=actual_source)


def local_adapter_path(spec: AdapterSpec, root: Path | None = None) -> Path:
    """Return a safe local path only for a local spec, never for a Hub ID."""
    if spec.source != "local":
        raise ValueError(f"{spec.slug} is a remote Hub adapter, not a local directory")
    path = ((root or repository_root()) / "lora_adapters" / spec.slug).resolve()
    expected_root = ((root or repository_root()) / "lora_adapters").resolve()
    if not path.is_relative_to(expected_root):
        raise ValueError("Adapter path escapes lora_adapters")
    return path
