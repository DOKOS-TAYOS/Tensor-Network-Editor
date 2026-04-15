"""Project-local storage for promoted static templates."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ._io import read_utf8_text, write_utf8_text
from ._template_catalog import (
    TemplateDefinition,
    _validate_template_name,
    build_static_template_definition,
)
from .errors import PackageIOError, SerializationError
from .models import NetworkSpec
from .serialization import deserialize_spec, serialize_spec
from .types import JSONValue, StrPath
from .validation import ensure_valid_spec

PROJECT_TEMPLATE_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectTemplateEntry:
    """One static template persisted in the project-level catalog."""

    name: str
    display_name: str
    spec: NetworkSpec

    @property
    def definition(self) -> TemplateDefinition:
        """Return the static template definition exposed to the editor."""
        return build_static_template_definition(
            self.name,
            self.display_name,
            self.spec,
            source="project",
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize one project template entry to the catalog payload."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "spec": cast(JSONValue, serialize_spec(self.spec)),
        }


@dataclass(frozen=True)
class ProjectTemplateCatalog:
    """Loaded project-local static templates together with load warnings."""

    path: Path
    entries: dict[str, ProjectTemplateEntry]
    warnings: list[str]


def resolve_project_template_catalog_path(
    template_catalog_path: StrPath | None = None,
) -> Path:
    """Resolve the project-local template catalog path."""
    if template_catalog_path is None:
        return Path.cwd() / ".tensor-network-editor" / "templates.json"
    return Path(template_catalog_path)


def derive_project_template_display_name(template_name: str) -> str:
    """Derive a human-friendly display name from one template id."""
    normalized_name = _validate_template_name(template_name)
    return " ".join(segment.capitalize() for segment in normalized_name.split("_"))


def load_project_template_catalog(
    template_catalog_path: StrPath | None = None,
) -> ProjectTemplateCatalog:
    """Load the project-local static template catalog from disk."""
    catalog_path = resolve_project_template_catalog_path(template_catalog_path)
    if not catalog_path.exists():
        return ProjectTemplateCatalog(path=catalog_path, entries={}, warnings=[])

    try:
        payload = json.loads(
            read_utf8_text(
                catalog_path,
                description="project template catalog JSON",
            )
        )
    except PackageIOError as exc:
        return ProjectTemplateCatalog(
            path=catalog_path,
            entries={},
            warnings=[str(exc)],
        )
    except json.JSONDecodeError as exc:
        return ProjectTemplateCatalog(
            path=catalog_path,
            entries={},
            warnings=[
                f"Could not parse the project template catalog at '{catalog_path}': {exc.msg}"
            ],
        )

    if not isinstance(payload, dict):
        return ProjectTemplateCatalog(
            path=catalog_path,
            entries={},
            warnings=[
                f"Project template catalog '{catalog_path}' must contain a JSON object."
            ],
        )

    schema_version = payload.get("schema_version")
    if schema_version != PROJECT_TEMPLATE_CATALOG_SCHEMA_VERSION:
        return ProjectTemplateCatalog(
            path=catalog_path,
            entries={},
            warnings=["Project template catalog schema version is not supported."],
        )

    raw_entries = payload.get("templates")
    if not isinstance(raw_entries, list):
        return ProjectTemplateCatalog(
            path=catalog_path,
            entries={},
            warnings=[
                f"Project template catalog '{catalog_path}' must contain a 'templates' list."
            ],
        )

    entries: dict[str, ProjectTemplateEntry] = {}
    warnings: list[str] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            warnings.append(
                f"Skipped project template entry #{index + 1}: expected an object."
            )
            continue
        try:
            entry = _parse_project_template_entry(raw_entry)
        except (SerializationError, ValueError) as exc:
            warnings.append(f"Skipped project template entry #{index + 1}: {exc}")
            continue
        if entry.name in entries:
            warnings.append(f"Skipped duplicated project template '{entry.name}'.")
            continue
        entries[entry.name] = entry

    return ProjectTemplateCatalog(
        path=catalog_path,
        entries=entries,
        warnings=warnings,
    )


def append_project_template(
    template_catalog_path: StrPath | None,
    template_name: str,
    spec: NetworkSpec,
    *,
    overwrite: bool = False,
    reserved_names: set[str] | None = None,
) -> ProjectTemplateCatalog:
    """Append one new project-local template and persist the catalog."""
    catalog = load_project_template_catalog(template_catalog_path)
    normalized_name = _validate_template_name(template_name)
    _validate_project_template_destination(
        catalog.entries,
        normalized_name,
        reserved_names=reserved_names,
        overwrite=overwrite,
    )
    entry = _build_project_template_entry(normalized_name, spec)
    next_entries = dict(catalog.entries)
    next_entries[normalized_name] = entry
    save_project_template_catalog(catalog.path, next_entries)
    return load_project_template_catalog(catalog.path)


def rename_project_template(
    template_catalog_path: StrPath | None,
    template_name: str,
    new_template_name: str,
    *,
    overwrite: bool = False,
    reserved_names: set[str] | None = None,
) -> ProjectTemplateCatalog:
    """Rename one persisted project-local template and reload the catalog."""
    catalog = load_project_template_catalog(template_catalog_path)
    normalized_name = _validate_template_name(template_name)
    normalized_new_name = _validate_template_name(new_template_name)
    if normalized_name not in catalog.entries:
        raise ValueError(f"Unknown project template '{normalized_name}'.")
    if normalized_new_name != normalized_name:
        _validate_project_template_destination(
            catalog.entries,
            normalized_new_name,
            reserved_names=reserved_names,
            overwrite=overwrite,
        )

    current_entry = catalog.entries[normalized_name]
    renamed_entry = _build_project_template_entry(
        normalized_new_name,
        current_entry.spec,
    )
    next_entries: dict[str, ProjectTemplateEntry] = {}
    for entry_name, entry in catalog.entries.items():
        if (
            overwrite
            and normalized_new_name != normalized_name
            and entry_name == normalized_new_name
        ):
            continue
        if entry_name == normalized_name:
            next_entries[normalized_new_name] = renamed_entry
            continue
        next_entries[entry_name] = entry
    save_project_template_catalog(catalog.path, next_entries)
    return load_project_template_catalog(catalog.path)


def delete_project_template(
    template_catalog_path: StrPath | None,
    template_name: str,
) -> ProjectTemplateCatalog:
    """Delete one persisted project-local template and reload the catalog."""
    catalog = load_project_template_catalog(template_catalog_path)
    normalized_name = _validate_template_name(template_name)
    if normalized_name not in catalog.entries:
        raise ValueError(f"Unknown project template '{normalized_name}'.")
    next_entries = {
        entry_name: entry
        for entry_name, entry in catalog.entries.items()
        if entry_name != normalized_name
    }
    save_project_template_catalog(catalog.path, next_entries)
    return load_project_template_catalog(catalog.path)


def save_project_template_catalog(
    catalog_path: StrPath,
    entries: dict[str, ProjectTemplateEntry],
) -> None:
    """Write the project-local template catalog to disk."""
    target_path = Path(catalog_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROJECT_TEMPLATE_CATALOG_SCHEMA_VERSION,
        "templates": [entry.to_dict() for entry in entries.values()],
    }
    write_utf8_text(
        target_path,
        json.dumps(payload, indent=2),
        description="project template catalog JSON",
    )


def _parse_project_template_entry(payload: dict[str, object]) -> ProjectTemplateEntry:
    """Parse one serialized project template catalog entry."""
    raw_name = payload.get("name")
    if not isinstance(raw_name, str):
        raise ValueError("Missing template 'name'.")
    normalized_name = _validate_template_name(raw_name)

    raw_display_name = payload.get("display_name")
    display_name = (
        raw_display_name.strip()
        if isinstance(raw_display_name, str) and raw_display_name.strip()
        else derive_project_template_display_name(normalized_name)
    )

    raw_spec = payload.get("spec")
    if not isinstance(raw_spec, dict):
        raise SerializationError("Missing template 'spec' payload.")
    spec = deserialize_spec(cast(dict[str, object], raw_spec))
    return ProjectTemplateEntry(
        name=normalized_name,
        display_name=display_name,
        spec=spec,
    )


def _build_project_template_entry(
    template_name: str,
    spec: NetworkSpec,
) -> ProjectTemplateEntry:
    """Normalize one project template entry before writing it to disk."""
    normalized_name = _validate_template_name(template_name)
    display_name = derive_project_template_display_name(normalized_name)
    normalized_spec = ensure_valid_spec(deepcopy(spec))
    normalized_spec.name = display_name
    return ProjectTemplateEntry(
        name=normalized_name,
        display_name=display_name,
        spec=normalized_spec,
    )


def _validate_project_template_destination(
    entries: dict[str, ProjectTemplateEntry],
    template_name: str,
    *,
    reserved_names: set[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Validate one destination project template name for write operations."""
    if template_name in (reserved_names or set()):
        raise ValueError(
            f"Template '{template_name}' is already registered as a global template."
        )
    if template_name in entries and not overwrite:
        raise ValueError(f"Template '{template_name}' is already registered.")
