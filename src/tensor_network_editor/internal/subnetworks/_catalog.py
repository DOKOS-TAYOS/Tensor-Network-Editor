"""Project-local and shared catalogs for reusable subnetworks."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ...errors import PackageIOError, SerializationError
from ...models import NetworkSpec
from ...types import JSONValue, StrPath
from ...validation import ensure_valid_spec
from .._logging import log_branch, log_operation
from ..io._io import read_utf8_text, write_utf8_text
from ..io._serialization import deserialize_spec, serialize_spec

SUBNETWORK_CATALOG_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubnetworkCatalogEntry:
    """One reusable subnetwork persisted in a catalog."""

    name: str
    display_name: str
    tags: list[str]
    spec: NetworkSpec

    def to_definition(
        self, *, source: Literal["project", "shared"]
    ) -> dict[str, JSONValue]:
        """Return bootstrap metadata for one saved reusable subnetwork."""
        return {
            "display_name": self.display_name,
            "tags": cast(JSONValue, list(self.tags)),
            "source": source,
            "tensor_count": len(self.spec.tensors),
            "edge_count": len(self.spec.edges),
            "spec": cast(JSONValue, serialize_spec(self.spec)),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize one catalog entry to the on-disk JSON structure."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "tags": cast(JSONValue, list(self.tags)),
            "spec": cast(JSONValue, serialize_spec(self.spec)),
        }


@dataclass(frozen=True)
class SubnetworkCatalog:
    """Loaded reusable subnetworks together with load warnings."""

    path: Path
    entries: dict[str, SubnetworkCatalogEntry]
    warnings: list[str]


def resolve_project_subnetwork_catalog_path(
    subnetwork_catalog_path: StrPath | None = None,
) -> Path:
    """Resolve the reusable-subnetwork catalog path."""
    if subnetwork_catalog_path is None:
        return Path.cwd() / ".tensor-network-editor" / "subnetworks.json"
    return Path(subnetwork_catalog_path)


def derive_project_subnetwork_display_name(subnetwork_name: str) -> str:
    """Derive a human-friendly display name from a saved subnetwork id."""
    normalized_name = _validate_subnetwork_name(subnetwork_name)
    return " ".join(segment.capitalize() for segment in normalized_name.split("_"))


def normalize_subnetwork_tags(raw_tags: Sequence[str] | None) -> list[str]:
    """Normalize reusable-subnetwork tags using the package tag conventions."""
    if raw_tags is None:
        return []
    return sorted(
        {
            str(raw_tag).strip()
            for raw_tag in raw_tags
            if isinstance(raw_tag, str) and raw_tag.strip()
        }
    )


def load_project_subnetwork_catalog(
    subnetwork_catalog_path: StrPath | None = None,
) -> SubnetworkCatalog:
    """Load the reusable-subnetwork catalog from disk."""
    catalog_path = resolve_project_subnetwork_catalog_path(subnetwork_catalog_path)
    with log_operation(
        LOGGER,
        "Reusable subnetwork catalog load",
        context={"path": catalog_path},
        emit_start=False,
    ) as success_context:
        if not catalog_path.exists():
            success_context["status"] = "missing"
            log_branch(LOGGER, "Reusable subnetwork catalog not found on disk")
            return SubnetworkCatalog(path=catalog_path, entries={}, warnings=[])

        try:
            payload = json.loads(
                read_utf8_text(
                    catalog_path,
                    description="reusable subnetwork catalog JSON",
                )
            )
        except PackageIOError as exc:
            log_branch(
                LOGGER,
                f"Reusable subnetwork catalog read failed: {exc}",
                level=logging.WARNING,
            )
            return SubnetworkCatalog(path=catalog_path, entries={}, warnings=[str(exc)])
        except json.JSONDecodeError as exc:
            warning = f"Could not parse the reusable subnetwork catalog at '{catalog_path}': {exc.msg}"
            log_branch(LOGGER, warning, level=logging.WARNING)
            return SubnetworkCatalog(
                path=catalog_path,
                entries={},
                warnings=[warning],
            )

        if not isinstance(payload, dict):
            warning = f"Reusable subnetwork catalog '{catalog_path}' must contain a JSON object."
            log_branch(LOGGER, warning, level=logging.WARNING)
            return SubnetworkCatalog(
                path=catalog_path,
                entries={},
                warnings=[warning],
            )

        schema_version = payload.get("schema_version")
        if schema_version != SUBNETWORK_CATALOG_SCHEMA_VERSION:
            warning = "Reusable subnetwork catalog schema version is not supported."
            log_branch(LOGGER, warning, level=logging.WARNING)
            return SubnetworkCatalog(
                path=catalog_path,
                entries={},
                warnings=[warning],
            )

        raw_entries = payload.get("subnetworks")
        if not isinstance(raw_entries, list):
            warning = f"Reusable subnetwork catalog '{catalog_path}' must contain a 'subnetworks' list."
            log_branch(LOGGER, warning, level=logging.WARNING)
            return SubnetworkCatalog(
                path=catalog_path,
                entries={},
                warnings=[warning],
            )

        entries: dict[str, SubnetworkCatalogEntry] = {}
        warnings: list[str] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                warning = f"Skipped reusable subnetwork entry #{index + 1}: expected an object."
                warnings.append(warning)
                log_branch(LOGGER, warning, level=logging.WARNING)
                continue
            try:
                entry = _parse_project_subnetwork_entry(raw_entry)
            except (SerializationError, ValueError) as exc:
                warning = f"Skipped reusable subnetwork entry #{index + 1}: {exc}"
                warnings.append(warning)
                log_branch(LOGGER, warning, level=logging.WARNING)
                continue
            if entry.name in entries:
                warning = f"Skipped duplicated reusable subnetwork '{entry.name}'."
                warnings.append(warning)
                log_branch(LOGGER, warning, level=logging.WARNING)
                continue
            entries[entry.name] = entry

        success_context["status"] = len(entries)
        success_context["warning_count"] = len(warnings)
        return SubnetworkCatalog(path=catalog_path, entries=entries, warnings=warnings)


def append_project_subnetwork(
    subnetwork_catalog_path: StrPath | None,
    subnetwork_name: str,
    spec: NetworkSpec,
    *,
    tags: Sequence[str] | None = None,
    overwrite: bool = False,
) -> SubnetworkCatalog:
    """Append one reusable subnetwork to the project-local catalog."""
    with log_operation(
        LOGGER,
        "Reusable subnetwork catalog append",
        context={"subnetwork_name": subnetwork_name, "tag_count": len(tags or ())},
    ):
        catalog = load_project_subnetwork_catalog(subnetwork_catalog_path)
        normalized_name = _validate_subnetwork_name(subnetwork_name)
        _validate_project_subnetwork_destination(
            catalog.entries,
            normalized_name,
            overwrite=overwrite,
        )
        entry = _build_project_subnetwork_entry(normalized_name, spec, tags=tags)
        next_entries = dict(catalog.entries)
        next_entries[normalized_name] = entry
        save_project_subnetwork_catalog(catalog.path, next_entries)
        return load_project_subnetwork_catalog(catalog.path)


def rename_project_subnetwork(
    subnetwork_catalog_path: StrPath | None,
    subnetwork_name: str,
    new_subnetwork_name: str,
    *,
    overwrite: bool = False,
) -> SubnetworkCatalog:
    """Rename one persisted reusable subnetwork and reload the catalog."""
    with log_operation(
        LOGGER,
        "Reusable subnetwork catalog rename",
        context={
            "subnetwork_name": subnetwork_name,
            "selected_subnetwork": new_subnetwork_name,
        },
    ):
        catalog = load_project_subnetwork_catalog(subnetwork_catalog_path)
        normalized_name = _validate_subnetwork_name(subnetwork_name)
        normalized_new_name = _validate_subnetwork_name(new_subnetwork_name)
        if normalized_name not in catalog.entries:
            raise ValueError(f"Unknown reusable subnetwork '{normalized_name}'.")
        if normalized_new_name != normalized_name:
            _validate_project_subnetwork_destination(
                catalog.entries,
                normalized_new_name,
                overwrite=overwrite,
            )

        current_entry = catalog.entries[normalized_name]
        renamed_entry = _build_project_subnetwork_entry(
            normalized_new_name,
            current_entry.spec,
            tags=current_entry.tags,
        )
        next_entries: dict[str, SubnetworkCatalogEntry] = {}
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
        save_project_subnetwork_catalog(catalog.path, next_entries)
        return load_project_subnetwork_catalog(catalog.path)


def delete_project_subnetwork(
    subnetwork_catalog_path: StrPath | None,
    subnetwork_name: str,
) -> SubnetworkCatalog:
    """Delete one persisted reusable subnetwork and reload the catalog."""
    with log_operation(
        LOGGER,
        "Reusable subnetwork catalog delete",
        context={"subnetwork_name": subnetwork_name},
    ):
        catalog = load_project_subnetwork_catalog(subnetwork_catalog_path)
        normalized_name = _validate_subnetwork_name(subnetwork_name)
        if normalized_name not in catalog.entries:
            raise ValueError(f"Unknown reusable subnetwork '{normalized_name}'.")
        next_entries = {
            entry_name: entry
            for entry_name, entry in catalog.entries.items()
            if entry_name != normalized_name
        }
        save_project_subnetwork_catalog(catalog.path, next_entries)
        return load_project_subnetwork_catalog(catalog.path)


def save_project_subnetwork_catalog(
    catalog_path: StrPath,
    entries: dict[str, SubnetworkCatalogEntry],
) -> None:
    """Write the reusable-subnetwork catalog to disk."""
    target_path = Path(catalog_path)
    with log_operation(
        LOGGER,
        "Reusable subnetwork catalog save",
        context={"path": target_path},
        start_level=logging.DEBUG,
        success_level=logging.DEBUG,
        emit_start=False,
    ) as success_context:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PackageIOError(
                "Could not create parent directory for reusable subnetwork catalog JSON "
                f"at '{target_path.parent}': {exc}"
            ) from exc
        payload = {
            "schema_version": SUBNETWORK_CATALOG_SCHEMA_VERSION,
            "subnetworks": [entry.to_dict() for entry in entries.values()],
        }
        write_utf8_text(
            target_path,
            json.dumps(payload, indent=2),
            description="reusable subnetwork catalog JSON",
        )
        success_context["status"] = len(entries)


def _validate_subnetwork_name(subnetwork_name: str) -> str:
    """Validate and normalize one reusable-subnetwork name."""
    normalized_name = str(subnetwork_name).strip()
    if (
        not normalized_name
        or not normalized_name[0].islower()
        or any(
            not character.islower() and not character.isdigit() and character != "_"
            for character in normalized_name
        )
    ):
        raise ValueError(
            "Reusable subnetwork names must start with a lowercase letter and contain only lowercase letters, digits, and underscores."
        )
    return normalized_name


def _parse_project_subnetwork_entry(
    payload: Mapping[str, object],
) -> SubnetworkCatalogEntry:
    """Parse one serialized reusable-subnetwork catalog entry."""
    raw_name = payload.get("name")
    if not isinstance(raw_name, str):
        raise ValueError("Missing reusable subnetwork 'name'.")
    normalized_name = _validate_subnetwork_name(raw_name)

    raw_display_name = payload.get("display_name")
    display_name = (
        raw_display_name.strip()
        if isinstance(raw_display_name, str) and raw_display_name.strip()
        else derive_project_subnetwork_display_name(normalized_name)
    )

    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("Reusable subnetwork 'tags' must be a list when provided.")
    tags = normalize_subnetwork_tags(raw_tags)

    raw_spec = payload.get("spec")
    if not isinstance(raw_spec, dict):
        raise SerializationError("Missing reusable subnetwork 'spec' payload.")
    spec = deserialize_spec(raw_spec)
    return SubnetworkCatalogEntry(
        name=normalized_name,
        display_name=display_name,
        tags=tags,
        spec=spec,
    )


def _build_project_subnetwork_entry(
    subnetwork_name: str,
    spec: NetworkSpec,
    *,
    tags: Sequence[str] | None = None,
) -> SubnetworkCatalogEntry:
    """Normalize one reusable-subnetwork entry before writing it to disk."""
    normalized_name = _validate_subnetwork_name(subnetwork_name)
    display_name = derive_project_subnetwork_display_name(normalized_name)
    normalized_spec = ensure_valid_spec(deepcopy(spec))
    normalized_spec.name = display_name
    return SubnetworkCatalogEntry(
        name=normalized_name,
        display_name=display_name,
        tags=normalize_subnetwork_tags(tags),
        spec=normalized_spec,
    )


def _validate_project_subnetwork_destination(
    entries: dict[str, SubnetworkCatalogEntry],
    subnetwork_name: str,
    *,
    overwrite: bool = False,
) -> None:
    """Validate one destination reusable-subnetwork name for write operations."""
    if subnetwork_name in entries and not overwrite:
        raise ValueError(
            f"Reusable subnetwork '{subnetwork_name}' is already registered."
        )
