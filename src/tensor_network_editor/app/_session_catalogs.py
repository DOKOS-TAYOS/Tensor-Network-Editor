"""Catalog support helpers shared by editor sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from ..internal.subnetworks._catalog import (
    SubnetworkCatalog,
    SubnetworkCatalogEntry,
    append_project_subnetwork,
    delete_project_subnetwork,
    load_project_subnetwork_catalog,
    rename_project_subnetwork,
)
from ..internal.templates._project_templates import (
    ProjectTemplateCatalog,
    ProjectTemplateEntry,
    append_project_template,
    delete_project_template,
    derive_project_template_display_name,
    load_project_template_catalog,
    rename_project_template,
)
from ..models import NetworkSpec
from ..templates import list_template_names, serialize_template_definitions
from ..types import JSONValue, StrPath


@dataclass(slots=True)
class SessionCatalogState:
    """Mutable project/shared catalog state attached to one editor session."""

    project_template_catalog: ProjectTemplateCatalog
    project_subnetwork_catalog: SubnetworkCatalog
    shared_subnetwork_catalog: SubnetworkCatalog | None

    @classmethod
    def load(
        cls,
        *,
        template_catalog_path: StrPath | None,
        subnetwork_catalog_path: StrPath | None,
        shared_subnetwork_catalog_path: StrPath | None,
    ) -> SessionCatalogState:
        """Load all catalog sources used by one editor session."""
        project_template_catalog = load_project_template_catalog(
            template_catalog_path,
            reserved_names=set(list_template_names()),
        )
        project_subnetwork_catalog = load_project_subnetwork_catalog(
            subnetwork_catalog_path
        )
        shared_subnetwork_catalog = (
            load_project_subnetwork_catalog(shared_subnetwork_catalog_path)
            if shared_subnetwork_catalog_path is not None
            else None
        )
        return cls(
            project_template_catalog=project_template_catalog,
            project_subnetwork_catalog=project_subnetwork_catalog,
            shared_subnetwork_catalog=shared_subnetwork_catalog,
        )

    @property
    def template_catalog_path(self) -> StrPath | None:
        """Return the resolved project-template catalog path."""
        return self.project_template_catalog.path

    @property
    def subnetwork_catalog_path(self) -> StrPath | None:
        """Return the resolved project-subnetwork catalog path."""
        return self.project_subnetwork_catalog.path

    @property
    def shared_subnetwork_catalog_path(self) -> StrPath | None:
        """Return the resolved shared-subnetwork catalog path."""
        if self.shared_subnetwork_catalog is None:
            return None
        return self.shared_subnetwork_catalog.path

    @property
    def project_template_entries(self) -> Mapping[str, ProjectTemplateEntry]:
        """Return the project-local static template entries keyed by name."""
        return self.project_template_catalog.entries

    @property
    def template_catalog_warnings(self) -> list[str]:
        """Return any warnings raised while loading the local template catalog."""
        return list(self.project_template_catalog.warnings)

    @property
    def project_subnetwork_entries(self) -> Mapping[str, SubnetworkCatalogEntry]:
        """Return the project-local reusable subnetwork entries keyed by name."""
        return self.project_subnetwork_catalog.entries

    @property
    def shared_subnetwork_entries(self) -> Mapping[str, SubnetworkCatalogEntry]:
        """Return the shared reusable subnetwork entries keyed by name."""
        if self.shared_subnetwork_catalog is None:
            return {}
        return self.shared_subnetwork_catalog.entries

    @property
    def subnetwork_catalog_warnings(self) -> list[str]:
        """Return any warnings raised while loading reusable-subnetwork catalogs."""
        warnings = list(self.project_subnetwork_catalog.warnings)
        if self.shared_subnetwork_catalog is not None:
            warnings.extend(self.shared_subnetwork_catalog.warnings)
            warnings.extend(
                [
                    f"Project reusable subnetwork '{name}' shadows the shared catalog entry."
                    for name in self.project_subnetwork_catalog.entries
                    if name in self.shared_subnetwork_catalog.entries
                ]
            )
        return warnings

    def list_available_template_names(self) -> list[str]:
        """Return the merged project-local and globally registered templates."""
        return list(self.project_template_catalog.entries) + list_template_names()

    def list_global_template_names(self) -> list[str]:
        """Return the globally registered template names only."""
        return list_template_names()

    def serialize_available_template_definitions(
        self,
    ) -> dict[str, dict[str, JSONValue]]:
        """Return serialized template definitions for the current session."""
        definitions = {
            template_name: entry.definition.to_dict()
            for template_name, entry in self.project_template_catalog.entries.items()
        }
        definitions.update(serialize_template_definitions())
        return definitions

    def list_available_subnetwork_names(self) -> list[str]:
        """Return merged project-local and shared reusable subnetworks."""
        shared_names = (
            list(self.shared_subnetwork_catalog.entries)
            if self.shared_subnetwork_catalog is not None
            else []
        )
        return list(self.project_subnetwork_catalog.entries) + [
            name
            for name in shared_names
            if name not in self.project_subnetwork_catalog.entries
        ]

    def serialize_available_subnetwork_definitions(
        self,
    ) -> dict[str, dict[str, JSONValue]]:
        """Return serialized reusable-subnetwork definitions for the editor."""
        definitions = {
            subnetwork_name: entry.to_definition(source="project")
            for subnetwork_name, entry in self.project_subnetwork_catalog.entries.items()
        }
        if self.shared_subnetwork_catalog is not None:
            for (
                subnetwork_name,
                entry,
            ) in self.shared_subnetwork_catalog.entries.items():
                if subnetwork_name in definitions:
                    continue
                definitions[subnetwork_name] = entry.to_definition(source="shared")
        return definitions

    def has_project_template(self, template_name: str) -> bool:
        """Return whether the session exposes a project-local template name."""
        return template_name in self.project_template_catalog.entries

    def has_global_template(self, template_name: str) -> bool:
        """Return whether the session exposes a globally registered template."""
        return template_name in list_template_names()

    def has_project_subnetwork(self, subnetwork_name: str) -> bool:
        """Return whether the session exposes a project-local reusable subnetwork."""
        return subnetwork_name in self.project_subnetwork_catalog.entries

    def build_project_template(self, template_name: str) -> NetworkSpec:
        """Build a copied project-local template spec for insertion."""
        try:
            entry = self.project_template_catalog.entries[template_name]
        except KeyError as exc:
            raise ValueError(f"Unknown template '{template_name}'.") from exc
        return deepcopy(entry.spec)

    def build_project_template_display_name(self, template_name: str) -> str:
        """Return the derived display name used for one promoted template."""
        return derive_project_template_display_name(template_name)

    def build_saved_subnetwork(self, subnetwork_name: str) -> NetworkSpec:
        """Build a copied saved reusable subnetwork spec for insertion."""
        if subnetwork_name in self.project_subnetwork_catalog.entries:
            return deepcopy(
                self.project_subnetwork_catalog.entries[subnetwork_name].spec
            )
        if self.shared_subnetwork_catalog is not None and (
            subnetwork_name in self.shared_subnetwork_catalog.entries
        ):
            return deepcopy(
                self.shared_subnetwork_catalog.entries[subnetwork_name].spec
            )
        raise ValueError(f"Unknown reusable subnetwork '{subnetwork_name}'.")

    def save_project_subnetwork(
        self,
        subnetwork_name: str,
        spec: NetworkSpec,
        *,
        tags: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Persist one reusable subnetwork and reload the project catalog."""
        self.project_subnetwork_catalog = append_project_subnetwork(
            self.subnetwork_catalog_path,
            subnetwork_name,
            spec,
            tags=tags,
            overwrite=overwrite,
        )

    def rename_project_subnetwork(
        self,
        subnetwork_name: str,
        new_subnetwork_name: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Rename one project-local reusable subnetwork and reload the catalog."""
        self.project_subnetwork_catalog = rename_project_subnetwork(
            self.subnetwork_catalog_path,
            subnetwork_name,
            new_subnetwork_name,
            overwrite=overwrite,
        )

    def delete_project_subnetwork(self, subnetwork_name: str) -> None:
        """Delete one project-local reusable subnetwork and reload the catalog."""
        self.project_subnetwork_catalog = delete_project_subnetwork(
            self.subnetwork_catalog_path,
            subnetwork_name,
        )

    def save_project_template(
        self,
        template_name: str,
        spec: NetworkSpec,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist one new project-local static template and reload the catalog."""
        self.project_template_catalog = append_project_template(
            self.template_catalog_path,
            template_name,
            spec,
            overwrite=overwrite,
            reserved_names=set(self.list_global_template_names()),
        )

    def rename_project_template(
        self,
        template_name: str,
        new_template_name: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Rename one project-local static template and reload the catalog."""
        self.project_template_catalog = rename_project_template(
            self.template_catalog_path,
            template_name,
            new_template_name,
            overwrite=overwrite,
            reserved_names=set(self.list_global_template_names()),
        )

    def delete_project_template(self, template_name: str) -> None:
        """Delete one project-local static template entry and reload the catalog."""
        self.project_template_catalog = delete_project_template(
            self.template_catalog_path,
            template_name,
            reserved_names=set(self.list_global_template_names()),
        )
