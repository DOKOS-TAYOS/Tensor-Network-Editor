from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app._services import build_bootstrap_payload
from tensor_network_editor.app.session import EditorSession
from tests.factories import build_sample_spec


def _payload_subnetworks(payload: JsonDict) -> list[str]:
    return cast(list[str], payload["subnetworks"])


def _payload_subnetwork_definitions(payload: JsonDict) -> JsonDict:
    return cast(JsonDict, payload["subnetwork_definitions"])


def _payload_subnetwork_definition(payload: JsonDict, name: str) -> JsonDict:
    return cast(JsonDict, _payload_subnetwork_definitions(payload)[name])


def _payload_subnetwork_warnings(payload: JsonDict) -> list[str]:
    return cast(list[str], payload["subnetwork_catalog_warnings"])


def test_build_bootstrap_payload_includes_empty_subnetwork_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    payload = build_bootstrap_payload(EditorSession())

    assert payload["subnetworks"] == []
    assert payload["subnetwork_definitions"] == {}
    assert payload["subnetwork_catalog_warnings"] == []


def test_project_subnetwork_catalog_entries_are_loaded_per_session(
    tmp_path: Path,
) -> None:
    catalog_module = import_module(
        "tensor_network_editor.internal.subnetworks._catalog"
    )
    append_project_subnetwork = catalog_module.append_project_subnetwork

    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"
    spec = build_sample_spec()
    spec.notes = []
    spec.contraction_plan = None
    append_project_subnetwork(
        catalog_path,
        "project_pair",
        spec,
        tags=[" block ", "alpha", "block"],
    )

    payload = build_bootstrap_payload(
        EditorSession(subnetwork_catalog_path=catalog_path)
    )
    definition = _payload_subnetwork_definition(payload, "project_pair")

    assert _payload_subnetworks(payload) == ["project_pair"]
    assert definition["display_name"] == "Project Pair"
    assert definition["source"] == "project"
    assert definition["tags"] == ["alpha", "block"]
    assert _payload_subnetwork_warnings(payload) == []


def test_default_subnetwork_catalog_loads_from_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_module = import_module(
        "tensor_network_editor.internal.subnetworks._catalog"
    )
    append_project_subnetwork = catalog_module.append_project_subnetwork

    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"
    spec = build_sample_spec()
    spec.notes = []
    spec.contraction_plan = None
    append_project_subnetwork(catalog_path, "project_pair", spec)
    monkeypatch.chdir(tmp_path)

    payload = build_bootstrap_payload(EditorSession())

    assert _payload_subnetworks(payload) == ["project_pair"]


def test_project_subnetwork_catalog_shadows_shared_entry_with_warning(
    tmp_path: Path,
) -> None:
    catalog_module = import_module(
        "tensor_network_editor.internal.subnetworks._catalog"
    )
    append_project_subnetwork = catalog_module.append_project_subnetwork

    shared_catalog_path = tmp_path / "shared" / "subnetworks.json"
    project_catalog_path = tmp_path / "project" / "subnetworks.json"
    shared_spec = build_sample_spec()
    shared_spec.name = "Shared Pair"
    shared_spec.notes = []
    shared_spec.contraction_plan = None
    project_spec = build_sample_spec()
    project_spec.name = "Project Pair"
    project_spec.notes = []
    project_spec.contraction_plan = None

    append_project_subnetwork(shared_catalog_path, "pair", shared_spec, tags=["shared"])
    append_project_subnetwork(
        project_catalog_path,
        "pair",
        project_spec,
        tags=["project"],
    )

    payload = build_bootstrap_payload(
        EditorSession(
            subnetwork_catalog_path=project_catalog_path,
            shared_subnetwork_catalog_path=shared_catalog_path,
        )
    )
    definition = _payload_subnetwork_definition(payload, "pair")
    warnings = _payload_subnetwork_warnings(payload)

    assert _payload_subnetworks(payload) == ["pair"]
    assert definition["display_name"] == "Pair"
    assert definition["source"] == "project"
    assert definition["tags"] == ["project"]
    assert warnings
    assert "shared" in warnings[0].lower()
