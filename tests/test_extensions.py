from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from tensor_network_editor._project_templates import (
    append_project_template,
    delete_project_template,
    load_project_template_catalog,
    rename_project_template,
)
from tensor_network_editor._template_catalog import (
    _reset_template_registry_for_tests,
    list_template_names,
)
from tensor_network_editor.api import generate_code
from tensor_network_editor.app._protocol import JsonDict, resolve_engine
from tensor_network_editor.app._services import build_bootstrap_payload
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.codegen.base import CodeGenerator
from tensor_network_editor.codegen.registry import (
    _reset_generator_registry_for_tests,
    list_generator_names,
    register_generator,
)
from tensor_network_editor.errors import PackageIOError
from tensor_network_editor.models import (
    CodegenResult,
    NetworkSpec,
    TensorCollectionFormat,
)
from tensor_network_editor.serialization import SCHEMA_VERSION, serialize_spec
from tensor_network_editor.templates import (
    TemplateDefinition,
    TemplateParameters,
    build_template_spec,
    register_static_template,
    register_template,
    serialize_template_definitions,
)
from tests.factories import build_sample_spec


def _payload_templates(payload: JsonDict) -> list[str]:
    return cast(list[str], payload["templates"])


def _payload_template_definitions(payload: JsonDict) -> JsonDict:
    return cast(JsonDict, payload["template_definitions"])


def _payload_warnings(payload: JsonDict) -> list[str]:
    return cast(list[str], payload["template_catalog_warnings"])


def _payload_engines(payload: JsonDict) -> list[str]:
    return cast(list[str], payload["engines"])


class DummyCodeGenerator(CodeGenerator):
    """Small generator used to exercise the public registry hooks."""

    engine: str = "dummy_engine"

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        return CodegenResult(
            engine=self.engine,
            code=(
                f"# dummy export for {spec.name}\n"
                f"# collection_format={collection_format.value}\n"
            ),
            warnings=[],
            artifacts={"tensor_count": len(spec.tensors)},
        )


@pytest.fixture(autouse=True)
def reset_extension_registries() -> Iterator[None]:
    _reset_generator_registry_for_tests()
    _reset_template_registry_for_tests()
    yield
    _reset_generator_registry_for_tests()
    _reset_template_registry_for_tests()


def test_builtin_extension_registries_are_seeded() -> None:
    assert list_generator_names() == [
        "tensornetwork",
        "quimb",
        "tensorkrowch",
        "einsum_numpy",
        "einsum_torch",
    ]
    assert list_template_names() == [
        "mps",
        "mpo",
        "peps_2x2",
        "mera",
        "binary_tree",
    ]


def test_register_generator_supports_custom_engine_name() -> None:
    spec = NetworkSpec(name="custom export")
    register_generator("dummy_engine", DummyCodeGenerator())

    result = generate_code(
        spec,
        engine="dummy_engine",
        collection_format=TensorCollectionFormat.DICT,
    )

    assert result.engine == "dummy_engine"
    assert "collection_format=dict" in result.code
    assert result.artifacts == {"tensor_count": 0}


def test_register_generator_rejects_duplicate_name_without_overwrite() -> None:
    register_generator("dummy_engine", DummyCodeGenerator())

    with pytest.raises(ValueError, match="already registered"):
        register_generator("dummy_engine", DummyCodeGenerator())


def test_register_template_supports_custom_template_name() -> None:
    definition = TemplateDefinition(
        name="custom_pair",
        display_name="Custom Pair",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=2,
            bond_dimension=5,
            physical_dimension=3,
        ),
    )

    def build_custom_pair(parameters: TemplateParameters) -> NetworkSpec:
        return NetworkSpec(
            id="template_custom_pair",
            name=f"Custom Pair ({parameters.graph_size})",
        )

    register_template("custom_pair", definition, build_custom_pair)

    spec = build_template_spec("custom_pair")

    assert spec.name == "Custom Pair (2)"
    assert list_template_names()[-1] == "custom_pair"


def test_register_static_template_supports_fixed_network_specs() -> None:
    static_spec = NetworkSpec(
        id="fixed_fragment",
        name="Fixed Fragment",
    )

    register_static_template("fixed_fragment", "Fixed Fragment", static_spec)

    spec = build_template_spec(
        "fixed_fragment",
        TemplateParameters(
            graph_size=99,
            bond_dimension=77,
            physical_dimension=55,
        ),
    )
    definitions = serialize_template_definitions()

    assert spec.name == "Fixed Fragment"
    assert spec.id == "fixed_fragment"
    assert list_template_names()[-1] == "fixed_fragment"
    assert definitions["fixed_fragment"]["supports_parameters"] is False


def test_project_template_catalog_entries_are_loaded_per_session(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    promoted_spec = build_sample_spec()
    promoted_spec.name = "Project Pair"
    promoted_spec.notes = []
    promoted_spec.contraction_plan = None
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [
                    {
                        "name": "project_pair",
                        "display_name": "Project Pair",
                        "spec": serialize_spec(promoted_spec),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    session = EditorSession(template_catalog_path=catalog_path)
    payload = build_bootstrap_payload(session)
    templates = _payload_templates(payload)
    template_definitions = _payload_template_definitions(payload)
    warnings = _payload_warnings(payload)

    assert templates[0] == "project_pair"
    assert (
        cast(JsonDict, template_definitions["project_pair"])["supports_parameters"]
        is False
    )
    assert cast(JsonDict, template_definitions["project_pair"])["source"] == "project"
    assert cast(JsonDict, template_definitions["mps"])["source"] == "global"
    assert warnings == []
    assert build_template_spec("mps").name == "MPS"


def test_project_template_catalog_warnings_skip_invalid_entries(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    promoted_spec = build_sample_spec()
    promoted_spec.name = "Valid Project Pair"
    promoted_spec.notes = []
    promoted_spec.contraction_plan = None
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [
                    {
                        "name": "bad name",
                        "display_name": "Bad Name",
                        "spec": {"schema_version": SCHEMA_VERSION, "network": {}},
                    },
                    {
                        "name": "valid_project_pair",
                        "display_name": "Valid Project Pair",
                        "spec": serialize_spec(promoted_spec),
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    session = EditorSession(template_catalog_path=catalog_path)
    payload = build_bootstrap_payload(session)
    templates = _payload_templates(payload)
    warnings = _payload_warnings(payload)

    assert "valid_project_pair" in templates
    assert "bad name" not in templates
    assert warnings


def test_project_template_catalog_entries_do_not_leak_between_sessions(
    tmp_path: Path,
) -> None:
    left_catalog_path = tmp_path / "left" / ".tensor-network-editor" / "templates.json"
    right_catalog_path = (
        tmp_path / "right" / ".tensor-network-editor" / "templates.json"
    )
    promoted_spec = build_sample_spec()
    promoted_spec.name = "Project Left Pair"
    promoted_spec.notes = []
    promoted_spec.contraction_plan = None
    left_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    left_catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [
                    {
                        "name": "project_left_pair",
                        "display_name": "Project Left Pair",
                        "spec": serialize_spec(promoted_spec),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    left_session = EditorSession(template_catalog_path=left_catalog_path)
    right_session = EditorSession(template_catalog_path=right_catalog_path)

    assert "project_left_pair" in _payload_templates(
        build_bootstrap_payload(left_session)
    )
    assert "project_left_pair" not in _payload_templates(
        build_bootstrap_payload(right_session)
    )


def test_project_template_catalog_skips_names_that_collide_with_global_templates(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    promoted_spec = build_sample_spec()
    promoted_spec.name = "Shadow MPS"
    promoted_spec.notes = []
    promoted_spec.contraction_plan = None
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [
                    {
                        "name": "mps",
                        "display_name": "Shadow MPS",
                        "spec": serialize_spec(promoted_spec),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = build_bootstrap_payload(EditorSession(template_catalog_path=catalog_path))
    templates = _payload_templates(payload)
    template_definitions = _payload_template_definitions(payload)
    warnings = _payload_warnings(payload)

    assert templates.count("mps") == 1
    assert cast(JsonDict, template_definitions["mps"])["display_name"] == "MPS"
    assert cast(JsonDict, template_definitions["mps"])["source"] == "global"
    assert warnings
    assert "global template" in warnings[0]


def test_project_template_catalog_rewrites_away_global_name_collisions_on_save(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    colliding_spec = build_sample_spec()
    colliding_spec.name = "Shadow MPS"
    colliding_spec.notes = []
    colliding_spec.contraction_plan = None
    valid_spec = build_sample_spec()
    valid_spec.id = "project_pair"
    valid_spec.name = "Project Pair"
    valid_spec.notes = []
    valid_spec.contraction_plan = None
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [
                    {
                        "name": "mps",
                        "display_name": "Shadow MPS",
                        "spec": serialize_spec(colliding_spec),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    append_project_template(catalog_path, "project_pair", valid_spec)
    persisted_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload = build_bootstrap_payload(EditorSession(template_catalog_path=catalog_path))
    templates = _payload_templates(payload)
    warnings = _payload_warnings(payload)
    persisted_templates = cast(list[JsonDict], persisted_payload["templates"])

    assert [entry["name"] for entry in persisted_templates] == ["project_pair"]
    assert templates[0] == "project_pair"
    assert templates.count("mps") == 1
    assert warnings == []


def test_project_template_catalog_entries_can_be_renamed_preserving_order(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    first_spec = build_sample_spec()
    first_spec.id = "first_fragment"
    first_spec.name = "First Fragment"
    first_spec.notes = []
    first_spec.contraction_plan = None
    second_spec = build_sample_spec()
    second_spec.id = "second_fragment"
    second_spec.name = "Second Fragment"
    second_spec.notes = []
    second_spec.contraction_plan = None

    append_project_template(catalog_path, "first_fragment", first_spec)
    append_project_template(catalog_path, "second_fragment", second_spec)
    renamed_catalog = rename_project_template(
        catalog_path,
        "first_fragment",
        "renamed_fragment",
        reserved_names=set(list_template_names()),
    )
    reloaded_catalog = load_project_template_catalog(catalog_path)
    payload = build_bootstrap_payload(EditorSession(template_catalog_path=catalog_path))
    templates = _payload_templates(payload)
    template_definitions = _payload_template_definitions(payload)

    assert list(renamed_catalog.entries) == ["renamed_fragment", "second_fragment"]
    assert list(reloaded_catalog.entries) == ["renamed_fragment", "second_fragment"]
    assert (
        reloaded_catalog.entries["renamed_fragment"].display_name == "Renamed Fragment"
    )
    assert reloaded_catalog.entries["renamed_fragment"].spec.name == "Renamed Fragment"
    assert templates[:2] == ["renamed_fragment", "second_fragment"]
    assert (
        cast(JsonDict, template_definitions["renamed_fragment"])["source"] == "project"
    )


def test_project_template_catalog_entries_can_be_deleted_without_leaking(
    tmp_path: Path,
) -> None:
    left_catalog_path = tmp_path / "left" / ".tensor-network-editor" / "templates.json"
    right_catalog_path = (
        tmp_path / "right" / ".tensor-network-editor" / "templates.json"
    )
    promoted_spec = build_sample_spec()
    promoted_spec.id = "project_left_pair"
    promoted_spec.name = "Project Left Pair"
    promoted_spec.notes = []
    promoted_spec.contraction_plan = None

    append_project_template(left_catalog_path, "project_left_pair", promoted_spec)
    delete_project_template(left_catalog_path, "project_left_pair")
    left_payload = build_bootstrap_payload(
        EditorSession(template_catalog_path=left_catalog_path)
    )
    right_payload = build_bootstrap_payload(
        EditorSession(template_catalog_path=right_catalog_path)
    )
    left_templates = _payload_templates(left_payload)
    right_templates = _payload_templates(right_payload)
    left_template_definitions = _payload_template_definitions(left_payload)

    assert "project_left_pair" not in left_templates
    assert "project_left_pair" not in right_templates
    assert cast(JsonDict, left_template_definitions["mps"])["source"] == "global"


def test_project_template_catalog_overwrite_replaces_only_project_entries(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    first_spec = build_sample_spec()
    first_spec.id = "first_fragment"
    first_spec.name = "Project Pair"
    first_spec.notes = []
    first_spec.contraction_plan = None
    replacement_spec = build_sample_spec()
    replacement_spec.id = "replacement_fragment"
    replacement_spec.name = "Project Pair"
    replacement_spec.tensors[0].name = "Replacement A"
    replacement_spec.notes = []
    replacement_spec.contraction_plan = None

    append_project_template(catalog_path, "project_pair", first_spec)
    overwritten_catalog = append_project_template(
        catalog_path,
        "project_pair",
        replacement_spec,
        overwrite=True,
        reserved_names=set(),
    )

    assert overwritten_catalog.entries["project_pair"].spec.id == "replacement_fragment"
    assert (
        overwritten_catalog.entries["project_pair"].spec.tensors[0].name
        == "Replacement A"
    )

    with pytest.raises(ValueError, match="global"):
        append_project_template(
            catalog_path,
            "mps",
            replacement_spec,
            overwrite=True,
            reserved_names=set(list_template_names()),
        )


def test_append_project_template_wraps_catalog_parent_creation_errors(
    tmp_path: Path,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    promoted_spec = build_sample_spec()

    with pytest.raises(
        PackageIOError,
        match="Could not create parent directory for project template catalog JSON",
    ):
        append_project_template(
            blocked_parent / "templates.json",
            "project_pair",
            promoted_spec,
        )


def test_bootstrap_payload_and_protocol_reflect_registered_extensions() -> None:
    definition = TemplateDefinition(
        name="custom_pair",
        display_name="Custom Pair",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=2,
            bond_dimension=2,
            physical_dimension=2,
        ),
    )

    def build_custom_pair(parameters: TemplateParameters) -> NetworkSpec:
        return NetworkSpec(
            id="template_custom_pair",
            name=f"Custom Pair ({parameters.graph_size})",
        )

    register_generator("dummy_engine", DummyCodeGenerator())
    register_template("custom_pair", definition, build_custom_pair)
    session = EditorSession(default_engine="dummy_engine")

    payload = build_bootstrap_payload(session)
    engines = _payload_engines(payload)
    templates = _payload_templates(payload)

    assert payload["default_engine"] == "dummy_engine"
    assert "dummy_engine" in engines
    assert templates[-1] == "custom_pair"
    assert resolve_engine({"engine": "dummy_engine"}, "dummy_engine") == "dummy_engine"
