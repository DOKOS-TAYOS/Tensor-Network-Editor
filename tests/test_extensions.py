from __future__ import annotations

from collections.abc import Iterator

import pytest

from tensor_network_editor._template_catalog import (
    _reset_template_registry_for_tests,
    list_template_names,
)
from tensor_network_editor.api import generate_code
from tensor_network_editor.app._protocol import resolve_engine
from tensor_network_editor.app._services import build_bootstrap_payload
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.codegen.base import CodeGenerator
from tensor_network_editor.codegen.registry import (
    _reset_generator_registry_for_tests,
    list_generator_names,
    register_generator,
)
from tensor_network_editor.models import (
    CodegenResult,
    NetworkSpec,
    TensorCollectionFormat,
)
from tensor_network_editor.templates import (
    TemplateDefinition,
    TemplateParameters,
    build_template_spec,
    register_template,
)


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

    assert payload["default_engine"] == "dummy_engine"
    assert "dummy_engine" in payload["engines"]
    assert payload["templates"][-1] == "custom_pair"
    assert resolve_engine({"engine": "dummy_engine"}, "dummy_engine") == "dummy_engine"
