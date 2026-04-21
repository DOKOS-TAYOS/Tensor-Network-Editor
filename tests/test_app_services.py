from __future__ import annotations

from importlib import import_module
from typing import cast
from unittest.mock import patch

import tensor_network_editor.app._services as app_services_module
import tensor_network_editor.codegen.einsum as einsum_codegen_module
import tensor_network_editor.internal.analysis._contraction_analysis as contraction_analysis_module
from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app._services import (
    analyze_serialized_contraction,
    build_bootstrap_payload,
    build_template_from_payload,
    complete_session_request,
    generate_session_request,
)
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.codegen.registry import engine_name_to_text
from tensor_network_editor.models import CodegenResult, EditorResult, NetworkSpec
from tensor_network_editor.models import EngineName as SessionEngineName
from tensor_network_editor.serialization import SCHEMA_VERSION
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_linear_periodic_chain_spec,
    build_tree_periodic_tree_spec,
)


def test_build_bootstrap_payload_matches_session_contract(
    editor_session: EditorSession,
) -> None:
    payload = build_bootstrap_payload(editor_session)

    assert payload["default_engine"] == engine_name_to_text(
        editor_session.default_engine
    )
    assert payload["default_collection_format"] == (
        editor_session.default_collection_format.value
    )
    assert payload["templates"] == list(cast(JsonDict, payload["template_definitions"]))
    annotation_definitions = cast(JsonDict, payload["annotation_definitions"])
    tensor_annotations = cast(list[JsonDict], annotation_definitions["tensor"])
    index_annotations = cast(list[JsonDict], annotation_definitions["index"])
    assert list(annotation_definitions) == ["tensor", "index"]
    assert tensor_annotations[0]["key"] == "role"
    assert tensor_annotations[0]["label"] == "Tensor role"
    assert index_annotations[0]["key"] == "leg_kind"
    assert index_annotations[0]["label"] == "Leg kind"


def test_app_services_module_reexports_split_service_helpers() -> None:
    bootstrap_module = import_module("tensor_network_editor.app._bootstrap_payloads")
    session_module = import_module("tensor_network_editor.app._session_requests")
    template_module = import_module("tensor_network_editor.app._template_services")
    analysis_module = import_module("tensor_network_editor.app._analysis_services")
    subnetwork_module = import_module("tensor_network_editor.app._subnetwork_services")

    assert build_bootstrap_payload is bootstrap_module.build_bootstrap_payload
    assert generate_session_request is session_module.generate_session_request
    assert complete_session_request is session_module.complete_session_request
    assert build_template_from_payload is template_module.build_template_from_payload
    assert (
        analyze_serialized_contraction
        is app_services_module.analyze_serialized_contraction
    )
    assert (
        analysis_module.analyze_serialized_contraction.__module__
        == "tensor_network_editor.app._analysis_services"
    )
    assert (
        app_services_module.extract_serialized_subnetwork
        is subnetwork_module.extract_serialized_subnetwork
    )


def test_generate_session_request_matches_session_generate(
    editor_session: EditorSession,
    serialized_sample_spec: JsonDict,
) -> None:
    result = generate_session_request(
        editor_session,
        serialized_sample_spec,
        editor_session.default_engine,
        editor_session.default_collection_format,
    )

    assert isinstance(result, CodegenResult)
    assert result.engine is editor_session.default_engine


def test_complete_session_request_matches_session_complete(
    editor_session: EditorSession,
    serialized_sample_spec: JsonDict,
) -> None:
    result = complete_session_request(
        editor_session,
        serialized_sample_spec,
        editor_session.default_engine,
        editor_session.default_collection_format,
    )

    assert isinstance(result, EditorResult)
    assert result.confirmed is True


def test_build_template_from_payload_returns_network_spec(
    editor_session: EditorSession,
) -> None:
    spec = build_template_from_payload(
        editor_session,
        "binary_tree",
        {
            "graph_size": 4,
            "bond_dimension": 8,
            "physical_dimension": 5,
        },
    )

    assert isinstance(spec, NetworkSpec)
    assert spec.name == "Binary Tree depth 4"


def test_analyze_serialized_contraction_returns_structured_result(
    serialized_sample_spec: JsonDict,
) -> None:
    result = analyze_serialized_contraction(serialized_sample_spec)

    assert result.network_output_shape == (2, 4)


def test_analyze_serialized_contraction_does_not_revalidate_in_analysis(
    serialized_sample_spec: JsonDict,
) -> None:
    with (
        patch(
            "tensor_network_editor.app._services.validate_spec",
            wraps=app_services_module.validate_spec,
        ) as validate_spec_mock,
        patch(
            "tensor_network_editor.internal.analysis._contraction_analysis.ensure_valid_spec",
            wraps=contraction_analysis_module.ensure_valid_spec,
        ) as ensure_valid_spec_mock,
    ):
        result = analyze_serialized_contraction(serialized_sample_spec)

    assert result.network_output_shape == (2, 4)
    assert validate_spec_mock.call_count == 1
    assert ensure_valid_spec_mock.call_count == 0


def test_generate_session_request_passes_prevalidated_specs_to_generators(
    editor_session: EditorSession,
    serialized_sample_spec: JsonDict,
) -> None:
    prepare_network_calls: list[bool] = []
    original_prepare_network = einsum_codegen_module.prepare_network

    def counting_prepare_network(
        spec: NetworkSpec,
        *,
        validate: bool = True,
    ) -> object:
        prepare_network_calls.append(validate)
        return original_prepare_network(spec, validate=validate)

    with patch(
        "tensor_network_editor.codegen.backends.einsum.prepare_network",
        side_effect=counting_prepare_network,
    ):
        result = generate_session_request(
            editor_session,
            serialized_sample_spec,
            SessionEngineName.EINSUM_NUMPY,
            editor_session.default_collection_format,
        )

    assert isinstance(result, CodegenResult)
    assert prepare_network_calls == [False]


def test_build_bootstrap_payload_preserves_linear_periodic_chain_specs() -> None:
    session = EditorSession(
        initial_spec=build_linear_periodic_chain_spec(),
        default_engine=SessionEngineName.TENSORNETWORK,
    )

    payload = build_bootstrap_payload(session)
    spec_payload = cast(JsonDict, payload["spec"])
    network_payload = cast(JsonDict, spec_payload["network"])
    chain = cast(JsonDict, network_payload["linear_periodic_chain"])
    periodic_cell = cast(JsonDict, chain["periodic_cell"])
    contraction_plan = cast(JsonDict, periodic_cell["contraction_plan"])
    steps = cast(list[JsonDict], contraction_plan["steps"])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert chain["active_cell"] == "periodic"
    assert steps[0]["id"] == "periodic_contract_internal"


def test_build_bootstrap_payload_preserves_grid_periodic_grid_specs() -> None:
    session = EditorSession(
        initial_spec=build_grid_periodic_grid_spec(),
        default_engine=SessionEngineName.TENSORNETWORK,
    )

    payload = build_bootstrap_payload(session)
    spec_payload = cast(JsonDict, payload["spec"])
    network_payload = cast(JsonDict, spec_payload["network"])
    grid = cast(JsonDict, network_payload["grid_periodic_grid"])
    center_cell = cast(JsonDict, grid["center_cell"])
    tensors = cast(list[JsonDict], center_cell["tensors"])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert grid["active_cell"] == "center"
    assert any(tensor["grid_periodic_role"] == "left" for tensor in tensors)


def test_build_bootstrap_payload_preserves_tree_periodic_tree_specs() -> None:
    session = EditorSession(
        initial_spec=build_tree_periodic_tree_spec(),
        default_engine=SessionEngineName.TENSORNETWORK,
    )

    payload = build_bootstrap_payload(session)
    spec_payload = cast(JsonDict, payload["spec"])
    network_payload = cast(JsonDict, spec_payload["network"])
    tree = cast(JsonDict, network_payload["tree_periodic_tree"])
    branch_cell = cast(JsonDict, tree["branch_cell"])
    tensors = cast(list[JsonDict], branch_cell["tensors"])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert tree["active_cell"] == "branch"
    assert tree["branching_factor"] == 3
    assert any(tensor["tree_periodic_role"] == "parent" for tensor in tensors)
