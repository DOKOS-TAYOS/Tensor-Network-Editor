from __future__ import annotations

from typing import cast

from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app._services import (
    analyze_serialized_contraction,
    build_bootstrap_payload,
    build_template_from_payload,
    complete_session_request,
    generate_session_request,
)
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import CodegenResult, EditorResult, NetworkSpec
from tensor_network_editor.models import EngineName as SessionEngineName
from tensor_network_editor.serialization import SCHEMA_VERSION
from tests.factories import build_linear_periodic_chain_spec


def test_build_bootstrap_payload_matches_session_contract(
    editor_session: EditorSession,
) -> None:
    payload = build_bootstrap_payload(editor_session)

    assert payload["default_engine"] == editor_session.default_engine.value
    assert payload["default_collection_format"] == (
        editor_session.default_collection_format.value
    )
    assert payload["templates"] == list(
        cast(dict[str, object], payload["template_definitions"])
    )


def test_generate_session_request_matches_session_generate(
    editor_session: EditorSession,
    serialized_sample_spec: dict[str, object],
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
    serialized_sample_spec: dict[str, object],
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
    serialized_sample_spec: dict[str, object],
) -> None:
    result = analyze_serialized_contraction(serialized_sample_spec)

    assert result.network_output_shape == (2, 4)


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
