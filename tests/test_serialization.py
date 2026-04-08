from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from tensor_network_editor.errors import SerializationError, SpecValidationError
from tensor_network_editor.models import NetworkSpec
from tensor_network_editor.serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    serialize_spec,
)
from tensor_network_editor.types import JSONValue
from tests.factories import (
    build_linear_periodic_chain_spec,
    build_sample_spec_with_view_snapshots,
)


def test_serialize_spec_wraps_valid_network_with_schema(
    sample_spec: NetworkSpec,
) -> None:
    payload = serialize_spec(sample_spec)
    network_payload = cast(dict[str, JSONValue], payload["network"])
    notes_payload = cast(list[JSONValue], network_payload["notes"])
    first_note = cast(dict[str, JSONValue], notes_payload[0])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert network_payload["id"] == sample_spec.id
    assert first_note["text"] == "Check the contraction order"


def test_serialize_spec_rejects_invalid_network(sample_spec: NetworkSpec) -> None:
    sample_spec.tensors[0].name = "   "

    with pytest.raises(SpecValidationError):
        serialize_spec(sample_spec)


def test_deserialize_spec_round_trips_valid_payload(
    serialized_sample_spec: dict[str, object],
) -> None:
    restored = deserialize_spec(serialized_sample_spec)

    assert restored.id == "network_demo"
    assert [tensor.name for tensor in restored.tensors] == ["A", "B"]
    assert restored.contraction_plan is not None
    assert restored.contraction_plan.steps[0].id == "step_contract_ab"


def test_serialize_spec_preserves_contraction_view_snapshots() -> None:
    payload = serialize_spec(build_sample_spec_with_view_snapshots())

    network_payload = cast(dict[str, JSONValue], payload["network"])
    contraction_plan_payload = cast(
        dict[str, JSONValue], network_payload["contraction_plan"]
    )
    view_snapshots = cast(list[JSONValue], contraction_plan_payload["view_snapshots"])
    latest_snapshot = cast(dict[str, JSONValue], view_snapshots[-1])
    operand_layouts = cast(list[JSONValue], latest_snapshot["operand_layouts"])
    latest_layout = cast(dict[str, JSONValue], operand_layouts[0])
    latest_size = cast(dict[str, JSONValue], latest_layout["size"])

    assert len(view_snapshots) == 2
    assert latest_snapshot["applied_step_count"] == 1
    assert latest_layout["operand_id"] == "step_contract_ab"
    assert latest_size["width"] == 230.0


def test_serialize_spec_preserves_linear_periodic_chain_payload() -> None:
    payload = serialize_spec(build_linear_periodic_chain_spec())

    network_payload = cast(dict[str, JSONValue], payload["network"])
    chain_payload = cast(dict[str, JSONValue], network_payload["linear_periodic_chain"])
    periodic_cell_payload = cast(dict[str, JSONValue], chain_payload["periodic_cell"])
    periodic_tensors = cast(list[JSONValue], periodic_cell_payload["tensors"])
    boundary_tensor = cast(dict[str, JSONValue], periodic_tensors[2])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert chain_payload["active_cell"] == "periodic"
    assert boundary_tensor["linear_periodic_role"] == "previous"


def test_deserialize_spec_can_skip_validation(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    network_payload = cast(dict[str, object], payload["network"])
    tensors_payload = cast(list[object], network_payload["tensors"])
    first_tensor = cast(dict[str, object], tensors_payload[0])
    first_tensor["name"] = "   "

    restored = deserialize_spec(payload, validate=False)

    assert restored.tensors[0].name == "   "


def test_deserialize_spec_rejects_missing_schema_version() -> None:
    with pytest.raises(SerializationError, match="schema version"):
        deserialize_spec({"network": {}})


def test_deserialize_spec_rejects_boolean_schema_version() -> None:
    with pytest.raises(SerializationError, match="schema version"):
        deserialize_spec({"schema_version": True, "network": {}})


def test_deserialize_spec_rejects_non_integral_schema_version(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    payload["schema_version"] = 3.9

    with pytest.raises(SerializationError, match="schema version"):
        deserialize_spec(payload)


def test_deserialize_spec_rejects_unsupported_schema_version(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    payload["schema_version"] = SCHEMA_VERSION - 1

    with pytest.raises(SerializationError, match="Unsupported schema version"):
        deserialize_spec(payload)


def test_deserialize_spec_rejects_non_object_network_payload() -> None:
    with pytest.raises(SerializationError, match="'network' object"):
        deserialize_spec({"schema_version": SCHEMA_VERSION, "network": []})


def test_deserialize_spec_rejects_malformed_network_payload(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    network_payload = cast(dict[str, object], payload["network"])
    tensors_payload = cast(list[object], network_payload["tensors"])
    first_tensor = cast(dict[str, object], tensors_payload[0])
    del first_tensor["id"]

    with pytest.raises(SerializationError, match="malformed network object"):
        deserialize_spec(payload)


def test_deserialize_spec_round_trips_linear_periodic_chain() -> None:
    restored = deserialize_spec(
        serialize_spec_payload(build_linear_periodic_chain_spec())
    )

    assert restored.linear_periodic_chain is not None
    assert restored.linear_periodic_chain.active_cell.value == "periodic"
    assert (
        restored.linear_periodic_chain.initial_cell.tensors[1].linear_periodic_role
        is not None
    )
    assert (
        restored.linear_periodic_chain.initial_cell.tensors[
            1
        ].linear_periodic_role.value
        == "next"
    )
    assert restored.linear_periodic_chain.periodic_cell.contraction_plan is not None
    assert (
        restored.linear_periodic_chain.periodic_cell.contraction_plan.steps[0].id
        == "periodic_contract_internal"
    )


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        ("name", False),
        ("tensors.0.name", 123),
        ("tensors.0.indices.0.name", None),
        ("notes.0.text", 7),
        ("contraction_plan.name", []),
        ("contraction_plan.steps.0.left_operand_id", 9),
        ("linear_periodic_chain.active_cell", []),
    ],
)
def test_deserialize_spec_rejects_non_string_text_fields(
    serialized_sample_spec: dict[str, object],
    field_path: str,
    value: object,
) -> None:
    payload = deepcopy(serialized_sample_spec)
    if field_path.startswith("linear_periodic_chain."):
        payload = serialize_spec_payload(build_linear_periodic_chain_spec())
    current = cast(dict[str, object], payload["network"])
    path_parts = field_path.split(".")
    for path_part in path_parts[:-1]:
        if path_part.isdigit():
            current = cast(
                dict[str, object], cast(list[object], current)[int(path_part)]
            )
            continue
        current = cast(dict[str, object], current[path_part])
    last_part = path_parts[-1]
    if last_part.isdigit():
        cast(list[object], current)[int(last_part)] = value
    else:
        current[last_part] = value

    with pytest.raises(SerializationError, match="malformed network object"):
        deserialize_spec(payload)


def serialize_spec_payload(spec: NetworkSpec) -> dict[str, object]:
    return cast(dict[str, object], serialize_spec(spec))
