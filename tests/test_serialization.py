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
from tests.factories import build_sample_spec_with_view_snapshots


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
