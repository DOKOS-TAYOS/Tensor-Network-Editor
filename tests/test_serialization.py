from __future__ import annotations

from copy import deepcopy

import pytest

from tensor_network_editor.errors import SerializationError, SpecValidationError
from tensor_network_editor.models import NetworkSpec
from tensor_network_editor.serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    serialize_spec,
)


def test_serialize_spec_wraps_valid_network_with_schema(
    sample_spec: NetworkSpec,
) -> None:
    payload = serialize_spec(sample_spec)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["network"]["id"] == sample_spec.id
    assert payload["network"]["notes"][0]["text"] == "Check the contraction order"


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


def test_deserialize_spec_can_skip_validation(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    payload["network"]["tensors"][0]["name"] = "   "

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
    del payload["network"]["tensors"][0]["id"]

    with pytest.raises(SerializationError, match="malformed network object"):
        deserialize_spec(payload)
