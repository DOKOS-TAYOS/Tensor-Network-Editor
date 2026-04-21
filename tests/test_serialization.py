from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from tensor_network_editor.errors import SerializationError, SpecValidationError
from tensor_network_editor.models import NetworkSpec, TensorDataMode, TensorDataSpec
from tensor_network_editor.serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    serialize_spec,
)
from tensor_network_editor.types import JSONValue
from tests.factories import (
    build_grid_periodic_grid_spec,
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


def test_serialize_spec_preserves_tensor_data_payload() -> None:
    spec = build_sample_spec_with_view_snapshots()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.FILL,
        fill_value=2.5,
    )

    payload = serialize_spec(spec)
    network_payload = cast(dict[str, JSONValue], payload["network"])
    tensors_payload = cast(list[JSONValue], network_payload["tensors"])
    first_tensor = cast(dict[str, JSONValue], tensors_payload[0])
    tensor_data_payload = cast(dict[str, JSONValue], first_tensor["tensor_data"])

    assert payload["schema_version"] == 5
    assert tensor_data_payload == {
        "mode": "fill",
        "fill_value": 2.5,
    }


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


def test_deserialize_spec_accepts_schema_version_4_without_tensor_data(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = deepcopy(serialized_sample_spec)
    payload["schema_version"] = 4

    restored = deserialize_spec(payload)

    assert restored.tensors[0].tensor_data is None


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


def test_serialize_spec_preserves_grid_periodic_grid_payload() -> None:
    payload = serialize_spec(build_grid_periodic_grid_spec())

    network_payload = cast(dict[str, JSONValue], payload["network"])
    grid_payload = cast(dict[str, JSONValue], network_payload["grid_periodic_grid"])
    center_cell_payload = cast(dict[str, JSONValue], grid_payload["center_cell"])
    center_tensors = cast(list[JSONValue], center_cell_payload["tensors"])
    left_boundary_tensor = cast(
        dict[str, JSONValue],
        next(
            tensor_payload
            for tensor_payload in center_tensors
            if cast(dict[str, JSONValue], tensor_payload)["grid_periodic_role"]
            == "left"
        ),
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert grid_payload["active_cell"] == "center"
    assert left_boundary_tensor["grid_periodic_role"] == "left"


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
    payload["schema_version"] = 3

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


def test_deserialize_spec_round_trips_grid_periodic_grid() -> None:
    restored = deserialize_spec(serialize_spec_payload(build_grid_periodic_grid_spec()))

    assert restored.grid_periodic_grid is not None
    assert restored.grid_periodic_grid.active_cell.value == "center"
    center_boundary = next(
        tensor
        for tensor in restored.grid_periodic_grid.center_cell.tensors
        if tensor.grid_periodic_role is not None
        and tensor.grid_periodic_role.value == "left"
    )
    assert center_boundary.grid_periodic_role is not None
    assert center_boundary.grid_periodic_role.value == "left"


def test_deserialize_spec_round_trips_tree_periodic_tree() -> None:
    restored = deserialize_spec(build_tree_periodic_tree_payload(), validate=False)

    assert hasattr(restored, "tree_periodic_tree")
    tree_payload = restored.tree_periodic_tree
    assert tree_payload is not None
    assert tree_payload.active_cell.value == "branch"
    assert tree_payload.branching_factor == 3


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


def build_tree_periodic_tree_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "network": {
            "id": "network_tree_periodic",
            "name": "tree-periodic-tree",
            "tensors": [],
            "groups": [],
            "edges": [],
            "notes": [],
            "contraction_plan": None,
            "linear_periodic_chain": None,
            "grid_periodic_grid": None,
            "tree_periodic_tree": {
                "active_cell": "branch",
                "branching_factor": 3,
                "root_cell": {
                    "tensors": [
                        {
                            "id": "root_tensor",
                            "name": "Root",
                            "position": {"x": 220.0, "y": 120.0},
                            "size": {"width": 180.0, "height": 108.0},
                            "indices": [
                                {
                                    "id": "root_child_0",
                                    "name": "child_0",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "root_child_1",
                                    "name": "child_1",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "root_child_2",
                                    "name": "child_2",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                            ],
                            "linear_periodic_role": None,
                            "grid_periodic_role": None,
                            "metadata": {},
                        },
                        *[
                            {
                                "id": f"root_child_boundary_{child_index}",
                                "name": f"Child {child_index}",
                                "position": {
                                    "x": 120.0 + child_index * 100.0,
                                    "y": 320.0,
                                },
                                "size": {"width": 180.0, "height": 108.0},
                                "indices": [
                                    {
                                        "id": f"root_child_slot_{child_index}",
                                        "name": f"slot_{child_index}",
                                        "dimension": 2,
                                        "offset": {"x": 0.0, "y": 0.0},
                                        "metadata": {},
                                    }
                                ],
                                "linear_periodic_role": None,
                                "grid_periodic_role": None,
                                "tree_periodic_role": "child",
                                "tree_periodic_child_index": child_index,
                                "metadata": {},
                            }
                            for child_index in range(3)
                        ],
                    ],
                    "groups": [],
                    "edges": [
                        {
                            "id": f"root_edge_{child_index}",
                            "name": f"root_edge_{child_index}",
                            "left": {
                                "tensor_id": "root_tensor",
                                "index_id": f"root_child_{child_index}",
                            },
                            "right": {
                                "tensor_id": f"root_child_boundary_{child_index}",
                                "index_id": f"root_child_slot_{child_index}",
                            },
                            "metadata": {},
                        }
                        for child_index in range(3)
                    ],
                    "notes": [],
                    "contraction_plan": None,
                    "metadata": {},
                },
                "branch_cell": {
                    "tensors": [
                        {
                            "id": "branch_tensor",
                            "name": "Branch",
                            "position": {"x": 220.0, "y": 220.0},
                            "size": {"width": 180.0, "height": 108.0},
                            "indices": [
                                {
                                    "id": "branch_parent",
                                    "name": "parent",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "branch_child_0",
                                    "name": "child_0",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "branch_child_1",
                                    "name": "child_1",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "branch_child_2",
                                    "name": "child_2",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                            ],
                            "linear_periodic_role": None,
                            "grid_periodic_role": None,
                            "metadata": {},
                        },
                        {
                            "id": "branch_parent_boundary",
                            "name": "Parent",
                            "position": {"x": 220.0, "y": 40.0},
                            "size": {"width": 180.0, "height": 108.0},
                            "indices": [
                                {
                                    "id": "branch_parent_slot",
                                    "name": "parent_slot",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                }
                            ],
                            "linear_periodic_role": None,
                            "grid_periodic_role": None,
                            "tree_periodic_role": "parent",
                            "tree_periodic_child_index": None,
                            "metadata": {},
                        },
                        *[
                            {
                                "id": f"branch_child_boundary_{child_index}",
                                "name": f"Child {child_index}",
                                "position": {
                                    "x": 120.0 + child_index * 100.0,
                                    "y": 400.0,
                                },
                                "size": {"width": 180.0, "height": 108.0},
                                "indices": [
                                    {
                                        "id": f"branch_child_slot_{child_index}",
                                        "name": f"slot_{child_index}",
                                        "dimension": 2,
                                        "offset": {"x": 0.0, "y": 0.0},
                                        "metadata": {},
                                    }
                                ],
                                "linear_periodic_role": None,
                                "grid_periodic_role": None,
                                "tree_periodic_role": "child",
                                "tree_periodic_child_index": child_index,
                                "metadata": {},
                            }
                            for child_index in range(3)
                        ],
                    ],
                    "groups": [],
                    "edges": [
                        {
                            "id": "branch_edge_parent",
                            "name": "branch_edge_parent",
                            "left": {
                                "tensor_id": "branch_parent_boundary",
                                "index_id": "branch_parent_slot",
                            },
                            "right": {
                                "tensor_id": "branch_tensor",
                                "index_id": "branch_parent",
                            },
                            "metadata": {},
                        },
                        *[
                            {
                                "id": f"branch_edge_{child_index}",
                                "name": f"branch_edge_{child_index}",
                                "left": {
                                    "tensor_id": "branch_tensor",
                                    "index_id": f"branch_child_{child_index}",
                                },
                                "right": {
                                    "tensor_id": f"branch_child_boundary_{child_index}",
                                    "index_id": f"branch_child_slot_{child_index}",
                                },
                                "metadata": {},
                            }
                            for child_index in range(3)
                        ],
                    ],
                    "notes": [],
                    "contraction_plan": None,
                    "metadata": {},
                },
                "leaf_cell": {
                    "tensors": [
                        {
                            "id": "leaf_tensor",
                            "name": "Leaf",
                            "position": {"x": 220.0, "y": 220.0},
                            "size": {"width": 180.0, "height": 108.0},
                            "indices": [
                                {
                                    "id": "leaf_parent",
                                    "name": "parent",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                                {
                                    "id": "leaf_phys",
                                    "name": "phys",
                                    "dimension": 3,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                },
                            ],
                            "linear_periodic_role": None,
                            "grid_periodic_role": None,
                            "metadata": {},
                        },
                        {
                            "id": "leaf_parent_boundary",
                            "name": "Parent",
                            "position": {"x": 220.0, "y": 40.0},
                            "size": {"width": 180.0, "height": 108.0},
                            "indices": [
                                {
                                    "id": "leaf_parent_slot",
                                    "name": "parent_slot",
                                    "dimension": 2,
                                    "offset": {"x": 0.0, "y": 0.0},
                                    "metadata": {},
                                }
                            ],
                            "linear_periodic_role": None,
                            "grid_periodic_role": None,
                            "tree_periodic_role": "parent",
                            "tree_periodic_child_index": None,
                            "metadata": {},
                        },
                    ],
                    "groups": [],
                    "edges": [
                        {
                            "id": "leaf_edge_parent",
                            "name": "leaf_edge_parent",
                            "left": {
                                "tensor_id": "leaf_parent_boundary",
                                "index_id": "leaf_parent_slot",
                            },
                            "right": {
                                "tensor_id": "leaf_tensor",
                                "index_id": "leaf_parent",
                            },
                            "metadata": {},
                        }
                    ],
                    "notes": [],
                    "contraction_plan": None,
                    "metadata": {},
                },
                "metadata": {},
            },
            "metadata": {},
        },
    }
