from __future__ import annotations

from tensor_network_editor.canonicalization import canonicalize_spec
from tests.factories import build_sample_spec, build_three_tensor_complete_plan_spec


def test_canonicalize_spec_sorts_entities_and_normalizes_tags() -> None:
    spec = build_sample_spec()
    spec.metadata = {"zeta": {"tags": [" beta ", "alpha", "beta"]}, "alpha": 1}
    spec.tensors.reverse()
    spec.groups[0].metadata = {"tags": [" block ", "alpha", "block"]}
    spec.notes.append(spec.notes.pop(0))
    spec.edges[0].metadata = {"tags": [" bond ", "alpha", "bond"]}

    canonical = canonicalize_spec(spec)
    assert canonical.contraction_plan is not None

    assert canonical is not spec
    assert [tensor.id for tensor in canonical.tensors] == ["tensor_a", "tensor_b"]
    assert [edge.id for edge in canonical.edges] == ["edge_x"]
    assert [group.id for group in canonical.groups] == ["group_demo"]
    assert [note.id for note in canonical.notes] == ["note_demo"]
    assert list(canonical.metadata.keys()) == ["alpha", "zeta"]
    assert canonical.metadata["zeta"] == {"tags": ["alpha", "beta"]}
    assert canonical.groups[0].metadata["tags"] == ["alpha", "block"]
    assert canonical.edges[0].metadata["tags"] == ["alpha", "bond"]
    assert [step.id for step in canonical.contraction_plan.steps] == [
        "step_contract_ab"
    ]


def test_canonicalize_spec_rewrites_ids_deterministically() -> None:
    spec = build_three_tensor_complete_plan_spec()
    spec.metadata = {"tags": [" z ", "a", "z"]}
    spec.tensors.reverse()
    spec.edges.reverse()
    spec.groups = []
    spec.notes = []

    canonical = canonicalize_spec(spec, deterministic_ids=True)
    assert canonical.contraction_plan is not None

    assert canonical.id == "network_001"
    assert [tensor.id for tensor in canonical.tensors] == [
        "tensor_001",
        "tensor_002",
        "tensor_003",
    ]
    assert [index.id for index in canonical.tensors[0].indices] == [
        "index_001",
        "index_002",
    ]
    assert canonical.edges[0].id == "edge_001"
    assert canonical.edges[0].left.tensor_id == "tensor_001"
    assert canonical.edges[0].right.tensor_id == "tensor_002"
    assert canonical.contraction_plan.id == "plan_001"
    assert [step.id for step in canonical.contraction_plan.steps] == [
        "step_001",
        "step_002",
    ]
    assert canonical.contraction_plan.steps[1].left_operand_id == "step_001"
    assert canonical.contraction_plan.steps[1].right_operand_id == "tensor_003"
    assert canonical.metadata["tags"] == ["a", "z"]
