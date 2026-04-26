from __future__ import annotations

from importlib import import_module
from typing import cast
from unittest.mock import patch

import pytest

import tensor_network_editor as tne
from tensor_network_editor.analysis import analyze_contraction, analyze_spec
from tensor_network_editor.builder import NetworkBuilder
from tensor_network_editor.canonicalization import canonicalize_spec
from tensor_network_editor.linting import lint_spec
from tensor_network_editor.models import (
    DiffEntityChanges,
    LintIssue,
    LintReport,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    SpecDiffResult,
)
from tensor_network_editor.templates import (
    build_template_spec,
    list_template_names,
)
from tensor_network_editor.types import JSONValue
from tests.factories import (
    build_linear_periodic_partial_carry_chain_spec,
    build_sample_spec,
    build_three_tensor_spec,
    build_tree_periodic_tree_spec,
)


def test_package_root_exports_headless_entry_points() -> None:
    assert tne.analyze_spec is analyze_spec
    assert tne.analyze_contraction is analyze_contraction
    assert tne.canonicalize_spec is canonicalize_spec
    assert tne.NetworkBuilder is NetworkBuilder
    assert tne.lint_spec is lint_spec
    assert callable(tne.diff_specs)
    assert callable(tne.semantic_diff_specs)
    assert tne.build_template_spec is build_template_spec
    assert tne.list_template_names is list_template_names
    assert not hasattr(tne, "register_static_template")


def test_lint_models_preserve_public_payload_shape() -> None:
    report = LintReport(
        issues=[
            LintIssue(
                severity="warning",
                code="suspicious-open-leg",
                message="Index 'i' is open and may be missing an edge.",
                path="tensors.tensor_a.indices.tensor_a_i",
                suggestion="Connect it or mark it explicitly in metadata.",
            )
        ]
    )

    assert report.has_warnings is True
    assert report.to_dict() == {
        "issues": [
            {
                "severity": "warning",
                "code": "suspicious-open-leg",
                "message": "Index 'i' is open and may be missing an edge.",
                "path": "tensors.tensor_a.indices.tensor_a_i",
                "suggestion": "Connect it or mark it explicitly in metadata.",
            }
        ]
    }


def test_spec_diff_result_serializes_entity_changes() -> None:
    result = SpecDiffResult(
        tensor=DiffEntityChanges(added=["tensor_c"], removed=["tensor_a"]),
        edge=DiffEntityChanges(changed=["edge_x"]),
    )

    assert result.to_dict() == {
        "tensor": {
            "added": ["tensor_c"],
            "removed": ["tensor_a"],
            "changed": [],
        },
        "edge": {
            "added": [],
            "removed": [],
            "changed": ["edge_x"],
        },
        "group": {
            "added": [],
            "removed": [],
            "changed": [],
        },
        "note": {
            "added": [],
            "removed": [],
            "changed": [],
        },
        "plan": {
            "added": [],
            "removed": [],
            "changed": [],
        },
    }


def test_semantic_spec_diff_result_serializes_entries() -> None:
    result = SemanticSpecDiffResult(
        entries=[
            SemanticDiffEntry(
                entity_type="tensor",
                entity_id="tensor_a",
                change_type="changed",
                summary="Tensor changed.",
                field_changes=[
                    SemanticFieldChange(
                        path="metadata.tags",
                        before=["alpha"],
                        after=["alpha", "beta"],
                    )
                ],
            )
        ]
    )

    assert result.to_dict() == {
        "entries": [
            {
                "entity_type": "tensor",
                "entity_id": "tensor_a",
                "change_type": "changed",
                "summary": "Tensor changed.",
                "field_changes": [
                    {
                        "path": "metadata.tags",
                        "before": ["alpha"],
                        "after": ["alpha", "beta"],
                    }
                ],
            }
        ]
    }


def test_analyze_spec_returns_network_and_contraction_sections() -> None:
    report = analyze_spec(build_three_tensor_spec())

    assert report.network.tensor_count == 3
    assert report.network.edge_count == 2
    assert report.network.open_index_count == 2
    assert report.contraction is not None
    payload = report.to_dict()
    contraction_payload = cast(dict[str, JSONValue], payload["contraction"])
    manual_payload = cast(dict[str, JSONValue], contraction_payload["manual"])
    summary_payload = cast(dict[str, JSONValue], manual_payload["summary"])
    assert summary_payload["total_estimated_flops"] == 60


def test_analyze_spec_passes_memory_dtype_to_contraction_analysis() -> None:
    spec = build_three_tensor_spec()

    with patch(
        "tensor_network_editor.analysis._analyze_validated_contraction",
        return_value=None,
    ) as analyze_mock:
        report = analyze_spec(spec, memory_dtype="float32")

    assert report.contraction is None
    analyze_mock.assert_called_once()
    assert analyze_mock.call_args.kwargs["memory_dtype"] == "float32"


def test_analyze_spec_defaults_memory_dtype_to_float64() -> None:
    spec = build_three_tensor_spec()

    with patch(
        "tensor_network_editor.analysis._analyze_validated_contraction",
        return_value=None,
    ) as analyze_mock:
        analyze_spec(spec)

    analyze_mock.assert_called_once()
    assert analyze_mock.call_args.kwargs["memory_dtype"] == "float64"


def test_analyze_spec_reuses_validation_and_analysis_for_standard_specs() -> None:
    spec = build_three_tensor_spec()
    analysis_module = __import__(
        "tensor_network_editor.analysis",
        fromlist=["analyze_network"],
    )

    with (
        patch(
            "tensor_network_editor.analysis.ensure_valid_spec",
            wraps=analysis_module.ensure_valid_spec,
        ) as ensure_valid_spec_mock,
        patch(
            "tensor_network_editor.analysis.analyze_network",
            wraps=analysis_module.analyze_network,
        ) as analysis_analyze_network_mock,
    ):
        report = analyze_spec(spec)

    assert report.contraction is not None
    assert ensure_valid_spec_mock.call_count == 1
    assert analysis_analyze_network_mock.call_count == 1


def test_analyze_spec_reuses_validation_for_linear_periodic_specs() -> None:
    spec = build_linear_periodic_partial_carry_chain_spec()
    analysis_module = __import__(
        "tensor_network_editor.analysis",
        fromlist=["analyze_network"],
    )

    with (
        patch(
            "tensor_network_editor.analysis.ensure_valid_spec",
            wraps=analysis_module.ensure_valid_spec,
        ) as ensure_valid_spec_mock,
        patch(
            "tensor_network_editor.analysis.analyze_network",
            wraps=analysis_module.analyze_network,
        ) as analysis_analyze_network_mock,
    ):
        report = analyze_spec(spec)

    assert report.contraction is not None
    assert ensure_valid_spec_mock.call_count == 1
    assert analysis_analyze_network_mock.call_count == 1


def test_analyze_spec_uses_active_tree_periodic_cell() -> None:
    report = analyze_spec(build_tree_periodic_tree_spec())

    assert report.network.tensor_count == 5
    assert report.network.edge_count == 4
    assert report.network.open_index_count == 0


def test_diff_specs_compares_entities_by_stable_ids() -> None:
    before = build_sample_spec()
    after = build_sample_spec()
    after.tensors[0].name = "Tensor renamed"
    after.groups.clear()
    after.notes[0].text = "Updated note"

    result = tne.diff_specs(before, after)

    assert result.tensor.changed == ["tensor_a"]
    assert result.group.removed == ["group_demo"]
    assert result.note.changed == ["note_demo"]


def test_list_template_names_is_available_from_public_templates_module() -> None:
    assert list_template_names() == [
        "mps",
        "mpo",
        "peps_2x2",
        "mera",
        "binary_tree",
        "ttn",
        "pepo",
        "heisenberg_mps",
        "ising_mps",
        "transverse_ising_mpo",
        "tebd_gate_layer",
    ]


def test_removed_compatibility_modules_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.api")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.serialization")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.diffing")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.codegen.common")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.codegen.base")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.codegen.einsum")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.codegen.modes._grid_periodic_shared")

    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.codegen.modes._tree_periodic_shared")
