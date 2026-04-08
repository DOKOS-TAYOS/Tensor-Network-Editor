from __future__ import annotations

import tensor_network_editor as tne
from tensor_network_editor.analysis import analyze_contraction, analyze_spec
from tensor_network_editor.diffing import DiffEntityChanges, SpecDiffResult, diff_specs
from tensor_network_editor.linting import LintIssue, LintReport, lint_spec
from tensor_network_editor.templates import build_template_spec, list_template_names
from tests.factories import build_sample_spec, build_three_tensor_spec


def test_package_root_exports_headless_entry_points() -> None:
    assert tne.analyze_spec is analyze_spec
    assert tne.analyze_contraction is analyze_contraction
    assert tne.lint_spec is lint_spec
    assert tne.diff_specs is diff_specs
    assert tne.build_template_spec is build_template_spec
    assert tne.list_template_names is list_template_names


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


def test_analyze_spec_returns_network_and_contraction_sections() -> None:
    report = analyze_spec(build_three_tensor_spec())

    assert report.network.tensor_count == 3
    assert report.network.edge_count == 2
    assert report.network.open_index_count == 2
    assert report.contraction is not None
    assert (
        report.to_dict()["contraction"]["manual"]["summary"]["total_estimated_flops"]
        == 60
    )


def test_diff_specs_compares_entities_by_stable_ids() -> None:
    before = build_sample_spec()
    after = build_sample_spec()
    after.tensors[0].name = "Tensor renamed"
    after.groups.clear()
    after.notes[0].text = "Updated note"

    result = diff_specs(before, after)

    assert result.tensor.changed == ["tensor_a"]
    assert result.group.removed == ["group_demo"]
    assert result.note.changed == ["note_demo"]


def test_list_template_names_is_available_from_public_templates_module() -> None:
    assert list_template_names() == ["mps", "mpo", "peps_2x2", "mera", "binary_tree"]
