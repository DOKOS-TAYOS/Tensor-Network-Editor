from __future__ import annotations

from tensor_network_editor.linting import lint_spec
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from tests.factories import build_sample_spec, build_three_tensor_spec


def lint_codes(
    spec: NetworkSpec,
    *,
    max_tensor_rank: int = 6,
    max_tensor_cardinality: int = 4096,
) -> set[str]:
    report = lint_spec(
        spec,
        max_tensor_rank=max_tensor_rank,
        max_tensor_cardinality=max_tensor_cardinality,
    )
    return {issue.code for issue in report.issues}


def test_lint_spec_reports_disconnected_components() -> None:
    spec = build_sample_spec()
    spec.tensors.append(
        TensorSpec(
            id="tensor_c",
            name="C",
            position=CanvasPosition(x=560.0, y=120.0),
            indices=[IndexSpec(id="tensor_c_k", name="k", dimension=2)],
        )
    )

    assert "disconnected-components" in lint_codes(spec)


def test_lint_spec_reports_suspicious_open_legs_large_tensors_and_empty_groups() -> (
    None
):
    spec = build_sample_spec()
    spec.groups.append(GroupSpec(id="group_empty", name="Empty group", tensor_ids=[]))

    codes = lint_codes(spec, max_tensor_rank=1, max_tensor_cardinality=5)

    assert "suspicious-open-index" in codes
    assert "large-tensor-rank" in codes
    assert "large-tensor-cardinality" in codes
    assert "empty-group" in codes


def test_lint_spec_reports_uninformative_names() -> None:
    spec = build_sample_spec()
    spec.tensors[0].name = "Tensor"
    spec.groups[0].name = "Group"

    codes = lint_codes(spec)

    assert "uninformative-name" in codes


def test_lint_spec_reports_incomplete_manual_plan() -> None:
    spec = build_three_tensor_spec()

    assert "incomplete-manual-plan" in lint_codes(spec)


def test_lint_spec_detects_invalidated_manual_suffix_best_effort() -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_invalid_suffix",
        name="Invalid suffix",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_bad",
                left_operand_id="tensor_a",
                right_operand_id="tensor_c",
            ),
        ],
    )

    assert "invalidated-manual-suffix" in lint_codes(spec)
