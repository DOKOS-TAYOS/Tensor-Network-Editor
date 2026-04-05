from __future__ import annotations

import unittest

from tensor_network_editor._contraction_analysis import analyze_contraction
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)


def build_three_tensor_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_chain",
        name="chain",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_y", name="y", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=400.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=5),
                    IndexSpec(id="tensor_c_j", name="j", dimension=7),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            ),
            EdgeSpec(
                id="edge_y",
                name="bond_y",
                left=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_y"),
                right=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_y"),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="plan_chain",
            name="Chain path",
            steps=[
                ContractionStepSpec(
                    id="step_ab",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                )
            ],
        ),
    )


class ContractionAnalysisTests(unittest.TestCase):
    def test_analyze_contraction_reports_manual_pairwise_costs(self) -> None:
        from tests.test_api import build_sample_spec

        result = analyze_contraction(build_sample_spec())

        self.assertEqual(result.manual.status, "complete")
        self.assertEqual(len(result.manual.steps), 1)
        self.assertEqual(result.manual.steps[0].estimated_flops, 24)
        self.assertEqual(result.manual.steps[0].intermediate_size, 8)
        self.assertEqual(result.manual.summary.final_shape, (2, 4))
        self.assertEqual(result.manual.summary.peak_intermediate_size, 8)

    def test_analyze_contraction_marks_incomplete_manual_plan(self) -> None:
        result = analyze_contraction(build_three_tensor_spec())

        self.assertEqual(result.manual.status, "incomplete")
        self.assertEqual(result.manual.summary.total_estimated_flops, 30)
        self.assertIsNone(result.manual.summary.final_shape)
        self.assertEqual(result.manual.summary.peak_intermediate_size, 10)
        self.assertEqual(
            result.manual.summary.remaining_operand_ids, ("step_ab", "tensor_c")
        )


if __name__ == "__main__":
    unittest.main()
