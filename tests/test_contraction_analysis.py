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

        self.assertEqual(result.network_output_shape, (2, 4))
        self.assertEqual(result.manual.status, "complete")
        self.assertEqual(len(result.manual.steps), 1)
        self.assertEqual(result.manual.steps[0].estimated_flops, 48)
        self.assertEqual(result.manual.steps[0].estimated_macs, 24)
        self.assertEqual(result.manual.steps[0].intermediate_size, 8)
        self.assertEqual(result.manual.summary.total_estimated_flops, 48)
        self.assertEqual(result.manual.summary.total_estimated_macs, 24)
        self.assertEqual(result.manual.summary.final_shape, (2, 4))
        self.assertEqual(result.manual.summary.peak_intermediate_size, 8)
        self.assertIsNotNone(result.automatic_global)
        self.assertIsNotNone(result.automatic_local)

    def test_analyze_contraction_marks_incomplete_manual_plan(self) -> None:
        result = analyze_contraction(build_three_tensor_spec())

        self.assertEqual(result.network_output_shape, (2, 7))
        self.assertEqual(result.manual.status, "incomplete")
        self.assertEqual(result.manual.steps[0].estimated_flops, 60)
        self.assertEqual(result.manual.steps[0].estimated_macs, 30)
        self.assertEqual(result.manual.summary.total_estimated_flops, 60)
        self.assertEqual(result.manual.summary.total_estimated_macs, 30)
        self.assertEqual(result.manual.summary.final_shape, (2, 5))
        self.assertEqual(result.manual.summary.peak_intermediate_size, 10)
        self.assertEqual(
            result.manual.summary.remaining_operand_ids, ("step_ab", "tensor_c")
        )
        self.assertIn(result.automatic_global.status, {"complete", "unavailable"})
        self.assertIn(result.automatic_local.status, {"complete", "unavailable"})
        if result.automatic_local.status == "complete":
            self.assertEqual(len(result.automatic_local.steps), 1)
            self.assertEqual(result.automatic_local.summary.total_estimated_flops, 60)
            self.assertEqual(result.automatic_local.summary.total_estimated_macs, 30)
            self.assertEqual(
                {
                    result.automatic_local.steps[0].left_operand_id,
                    result.automatic_local.steps[0].right_operand_id,
                },
                {"tensor_a", "tensor_b"},
            )

    def test_analyze_contraction_accepts_multi_step_manual_plan(self) -> None:
        spec = build_three_tensor_spec()
        spec.contraction_plan = ContractionPlanSpec(
            id="plan_chain_complete",
            name="Chain complete",
            steps=[
                ContractionStepSpec(
                    id="step_ab",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                ),
                ContractionStepSpec(
                    id="step_abc",
                    left_operand_id="step_ab",
                    right_operand_id="tensor_c",
                ),
            ],
        )

        result = analyze_contraction(spec)

        self.assertEqual(result.manual.status, "complete")
        self.assertEqual(len(result.manual.steps), 2)
        self.assertEqual(result.manual.steps[1].left_operand_id, "step_ab")
        self.assertEqual(result.manual.steps[1].right_operand_id, "tensor_c")
        self.assertEqual(result.manual.steps[1].estimated_flops, 140)
        self.assertEqual(result.manual.steps[1].estimated_macs, 70)
        self.assertEqual(result.manual.summary.total_estimated_flops, 200)
        self.assertEqual(result.manual.summary.total_estimated_macs, 100)
        self.assertEqual(result.manual.summary.peak_intermediate_size, 14)
        self.assertEqual(result.manual.summary.final_shape, (2, 7))
        self.assertEqual(result.manual.summary.remaining_operand_ids, ("step_abc",))

    def test_automatic_summaries_do_not_expose_final_shape(self) -> None:
        result = analyze_contraction(build_three_tensor_spec())

        self.assertFalse(hasattr(result.automatic_global.summary, "final_shape"))
        self.assertFalse(hasattr(result.automatic_local.summary, "final_shape"))

    def test_matrix_multiplication_counts_two_flops_per_mac(self) -> None:
        spec = NetworkSpec(
            id="network_mm",
            name="matrix multiply",
            tensors=[
                TensorSpec(
                    id="tensor_a",
                    name="A",
                    position=CanvasPosition(x=80.0, y=120.0),
                    indices=[
                        IndexSpec(id="tensor_a_i", name="i", dimension=2),
                        IndexSpec(id="tensor_a_k", name="k", dimension=2),
                    ],
                ),
                TensorSpec(
                    id="tensor_b",
                    name="B",
                    position=CanvasPosition(x=240.0, y=120.0),
                    indices=[
                        IndexSpec(id="tensor_b_k", name="k", dimension=2),
                        IndexSpec(id="tensor_b_j", name="j", dimension=2),
                    ],
                ),
            ],
            edges=[
                EdgeSpec(
                    id="edge_k",
                    name="bond_k",
                    left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_k"),
                    right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_k"),
                )
            ],
            contraction_plan=ContractionPlanSpec(
                id="plan_mm",
                name="Matrix multiply",
                steps=[
                    ContractionStepSpec(
                        id="step_ab",
                        left_operand_id="tensor_a",
                        right_operand_id="tensor_b",
                    )
                ],
            ),
        )

        result = analyze_contraction(spec)

        self.assertEqual(result.manual.steps[0].estimated_macs, 8)
        self.assertEqual(result.manual.steps[0].estimated_flops, 16)


if __name__ == "__main__":
    unittest.main()
