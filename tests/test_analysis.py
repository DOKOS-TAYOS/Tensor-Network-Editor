from __future__ import annotations

import unittest

from tensor_network_editor._analysis import analyze_network
from tensor_network_editor.errors import SpecValidationError
from tensor_network_editor.models import IndexSpec
from tests.test_api import build_sample_spec


class NetworkAnalysisTests(unittest.TestCase):
    def test_analyze_network_builds_lookup_maps_and_open_indices(self) -> None:
        spec = build_sample_spec()

        analysis = analyze_network(spec)

        self.assertEqual(analysis.spec.id, "network_demo")
        self.assertEqual(sorted(analysis.tensor_map), ["tensor_a", "tensor_b"])
        self.assertEqual(
            sorted(analysis.index_map),
            ["tensor_a_i", "tensor_a_x", "tensor_b_j", "tensor_b_x"],
        )
        self.assertEqual(analysis.connected_index_ids, {"tensor_a_x", "tensor_b_x"})
        self.assertEqual(
            [(tensor.id, index.id) for tensor, index in analysis.open_indices],
            [("tensor_a", "tensor_a_i"), ("tensor_b", "tensor_b_j")],
        )
        left_index = analysis.left_index_by_edge_id["edge_x"]
        right_index = analysis.right_index_by_edge_id["edge_x"]
        self.assertIsNotNone(left_index)
        self.assertIsNotNone(right_index)
        assert left_index is not None
        assert right_index is not None
        self.assertEqual(left_index.id, "tensor_a_x")
        self.assertEqual(right_index.id, "tensor_b_x")

    def test_analyze_network_can_validate_before_building_lookups(self) -> None:
        spec = build_sample_spec()
        spec.tensors[0].indices[0] = IndexSpec(id="tensor_a_i", name="", dimension=2)

        with self.assertRaises(SpecValidationError):
            analyze_network(spec, validate=True)


if __name__ == "__main__":
    unittest.main()
