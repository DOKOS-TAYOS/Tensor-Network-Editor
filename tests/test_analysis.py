from __future__ import annotations

import unittest

from tensor_network_editor._analysis import analyze_network
from tests.test_api import build_sample_spec


class NetworkAnalysisTests(unittest.TestCase):
    def test_analyze_network_builds_lookup_maps_and_open_indices(self) -> None:
        spec = build_sample_spec()

        analysis = analyze_network(spec)

        self.assertEqual(analysis.spec.id, "network_demo")
        self.assertEqual(sorted(analysis.tensor_map), ["tensor_a", "tensor_b"])
        self.assertEqual(sorted(analysis.index_map), ["tensor_a_i", "tensor_a_x", "tensor_b_j", "tensor_b_x"])
        self.assertEqual(analysis.connected_index_ids, {"tensor_a_x", "tensor_b_x"})
        self.assertEqual(
            [(tensor.id, index.id) for tensor, index in analysis.open_indices],
            [("tensor_a", "tensor_a_i"), ("tensor_b", "tensor_b_j")],
        )
        self.assertEqual(analysis.left_index_by_edge_id["edge_x"].id, "tensor_a_x")
        self.assertEqual(analysis.right_index_by_edge_id["edge_x"].id, "tensor_b_x")


if __name__ == "__main__":
    unittest.main()
