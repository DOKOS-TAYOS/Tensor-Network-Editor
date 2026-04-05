from __future__ import annotations

import unittest

from tensor_network_editor._templates import (
    TemplateParameters,
    build_template_spec,
    list_template_names,
)
from tensor_network_editor.validation import ensure_valid_spec


class LayoutAndTemplateTests(unittest.TestCase):
    def test_template_catalog_exposes_expected_names(self) -> None:
        names = list_template_names()

        self.assertEqual(names, ["mps", "mpo", "peps_2x2", "mera", "binary_tree"])

    def test_all_templates_build_valid_specs(self) -> None:
        for template_name in list_template_names():
            with self.subTest(template_name=template_name):
                spec = build_template_spec(template_name)
                validated = ensure_valid_spec(spec)
                self.assertGreater(len(validated.tensors), 0)

    def test_templates_use_more_generous_spacing(self) -> None:
        expected_min_spans = {
            "mps": (900.0, 0.0),
            "mpo": (930.0, 0.0),
            "peps_2x2": (300.0, 260.0),
            "mera": (560.0, 420.0),
            "binary_tree": (560.0, 420.0),
        }

        for template_name, (min_width, min_height) in expected_min_spans.items():
            with self.subTest(template_name=template_name):
                spec = build_template_spec(template_name)
                x_positions = [tensor.position.x for tensor in spec.tensors]
                y_positions = [tensor.position.y for tensor in spec.tensors]
                self.assertGreaterEqual(max(x_positions) - min(x_positions), min_width)
                self.assertGreaterEqual(max(y_positions) - min(y_positions), min_height)

    def test_templates_expose_expected_index_sets_per_tensor(self) -> None:
        expected_index_names = {
            "mps": {
                "A1": {"right", "phys"},
                "A2": {"left", "right", "phys"},
                "A3": {"left", "right", "phys"},
                "A4": {"left", "phys"},
            },
            "mpo": {
                "W1": {"right", "bra", "ket"},
                "W2": {"left", "right", "bra", "ket"},
                "W3": {"left", "right", "bra", "ket"},
                "W4": {"left", "bra", "ket"},
            },
            "peps_2x2": {
                "A": {"right", "down", "phys"},
                "B": {"left", "down", "phys"},
                "C": {"right", "up", "phys"},
                "D": {"left", "up", "phys"},
            },
            "mera": {
                "Top": {"left", "right"},
                "Mid L": {"up", "left", "down"},
                "Mid R": {"up", "down", "right"},
                "Leaf L": {"up", "phys"},
                "Leaf M": {"left", "right", "phys"},
                "Leaf R": {"up", "phys"},
            },
            "binary_tree": {
                "Root": {"left", "right"},
                "Left": {"up", "left", "right"},
                "Right": {"up", "left", "right"},
                "LL": {"up", "phys"},
                "LR": {"up", "phys"},
                "RL": {"up", "phys"},
                "RR": {"up", "phys"},
            },
        }

        for template_name, expected_tensors in expected_index_names.items():
            with self.subTest(template_name=template_name):
                spec = build_template_spec(template_name)
                actual_tensors = {
                    tensor.name: {index.name for index in tensor.indices}
                    for tensor in spec.tensors
                }
                self.assertEqual(actual_tensors, expected_tensors)

    def test_mps_template_accepts_custom_length_and_dimensions(self) -> None:
        spec = build_template_spec(
            "mps",
            TemplateParameters(
                graph_size=5,
                bond_dimension=7,
                physical_dimension=11,
            ),
        )

        self.assertEqual(spec.name, "MPS (5 sites)")
        self.assertEqual(len(spec.tensors), 5)
        self.assertEqual(len(spec.edges), 4)
        self.assertEqual(
            [tensor.name for tensor in spec.tensors], ["A1", "A2", "A3", "A4", "A5"]
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name in {"left", "right"}
            },
            {7},
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name == "phys"
            },
            {11},
        )

    def test_peps_template_accepts_custom_side_length_and_dimensions(self) -> None:
        spec = build_template_spec(
            "peps_2x2",
            TemplateParameters(
                graph_size=3,
                bond_dimension=5,
                physical_dimension=2,
            ),
        )

        self.assertEqual(spec.name, "PEPS 3x3")
        self.assertEqual(len(spec.tensors), 9)
        self.assertEqual(len(spec.edges), 12)
        center_tensor = next(tensor for tensor in spec.tensors if tensor.name == "B2")
        self.assertEqual(
            {index.name for index in center_tensor.indices},
            {"left", "right", "up", "down", "phys"},
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name in {"left", "right", "up", "down"}
            },
            {5},
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name == "phys"
            },
            {2},
        )

    def test_mera_template_accepts_custom_depth_and_dimensions(self) -> None:
        spec = build_template_spec(
            "mera",
            TemplateParameters(
                graph_size=4,
                bond_dimension=6,
                physical_dimension=3,
            ),
        )

        self.assertEqual(spec.name, "MERA depth 4")
        self.assertEqual(len(spec.tensors), 10)
        self.assertEqual(len(spec.edges), 12)
        bottom_layer = [
            tensor for tensor in spec.tensors if tensor.name.startswith("L4-")
        ]
        self.assertEqual(len(bottom_layer), 4)
        self.assertTrue(
            all(
                any(index.name == "phys" for index in tensor.indices)
                for tensor in bottom_layer
            )
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name != "phys"
            },
            {6},
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in bottom_layer
                for index in tensor.indices
                if index.name == "phys"
            },
            {3},
        )

    def test_binary_tree_template_accepts_custom_depth_and_dimensions(self) -> None:
        spec = build_template_spec(
            "binary_tree",
            TemplateParameters(
                graph_size=4,
                bond_dimension=8,
                physical_dimension=5,
            ),
        )

        self.assertEqual(spec.name, "Binary Tree depth 4")
        self.assertEqual(len(spec.tensors), 15)
        self.assertEqual(len(spec.edges), 14)
        leaves = [tensor for tensor in spec.tensors if tensor.name.startswith("L4-")]
        self.assertEqual(len(leaves), 8)
        self.assertTrue(
            all(
                any(index.name == "phys" for index in tensor.indices)
                for tensor in leaves
            )
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in spec.tensors
                for index in tensor.indices
                if index.name != "phys"
            },
            {8},
        )
        self.assertEqual(
            {
                index.dimension
                for tensor in leaves
                for index in tensor.indices
                if index.name == "phys"
            },
            {5},
        )
