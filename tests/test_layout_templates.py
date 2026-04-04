from __future__ import annotations

import unittest

from tensor_network_editor._templates import build_template_spec, list_template_names
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
