from __future__ import annotations

import logging
import unittest
from pathlib import Path
from uuid import uuid4

import tensor_network_editor
from tensor_network_editor.api import generate_code, load_spec, save_spec
from tensor_network_editor.errors import PackageIOError, SerializationError
from tensor_network_editor.models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)


def build_sample_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_demo",
        name="demo",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=120.0, y=160.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=360.0, y=160.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_j", name="j", dimension=4),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            )
        ],
    )


def build_output_path(filename: str) -> Path:
    output_dir = Path.cwd() / ".test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{uuid4().hex}_{filename}"


class PublicApiTests(unittest.TestCase):
    def test_package_logger_uses_null_handler(self) -> None:
        package_logger = logging.getLogger("tensor_network_editor")

        self.assertTrue(
            any(
                isinstance(handler, logging.NullHandler)
                for handler in package_logger.handlers
            )
        )

    def test_package_root_exposes_canonical_public_api_only(self) -> None:
        self.assertTrue(hasattr(tensor_network_editor, "launch_tensor_network_editor"))
        self.assertFalse(hasattr(tensor_network_editor, "tensor_network_creation"))

    def test_generate_code_returns_codegen_result_for_each_engine(self) -> None:
        spec = build_sample_spec()

        for engine in EngineName:
            with self.subTest(engine=engine):
                result = generate_code(spec, engine=engine)
                self.assertEqual(result.engine, engine)
                self.assertTrue(result.code)
                self.assertIsInstance(result.warnings, list)

    def test_generate_code_writes_code_to_requested_path(self) -> None:
        spec = build_sample_spec()
        output_path = build_output_path("generated_network.py")
        try:
            result = generate_code(spec, engine=EngineName.EINSUM, path=output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), result.code)
        finally:
            output_path.unlink(missing_ok=True)

    def test_generate_code_wraps_file_write_failures(self) -> None:
        spec = build_sample_spec()
        missing_parent_path = (
            Path.cwd() / ".test_output" / "missing_dir" / "generated_network.py"
        )

        with self.assertRaises(PackageIOError):
            generate_code(spec, engine=EngineName.EINSUM, path=missing_parent_path)

    def test_save_and_load_spec_round_trip_preserves_tensor_order(self) -> None:
        spec = build_sample_spec()
        spec_path = build_output_path("network.json")
        try:
            save_spec(spec, spec_path)
            loaded_spec = load_spec(spec_path)
        finally:
            spec_path.unlink(missing_ok=True)

        self.assertEqual(
            [tensor.id for tensor in loaded_spec.tensors], ["tensor_a", "tensor_b"]
        )
        self.assertEqual(loaded_spec.edges[0].name, "bond_x")

    def test_save_spec_wraps_file_write_failures(self) -> None:
        spec = build_sample_spec()
        missing_parent_path = Path.cwd() / ".test_output" / "missing_dir" / "network.json"

        with self.assertRaises(PackageIOError):
            save_spec(spec, missing_parent_path)

    def test_load_spec_wraps_missing_file_failures(self) -> None:
        missing_path = Path.cwd() / ".test_output" / "does_not_exist.json"

        with self.assertRaises(PackageIOError):
            load_spec(missing_path)

    def test_load_spec_wraps_invalid_json_failures(self) -> None:
        invalid_path = build_output_path("invalid_network.json")
        invalid_path.write_text("{not json}", encoding="utf-8")

        try:
            with self.assertRaises(SerializationError):
                load_spec(invalid_path)
        finally:
            invalid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
