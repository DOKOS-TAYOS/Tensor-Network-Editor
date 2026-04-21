from __future__ import annotations

import logging
from pathlib import Path

import pytest

import tensor_network_editor
from tensor_network_editor.api import (
    generate_code,
    load_spec,
    load_spec_from_python_code,
    save_spec,
)
from tensor_network_editor.errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
)
from tensor_network_editor.models import (
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    TensorDataMode,
    TensorDataSpec,
)
from tests.conftest import distribution_for_checkout_import_or_skip
from tests.factories import (
    build_outer_product_plan_spec,
    build_sample_spec_with_view_snapshots,
    build_three_tensor_complete_plan_spec,
    build_three_tensor_hyperedge_spec,
    build_three_tensor_spec,
    build_three_tensor_spec_without_plan,
)


def test_package_version_matches_installed_metadata() -> None:
    distribution = distribution_for_checkout_import_or_skip(tensor_network_editor)

    assert tensor_network_editor.__version__ == distribution.version


def test_package_logger_uses_null_handler() -> None:
    package_logger = logging.getLogger("tensor_network_editor")

    assert any(
        isinstance(handler, logging.NullHandler) for handler in package_logger.handlers
    )


def test_package_root_exports_supported_public_api() -> None:
    assert set(tensor_network_editor.__all__) == {
        "CanvasPosition",
        "CanvasNoteSpec",
        "CodeGenerationError",
        "CodegenResult",
        "ContractionOperandLayoutSpec",
        "ContractionPlanSpec",
        "ContractionStepSpec",
        "ContractionViewSnapshotSpec",
        "EdgeEndpointRef",
        "EdgeSpec",
        "EditorResult",
        "EngineName",
        "GroupSpec",
        "HyperedgeSpec",
        "IndexSpec",
        "NetworkSpec",
        "TensorCollectionFormat",
        "TensorDataMode",
        "TensorDataSpec",
        "TensorSize",
        "TensorSpec",
        "__version__",
        "analyze_contraction",
        "analyze_spec",
        "build_template_spec",
        "canonicalize_spec",
        "diff_specs",
        "generate_code",
        "lint_spec",
        "list_generator_names",
        "launch_tensor_network_editor",
        "list_template_names",
        "load_spec",
        "load_spec_from_python_code",
        "register_generator",
        "register_static_template",
        "register_template",
        "save_spec",
        "SemanticDiffEntry",
        "SemanticFieldChange",
        "SemanticSpecDiffResult",
        "semantic_diff_specs",
        "validate_spec",
    }
    assert tensor_network_editor.generate_code is generate_code
    assert tensor_network_editor.load_spec is load_spec
    assert (
        tensor_network_editor.load_spec_from_python_code is load_spec_from_python_code
    )
    assert tensor_network_editor.save_spec is save_spec
    assert not hasattr(tensor_network_editor, "tensor_network_creation")


@pytest.mark.parametrize("engine", list(EngineName))
def test_generate_code_returns_codegen_result_for_each_engine(
    sample_spec: NetworkSpec,
    engine: EngineName,
) -> None:
    result = generate_code(sample_spec, engine=engine)

    assert result.engine is engine
    assert result.code
    assert isinstance(result.warnings, list)


def test_generate_code_can_print_and_write_code(
    sample_spec: NetworkSpec,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "generated_network.py"

    result = generate_code(
        sample_spec,
        engine=EngineName.EINSUM_NUMPY,
        print_code=True,
        path=output_path,
    )

    assert output_path.read_text(encoding="utf-8") == result.code
    assert capsys.readouterr().out == f"{result.code}\n"


def test_generate_code_wraps_file_write_failures(sample_spec: NetworkSpec) -> None:
    missing_parent_path = Path(".test_output") / "missing_dir" / "generated_network.py"

    with pytest.raises(PackageIOError):
        generate_code(
            sample_spec,
            engine=EngineName.EINSUM_NUMPY,
            path=missing_parent_path,
        )


def test_save_and_load_spec_round_trip_preserves_structure(
    sample_spec: NetworkSpec,
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "network.json"

    save_spec(sample_spec, spec_path)
    loaded_spec = load_spec(spec_path)

    assert [tensor.id for tensor in loaded_spec.tensors] == ["tensor_a", "tensor_b"]
    assert loaded_spec.edges[0].name == "bond_x"
    assert loaded_spec.tensors[0].size.width == 200.0
    assert loaded_spec.groups[0].tensor_ids == ["tensor_a", "tensor_b"]
    assert loaded_spec.notes[0].text == "Check the contraction order"
    assert loaded_spec.contraction_plan is not None
    assert loaded_spec.contraction_plan.steps[0].left_operand_id == "tensor_a"


def test_save_and_load_spec_round_trip_preserves_contraction_view_snapshots(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "network-with-snapshots.json"
    sample_spec = build_sample_spec_with_view_snapshots()

    save_spec(sample_spec, spec_path)
    loaded_spec = load_spec(spec_path)

    assert loaded_spec.contraction_plan is not None
    assert len(loaded_spec.contraction_plan.view_snapshots) == 2
    assert loaded_spec.contraction_plan.view_snapshots[1].applied_step_count == 1
    assert (
        loaded_spec.contraction_plan.view_snapshots[1].operand_layouts[0].operand_id
        == "step_contract_ab"
    )
    assert (
        loaded_spec.contraction_plan.view_snapshots[1].operand_layouts[0].size.width
        == 230.0
    )


def test_load_spec_round_trips_generated_python_file(
    tmp_path: Path,
) -> None:
    sample_spec = build_three_tensor_spec_without_plan()
    spec_path = tmp_path / "network_roundtrip.py"
    generate_code(
        sample_spec,
        engine=EngineName.TENSORNETWORK,
        collection_format=TensorCollectionFormat.DICT,
        path=spec_path,
    )

    loaded_spec = load_spec(spec_path)

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5), (5, 7)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x", "bond_y"]
    assert loaded_spec.groups == []
    assert loaded_spec.notes == []
    assert loaded_spec.contraction_plan is None


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    "collection_format",
    list(TensorCollectionFormat),
)
def test_load_spec_from_python_code_round_trips_generated_source(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> None:
    sample_spec = build_three_tensor_spec_without_plan()
    result = generate_code(
        sample_spec,
        engine=engine,
        collection_format=collection_format,
    )

    loaded_spec = load_spec_from_python_code(result.code)
    expected_edge_names = (
        ["b", "c"]
        if engine in {EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH}
        else ["bond_x", "bond_y"]
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5), (5, 7)]
    assert [edge.name for edge in loaded_spec.edges] == expected_edge_names
    assert loaded_spec.groups == []
    assert loaded_spec.notes == []
    assert loaded_spec.contraction_plan is None


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_round_trips_tensor_data(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_spec_without_plan()
    sample_spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    sample_spec.tensors[1].tensor_data = TensorDataSpec(
        mode=TensorDataMode.FILL,
        fill_value=1.5,
    )
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.tensors[0].tensor_data == TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert loaded_spec.tensors[1].tensor_data == TensorDataSpec(
        mode=TensorDataMode.FILL,
        fill_value=1.5,
    )


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_lowers_hyperedges_to_binary_network(
    engine: EngineName,
) -> None:
    result = generate_code(build_three_tensor_hyperedge_spec(), engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.hyperedges == []
    assert len(loaded_spec.tensors) == 4
    assert len(loaded_spec.edges) == 3
    assert any(tensor.shape == (3, 3, 3) for tensor in loaded_spec.tensors)


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_round_trips_manual_plan_steps(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_spec()
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5), (5, 7)]
    assert len(loaded_spec.edges) == 2
    assert loaded_spec.contraction_plan is not None
    assert loaded_spec.contraction_plan.id == "imported_contraction_plan"
    assert loaded_spec.contraction_plan.name == "Imported manual contraction path"
    assert loaded_spec.contraction_plan.view_snapshots == []
    assert [step.id for step in loaded_spec.contraction_plan.steps] == ["step_ab"]
    assert loaded_spec.contraction_plan.steps[0].left_operand_id == "tensor_a"
    assert loaded_spec.contraction_plan.steps[0].right_operand_id == "tensor_b"


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_round_trips_chained_manual_plan_steps(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_complete_plan_spec()
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.contraction_plan is not None
    assert [step.id for step in loaded_spec.contraction_plan.steps] == [
        "step_ab",
        "step_abc",
    ]
    assert loaded_spec.contraction_plan.steps[1].left_operand_id == "step_ab"
    assert loaded_spec.contraction_plan.steps[1].right_operand_id == "tensor_c"


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_rejects_malformed_manual_step_markup(
    engine: EngineName,
) -> None:
    result = generate_code(build_three_tensor_spec(), engine=engine)
    malformed_code = result.code.replace(
        "# Manual step step_ab",
        "# Manual step step_ab | left=tensor_a",
        1,
    )

    with pytest.raises(SerializationError, match="manual step"):
        load_spec_from_python_code(malformed_code)


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    "collection_format",
    list(TensorCollectionFormat),
)
def test_load_spec_from_python_code_round_trips_empty_network(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> None:
    result = generate_code(
        NetworkSpec(id="network_empty", name="empty network"),
        engine=engine,
        collection_format=collection_format,
    )

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.tensors == []
    assert loaded_spec.edges == []
    assert loaded_spec.groups == []
    assert loaded_spec.notes == []
    assert loaded_spec.contraction_plan is None


def test_generate_code_reports_backend_specific_codegen_errors() -> None:
    with pytest.raises(CodeGenerationError, match="TensorKrowch"):
        generate_code(
            build_outer_product_plan_spec(),
            engine=EngineName.TENSORKROWCH,
        )


def test_load_spec_rejects_unsupported_python_code(tmp_path: Path) -> None:
    spec_path = tmp_path / "unsupported.py"
    spec_path.write_text("print('hello')\n", encoding="utf-8")

    with pytest.raises(SerializationError, match="generated Python code"):
        load_spec(spec_path)


def test_save_spec_wraps_file_write_failures(sample_spec: NetworkSpec) -> None:
    missing_parent_path = Path(".test_output") / "missing_dir" / "network.json"

    with pytest.raises(PackageIOError):
        save_spec(sample_spec, missing_parent_path)


def test_load_spec_wraps_missing_file_failures(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(PackageIOError):
        load_spec(missing_path)


def test_load_spec_wraps_invalid_json_failures(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid_network.json"
    invalid_path.write_text("{not json}", encoding="utf-8")

    with pytest.raises(SerializationError):
        load_spec(invalid_path)
