from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest

import tensor_network_editor
from tensor_network_editor import generate_code as _generate_code
from tensor_network_editor.editor import EditorLaunchOptions, EditorUiMode, open_editor
from tensor_network_editor.errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
)
from tensor_network_editor.internal.io._serialization import (
    deserialize_spec_from_python_code_result,
)
from tensor_network_editor.io import (
    PythonLoadOptions,
    load_python_spec,
)
from tensor_network_editor.io import (
    load_spec as _load_spec,
)
from tensor_network_editor.io import (
    save_spec as _save_spec,
)
from tensor_network_editor.models import (
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    TensorDataMode,
    TensorDataSpec,
)
from tensor_network_editor.rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    render_spec_dot,
    render_spec_mermaid,
    render_spec_pdf,
    render_spec_svg,
    render_spec_tikz,
)
from tests.conftest import distribution_for_checkout_import_or_skip
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_outer_product_plan_spec,
    build_sample_spec_with_view_snapshots,
    build_three_tensor_complete_plan_spec,
    build_three_tensor_hyperedge_spec,
    build_three_tensor_spec,
    build_three_tensor_spec_without_plan,
    build_tree_periodic_tree_spec,
)
from tests.optional_backends import require_light_optional_modules

PythonSourceProfile = Literal["auto", "generated", "quimb", "tensornetwork", "einsum"]
PythonImportMode = Literal["static", "live"]


def generate_code(
    spec: NetworkSpec,
    *,
    engine: EngineName,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    print_code: bool = False,
    path: Path | str | None = None,
    output_path: Path | str | None = None,
):
    return _generate_code(
        spec,
        engine=engine,
        collection_format=collection_format,
        output_path=output_path if output_path is not None else path,
        print_code=print_code,
    )


def load_spec(
    path: Path | str,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: Literal["auto", "simple", "best_available"] = "auto",
    python_object_name: str | None = None,
) -> NetworkSpec:
    return _load_spec(
        path,
        python=PythonLoadOptions(
            source_profile=source_profile,
            import_mode=python_import_mode,
            reconstruction_level=python_reconstruction_level,
            object_name=python_object_name,
        ),
    )


def load_spec_from_python_code(
    code: str,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: Literal["auto", "simple", "best_available"] = "auto",
    python_object_name: str | None = None,
) -> NetworkSpec:
    return load_python_spec(
        code,
        python=PythonLoadOptions(
            source_profile=source_profile,
            import_mode=python_import_mode,
            reconstruction_level=python_reconstruction_level,
            object_name=python_object_name,
        ),
    )


def save_spec(spec: NetworkSpec, path: Path | str) -> None:
    _save_spec(spec, path=path)


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
        "CodegenResult",
        "ContractionOperandLayoutSpec",
        "ContractionPlanSpec",
        "ContractionStepSpec",
        "ContractionViewSnapshotSpec",
        "EdgeEndpointRef",
        "EdgeSpec",
        "EditorLaunchOptions",
        "EditorThemeName",
        "EditorUiMode",
        "EditorResult",
        "EngineName",
        "DotRenderOptions",
        "GroupSpec",
        "HyperedgeSpec",
        "IndexHandle",
        "IndexSpec",
        "NetworkBuilder",
        "NetworkSpec",
        "PythonLoadOptions",
        "SvgRenderOptions",
        "TikzRenderOptions",
        "TensorCollectionFormat",
        "TensorDataMode",
        "TensorDataSpec",
        "TensorHandle",
        "TensorSize",
        "TensorSpec",
        "ValidationIssue",
        "__version__",
        "analyze_contraction",
        "analyze_spec",
        "build_template_spec",
        "canonicalize_spec",
        "diff_specs",
        "generate_code",
        "lint_spec",
        "list_template_names",
        "load_python_spec",
        "load_spec",
        "open_editor",
        "render_spec_dot",
        "render_spec_mermaid",
        "render_spec_pdf",
        "render_spec_svg",
        "render_spec_png",
        "render_spec_tikz",
        "save_spec",
        "semantic_diff_specs",
        "validate_spec",
    }
    assert tensor_network_editor.generate_code is _generate_code
    assert tensor_network_editor.open_editor is open_editor
    assert tensor_network_editor.load_spec is _load_spec
    assert tensor_network_editor.load_python_spec is load_python_spec
    assert tensor_network_editor.render_spec_svg is render_spec_svg
    from tensor_network_editor.rendering import render_spec_png

    assert tensor_network_editor.render_spec_png is render_spec_png
    assert tensor_network_editor.render_spec_pdf is render_spec_pdf
    assert tensor_network_editor.render_spec_tikz is render_spec_tikz
    assert tensor_network_editor.render_spec_dot is render_spec_dot
    assert tensor_network_editor.render_spec_mermaid is render_spec_mermaid
    assert tensor_network_editor.SvgRenderOptions is SvgRenderOptions
    assert tensor_network_editor.TikzRenderOptions is TikzRenderOptions
    assert tensor_network_editor.DotRenderOptions is DotRenderOptions
    assert tensor_network_editor.save_spec is _save_spec
    assert not hasattr(tensor_network_editor, "tensor_network_creation")
    assert not hasattr(tensor_network_editor, "launch_tensor_network_editor")
    assert not hasattr(tensor_network_editor, "load_spec_from_python_code")
    assert not hasattr(tensor_network_editor, "register_generator")
    assert not hasattr(tensor_network_editor, "register_template")
    assert not hasattr(tensor_network_editor, "register_static_template")
    assert not hasattr(tensor_network_editor, "list_generator_names")


def test_python_load_options_defaults_match_public_contract() -> None:
    options = PythonLoadOptions()

    assert options.source_profile == "auto"
    assert options.import_mode == "static"
    assert options.reconstruction_level == "auto"
    assert options.object_name is None


def test_editor_launch_options_defaults_match_public_contract() -> None:
    options = EditorLaunchOptions()

    assert options.default_engine is EngineName.TENSORKROWCH
    assert options.default_collection_format is TensorCollectionFormat.LIST
    assert options.theme == "dark"
    assert options.ui_mode is None
    assert options.open_browser is True
    assert options.host == "127.0.0.1"
    assert options.port == 0
    assert options.print_code is False
    assert options.code_path is None
    assert options.log_file_path is None
    assert options.log_file_max_bytes == 10_485_760
    assert options.log_file_backup_count == 5


def test_editor_launch_options_rejects_unknown_theme() -> None:
    with pytest.raises(ValueError, match="Unsupported editor theme 'sepia'"):
        EditorLaunchOptions(theme="sepia")  # type: ignore[arg-type]


def test_editor_ui_mode_type_alias_matches_public_contract() -> None:
    assert EditorUiMode == Literal["browser", "pywebview", "server"]


@pytest.mark.parametrize(
    ("ui_mode", "open_browser", "expected_message"),
    [
        ("browser", False, "ui_mode='browser' requires open_browser=True"),
        ("server", True, "ui_mode='server' requires open_browser=False"),
    ],
)
def test_editor_launch_options_rejects_conflicting_browser_flags(
    ui_mode: EditorUiMode,
    open_browser: bool,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        EditorLaunchOptions(ui_mode=ui_mode, open_browser=open_browser)


def test_open_editor_passes_editor_launch_options(sample_spec: NetworkSpec) -> None:
    launch_result = object()

    with patch(
        "tensor_network_editor.editor.launch_editor_session",
        return_value=launch_result,
    ) as launch_editor_session_mock:
        result = open_editor(
            sample_spec,
            options=EditorLaunchOptions(
                default_engine=EngineName.EINSUM_NUMPY,
                default_collection_format=TensorCollectionFormat.DICT,
                theme="colorblind",
                ui_mode="pywebview",
                open_browser=False,
                host="0.0.0.0",
                port=8123,
                print_code=True,
                code_path="generated.py",
                log_file_path="session.log",
                log_file_max_bytes=2048,
                log_file_backup_count=7,
                template_catalog_path="templates.json",
                subnetwork_catalog_path="subnetworks.json",
                shared_subnetwork_catalog_path="shared.json",
            ),
        )

    assert result is launch_result
    launch_editor_session_mock.assert_called_once_with(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
        default_collection_format=TensorCollectionFormat.DICT,
        theme="colorblind",
        ui_mode="pywebview",
        open_browser=False,
        host="0.0.0.0",
        port=8123,
        print_code=True,
        code_path="generated.py",
        log_file_path="session.log",
        log_file_max_bytes=2048,
        log_file_backup_count=7,
        template_catalog_path="templates.json",
        subnetwork_catalog_path="subnetworks.json",
        shared_subnetwork_catalog_path="shared.json",
        draft_path=None,
        _on_server_ready=None,
    )


def test_open_editor_theme_argument_overrides_options(
    sample_spec: NetworkSpec,
) -> None:
    launch_result = object()

    with patch(
        "tensor_network_editor.editor.launch_editor_session",
        return_value=launch_result,
    ) as launch_editor_session_mock:
        result = open_editor(
            sample_spec,
            theme="light",
            options=EditorLaunchOptions(theme="dark", open_browser=False),
        )

    assert result is launch_result
    assert launch_editor_session_mock.call_args.kwargs["theme"] == "light"


def test_open_editor_rejects_unknown_theme(sample_spec: NetworkSpec) -> None:
    with pytest.raises(ValueError, match="Unsupported editor theme 'sepia'"):
        open_editor(sample_spec, theme="sepia")  # type: ignore[arg-type]


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
        output_path=output_path,
    )

    assert output_path.read_text(encoding="utf-8") == result.code
    assert capsys.readouterr().out == f"{result.code}\n"


def test_generate_code_wraps_file_write_failures(sample_spec: NetworkSpec) -> None:
    missing_parent_path = Path(".test_output") / "missing_dir" / "generated_network.py"

    with pytest.raises(PackageIOError):
        generate_code(
            sample_spec,
            engine=EngineName.EINSUM_NUMPY,
            output_path=missing_parent_path,
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


def test_load_spec_imports_quimb_python_file_with_auto_detection(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "quimb_network.py"
    source_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import quimb.tensor as qtn",
                "",
                "a_data = np.zeros((2, 3))",
                "b_data = np.zeros((3, 5))",
                "tensor_a = qtn.Tensor(a_data, inds=('i', 'bond_x'), tags=('A',))",
                "tensor_b = qtn.Tensor(b_data, inds=('bond_x', 'j'), tags=('B',))",
                "network = qtn.TensorNetwork([tensor_a, tensor_b])",
            ]
        ),
        encoding="utf-8",
    )

    loaded_spec = load_spec(source_path)

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]
    assert loaded_spec.hyperedges == []
    assert loaded_spec.contraction_plan is None


@pytest.mark.optional_backend
def test_load_spec_imports_quimb_python_file_with_live_mode(
    tmp_path: Path,
) -> None:
    require_light_optional_modules(("numpy", "quimb"))
    source_path = tmp_path / "quimb_network_live.py"
    source_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import quimb.tensor as qtn",
                "",
                "def build_network() -> qtn.TensorNetwork:",
                "    left = qtn.Tensor(np.arange(6, dtype=float).reshape(2, 3), inds=('i', 'bond_x'), tags=('A',))",
                "    right = qtn.Tensor(np.full((3, 5), 1.5, dtype=float), inds=('bond_x', 'j'), tags=('B',))",
                "    return qtn.TensorNetwork([left, right])",
                "",
                "network = build_network()",
            ]
        ),
        encoding="utf-8",
    )

    loaded_spec = load_spec(
        source_path,
        python_import_mode="live",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]
    assert loaded_spec.tensors[0].tensor_data == TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
    )
    assert loaded_spec.tensors[1].tensor_data == TensorDataSpec(
        mode=TensorDataMode.FILL,
        fill_value=1.5,
    )


@pytest.mark.optional_backend
def test_load_spec_imports_live_python_file_with_relative_helper_import(
    tmp_path: Path,
) -> None:
    require_light_optional_modules(("numpy", "quimb"))
    helper_path = tmp_path / "helper_module.py"
    helper_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import quimb.tensor as qtn",
                "",
                "def build_network() -> qtn.TensorNetwork:",
                "    left = qtn.Tensor(np.ones((2, 3)), inds=('i', 'bond_x'), tags=('A',))",
                "    right = qtn.Tensor(np.ones((3, 5)), inds=('bond_x', 'j'), tags=('B',))",
                "    return qtn.TensorNetwork([left, right])",
            ]
        ),
        encoding="utf-8",
    )
    source_path = tmp_path / "quimb_network_from_helper.py"
    source_path.write_text(
        "\n".join(
            [
                "from helper_module import build_network",
                "",
                "network = build_network()",
            ]
        ),
        encoding="utf-8",
    )

    loaded_spec = load_spec(
        source_path,
        source_profile="quimb",
        python_import_mode="live",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]


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


def test_load_spec_from_python_code_live_generated_source_falls_back_to_static_parser() -> (
    None
):
    sample_spec = build_three_tensor_spec_without_plan()
    result = generate_code(sample_spec, engine=EngineName.TENSORKROWCH)

    with patch(
        "tensor_network_editor.internal.io._serialization.import_live_python_source",
        side_effect=SerializationError("No module named 'torch'"),
    ):
        parsed_result = deserialize_spec_from_python_code_result(
            result.code,
            source_profile="auto",
            python_import_mode="live",
        )

    assert [tensor.name for tensor in parsed_result.spec.tensors] == ["A", "B", "C"]
    assert [edge.name for edge in parsed_result.spec.edges] == ["bond_x", "bond_y"]
    assert parsed_result.warnings
    assert "live python import failed" in parsed_result.warnings[0].lower()
    assert "static parser" in parsed_result.warnings[0].lower()
    assert "no module named 'torch'" in parsed_result.warnings[0].lower()


def test_load_spec_from_python_code_live_import_does_not_fallback_for_ambiguous_globals() -> (
    None
):
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "tensor_a = qtn.Tensor(np.ones((2, 3)), inds=('i', 'j'), tags=('A',))",
            "tensor_b = qtn.Tensor(np.ones((2, 3)), inds=('k', 'l'), tags=('B',))",
        ]
    )

    with patch(
        "tensor_network_editor.internal.io._serialization.import_live_python_source",
        side_effect=SerializationError(
            "Live import found multiple compatible globals "
            "(tensor_a, tensor_b). Pass python_object_name to choose one."
        ),
    ):
        with pytest.raises(SerializationError, match="python_object_name"):
            load_spec_from_python_code(
                code,
                source_profile="quimb",
                python_import_mode="live",
            )


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
def test_load_spec_from_python_code_round_trips_external_tensor_data(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_spec_without_plan()
    sample_spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/a.npz",
        array_key="a",
    )
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.tensors[0].tensor_data == TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/a.npz",
        array_key="a",
    )


def test_load_spec_from_python_code_round_trips_external_pt_tensor_data() -> None:
    sample_spec = build_three_tensor_spec_without_plan()
    sample_spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/a.pt",
        array_key="weights",
    )
    result = generate_code(sample_spec, engine=EngineName.EINSUM_NUMPY)

    loaded_spec = load_spec_from_python_code(result.code)

    assert loaded_spec.tensors[0].tensor_data == TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/a.pt",
        array_key="weights",
    )


@pytest.mark.parametrize(
    ("spec_factory", "engine", "payload_attr"),
    [
        (
            build_linear_periodic_chain_spec,
            EngineName.EINSUM_NUMPY,
            "linear_periodic_chain",
        ),
        (
            build_linear_periodic_carry_chain_spec,
            EngineName.TENSORNETWORK,
            "linear_periodic_chain",
        ),
        (build_grid_periodic_grid_spec, EngineName.QUIMB, "grid_periodic_grid"),
        (build_tree_periodic_tree_spec, EngineName.EINSUM_NUMPY, "tree_periodic_tree"),
    ],
)
def test_load_spec_from_python_code_round_trips_periodic_generated_source(
    spec_factory: Callable[[], NetworkSpec],
    engine: EngineName,
    payload_attr: str,
) -> None:
    sample_spec = spec_factory()
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)

    assert "# TNE_SPEC_B64:" in result.code
    assert getattr(loaded_spec, payload_attr) is not None
    assert loaded_spec.to_dict() == sample_spec.to_dict()


def test_load_spec_from_python_code_keeps_legacy_periodic_export_error() -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=EngineName.QUIMB)
    legacy_code = "\n".join(
        line
        for line in result.code.splitlines()
        if not line.startswith("# TNE_SPEC_B64:")
    )

    with pytest.raises(SerializationError, match="linear periodic mode"):
        load_spec_from_python_code(legacy_code)


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_round_trips_hyperedges(
    engine: EngineName,
) -> None:
    result = generate_code(build_three_tensor_hyperedge_spec(), engine=engine)

    loaded_spec = load_spec_from_python_code(result.code)
    hyperedge = loaded_spec.hyperedges[0]
    tensor_name_by_id = {tensor.id: tensor.name for tensor in loaded_spec.tensors}

    assert len(loaded_spec.hyperedges) == 1
    assert hyperedge.name == "shared_h"
    assert len(hyperedge.endpoints) == 3
    assert sorted(
        tensor_name_by_id[endpoint.tensor_id] for endpoint in hyperedge.endpoints
    ) == ["A", "B", "C"]
    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert loaded_spec.edges == []


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
def test_load_spec_from_python_code_generated_simple_reconstruction_omits_manual_plan(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_spec()
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(
        result.code,
        python_reconstruction_level="simple",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert len(loaded_spec.edges) == 2
    assert loaded_spec.contraction_plan is None


@pytest.mark.parametrize("engine", list(EngineName))
def test_load_spec_from_python_code_generated_best_available_preserves_manual_plan(
    engine: EngineName,
) -> None:
    sample_spec = build_three_tensor_spec()
    result = generate_code(sample_spec, engine=engine)

    loaded_spec = load_spec_from_python_code(
        result.code,
        python_reconstruction_level="best_available",
    )

    assert loaded_spec.contraction_plan is not None
    assert [step.id for step in loaded_spec.contraction_plan.steps] == ["step_ab"]


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


def test_load_spec_from_python_code_imports_quimb_tensor_network_profile() -> None:
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            "tensor_a = qtn.Tensor(a_data, inds=('i', 'bond_x'), tags=('A',))",
            "tensor_b = qtn.Tensor(b_data, inds=('bond_x', 'j'), tags=('B',))",
            "network = qtn.TensorNetwork([tensor_a, tensor_b])",
        ]
    )

    loaded_spec = load_spec_from_python_code(code, source_profile="quimb")

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]
    assert loaded_spec.hyperedges == []
    assert loaded_spec.contraction_plan is None


def test_load_spec_from_python_code_auto_reconstruction_keeps_external_profiles_simple() -> (
    None
):
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            "tensor_a = qtn.Tensor(a_data, inds=('i', 'bond_x'), tags=('A',))",
            "tensor_b = qtn.Tensor(b_data, inds=('bond_x', 'j'), tags=('B',))",
            "network = qtn.TensorNetwork([tensor_a, tensor_b])",
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        python_reconstruction_level="auto",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert loaded_spec.contraction_plan is None


def test_load_spec_from_python_code_rejects_best_available_for_external_profiles() -> (
    None
):
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            "tensor_a = qtn.Tensor(a_data, inds=('i', 'bond_x'), tags=('A',))",
            "tensor_b = qtn.Tensor(b_data, inds=('bond_x', 'j'), tags=('B',))",
            "network = qtn.TensorNetwork([tensor_a, tensor_b])",
        ]
    )

    with pytest.raises(SerializationError, match="best_available"):
        load_spec_from_python_code(
            code,
            source_profile="quimb",
            python_reconstruction_level="best_available",
        )


@pytest.mark.optional_backend
def test_load_spec_from_python_code_live_imports_quimb_tensor_network() -> None:
    require_light_optional_modules(("numpy", "quimb"))
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "def build_network() -> qtn.TensorNetwork:",
            "    left = qtn.Tensor(np.ones((2, 3)), inds=('i', 'bond_x'), tags=('A',))",
            "    right = qtn.Tensor(np.ones((3, 5)), inds=('bond_x', 'j'), tags=('B',))",
            "    return qtn.TensorNetwork([left, right])",
            "",
            "network = build_network()",
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        python_import_mode="live",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]


@pytest.mark.optional_backend
def test_load_spec_from_python_code_live_import_rejects_best_available() -> None:
    require_light_optional_modules(("numpy", "quimb"))
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "left = qtn.Tensor(np.ones((2, 3)), inds=('i', 'bond_x'), tags=('A',))",
            "right = qtn.Tensor(np.ones((3, 5)), inds=('bond_x', 'j'), tags=('B',))",
            "network = qtn.TensorNetwork([left, right])",
        ]
    )

    with pytest.raises(SerializationError, match="best_available"):
        load_spec_from_python_code(
            code,
            python_import_mode="live",
            python_reconstruction_level="best_available",
        )


@pytest.mark.optional_backend
def test_load_spec_from_python_code_live_imports_tensornetwork_object() -> None:
    require_light_optional_modules(("numpy", "tensornetwork"))
    code = "\n".join(
        [
            "import numpy as np",
            "import tensornetwork as tn",
            "",
            "def build_network() -> list[tn.Node]:",
            "    left = tn.Node(np.ones((2, 3)), name='A', axis_names=['i', 'bond_x'])",
            "    right = tn.Node(np.arange(15, dtype=float).reshape(3, 5), name='B', axis_names=['bond_x', 'j'])",
            "    tn.connect(left['bond_x'], right['bond_x'], name='bond_x')",
            "    return [left, right]",
            "",
            "network = build_network()",
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        python_import_mode="live",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]
    assert loaded_spec.tensors[1].tensor_data == TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ],
    )


@pytest.mark.optional_backend
def test_load_spec_from_python_code_live_import_uses_explicit_object_name() -> None:
    require_light_optional_modules(("numpy", "quimb"))
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "def build_network(tag: str, bond: str) -> qtn.TensorNetwork:",
            "    left = qtn.Tensor(np.ones((2, 3)), inds=('i', bond), tags=(tag,))",
            "    right = qtn.Tensor(np.ones((3, 5)), inds=(bond, 'j'), tags=(f'{tag}R',))",
            "    return qtn.TensorNetwork([left, right])",
            "",
            "first_network = build_network('A', 'bond_a')",
            "second_network = build_network('B', 'bond_b')",
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        python_import_mode="live",
        python_object_name="second_network",
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["B", "BR"]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_b"]


@pytest.mark.optional_backend
def test_load_spec_from_python_code_live_import_rejects_ambiguous_globals() -> None:
    require_light_optional_modules(("numpy", "quimb"))
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "tensor_a = qtn.Tensor(np.ones((2, 3)), inds=('i', 'j'), tags=('A',))",
            "tensor_b = qtn.Tensor(np.ones((2, 3)), inds=('k', 'l'), tags=('B',))",
        ]
    )

    with pytest.raises(SerializationError, match="python_object_name"):
        load_spec_from_python_code(
            code,
            python_import_mode="live",
        )


def test_load_spec_from_python_code_imports_quimb_ampersand_chain_hyperedge() -> None:
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            "c_data = np.zeros((3, 7))",
            "tensor_a = qtn.Tensor(a_data, inds=('i', 'shared_h'), tags=('A',))",
            "tensor_b = qtn.Tensor(b_data, inds=('shared_h', 'j'), tags=('B',))",
            "tensor_c = qtn.Tensor(c_data, inds=('shared_h', 'k'), tags=('C',))",
            "network = tensor_a & tensor_b & tensor_c",
        ]
    )

    loaded_spec = load_spec_from_python_code(code)

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B", "C"]
    assert loaded_spec.edges == []
    assert len(loaded_spec.hyperedges) == 1
    assert loaded_spec.hyperedges[0].name == "shared_h"
    assert len(loaded_spec.hyperedges[0].endpoints) == 3


@pytest.mark.parametrize(
    ("statement", "source_profile"),
    [
        (
            "edge_x = tn.connect(node_a['bond_x'], node_b['bond_x'], name='bond_x')",
            "tensornetwork",
        ),
        (
            "bond_x = node_a['bond_x'] ^ node_b['bond_x']",
            "tensornetwork",
        ),
    ],
)
def test_load_spec_from_python_code_imports_tensornetwork_profile(
    statement: str,
    source_profile: str,
) -> None:
    code = "\n".join(
        [
            "import numpy as np",
            "import tensornetwork as tn",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            "node_a = tn.Node(a_data, name='A', axis_names=['i', 'bond_x'])",
            "node_b = tn.Node(b_data, name='B', axis_names=['bond_x', 'j'])",
            statement,
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        source_profile=cast(PythonSourceProfile, source_profile),
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["bond_x"]
    assert loaded_spec.hyperedges == []
    assert loaded_spec.contraction_plan is None


@pytest.mark.parametrize(
    ("statement", "source_profile"),
    [
        ("result = np.einsum('ab,bc->ac', a_data, b_data)", "einsum"),
        ("result = oe.contract('ab,bc->ac', a_data, b_data)", "einsum"),
    ],
)
def test_load_spec_from_python_code_imports_einsum_profiles(
    statement: str,
    source_profile: str,
) -> None:
    code = "\n".join(
        [
            "import numpy as np",
            "import opt_einsum as oe",
            "",
            "a_data = np.zeros((2, 3))",
            "b_data = np.zeros((3, 5))",
            statement,
        ]
    )

    loaded_spec = load_spec_from_python_code(
        code,
        source_profile=cast(PythonSourceProfile, source_profile),
    )

    assert [tensor.name for tensor in loaded_spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in loaded_spec.tensors] == [(2, 3), (3, 5)]
    assert [edge.name for edge in loaded_spec.edges] == ["b"]
    assert loaded_spec.hyperedges == []
    assert loaded_spec.contraction_plan is None


def test_load_spec_from_python_code_rejects_external_profiles_without_static_shapes() -> (
    None
):
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "shape = (2, 3)",
            "a_data = np.zeros(shape)",
            "tensor_a = qtn.Tensor(a_data, inds=('i', 'j'), tags=('A',))",
            "network = qtn.TensorNetwork([tensor_a])",
        ]
    )

    with pytest.raises(SerializationError, match="supported"):
        load_spec_from_python_code(code, source_profile="quimb")


def test_load_spec_from_python_code_live_import_rejects_generated_profile() -> None:
    code = "network = object()\n"

    with pytest.raises(SerializationError, match="live import"):
        load_spec_from_python_code(
            code,
            source_profile="generated",
            python_import_mode="live",
        )


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
