"""Standard array-backend renderers for linear-periodic code generation."""

from __future__ import annotations

from ....internal.modes._linear_periodic import (
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ....models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ...backends.quimb import QuimbCodeGenerator
from ...shared._linear_periodic_expressions import (
    _build_einsum_label_expression_map,
    _build_quimb_label_expression_map,
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from .array_common import (
    _einsum_generator_for_engine,
    _render_einsum_main_flow_lines,
    _render_einsum_manual_plan_lines,
    _render_einsum_shared_helper_lines,
    _render_quimb_main_flow_lines,
    _render_quimb_shared_helper_lines,
)
from .common import (
    _cell_from_chain,
    _RenderedCellHelper,
    render_linear_periodic_helper,
    render_linear_periodic_script,
)


def _generate_quimb_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for the ``quimb`` backend."""
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
    ]
    shared_helper_lines = _render_quimb_shared_helper_lines()
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_quimb_main_flow_lines()
    return CodegenResult(
        engine=EngineName.QUIMB,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                collection_format=collection_format,
            ).lines,
            periodic_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                collection_format=collection_format,
            ).lines,
            final_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                collection_format=collection_format,
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _generate_einsum_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
    ]
    shared_helper_lines = _render_einsum_shared_helper_lines(engine)
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_einsum_main_flow_lines(engine)
    return CodegenResult(
        engine=engine,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            periodic_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            final_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _render_quimb_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for ``quimb``."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
    generator = QuimbCodeGenerator()
    tensor_value_by_id = {
        tensor.spec.id: (
            f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
            f"tags={(tensor.spec.name, generator._operand_tag(tensor.spec.id))!r})"
        )
        for tensor in prepared.tensors
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    network_setup_lines = [
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name),
    ]
    assembly_lines: list[str]
    output_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        assembly_lines, output_lines = generator._render_manual_plan(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
        assembly_lines = [
            "network = qtn.TensorNetwork(network_tensors)",
            "",
            *assembly_lines,
        ]
        output_lines = [
            line
            for line in output_lines
            if not line.startswith("result = ") and not line.startswith("open_inds = ")
        ]
        output_lines.append("cell_tensors = list(remaining_operands.values())")
    else:
        assembly_lines = ["cell_tensors = list(network_tensors)"]
        output_lines = []
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    output_lines.append(
        "open_inds = " + _render_python_tuple_expression(local_open_expressions)
    )
    output_lines.append("result = cell_tensors[0] if len(cell_tensors) == 1 else None")
    output_lines.extend(
        [
            "return {",
            "    'tensors': cell_tensors,",
            "    'open_inds': open_inds,",
            "    'result': result,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
        CodeSection(title="Network setup", lines=network_setup_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(CodeSection(title="Manual contraction", lines=assembly_lines))
    else:
        sections.append(CodeSection(title="Cell assembly", lines=assembly_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )


def _render_einsum_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for an einsum backend."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
    generator = _einsum_generator_for_engine(engine)
    tensor_value_by_id = {
        tensor.spec.id: (
            f"{generator.module_alias}.zeros({tensor.spec.shape!r}"
            f"{generator.zero_initializer_suffix})"
        )
        for tensor in prepared.tensors
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    assembly_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        assembly_lines = _render_einsum_manual_plan_lines(
            prepared=prepared,
            engine=engine,
            collection_format=collection_format,
            collection_name=collection_name,
            label_expression_by_label=label_expression_by_label,
        )
    else:
        assembly_lines = [
            "cell_operands = []",
            "cell_operand_labels = []",
        ]
        for tensor in prepared.tensors:
            assembly_lines.append(
                "cell_operands.append("
                + tensor_collection_reference_by_id(
                    prepared,
                    tensor.spec.id,
                    collection_format,
                    collection_name,
                )
                + ")"
            )
            assembly_lines.append(
                "cell_operand_labels.append("
                + _render_python_list_expression(
                    [label_expression_by_label[index.label] for index in tensor.indices]
                )
                + ")"
            )
    output_lines = [
        "open_labels = "
        + _render_python_list_expression(
            [
                label_expression_by_label[index.label]
                for index in prepared.open_indices
                if index.spec.id not in interface_index_ids
            ]
        ),
    ]
    output_lines.extend(
        [
            "return {",
            "    'operands': cell_operands,",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(CodeSection(title="Manual contraction", lines=assembly_lines))
    else:
        sections.append(CodeSection(title="Cell assembly", lines=assembly_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )
