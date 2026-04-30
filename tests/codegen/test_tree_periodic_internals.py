from __future__ import annotations

from tensor_network_editor.models import (
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
)
from tests.factories import build_tree_periodic_tree_spec


def test_tree_periodic_array_renderer_facade_reexports_internal_entrypoint() -> None:
    from tensor_network_editor.codegen.modes import (
        _tree_periodic_array_renderers as facade,
    )
    from tensor_network_editor.codegen.modes._tree_periodic.array_common import (
        generate_array_tree_periodic_code as implementation,
    )

    assert facade.generate_array_tree_periodic_code is implementation


def test_tree_periodic_graph_renderer_facade_reexports_internal_entrypoint() -> None:
    from tensor_network_editor.codegen.modes import (
        _tree_periodic_graph_renderers as facade,
    )
    from tensor_network_editor.codegen.modes._tree_periodic.graph_common import (
        generate_graph_tree_periodic_code as implementation,
    )

    assert facade.generate_graph_tree_periodic_code is implementation


def test_tree_periodic_common_helpers_resolve_cells_and_child_ports() -> None:
    from tensor_network_editor.codegen.modes._tree_periodic.common import (
        _build_child_ports_by_index,
        _cell_from_tree,
        _render_parent_interface_validation,
        _render_tree_bottom_up_marker_lines,
    )
    from tensor_network_editor.internal.modes._tree_periodic import (
        build_tree_periodic_interface_ports,
    )

    tree = build_tree_periodic_tree_spec().tree_periodic_tree
    assert tree is not None

    assert _cell_from_tree(tree, TreePeriodicCellName.ROOT) is tree.root_cell
    assert _cell_from_tree(tree, TreePeriodicCellName.BRANCH) is tree.branch_cell
    assert _cell_from_tree(tree, TreePeriodicCellName.LEAF) is tree.leaf_cell

    child_ports_by_index = _build_child_ports_by_index(
        tree=tree,
        cell=tree.branch_cell,
        cell_name=TreePeriodicCellName.BRANCH,
    )
    assert tuple(child_ports_by_index) == tuple(range(tree.branching_factor))

    parent_ports = build_tree_periodic_interface_ports(
        tree.branch_cell,
        cell_name=TreePeriodicCellName.BRANCH,
        role=TreePeriodicTensorRole.PARENT,
    )
    assert _render_parent_interface_validation(()) == []
    assert _render_parent_interface_validation(parent_ports) == [
        f"if len(parent_interface) != {len(parent_ports)}:",
        "    raise ValueError('The provided parent interface does not match this tree cell.')",
    ]
    assert _render_tree_bottom_up_marker_lines() == [
        "",
        "# Manual tree cell plans are assembled from leaves toward the root.",
        "for level in range(n - 1, 0, -1):",
        "    pass",
    ]


def test_tree_periodic_array_helpers_keep_child_interfaces_and_backend_tensor_builders() -> (
    None
):
    from tensor_network_editor.codegen.modes._tree_periodic.array_einsum import (
        _render_einsum_cell_helper,
    )
    from tensor_network_editor.codegen.modes._tree_periodic.array_helpers import (
        _render_child_interface_lines,
    )
    from tensor_network_editor.codegen.modes._tree_periodic.common import (
        _build_child_ports_by_index,
    )

    tree = build_tree_periodic_tree_spec().tree_periodic_tree
    assert tree is not None

    child_ports_by_index = _build_child_ports_by_index(
        tree=tree,
        cell=tree.branch_cell,
        cell_name=TreePeriodicCellName.BRANCH,
    )
    child_interface_lines = _render_child_interface_lines(
        cell_name=TreePeriodicCellName.BRANCH,
        child_ports_by_index=child_ports_by_index,
    )
    assert child_interface_lines[0] == "child_interfaces = []"
    assert "child_interfaces.append([" in child_interface_lines[1]

    numpy_helper = _render_einsum_cell_helper(
        tree=tree,
        cell_name=TreePeriodicCellName.ROOT,
        helper_name="build_root_cell",
        helper_signature="",
        engine=EngineName.EINSUM_NUMPY,
        collection_format=TensorCollectionFormat.LIST,
    )
    torch_helper = _render_einsum_cell_helper(
        tree=tree,
        cell_name=TreePeriodicCellName.ROOT,
        helper_name="build_root_cell",
        helper_signature="",
        engine=EngineName.EINSUM_TORCH,
        collection_format=TensorCollectionFormat.LIST,
    )

    numpy_helper_body = "\n".join(numpy_helper.lines)
    torch_helper_body = "\n".join(torch_helper.lines)
    assert "np.zeros(" in numpy_helper_body
    assert "torch.zeros(" in torch_helper_body
    assert "np.zeros(" not in torch_helper_body


def test_tree_periodic_array_shared_helpers_build_context_and_sections() -> None:
    from tensor_network_editor.codegen.modes._tree_periodic.array_shared import (
        build_tree_array_cell_context,
        render_tree_array_tensor_sections,
    )

    tree = build_tree_periodic_tree_spec().tree_periodic_tree
    assert tree is not None

    context = build_tree_array_cell_context(
        tree=tree,
        cell_name=TreePeriodicCellName.ROOT,
        collection_format=TensorCollectionFormat.LIST,
    )
    tensor_collection_lines, tensor_construction_lines = (
        render_tree_array_tensor_sections(
            context=context,
            tensor_value_by_id={
                tensor.spec.id: f"value_{tensor.variable_name}"
                for tensor in context.prepared.tensors
            },
        )
    )

    assert context.collection_name == "tensors"
    assert context.parent_ports == ()
    assert tuple(context.child_ports_by_index) == tuple(range(tree.branching_factor))
    assert context.interface_index_ids
    assert tensor_collection_lines == ["tensors = []"]
    assert any(line.startswith("# Tensor ") for line in tensor_construction_lines)
    assert any("tensors.append(value_" in line for line in tensor_construction_lines)
