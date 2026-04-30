from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from tensor_network_editor import generate_code
from tensor_network_editor.codegen.backends.einsum_numpy import EinsumNumpyCodeGenerator
from tensor_network_editor.codegen.shared.common import (
    render_remaining_operands_mapping,
)
from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorCollectionFormat,
    TensorDataDType,
    TensorDataMode,
    TensorDataRandomDistribution,
    TensorDataSpec,
    TensorSpec,
)
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_grid_periodic_grid_spec_with_partial_plan,
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_outer_product_plan_spec,
    build_sample_spec,
    build_sample_spec_without_plan,
    build_three_tensor_complete_plan_spec,
    build_three_tensor_hyperedge_spec,
    build_three_tensor_spec,
    build_three_tensor_spec_without_plan,
    build_tree_periodic_tree_spec,
    build_tree_periodic_tree_spec_with_partial_plan,
)
from tests.optional_backends import require_engine_backend


def build_many_label_spec() -> NetworkSpec:
    tensors = []
    for tensor_index in range(18):
        tensors.append(
            TensorSpec(
                id=f"tensor_{tensor_index}",
                name=f"T{tensor_index}",
                position=CanvasPosition(x=float(tensor_index * 120), y=0.0),
                indices=[
                    IndexSpec(
                        id=f"tensor_{tensor_index}_i{index_offset}",
                        name=f"i{index_offset}",
                        dimension=2,
                    )
                    for index_offset in range(3)
                ],
            )
        )
    return NetworkSpec(id="many_labels", name="many labels", tensors=tensors)


def build_matrix_layout_spec() -> NetworkSpec:
    return NetworkSpec(
        id="matrix_layout",
        name="matrix layout",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=100.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=260.0, y=108.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_y", name="y", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=120.0, y=260.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=5),
                    IndexSpec(id="tensor_c_j", name="j", dimension=7),
                ],
            ),
        ],
    )


def build_disconnected_pairs_spec() -> NetworkSpec:
    return NetworkSpec(
        id="disconnected_pairs",
        name="disconnected pairs",
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
                    IndexSpec(id="tensor_b_j", name="j", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=440.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_k", name="k", dimension=7),
                    IndexSpec(id="tensor_c_y", name="y", dimension=11),
                ],
            ),
            TensorSpec(
                id="tensor_d",
                name="D",
                position=CanvasPosition(x=600.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_d_y", name="y", dimension=11),
                    IndexSpec(id="tensor_d_l", name="l", dimension=13),
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
                left=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_y"),
                right=EdgeEndpointRef(tensor_id="tensor_d", index_id="tensor_d_y"),
            ),
        ],
    )


def build_random_better_chain_spec() -> NetworkSpec:
    return NetworkSpec(
        id="random_better_chain",
        name="random better chain",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=100),
                    IndexSpec(id="tensor_a_x", name="x", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=2),
                    IndexSpec(id="tensor_b_y", name="y", dimension=100),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=400.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=100),
                    IndexSpec(id="tensor_c_j", name="j", dimension=2),
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
    )


def build_empty_spec() -> NetworkSpec:
    return NetworkSpec(id="empty_network", name="empty network")


def test_render_remaining_operands_mapping_renders_joined_display_names() -> None:
    lines = render_remaining_operands_mapping(
        operand_ids=("step_ab", "tensor_c"),
        source_tensor_ids_by_operand_id={
            "step_ab": ("tensor_a", "tensor_b"),
            "tensor_c": ("tensor_c",),
        },
        tensor_names_by_id={
            "tensor_a": "A",
            "tensor_b": "B",
            "tensor_c": "C",
        },
        base_operand_expressions={
            "tensor_a": "tensors[0]",
            "tensor_b": "tensors[1]",
            "tensor_c": "tensors[2]",
        },
        step_result_indexes={"step_ab": 0},
        latest_result_index=0,
    )

    assert lines == [
        "remaining_operands = {",
        "    'A-B': results_list[-1],",
        "    'C': tensors[2],",
        "}",
    ]


def _import_required_backend(engine: EngineName) -> None:
    """Skip execution tests when an optional backend is not installed."""
    require_engine_backend(engine)


def _execute_generated_code(
    code: str,
    *,
    n: int | None = None,
    m: int | None = None,
) -> dict[str, object]:
    """Execute generated code in a shared namespace and return that namespace."""
    namespace: dict[str, object] = {}
    if n is not None:
        namespace["n"] = n
    if m is not None:
        namespace["m"] = m
    exec(code, namespace, namespace)
    return namespace


class _FakeTensorKrowchEdge:
    """Minimal edge object for generated-code regression tests."""

    def __init__(
        self,
        node: _FakeTensorKrowchNode,
        axis_name: str,
        *,
        origin: tuple[str, str] | None = None,
    ) -> None:
        self.node1 = node
        self.axis1 = SimpleNamespace(name=axis_name)
        self.node2: _FakeTensorKrowchNode | None = None
        self.axis2: SimpleNamespace | None = None
        self.origin = origin or (node.name, axis_name)

    @classmethod
    def from_endpoints(
        cls,
        *,
        node1: _FakeTensorKrowchNode,
        axis1_name: str,
        node2: _FakeTensorKrowchNode | None = None,
        axis2_name: str | None = None,
        origin: tuple[str, str] | None = None,
    ) -> _FakeTensorKrowchEdge:
        """Build one edge with explicit endpoint ownership."""
        edge = cls(node1, axis1_name, origin=origin)
        if node2 is not None and axis2_name is not None:
            edge.attach_second(node2, axis2_name)
        return edge

    def attach_second(
        self,
        node: _FakeTensorKrowchNode,
        axis_name: str,
    ) -> None:
        self.node2 = node
        self.axis2 = SimpleNamespace(name=axis_name)

    def replace_endpoint(
        self,
        old_node: _FakeTensorKrowchNode,
        new_node: _FakeTensorKrowchNode,
        new_axis_name: str,
    ) -> None:
        if self.node1 is old_node:
            self.node1 = new_node
            self.axis1 = SimpleNamespace(name=new_axis_name)
            return
        if self.node2 is old_node:
            self.node2 = new_node
            self.axis2 = SimpleNamespace(name=new_axis_name)

    def is_dangling(self) -> bool:
        return self.node2 is None

    def axis_name_for_node(
        self,
        node: _FakeTensorKrowchNode,
    ) -> SimpleNamespace:
        """Return the endpoint axis metadata for ``node``."""
        if self.node1 is node:
            return self.axis1
        assert self.node2 is node
        assert self.axis2 is not None
        return self.axis2


class _FakeTensorKrowchNode:
    """Minimal node object for generated-code regression tests."""

    def __init__(
        self,
        *,
        tensor: object,
        axes_names: tuple[str, ...],
        name: str,
        network: object,
    ) -> None:
        del tensor, network
        self.name = name
        self.edges_by_axis_name = {
            axis_name: _FakeTensorKrowchEdge(self, axis_name)
            for axis_name in axes_names
        }
        self.pending_edges_by_axis_name: dict[str, _FakeTensorKrowchEdge] = {}
        self.pending_edge_owner_by_axis_name: dict[str, _FakeTensorKrowchNode] = {}
        self.axis_is_node1_by_axis_name = {axis_name: True for axis_name in axes_names}

    def __getitem__(self, axis_name: str) -> _FakeTensorKrowchEdge:
        if axis_name in self.edges_by_axis_name:
            return self.edges_by_axis_name[axis_name]
        return self.pending_edges_by_axis_name[axis_name]

    def reattach_edges(self, override: bool = False) -> None:
        for axis_name, edge in list(self.pending_edges_by_axis_name.items()):
            owner = self.pending_edge_owner_by_axis_name.pop(axis_name)
            owner_is_node1 = edge.node1 is owner
            if owner_is_node1:
                other_node = edge.node2
                other_axis_name = None if edge.axis2 is None else edge.axis2.name
            else:
                other_node = edge.node1
                other_axis_name = edge.axis1.name
            if override:
                if owner_is_node1:
                    edge.node1 = self
                    edge.axis1 = SimpleNamespace(name=axis_name)
                else:
                    edge.node2 = self
                    edge.axis2 = SimpleNamespace(name=axis_name)
                self.edges_by_axis_name[axis_name] = edge
            else:
                if owner_is_node1:
                    self.edges_by_axis_name[axis_name] = (
                        _FakeTensorKrowchEdge.from_endpoints(
                            node1=self,
                            axis1_name=axis_name,
                            node2=other_node,
                            axis2_name=other_axis_name,
                            origin=edge.origin,
                        )
                    )
                else:
                    assert other_node is not None
                    assert other_axis_name is not None
                    self.edges_by_axis_name[axis_name] = (
                        _FakeTensorKrowchEdge.from_endpoints(
                            node1=other_node,
                            axis1_name=other_axis_name,
                            node2=self,
                            axis2_name=axis_name,
                            origin=edge.origin,
                        )
                    )
            self.axis_is_node1_by_axis_name[axis_name] = owner_is_node1
        self.pending_edges_by_axis_name = {}


class _FakeTensorKrowchModule(ModuleType):
    """Tiny ``tensorkrowch`` double that exposes fragile axis ordering."""

    def __init__(self) -> None:
        super().__init__("tensorkrowch")
        self.Node = _FakeTensorKrowchNode
        self.TensorNetwork = _fake_tensorkrowch_network_factory

    @staticmethod
    def connect(
        left_edge: _FakeTensorKrowchEdge,
        right_edge: _FakeTensorKrowchEdge,
    ) -> _FakeTensorKrowchEdge:
        left_edge.attach_second(right_edge.node1, right_edge.axis1.name)
        right_edge.node1.edges_by_axis_name[right_edge.axis1.name] = left_edge
        right_edge.node1.axis_is_node1_by_axis_name[right_edge.axis1.name] = False
        return left_edge

    @staticmethod
    def contract_between(
        left_node: _FakeTensorKrowchNode,
        right_node: _FakeTensorKrowchNode,
    ) -> _FakeTensorKrowchNode:
        left_edges = set(left_node.edges_by_axis_name.values())
        right_edges = set(right_node.edges_by_axis_name.values())
        if not left_edges.intersection(right_edges):
            raise ValueError(
                f"No batch edges or shared edges between nodes {left_node.name} and {right_node.name} found"
            )
        shared_edges = left_edges.intersection(right_edges)
        surviving_edges_with_owner = [
            (edge, left_node)
            for edge in left_node.edges_by_axis_name.values()
            if edge not in shared_edges
        ] + [
            (edge, right_node)
            for edge in right_node.edges_by_axis_name.values()
            if edge not in shared_edges
        ]
        surviving_axis_names = _deduplicate_fake_tensorkrowch_axis_names(
            tuple(
                edge.axis_name_for_node(owner).name
                for edge, owner in surviving_edges_with_owner
            )
        )
        result = _FakeTensorKrowchNode(
            tensor=None,
            axes_names=surviving_axis_names,
            name=f"{left_node.name}_{right_node.name}",
            network=None,
        )
        result.edges_by_axis_name = {}
        result.pending_edges_by_axis_name = {}
        result.pending_edge_owner_by_axis_name = {}
        result.axis_is_node1_by_axis_name = {}
        for axis_name, (edge, owner) in zip(
            surviving_axis_names,
            surviving_edges_with_owner,
            strict=True,
        ):
            result.pending_edges_by_axis_name[axis_name] = edge
            result.pending_edge_owner_by_axis_name[axis_name] = owner
            result.axis_is_node1_by_axis_name[axis_name] = edge.node1 is owner
        return result


class _FakeTorchModule(ModuleType):
    """Tiny ``torch`` double for generated-code regression tests."""

    float32: object

    def __init__(self) -> None:
        super().__init__("torch")
        self.float32 = object()

    @staticmethod
    def zeros(
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> tuple[tuple[int, ...], object | None]:
        return (shape, dtype)


def _deduplicate_fake_tensorkrowch_axis_names(
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Mirror TensorKrowch suffixing for exact duplicate surviving axes."""
    base_names = [
        axis_name.rsplit("_", 1)[0]
        if axis_name.rsplit("_", 1)[-1].isdigit()
        else axis_name
        for axis_name in axis_names
    ]
    result: list[str] = []
    counts: dict[str, int] = {}
    for axis_name in base_names:
        index = counts.get(axis_name, 0)
        counts[axis_name] = index + 1
        if base_names.count(axis_name) == 1:
            result.append(axis_name)
        else:
            result.append(f"{axis_name}_{index}")
    return tuple(result)


def _fake_tensorkrowch_network_factory() -> object:
    """Return a placeholder TensorNetwork instance for generated code."""
    return SimpleNamespace(reset=lambda: None)


class _ResetAwareFakeTensorKrowchNetwork:
    """Minimal network object that can resync inherited resultant edges."""

    def __init__(self) -> None:
        self.nodes: list[_ResetAwareFakeTensorKrowchNode] = []

    def register(self, node: _ResetAwareFakeTensorKrowchNode) -> None:
        self.nodes.append(node)

    def reset(self) -> None:
        for node in self.nodes:
            node.reset_inherited_edges()


class _ResetAwareFakeTensorKrowchEdge:
    """Edge double that hides inherited-result connections until reset."""

    def __init__(
        self,
        node: _ResetAwareFakeTensorKrowchNode,
        axis_name: str,
    ) -> None:
        self.node1 = node
        self.axis1 = SimpleNamespace(name=axis_name)
        self.node2: _ResetAwareFakeTensorKrowchNode | None = None
        self.axis2: SimpleNamespace | None = None
        self.origin = (node.name, axis_name)
        self.inherited_source_by_result_node: dict[
            _ResetAwareFakeTensorKrowchNode,
            tuple[_ResetAwareFakeTensorKrowchNode, str],
        ] = {}

    def attach_second(
        self,
        node: _ResetAwareFakeTensorKrowchNode,
        axis_name: str,
    ) -> None:
        self.node2 = node
        self.axis2 = SimpleNamespace(name=axis_name)

    def replace_endpoint(
        self,
        old_node: _ResetAwareFakeTensorKrowchNode,
        new_node: _ResetAwareFakeTensorKrowchNode,
        new_axis_name: str,
    ) -> None:
        if self.node1 is old_node:
            self.inherited_source_by_result_node[new_node] = (
                old_node,
                self.axis1.name,
            )
            self.node1 = new_node
            self.axis1 = SimpleNamespace(name=new_axis_name)
            self._stale_other_resultant_endpoints(excluded_result_node=new_node)
            return
        if self.node2 is old_node:
            assert self.axis2 is not None
            self.inherited_source_by_result_node[new_node] = (
                old_node,
                self.axis2.name,
            )
            self.node2 = new_node
            self.axis2 = SimpleNamespace(name=new_axis_name)
            self._stale_other_resultant_endpoints(excluded_result_node=new_node)

    def materialize_leaf_endpoint_for_resultant(
        self,
        node: _ResetAwareFakeTensorKrowchNode,
    ) -> None:
        source = self.inherited_source_by_result_node.get(node)
        if source is None:
            return
        source_node, source_axis_name = source
        if self.node1 is node:
            self.node1 = source_node
            self.axis1 = SimpleNamespace(name=source_axis_name)
            return
        if self.node2 is node:
            self.node2 = source_node
            self.axis2 = SimpleNamespace(name=source_axis_name)

    def restore_resultant_endpoint(
        self,
        node: _ResetAwareFakeTensorKrowchNode,
        axis_name: str,
    ) -> None:
        source = self.inherited_source_by_result_node.get(node)
        if source is None:
            return
        source_node, source_axis_name = source
        if self.node1 is source_node and self.axis1.name == source_axis_name:
            self.node1 = node
            self.axis1 = SimpleNamespace(name=axis_name)
            return
        if (
            self.node2 is source_node
            and self.axis2 is not None
            and self.axis2.name == source_axis_name
        ):
            self.node2 = node
            self.axis2 = SimpleNamespace(name=axis_name)

    def _stale_other_resultant_endpoints(
        self,
        *,
        excluded_result_node: _ResetAwareFakeTensorKrowchNode,
    ) -> None:
        """Hide this edge from other inherited-result views until reset."""
        for result_node in tuple(self.inherited_source_by_result_node):
            if result_node is excluded_result_node:
                continue
            if self.node1 is result_node or self.node2 is result_node:
                self.materialize_leaf_endpoint_for_resultant(result_node)

    def is_dangling(self) -> bool:
        return self.node2 is None

    def axis_name_for_node(
        self,
        node: _ResetAwareFakeTensorKrowchNode,
    ) -> SimpleNamespace:
        if self.node1 is node:
            return self.axis1
        assert self.node2 is node
        assert self.axis2 is not None
        return self.axis2

    def connects_nodes(
        self,
        left_node: _ResetAwareFakeTensorKrowchNode,
        right_node: _ResetAwareFakeTensorKrowchNode,
    ) -> bool:
        return (self.node1 is left_node and self.node2 is right_node) or (
            self.node1 is right_node and self.node2 is left_node
        )


class _ResetAwareFakeTensorKrowchNode:
    """Node double that tracks resultant-edge visibility across resets."""

    def __init__(
        self,
        *,
        tensor: object,
        axes_names: tuple[str, ...],
        name: str,
        network: _ResetAwareFakeTensorKrowchNetwork | None,
    ) -> None:
        del tensor
        self.name = name
        self.network = network
        self.is_resultant = False
        self.edges_by_axis_name = {
            axis_name: _ResetAwareFakeTensorKrowchEdge(self, axis_name)
            for axis_name in axes_names
        }
        self.pending_edges_by_axis_name: dict[
            str,
            _ResetAwareFakeTensorKrowchEdge,
        ] = {}
        if network is not None:
            network.register(self)

    def __getitem__(self, axis_name: str) -> _ResetAwareFakeTensorKrowchEdge:
        if axis_name in self.edges_by_axis_name:
            return self.edges_by_axis_name[axis_name]
        return self.pending_edges_by_axis_name[axis_name]

    def reattach_edges(self) -> None:
        self.edges_by_axis_name.update(self.pending_edges_by_axis_name)
        self.pending_edges_by_axis_name = {}

    def reset_inherited_edges(self) -> None:
        for axis_name, edge in self.edges_by_axis_name.items():
            edge.restore_resultant_endpoint(self, axis_name)
        for axis_name, edge in self.pending_edges_by_axis_name.items():
            edge.restore_resultant_endpoint(self, axis_name)


class _ResetAwareFakeTensorKrowchModule(ModuleType):
    """TensorKrowch double that requires ``network.reset()`` for inherited edges."""

    def __init__(self) -> None:
        super().__init__("tensorkrowch")
        self.Node = _ResetAwareFakeTensorKrowchNode
        self.TensorNetwork = _reset_aware_fake_tensorkrowch_network_factory

    @staticmethod
    def connect(
        left_edge: _ResetAwareFakeTensorKrowchEdge,
        right_edge: _ResetAwareFakeTensorKrowchEdge,
    ) -> _ResetAwareFakeTensorKrowchEdge:
        if left_edge.is_dangling() and left_edge.node1.is_resultant:
            left_edge.materialize_leaf_endpoint_for_resultant(left_edge.node1)
        left_edge.attach_second(right_edge.node1, right_edge.axis1.name)
        right_edge.node1.edges_by_axis_name[right_edge.axis1.name] = left_edge
        return left_edge

    @staticmethod
    def contract_between(
        left_node: _ResetAwareFakeTensorKrowchNode,
        right_node: _ResetAwareFakeTensorKrowchNode,
    ) -> _ResetAwareFakeTensorKrowchNode:
        left_edges = set(left_node.edges_by_axis_name.values())
        right_edges = set(right_node.edges_by_axis_name.values())
        shared_edges = {
            edge
            for edge in left_edges.intersection(right_edges)
            if edge.connects_nodes(left_node, right_node)
        }
        if not shared_edges:
            raise ValueError(
                f"No batch edges or shared edges between nodes {left_node.name} and {right_node.name} found"
            )
        surviving_edges_with_owner = [
            (edge, right_node)
            for edge in right_node.edges_by_axis_name.values()
            if edge not in shared_edges
        ] + [
            (edge, left_node)
            for edge in left_node.edges_by_axis_name.values()
            if edge not in shared_edges
        ]
        surviving_axis_names = _deduplicate_fake_tensorkrowch_axis_names(
            tuple(
                edge.axis_name_for_node(owner).name
                for edge, owner in surviving_edges_with_owner
            )
        )
        result = _ResetAwareFakeTensorKrowchNode(
            tensor=None,
            axes_names=surviving_axis_names,
            name=f"{left_node.name}_{right_node.name}",
            network=left_node.network,
        )
        result.is_resultant = True
        result.edges_by_axis_name = {}
        result.pending_edges_by_axis_name = {}
        for axis_name, (edge, owner) in zip(
            surviving_axis_names,
            surviving_edges_with_owner,
            strict=True,
        ):
            edge.replace_endpoint(owner, result, axis_name)
            result.pending_edges_by_axis_name[axis_name] = edge
        return result


def _reset_aware_fake_tensorkrowch_network_factory() -> (
    _ResetAwareFakeTensorKrowchNetwork
):
    """Return a fake network that models inherited-edge reset semantics."""
    return _ResetAwareFakeTensorKrowchNetwork()


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            ["import tensornetwork as tn", "axis_names=['i', 'x']", "tn.connect("],
        ),
        (
            EngineName.QUIMB,
            [
                "import quimb.tensor as qtn",
                "qtn.Tensor(",
                "network = qtn.TensorNetwork(",
            ],
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "import tensorkrowch as tk",
                "network = tk.TensorNetwork()",
                "tk.connect(",
            ],
        ),
        (
            EngineName.EINSUM_NUMPY,
            [
                "import numpy as np",
                "np.zeros((2, 3)",
                "results_list.append(np.einsum(",
                "result = results_list[-1]",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "import torch",
                "torch.zeros((2, 3), dtype=torch.float32)",
                "results_list.append(torch.einsum(",
                "result = results_list[-1]",
            ],
        ),
    ],
)
def test_generate_code_emits_engine_specific_contracts(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    result = generate_code(build_sample_spec_without_plan(), engine=engine)

    assert result.engine is engine
    assert result.code.endswith("\n")
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            [
                "a_data = np.full((2, 3), 2.5, dtype=float)",
                "b_data = np.ones((3, 4), dtype=float)",
                "tn.Node(a_data, ",
                "tn.Node(b_data, ",
            ],
        ),
        (
            EngineName.QUIMB,
            [
                "a_data = np.full((2, 3), 2.5, dtype=float)",
                "b_data = np.ones((3, 4), dtype=float)",
                "qtn.Tensor(data=a_data, ",
                "qtn.Tensor(data=b_data, ",
            ],
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "a_data = torch.full((2, 3), 2.5, dtype=torch.float32)",
                "b_data = torch.ones((3, 4), dtype=torch.float32)",
                "tk.Node(tensor=a_data, ",
                "tk.Node(tensor=b_data, ",
            ],
        ),
        (
            EngineName.EINSUM_NUMPY,
            [
                "a_data = np.full((2, 3), 2.5, dtype=float)",
                "b_data = np.ones((3, 4), dtype=float)",
                "tensors.append(a_data)",
                "tensors.append(b_data)",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "a_data = torch.full((2, 3), 2.5, dtype=torch.float32)",
                "b_data = torch.ones((3, 4), dtype=torch.float32)",
                "tensors.append(a_data)",
                "tensors.append(b_data)",
            ],
        ),
    ],
)
def test_generate_code_uses_tensor_data_initializers(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    spec = build_sample_spec_without_plan()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.FILL,
        fill_value=2.5,
    )
    spec.tensors[1].tensor_data = TensorDataSpec(
        mode=TensorDataMode.ONES,
    )

    result = generate_code(spec, engine=engine)

    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.EINSUM_NUMPY,
            [
                "i_data = np.eye(3, dtype=np.float64)",
                "copy_data = np.zeros((3,) * 3, dtype=np.complex64)",
                "copy_data[(np.arange(3),) * 3] = 1",
                "r_rng = np.random.default_rng(123)",
                "r_data = r_rng.uniform(size=(2, 2)).astype(np.float32)",
                "z_data = np.zeros((2, 2), dtype=np.float32)",
                "f_data = np.full((2, 2), (1.25-0.5j), dtype=np.complex128)",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "i_data = torch.eye(3, dtype=torch.float64)",
                "copy_data = torch.zeros((3,) * 3, dtype=torch.complex64)",
                "copy_data.index_put_((torch.arange(3),) * 3, torch.ones(3, dtype=torch.complex64))",
                "r_generator = torch.Generator()",
                "r_generator.manual_seed(123)",
                "r_data = torch.rand((2, 2), generator=r_generator, dtype=torch.float32)",
                "z_data = torch.zeros((2, 2), dtype=torch.float32)",
                "f_data = torch.full((2, 2), (1.25-0.5j), dtype=torch.complex128)",
            ],
        ),
    ],
)
def test_generate_code_uses_extended_tensor_data_initializers(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    spec = NetworkSpec(
        id="extended_initializers",
        name="extended initializers",
        tensors=[
            TensorSpec(
                id="tensor_i",
                name="I",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="i_left", name="left", dimension=3),
                    IndexSpec(id="i_right", name="right", dimension=3),
                ],
                tensor_data=TensorDataSpec(
                    mode=TensorDataMode.IDENTITY,
                    dtype=TensorDataDType.FLOAT64,
                ),
            ),
            TensorSpec(
                id="tensor_copy",
                name="Copy",
                position=CanvasPosition(x=260.0, y=120.0),
                indices=[
                    IndexSpec(id="copy_a", name="a", dimension=3),
                    IndexSpec(id="copy_b", name="b", dimension=3),
                    IndexSpec(id="copy_c", name="c", dimension=3),
                ],
                tensor_data=TensorDataSpec(
                    mode=TensorDataMode.COPY,
                    dtype=TensorDataDType.COMPLEX64,
                ),
            ),
            TensorSpec(
                id="tensor_r",
                name="R",
                position=CanvasPosition(x=440.0, y=120.0),
                indices=[
                    IndexSpec(id="r_a", name="a", dimension=2),
                    IndexSpec(id="r_b", name="b", dimension=2),
                ],
                tensor_data=TensorDataSpec(
                    mode=TensorDataMode.RANDOM,
                    dtype=TensorDataDType.FLOAT32,
                    seed=123,
                    distribution=TensorDataRandomDistribution.UNIFORM,
                ),
            ),
            TensorSpec(
                id="tensor_z",
                name="Z",
                position=CanvasPosition(x=620.0, y=120.0),
                indices=[
                    IndexSpec(id="z_a", name="a", dimension=2),
                    IndexSpec(id="z_b", name="b", dimension=2),
                ],
                tensor_data=TensorDataSpec(
                    mode=TensorDataMode.ZEROS,
                    dtype=TensorDataDType.FLOAT32,
                ),
            ),
            TensorSpec(
                id="tensor_f",
                name="F",
                position=CanvasPosition(x=800.0, y=120.0),
                indices=[
                    IndexSpec(id="f_a", name="a", dimension=2),
                    IndexSpec(id="f_b", name="b", dimension=2),
                ],
                tensor_data=TensorDataSpec(
                    mode=TensorDataMode.FILL,
                    fill_value={"real": 1.25, "imag": -0.5},
                    dtype=TensorDataDType.COMPLEX128,
                ),
            ),
        ],
    )

    result = generate_code(spec, engine=engine)

    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            [
                "a_data = _load_external_tensor_data(",
                "'project_data/a.npy'",
                "expected_shape=(2, 3)",
                "array_key=None",
            ],
        ),
        (
            EngineName.QUIMB,
            [
                "a_data = _load_external_tensor_data(",
                "'project_data/a.npy'",
                "expected_shape=(2, 3)",
                "array_key=None",
            ],
        ),
        (
            EngineName.EINSUM_NUMPY,
            [
                "a_data = _load_external_tensor_data(",
                "'project_data/a.npy'",
                "expected_shape=(2, 3)",
                "array_key=None",
            ],
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "a_data = torch.as_tensor(_load_external_tensor_data(",
                "'project_data/a.npy'",
                "expected_shape=(2, 3)",
                "array_key=None",
                "dtype=torch.float64",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "a_data = torch.as_tensor(_load_external_tensor_data(",
                "'project_data/a.npy'",
                "expected_shape=(2, 3)",
                "array_key=None",
                "dtype=torch.float64",
            ],
        ),
    ],
)
def test_generate_code_uses_external_tensor_data_initializers(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    spec = build_sample_spec_without_plan()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="project_data/a.npy",
        dtype=TensorDataDType.FLOAT64,
    )

    result = generate_code(spec, engine=engine)

    for snippet in expected_snippets:
        assert snippet in result.code
    assert "def _load_external_tensor_data(" in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            [
                "import torch",
                "torch.load(path, map_location='cpu', weights_only=True)",
                "return data.detach().cpu().numpy()",
                "a_data = _load_external_tensor_data(",
                "'project_data/a.pt'",
                "array_key='weights'",
            ],
        ),
        (
            EngineName.QUIMB,
            [
                "import torch",
                "torch.load(path, map_location='cpu', weights_only=True)",
                "return data.detach().cpu().numpy()",
                "a_data = _load_external_tensor_data(",
                "'project_data/a.pt'",
                "array_key='weights'",
            ],
        ),
        (
            EngineName.EINSUM_NUMPY,
            [
                "import torch",
                "torch.load(path, map_location='cpu', weights_only=True)",
                "return data.detach().cpu().numpy()",
                "a_data = _load_external_tensor_data(",
                "'project_data/a.pt'",
                "array_key='weights'",
            ],
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "torch.load(path, map_location='cpu', weights_only=True)",
                "return data",
                "a_data = torch.as_tensor(_load_external_tensor_data(",
                "'project_data/a.pt'",
                "array_key='weights'",
                "dtype=torch.float64",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "torch.load(path, map_location='cpu', weights_only=True)",
                "return data",
                "a_data = torch.as_tensor(_load_external_tensor_data(",
                "'project_data/a.pt'",
                "array_key='weights'",
                "dtype=torch.float64",
            ],
        ),
    ],
)
def test_generate_code_uses_external_pt_tensor_data_initializers(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    spec = build_sample_spec_without_plan()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="project_data/a.pt",
        array_key="weights",
        dtype=TensorDataDType.FLOAT64,
    )

    result = generate_code(spec, engine=engine)

    for snippet in expected_snippets:
        assert snippet in result.code
    assert "suffix = Path(path).suffix.lower()" in result.code


def test_generate_code_anchors_relative_external_tensor_paths_to_base_path() -> None:
    spec = build_sample_spec_without_plan()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="project_data/a.npz",
        array_key="left",
    )

    result = generate_code(
        spec,
        engine=EngineName.EINSUM_NUMPY,
        external_data_base_path="C:/project/designs",
    )

    assert "'C:/project/designs/project_data/a.npz'" in result.code
    assert "array_key='left'" in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippets", "unexpected_snippet"),
    [
        (
            EngineName.TENSORNETWORK,
            [
                "copy_shared_h_data = np.zeros((3,) * 3, dtype=float)",
                "copy_shared_h_data[(np.arange(3),) * 3] = 1",
            ],
            "copy_shared_h_data = np.array([[[",
        ),
        (
            EngineName.QUIMB,
            [
                "copy_shared_h_data = np.zeros((3,) * 3, dtype=float)",
                "copy_shared_h_data[(np.arange(3),) * 3] = 1",
            ],
            "copy_shared_h_data = np.array([[[",
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "copy_shared_h_data = torch.zeros((3,) * 3, dtype=torch.float32)",
                "copy_shared_h_data.index_put_((torch.arange(3),) * 3, torch.ones(3, dtype=torch.float32))",
            ],
            "copy_shared_h_data = torch.tensor([[[",
        ),
        (
            EngineName.EINSUM_NUMPY,
            [
                "copy_shared_h_data = np.zeros((3,) * 3, dtype=float)",
                "copy_shared_h_data[(np.arange(3),) * 3] = 1",
            ],
            "copy_shared_h_data = np.array([[[",
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "copy_shared_h_data = torch.zeros((3,) * 3, dtype=torch.float32)",
                "copy_shared_h_data.index_put_((torch.arange(3),) * 3, torch.ones(3, dtype=torch.float32))",
            ],
            "copy_shared_h_data = torch.tensor([[[",
        ),
    ],
)
def test_generate_code_uses_compact_hyperedge_copy_tensor_initializers(
    engine: EngineName,
    expected_snippets: list[str],
    unexpected_snippet: str,
) -> None:
    result = generate_code(build_three_tensor_hyperedge_spec(), engine=engine)

    for snippet in expected_snippets:
        assert snippet in result.code
    assert unexpected_snippet not in result.code


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])", "tensor_rows[0].append("],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_generate_code_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    spec = build_matrix_layout_spec()

    result = generate_code(spec, engine=engine, collection_format=collection_format)

    assert result.engine is engine
    for snippet in expected_snippets:
        assert snippet in result.code
    assert container_name in result.code
    assert "# Tensor A" in result.code
    assert "# Tensor B" in result.code
    assert "_TNE_SPEC" not in result.code


def test_matrix_collection_format_groups_tensors_by_visual_rows() -> None:
    result = generate_code(
        build_matrix_layout_spec(),
        engine=EngineName.EINSUM_NUMPY,
        collection_format=TensorCollectionFormat.MATRIX,
    )

    assignment_start = result.code.index("tensor_rows = []")
    assignment_end = result.code.index("# Contraction")
    assignment = result.code[assignment_start:assignment_end]

    assert "tensor_rows = []" in assignment
    assert "tensor_rows.append([])" in assignment
    assert assignment.index("# Tensor A") < assignment.index("tensor_rows[0].append(")
    assert assignment.index("# Tensor B") < assignment.index(
        "tensor_rows[0].append(", assignment.index("# Tensor B")
    )
    assert assignment.index("# Tensor C") < assignment.index("tensor_rows[1].append(")
    assert (
        assignment.index("np.zeros((2, 3), dtype=float)")
        < assignment.index("np.zeros((3, 5), dtype=float)")
        < assignment.index("np.zeros((5, 7), dtype=float)")
    )
    assert "tensor_rows[0][0]" in result.code
    assert "tensor_rows[0][1]" in result.code
    assert "tensor_rows[1][0]" in result.code


def test_generate_code_does_not_emit_roundtrip_metadata() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "_TNE_SPEC" not in result.code
    assert "# Tensor A data" in result.code


def test_periodic_generate_code_emits_roundtrip_metadata_marker() -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.EINSUM_NUMPY,
    )

    assert "# TNE_SPEC_B64:" in result.code
    assert "# Tensor Network Editor linear periodic mode" in result.code
    assert result.code.index(
        "# Tensor Network Editor linear periodic mode"
    ) < result.code.index("# TNE_SPEC_B64:")


def test_periodic_generate_code_can_skip_roundtrip_metadata_marker() -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.EINSUM_NUMPY,
        include_roundtrip_metadata=False,
    )

    assert "# TNE_SPEC_B64:" not in result.code
    assert "# Tensor Network Editor linear periodic mode" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_generate_code_labels_shared_normal_sections(engine: EngineName) -> None:
    result = generate_code(build_sample_spec_without_plan(), engine=engine)

    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Outputs" in result.code
    assert result.code.index("# Tensor collection") < result.code.index(
        "# Tensor construction"
    )
    assert result.code.index("# Tensor construction") < result.code.index("# Outputs")


@pytest.mark.parametrize(
    ("engine", "expected_snippet"),
    [
        (EngineName.TENSORNETWORK, "results_list.append(tn.contract_between("),
        (EngineName.QUIMB, "results_list.append(network["),
        (EngineName.TENSORKROWCH, "results_list.append(tk.contract_between("),
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_generate_code_respects_manual_plan_steps(
    engine: EngineName,
    expected_snippet: str,
) -> None:
    result = generate_code(build_sample_spec(), engine=engine)

    assert "# Manual contraction" in result.code
    assert expected_snippet in result.code
    assert "results_list = []" in result.code
    assert result.code.index("# Manual contraction") < result.code.index(
        "results_list = []"
    )
    assert result.code.index("results_list = []") < result.code.index("# Outputs")
    assert "remaining_operands = {" in result.code
    assert "'A-B': results_list[-1]" in result.code
    assert "result = results_list[-1]" in result.code


def test_tensorkrowch_normal_codegen_does_not_emit_reattach_edges() -> None:
    result = generate_code(
        build_three_tensor_complete_plan_spec(),
        engine=EngineName.TENSORKROWCH,
    )

    assert "results_list.append(tk.contract_between(" in result.code
    assert "reattach_edges(" not in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_generate_code_keeps_partial_manual_plan_as_prefix(
    engine: EngineName,
) -> None:
    result = generate_code(build_three_tensor_spec(), engine=engine)

    assert "results_list = []" in result.code
    assert "remaining_operands = {" in result.code
    assert "'A-B': results_list[-1]" in result.code
    assert "'C': tensors[2]" in result.code
    assert "result =" not in result.code


@pytest.mark.parametrize(
    ("engine", "append_snippet"),
    [
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_einsum_codegen_without_plan_emits_pairwise_steps(
    engine: EngineName,
    append_snippet: str,
) -> None:
    result = generate_code(build_three_tensor_spec_without_plan(), engine=engine)

    assert "# Contraction" in result.code
    assert "results_list = []" in result.code
    assert result.code.count(append_snippet) == 2
    assert "result = results_list[-1]" in result.code
    assert (
        f"result = {'np' if engine is EngineName.EINSUM_NUMPY else 'torch'}.einsum("
        not in result.code
    )


@pytest.mark.parametrize(
    ("engine", "append_snippet"),
    [
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_einsum_codegen_without_plan_contracts_connected_pairs_first(
    engine: EngineName,
    append_snippet: str,
) -> None:
    with patch(
        "tensor_network_editor.codegen.backends.einsum._load_random_optimizer_tools",
        return_value=None,
    ):
        result = generate_code(build_disconnected_pairs_spec(), engine=engine)

    assert result.code.count(append_snippet) == 3
    first_connected_step = f"{append_snippet}'ab,bc->ac', tensors[0], tensors[1]))"
    second_connected_step = f"{append_snippet}'de,ef->df', tensors[2], tensors[3]))"

    assert first_connected_step in result.code
    assert second_connected_step in result.code
    assert result.code.index(first_connected_step) < result.code.index(
        second_connected_step
    )


@pytest.mark.parametrize(
    ("engine", "module_name"),
    [
        (EngineName.EINSUM_NUMPY, "np"),
        (EngineName.EINSUM_TORCH, "torch"),
    ],
)
def test_einsum_codegen_without_plan_prefers_better_random_route_when_available(
    engine: EngineName,
    module_name: str,
) -> None:
    class FakeRandomGreedy:
        def __init__(
            self,
            *,
            max_time: float | None = None,
            minimize: str = "flops",
        ) -> None:
            self.max_time = max_time
            self.minimize = minimize

    def fake_contract_path(
        equation: str,
        *operand_shapes: tuple[int, ...],
        shapes: bool = True,
        optimize: object | None = None,
    ) -> tuple[list[tuple[int, int]], object]:
        assert equation == "ab,bc,cd->ad"
        assert shapes is True
        assert operand_shapes == ((100, 2), (2, 100), (100, 2))
        assert isinstance(optimize, FakeRandomGreedy)
        assert optimize.max_time == 0.05
        assert optimize.minimize == "flops"
        return [(1, 2), (0, 1)], object()

    with patch(
        "tensor_network_editor.codegen.backends.einsum._load_random_optimizer_tools",
        return_value=(fake_contract_path, FakeRandomGreedy),
    ):
        result = generate_code(build_random_better_chain_spec(), engine=engine)

    first_step = f"results_list.append({module_name}.einsum('bc,cd->bd', tensors[1], tensors[2]))"
    second_step = f"results_list.append({module_name}.einsum('ab,bd->ad', tensors[0], results_list[-1]))"

    assert first_step in result.code
    assert second_step in result.code
    assert result.code.index(first_step) < result.code.index(second_step)


def test_einsum_codegen_reuses_random_route_for_repeated_signature() -> None:
    class FakeRandomGreedy:
        def __init__(
            self,
            *,
            max_time: float | None = None,
            minimize: str = "flops",
        ) -> None:
            self.max_time = max_time
            self.minimize = minimize

    call_count = 0

    def fake_contract_path(
        equation: str,
        *operand_shapes: tuple[int, ...],
        shapes: bool = True,
        optimize: object | None = None,
    ) -> tuple[list[tuple[int, int]], object]:
        nonlocal call_count
        assert equation == "ab,bc,cd->ad"
        assert shapes is True
        assert operand_shapes == ((100, 2), (2, 100), (100, 2))
        assert isinstance(optimize, FakeRandomGreedy)
        call_count += 1
        return [(1, 2), (0, 1)], object()

    generator = EinsumNumpyCodeGenerator()
    with patch(
        "tensor_network_editor.codegen.backends.einsum._load_random_optimizer_tools",
        return_value=(fake_contract_path, FakeRandomGreedy),
    ):
        first = generator.generate(build_random_better_chain_spec())
        second = generator.generate(build_random_better_chain_spec())

    assert first.code == second.code
    assert call_count == 1


def test_tensorkrowch_codegen_rejects_manual_outer_product_plan() -> None:
    with pytest.raises(CodeGenerationError, match="outer product"):
        generate_code(build_outer_product_plan_spec(), engine=EngineName.TENSORKROWCH)


def test_tensorkrowch_codegen_uses_edges_list_for_connections() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORKROWCH,
    )

    assert "edges_list = []" in result.code
    assert "edges_list.append((" in result.code
    assert "tk.connect(" in result.code


@pytest.mark.heavy_backend
def test_tensorkrowch_codegen_executes_when_tensor_names_contain_spaces() -> None:
    _import_required_backend(EngineName.TENSORKROWCH)
    spec = NetworkSpec(
        id="space_names",
        name="space names",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="Tensor A",
                position=CanvasPosition(x=0.0, y=0.0),
                indices=[IndexSpec(id="tensor_a_i", name="i", dimension=2)],
            )
        ],
    )

    result = generate_code(spec, engine=EngineName.TENSORKROWCH)
    namespace = _execute_generated_code(result.code)

    assert "tensors" in namespace


def test_tensornetwork_codegen_uses_edges_list_for_connections() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "edges_list = []" in result.code
    assert "edges_list.append(tn.connect(" in result.code
    assert "name='bond_x'" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.TENSORNETWORK, EngineName.TENSORKROWCH],
)
def test_linear_periodic_codegen_uses_cell_helpers_and_free_n_loop(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "def build_initial_cell(" in result.code
    assert "def build_periodic_cell(cell_index" in result.code
    assert "def build_final_cell(" in result.code
    assert "if n < 2:" in result.code
    assert "for cell_index in range(1, n - 1):" in result.code
    assert "connect_cell_interfaces(" in result.code
    assert "periodic_contract_internal" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_codegen_labels_shared_for_sections(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "# Shared helpers" in result.code
    assert "# Initial cell" in result.code
    assert "# Periodic cell" in result.code
    assert "# Final cell" in result.code
    assert "# Main loop" in result.code
    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Outputs" in result.code
    assert result.code.index("# Shared helpers") < result.code.index("# Initial cell")
    assert result.code.index("# Initial cell") < result.code.index("# Periodic cell")
    assert result.code.index("# Periodic cell") < result.code.index("# Final cell")
    assert result.code.index("# Final cell") < result.code.index("# Main loop")
    assert "def build_initial_cell() -> " in result.code
    assert "def build_periodic_cell(" in result.code
    assert ") -> " in result.code
    assert "def build_final_cell(" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_carry_codegen_labels_shared_for_sections(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_carry_chain_spec(), engine=engine)

    assert "# Shared helpers" in result.code
    assert "# Initial cell" in result.code
    assert "# Periodic cell" in result.code
    assert "# Final cell" in result.code
    assert "# Main loop" in result.code
    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Previous interface" in result.code
    assert "# Manual contraction" in result.code
    assert "# Outputs" in result.code
    assert "def build_initial_cell() -> " in result.code
    assert "previous_payload: dict[str, object]" in result.code


def test_linear_periodic_carry_tensorkrowch_codegen_tracks_boundary_edges_without_axis_order_assumptions() -> (
    None
):
    result = generate_code(
        build_linear_periodic_carry_chain_spec(),
        engine=EngineName.TENSORKROWCH,
    )
    fake_torch = _FakeTorchModule()
    fake_tensorkrowch = _FakeTensorKrowchModule()

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "tensorkrowch": fake_tensorkrowch,
        },
    ):
        namespace = _execute_generated_code(result.code, n=3)

    open_edges = namespace["open_edges"]
    assert isinstance(open_edges, list)
    assert len(open_edges) == 4
    assert [edge.origin for edge in open_edges] == [
        ("Initial", "phys"),
        ("PeriodicLeft", "phys_l"),
        ("PeriodicRight", "phys_r"),
        ("Final", "phys"),
    ]


def test_linear_periodic_carry_tensorkrowch_codegen_executes_when_periodic_cell_contracts_local_pair_before_previous_payload() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    assert spec.linear_periodic_chain.periodic_cell.contraction_plan is not None
    spec.linear_periodic_chain.periodic_cell.contraction_plan.steps = [
        ContractionStepSpec(
            id="periodic_contract_internal_first",
            left_operand_id="periodic_left_tensor",
            right_operand_id="periodic_right_tensor",
        ),
        ContractionStepSpec(
            id="periodic_consume_previous_second",
            left_operand_id="periodic_contract_internal_first",
            right_operand_id="__linear_previous__",
        ),
        ContractionStepSpec(
            id="periodic_carry_last",
            left_operand_id="periodic_consume_previous_second",
            right_operand_id="__linear_next__",
        ),
    ]
    result = generate_code(spec, engine=EngineName.TENSORKROWCH)
    fake_torch = _FakeTorchModule()
    fake_tensorkrowch = _FakeTensorKrowchModule()

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "tensorkrowch": fake_tensorkrowch,
        },
    ):
        namespace = _execute_generated_code(result.code, n=5)

    assert "result" in namespace
    assert "open_edges" in namespace


def test_linear_periodic_carry_tensorkrowch_codegen_materializes_result_edges_with_override() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    assert spec.linear_periodic_chain.periodic_cell.contraction_plan is not None
    spec.linear_periodic_chain.periodic_cell.contraction_plan.steps = [
        ContractionStepSpec(
            id="periodic_contract_internal_first",
            left_operand_id="periodic_left_tensor",
            right_operand_id="periodic_right_tensor",
        ),
        ContractionStepSpec(
            id="periodic_consume_previous_second",
            left_operand_id="periodic_contract_internal_first",
            right_operand_id="__linear_previous__",
        ),
        ContractionStepSpec(
            id="periodic_carry_last",
            left_operand_id="periodic_consume_previous_second",
            right_operand_id="__linear_next__",
        ),
    ]
    result = generate_code(spec, engine=EngineName.TENSORKROWCH)
    fake_torch = _FakeTorchModule()
    fake_tensorkrowch = _FakeTensorKrowchModule()

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "tensorkrowch": fake_tensorkrowch,
        },
    ):
        namespace = _execute_generated_code(result.code, n=5)

    assert "reattach_edges(override=True)" in result.code
    assert "network.reset()" not in result.code
    assert "open_edges.extend([tracked_edge_0, tracked_edge_1])" in result.code
    assert "outgoing_interface = [results_list[-1]['right']]" in result.code
    assert "result" in namespace
    assert "open_edges" in namespace


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_codegen_does_not_stringify_manual_blocks(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "['results_list = []'" not in result.code
    assert "['remaining_operands = {" not in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippet"),
    [
        (EngineName.QUIMB, "import quimb.tensor as qtn"),
        (EngineName.EINSUM_NUMPY, "result = np.einsum("),
        (EngineName.EINSUM_TORCH, "result = torch.einsum("),
    ],
)
def test_linear_periodic_codegen_supports_remaining_backends(
    engine: EngineName,
    expected_snippet: str,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "if n < 2:" in result.code
    assert "for cell_index in range(1, n - 1):" in result.code
    assert expected_snippet in result.code


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.TENSORNETWORK, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.TENSORKROWCH, marks=pytest.mark.heavy_backend),
    ],
)
def test_linear_periodic_codegen_executes_for_supported_backends(
    engine: EngineName,
) -> None:
    _import_required_backend(engine)

    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    assert "network_nodes" in namespace
    assert "open_edges" in namespace


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.TENSORNETWORK, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.TENSORKROWCH, marks=pytest.mark.heavy_backend),
    ],
)
@pytest.mark.parametrize(
    "spec_factory",
    [
        build_linear_periodic_carry_chain_spec,
        build_linear_periodic_partial_carry_chain_spec,
    ],
)
def test_linear_periodic_carry_codegen_executes_for_supported_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
) -> None:
    _import_required_backend(engine)

    result = generate_code(spec_factory(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    assert "network_nodes" in namespace
    assert "open_edges" in namespace
    assert "result" in namespace


def test_linear_periodic_carry_codegen_threads_interface_payloads() -> None:
    result = generate_code(
        build_linear_periodic_carry_chain_spec(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "previous_payload = build_initial_cell()" in result.code
    assert "'operand':" in result.code
    assert "'outgoing_interface':" in result.code
    assert "next_boundary_operand" not in result.code


@pytest.mark.parametrize(
    ("engine", "expected_names"),
    [
        pytest.param(
            EngineName.QUIMB,
            {"network", "open_inds"},
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            {"result"},
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            {"result"},
            marks=pytest.mark.heavy_backend,
        ),
    ],
)
def test_linear_periodic_codegen_executes_for_remaining_backends(
    engine: EngineName,
    expected_names: set[str],
) -> None:
    _import_required_backend(engine)

    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    for expected_name in expected_names:
        assert expected_name in namespace


@pytest.mark.parametrize(
    ("engine", "spec_factory", "expected_snippet"),
    [
        (
            EngineName.QUIMB,
            build_linear_periodic_carry_chain_spec,
            "network.contract_between(",
        ),
        (
            EngineName.QUIMB,
            build_linear_periodic_partial_carry_chain_spec,
            "network.contract_between(",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(np.einsum(",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_partial_carry_chain_spec,
            "results_list.append(np.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(torch.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_partial_carry_chain_spec,
            "results_list.append(torch.einsum(",
        ),
    ],
)
def test_linear_periodic_carry_codegen_supports_remaining_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
    expected_snippet: str,
) -> None:
    result = generate_code(spec_factory(), engine=engine)

    assert "previous_payload = build_initial_cell()" in result.code
    assert expected_snippet in result.code


@pytest.mark.parametrize(
    (
        "engine",
        "spec_factory",
        "expected_names",
        "expect_non_empty_remaining_operands",
    ),
    [
        pytest.param(
            EngineName.QUIMB,
            build_linear_periodic_carry_chain_spec,
            {"network", "open_inds", "result"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.QUIMB,
            build_linear_periodic_partial_carry_chain_spec,
            {"network", "open_inds", "result"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_carry_chain_spec,
            {"result", "remaining_operands"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_partial_carry_chain_spec,
            {"result", "remaining_operands"},
            True,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            build_linear_periodic_carry_chain_spec,
            {"result", "remaining_operands"},
            False,
            marks=pytest.mark.heavy_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            build_linear_periodic_partial_carry_chain_spec,
            {"result", "remaining_operands"},
            True,
            marks=pytest.mark.heavy_backend,
        ),
    ],
)
def test_linear_periodic_carry_codegen_executes_for_remaining_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
    expected_names: set[str],
    expect_non_empty_remaining_operands: bool,
) -> None:
    _import_required_backend(engine)

    result = generate_code(spec_factory(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    for expected_name in expected_names:
        assert expected_name in namespace
    if "remaining_operands" in expected_names:
        remaining_operands = namespace["remaining_operands"]
        assert isinstance(remaining_operands, dict)
        if expect_non_empty_remaining_operands:
            assert remaining_operands


@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])", "tensor_rows[0].append("],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_quimb_linear_periodic_codegen_supports_collection_formats(
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.QUIMB,
        collection_format=collection_format,
    )

    for snippet in expected_snippets:
        assert snippet in result.code
    assert container_name in result.code
    assert "def build_initial_cell()" in result.code
    assert "network_tensors = list(initial_cell['tensors'])" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])"],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_linear_periodic_codegen_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=engine,
        collection_format=collection_format,
    )

    assert container_name in result.code
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_grid_periodic_codegen_uses_cell_helpers_and_free_n_m_loops(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "def build_top_left_cell(" in result.code
    assert "def build_top_cell(column_index" in result.code
    assert "def build_center_cell(column_index: int, row_index: int)" in result.code
    assert (
        "def build_bottom_right_cell(column_index: int, row_index: int)" in result.code
    )
    assert "validate_grid_shape(n, m)" in result.code
    assert "if n < 2:" in result.code
    assert "if m < 2:" in result.code
    assert "for column_index in range(1, n - 1):" in result.code
    assert "for row_index in range(1, m - 1):" in result.code


@pytest.mark.parametrize(
    "engine",
    [
        EngineName.TENSORNETWORK,
        EngineName.TENSORKROWCH,
    ],
)
def test_grid_periodic_codegen_supports_graph_backends_without_execution(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "# Tensor Network Editor grid periodic mode" in result.code
    assert "connect_cell_interfaces(" in result.code
    assert "network_nodes.extend(" in result.code
    assert "open_edges.extend(" in result.code
    assert "build_bottom_right_cell(n - 1, m - 1)" in result.code
    if engine is EngineName.TENSORNETWORK:
        assert "import tensornetwork as tn" in result.code
        assert "tn.connect(" in result.code
    else:
        assert "import tensorkrowch as tk" in result.code
        assert "tk.connect(" in result.code


@pytest.mark.parametrize(
    "engine",
    [
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    ],
)
def test_grid_periodic_codegen_supports_remaining_backends_without_execution(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "# Tensor Network Editor grid periodic mode" in result.code
    assert "horizontal_label(" in result.code
    assert "vertical_label(" in result.code
    assert "build_bottom_right_cell(n - 1, m - 1)" in result.code
    assert "for column_index in range(1, n - 1):" in result.code
    assert "for row_index in range(1, m - 1):" in result.code
    if engine is EngineName.QUIMB:
        assert "import quimb.tensor as qtn" in result.code
        assert "network = qtn.TensorNetwork(network_tensors)" in result.code
    if engine is EngineName.EINSUM_NUMPY:
        assert "import numpy as np" in result.code
        assert "result = np.einsum(" in result.code
    if engine is EngineName.EINSUM_TORCH:
        assert "import torch" in result.code
        assert "result = torch.einsum(" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])"],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_grid_periodic_codegen_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_grid_periodic_grid_spec(),
        engine=engine,
        collection_format=collection_format,
    )

    assert container_name in result.code
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_grid_periodic_codegen_with_manual_border_plan_exports_partial_network(
    engine: EngineName,
) -> None:
    result = generate_code(
        build_grid_periodic_grid_spec_with_partial_plan(),
        engine=engine,
    )

    assert "remaining_operands = {" in result.code
    assert "result = next(iter(remaining_operands.values()))" in result.code
    assert "Manual grid cell plans may leave a partial network" in result.code
    assert "top_left_cell = build_top_left_cell()" in result.code
    assert "center_left_step" in result.code
    if engine is EngineName.EINSUM_NUMPY:
        assert "result = np.einsum(" not in result.code
    if engine is EngineName.EINSUM_TORCH:
        assert "result = torch.einsum(" not in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_tree_periodic_codegen_uses_tree_helpers_and_total_depth_loops(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "def build_root_cell(" in result.code
    assert (
        "def build_branch_cell(level: int, node_index: int, parent_interface:"
        in result.code
    )
    assert (
        "def build_leaf_cell(level: int, node_index: int, parent_interface:"
        in result.code
    )
    assert "validate_tree_depth(n)" in result.code
    assert "if n < 3:" in result.code
    assert "branching_factor = 3" in result.code
    assert "frontier = list(root_cell['child_interfaces'])" in result.code
    assert "for level in range(1, n - 1):" in result.code
    assert "build_leaf_cell(n - 1, node_index, parent_interface)" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.TENSORNETWORK, EngineName.TENSORKROWCH],
)
def test_tree_periodic_codegen_supports_graph_backends(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "# Tensor Network Editor tree periodic mode" in result.code
    assert "def connect_tree_interfaces(" in result.code
    assert "network_nodes.extend(branch_cell['nodes'])" in result.code
    assert "open_edges.extend(leaf_cell['open_edges'])" in result.code
    if engine is EngineName.TENSORNETWORK:
        assert "import tensornetwork as tn" in result.code
        assert "tn.connect(" in result.code
    else:
        assert "import tensorkrowch as tk" in result.code
        assert "tk.connect(" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.QUIMB, EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_tree_periodic_codegen_supports_array_backends(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "# Tensor Network Editor tree periodic mode" in result.code
    assert "child_interfaces = []" in result.code
    assert "branching_factor = 3" in result.code
    if engine is EngineName.QUIMB:
        assert "import quimb.tensor as qtn" in result.code
        assert "network = qtn.TensorNetwork(network_tensors)" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> str"
            in result.code
        )
    if engine is EngineName.EINSUM_NUMPY:
        assert "import numpy as np" in result.code
        assert "result = np.einsum(" in result.code
        assert "np.zeros(" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> int"
            in result.code
        )
    if engine is EngineName.EINSUM_TORCH:
        assert "import torch" in result.code
        assert "result = torch.einsum(" in result.code
        assert "torch.zeros(" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> int"
            in result.code
        )


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])"],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_tree_periodic_codegen_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_tree_periodic_tree_spec(),
        engine=engine,
        collection_format=collection_format,
    )

    assert container_name in result.code
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_tree_periodic_codegen_with_manual_border_plan_exports_partial_network_bottom_up(
    engine: EngineName,
) -> None:
    result = generate_code(
        build_tree_periodic_tree_spec_with_partial_plan(),
        engine=engine,
    )

    assert "remaining_operands = {" in result.code
    assert "result = next(iter(remaining_operands.values()))" in result.code
    assert (
        "Manual tree cell plans are assembled from leaves toward the root"
        in result.code
    )
    assert "for level in range(n - 1, 0, -1):" in result.code
    assert "branch_parent_step" in result.code
    if engine is EngineName.EINSUM_NUMPY:
        assert "result = np.einsum(" not in result.code
    if engine is EngineName.EINSUM_TORCH:
        assert "result = torch.einsum(" not in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_einsum_codegen_uses_integer_sublist_form_for_many_labels(
    engine: EngineName,
) -> None:
    result = generate_code(build_many_label_spec(), engine=engine)

    append_snippet = (
        "results_list.append(np.einsum("
        if engine is EngineName.EINSUM_NUMPY
        else "results_list.append(torch.einsum("
    )

    assert "integer-sublist form because the network uses many labels" in result.code
    assert append_snippet in result.code
    assert "result = results_list[-1]" in result.code
    assert "# Einsum equation:" not in result.code


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.EINSUM_NUMPY, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.EINSUM_TORCH, marks=pytest.mark.heavy_backend),
    ],
)
def test_einsum_codegen_executes_for_empty_network(engine: EngineName) -> None:
    result = generate_code(build_empty_spec(), engine=engine)
    namespace: dict[str, object] = {}
    _import_required_backend(engine)

    exec(result.code, namespace, namespace)

    assert "result" in namespace
