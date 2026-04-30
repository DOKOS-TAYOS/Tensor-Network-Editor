"""Linear-family internal template builders."""

from __future__ import annotations

from collections.abc import Callable

from ...models import EdgeSpec, NetworkSpec, TensorDataMode, TensorDataSpec, TensorSpec
from ._template_builders_common import (
    DOWN_OFFSET,
    HORIZONTAL_SPACING,
    LEFT_OFFSET,
    RIGHT_OFFSET,
    UP_OFFSET,
    TemplateIndexConfig,
    _annotate_physics_1d_indices,
    _build_zero_literal,
    _make_edge,
    _make_tensor,
    _resolve_graph_size,
    _set_nested_literal_value,
)
from ._template_catalog import TEMPLATE_DEFINITIONS, TemplateParameters

LinearChainSiteIndexBuilder = Callable[
    [int, int, TemplateParameters], list[TemplateIndexConfig]
]


def _build_mps_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build an MPS template with the requested site count and dimensions."""
    spec = _build_linear_chain_template(
        "mps",
        parameters,
        tensor_name_prefix="A",
        spacing=HORIZONTAL_SPACING,
        site_index_builder=_build_mps_site_indices,
        periodic=parameters.boundary_condition == "periodic",
    )
    return _apply_mps_template_configuration(spec, parameters)


def _build_mpo_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build an MPO template with the requested site count and dimensions."""
    spec = _build_linear_chain_template(
        "mpo",
        parameters,
        tensor_name_prefix="W",
        spacing=330.0,
        site_index_builder=_build_mpo_site_indices,
        periodic=parameters.boundary_condition == "periodic",
    )
    return _apply_mpo_template_configuration(spec, parameters)


def _build_linear_chain_template(
    template_name: str,
    parameters: TemplateParameters,
    *,
    tensor_name_prefix: str,
    spacing: float,
    site_index_builder: LinearChainSiteIndexBuilder,
    periodic: bool = False,
) -> NetworkSpec:
    """Build one left-to-right chain template from a per-site index factory."""
    length = _resolve_graph_size(parameters)
    tensors = [
        _make_tensor(
            f"tensor_{site_index}",
            f"{tensor_name_prefix}{site_index + 1}",
            spacing * site_index,
            0.0,
            site_index_builder(site_index, length, parameters),
        )
        for site_index in range(length)
    ]
    definition = TEMPLATE_DEFINITIONS[template_name]
    spec_name = (
        definition.display_name
        if length == definition.defaults.graph_size
        else f"{definition.display_name} ({length} {definition.graph_size_label.lower()})"
    )
    return NetworkSpec(
        id=f"template_{template_name}_{length}",
        name=spec_name,
        tensors=tensors,
        edges=_make_linear_chain_edges(tensors, periodic=periodic),
    )


def _build_mps_site_indices(
    site_index: int,
    length: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the named index layout for one MPS site."""
    tensor_indices: list[TemplateIndexConfig] = []
    if parameters.boundary_condition == "periodic" or site_index > 0:
        tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
    if parameters.boundary_condition == "periodic" or site_index < length - 1:
        tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
    tensor_indices.append(("phys", parameters.physical_dimension, DOWN_OFFSET))
    return tensor_indices


def _build_mpo_site_indices(
    site_index: int,
    length: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the named index layout for one MPO site."""
    tensor_indices: list[TemplateIndexConfig] = []
    if parameters.boundary_condition == "periodic" or site_index > 0:
        tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
    if parameters.boundary_condition == "periodic" or site_index < length - 1:
        tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
    tensor_indices.extend(
        [
            ("bra", parameters.physical_dimension, UP_OFFSET),
            ("ket", parameters.physical_dimension, DOWN_OFFSET),
        ]
    )
    return tensor_indices


def _make_linear_chain_edges(
    tensors: list[TensorSpec],
    *,
    periodic: bool = False,
) -> list[EdgeSpec]:
    """Return the right-to-left bonds between adjacent chain tensors."""
    edges = [
        _make_edge(
            f"edge_{site_index}_{site_index + 1}",
            tensors[site_index],
            "right",
            tensors[site_index + 1],
            "left",
        )
        for site_index in range(len(tensors) - 1)
    ]
    if periodic and len(tensors) > 1:
        edges.append(
            _make_edge(
                f"edge_{len(tensors)}_1",
                tensors[-1],
                "right",
                tensors[0],
                "left",
            )
        )
    return edges


def _apply_mps_template_configuration(
    spec: NetworkSpec,
    parameters: TemplateParameters,
) -> NetworkSpec:
    """Attach metadata and tensor initialization presets to the built MPS."""
    spec.metadata = {
        "template_name": "mps",
        "role": "state",
        "boundary_condition": parameters.boundary_condition,
        "symmetry": parameters.symmetry,
        "initial_state": parameters.initial_state,
    }
    for tensor_index, tensor in enumerate(spec.tensors):
        tensor.metadata = {
            "role": "state",
            "family": "mps",
            "symmetry": parameters.symmetry,
            "initial_state": parameters.initial_state,
        }
        _annotate_physics_1d_indices(tensor, symmetry=parameters.symmetry)
        tensor.tensor_data = _build_mps_tensor_data(
            tensor,
            tensor_index=tensor_index,
            parameters=parameters,
        )
    return spec


def _apply_mpo_template_configuration(
    spec: NetworkSpec,
    parameters: TemplateParameters,
) -> NetworkSpec:
    """Attach semantic MPO metadata to the built operator chain."""
    spec.metadata = {
        "template_name": "mpo",
        "role": "operator",
        "boundary_condition": parameters.boundary_condition,
        "j": parameters.j,
        "h": parameters.h,
    }
    for tensor in spec.tensors:
        tensor.metadata = {
            "role": "operator",
            "family": "mpo",
            "boundary_condition": parameters.boundary_condition,
            "j": parameters.j,
            "h": parameters.h,
        }
        _annotate_physics_1d_indices(tensor, symmetry="none")
    return spec


def _build_mps_tensor_data(
    tensor: TensorSpec,
    *,
    tensor_index: int,
    parameters: TemplateParameters,
) -> TensorDataSpec:
    """Return the tensor-data initializer matching the chosen MPS preset."""
    if parameters.initial_state == "zeros":
        return TensorDataSpec(mode=TensorDataMode.ZEROS)
    if parameters.initial_state == "random":
        return TensorDataSpec(
            mode=TensorDataMode.RANDOM,
            seed=tensor_index,
        )
    if parameters.initial_state == "all_up":
        return _build_mps_literal_state_tensor_data(tensor, basis_index=0)
    if parameters.initial_state == "all_down":
        return _build_mps_literal_state_tensor_data(tensor, basis_index=1)
    return _build_mps_literal_state_tensor_data(
        tensor,
        basis_index=tensor_index % 2,
    )


def _build_mps_literal_state_tensor_data(
    tensor: TensorSpec,
    *,
    basis_index: int,
) -> TensorDataSpec:
    """Build one explicit basis-state tensor embedded into the current shape."""
    values = _build_zero_literal(list(tensor.shape))
    _set_nested_literal_value(
        values,
        [0] * (len(tensor.shape) - 1) + [basis_index],
        1.0,
    )
    return TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=values,
    )
