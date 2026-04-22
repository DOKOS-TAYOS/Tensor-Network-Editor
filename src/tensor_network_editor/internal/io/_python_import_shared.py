"""Shared builders used by static and live Python importers."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SerializationError
from ...models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    HyperedgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorDataSpec,
    TensorSize,
    TensorSpec,
)
from ._python_roundtrip_helpers import sanitize_identifier


@dataclass(slots=True)
class ImportedTensor:
    """Tensor information recovered from one supported Python object."""

    reference: str
    name: str
    shape: tuple[int, ...]
    index_labels: tuple[str, ...]
    tensor_data: TensorDataSpec | None = None


@dataclass(slots=True, frozen=True)
class ExplicitConnection:
    """One explicit binary connection recovered from source or runtime."""

    name: str
    left_reference: str
    left_index_name: str
    right_reference: str
    right_index_name: str


def build_network_from_shared_labels(
    *,
    tensors_by_reference: dict[str, ImportedTensor],
    tensor_order: list[str],
    allow_hyperedges: bool,
) -> NetworkSpec:
    """Build a ``NetworkSpec`` by inferring connections from shared labels."""
    tensor_specs, index_lookup = build_imported_tensor_specs(
        tensors_by_reference=tensors_by_reference,
        tensor_order=tensor_order,
    )
    edge_specs: list[EdgeSpec] = []
    hyperedge_specs: list[HyperedgeSpec] = []
    label_occurrences: dict[str, list[tuple[str, str]]] = {}
    for reference in tensor_order:
        tensor = tensors_by_reference[reference]
        for index_label in tensor.index_labels:
            tensor_id, index_id = index_lookup[(reference, index_label)]
            label_occurrences.setdefault(index_label, []).append((tensor_id, index_id))
    edge_counter = 1
    hyperedge_counter = 1
    for label, occurrences in label_occurrences.items():
        if len(occurrences) == 2:
            edge_specs.append(
                EdgeSpec(
                    id=f"edge_{edge_counter}",
                    name=label,
                    left=EdgeEndpointRef(
                        tensor_id=occurrences[0][0],
                        index_id=occurrences[0][1],
                    ),
                    right=EdgeEndpointRef(
                        tensor_id=occurrences[1][0],
                        index_id=occurrences[1][1],
                    ),
                )
            )
            edge_counter += 1
        elif len(occurrences) > 2:
            if not allow_hyperedges:
                raise SerializationError(
                    "The supported Python importer cannot infer non-binary shared labels for this profile."
                )
            hyperedge_specs.append(
                HyperedgeSpec(
                    id=f"hyperedge_{hyperedge_counter}",
                    name=label,
                    endpoints=[
                        EdgeEndpointRef(tensor_id=tensor_id, index_id=index_id)
                        for tensor_id, index_id in occurrences
                    ],
                )
            )
            hyperedge_counter += 1
    return NetworkSpec(
        id="imported_python_network",
        name="Imported Python Network",
        tensors=tensor_specs,
        edges=edge_specs,
        hyperedges=hyperedge_specs,
        groups=[],
        notes=[],
        contraction_plan=None,
    )


def build_network_from_explicit_connections(
    *,
    tensors_by_reference: dict[str, ImportedTensor],
    tensor_order: list[str],
    explicit_connections: list[ExplicitConnection],
) -> NetworkSpec:
    """Build a ``NetworkSpec`` from explicit named binary connections."""
    tensor_specs, index_lookup = build_imported_tensor_specs(
        tensors_by_reference=tensors_by_reference,
        tensor_order=tensor_order,
    )
    edge_specs: list[EdgeSpec] = []
    for edge_index, connection in enumerate(explicit_connections, start=1):
        left_lookup = index_lookup.get(
            (connection.left_reference, connection.left_index_name)
        )
        right_lookup = index_lookup.get(
            (connection.right_reference, connection.right_index_name)
        )
        if left_lookup is None or right_lookup is None:
            raise SerializationError(
                "The supported TensorNetwork importer references an unknown node axis."
            )
        edge_specs.append(
            EdgeSpec(
                id=f"edge_{edge_index}",
                name=connection.name,
                left=EdgeEndpointRef(tensor_id=left_lookup[0], index_id=left_lookup[1]),
                right=EdgeEndpointRef(
                    tensor_id=right_lookup[0],
                    index_id=right_lookup[1],
                ),
            )
        )
    return NetworkSpec(
        id="imported_python_network",
        name="Imported Python Network",
        tensors=tensor_specs,
        edges=edge_specs,
        hyperedges=[],
        groups=[],
        notes=[],
        contraction_plan=None,
    )


def build_imported_tensor_specs(
    *,
    tensors_by_reference: dict[str, ImportedTensor],
    tensor_order: list[str],
) -> tuple[list[TensorSpec], dict[tuple[str, str], tuple[str, str]]]:
    """Build imported tensors and an index lookup keyed by reference and label."""
    tensor_specs: list[TensorSpec] = []
    index_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    used_tensor_ids: set[str] = set()
    for tensor_index, reference in enumerate(tensor_order, start=1):
        imported_tensor = tensors_by_reference[reference]
        if len(imported_tensor.shape) != len(imported_tensor.index_labels):
            raise SerializationError(
                "The supported Python importer requires tensor shapes to match their named indices."
            )
        tensor_id = unique_identifier(
            preferred_identifier=reference,
            used_identifiers=used_tensor_ids,
            fallback_identifier=f"tensor_{tensor_index}",
        )
        index_specs: list[IndexSpec] = []
        for label_index, (index_label, dimension) in enumerate(
            zip(imported_tensor.index_labels, imported_tensor.shape, strict=True),
            start=1,
        ):
            index_id = f"{tensor_id}_index_{label_index}"
            index_specs.append(
                IndexSpec(
                    id=index_id,
                    name=index_label,
                    dimension=dimension,
                )
            )
            index_lookup[(reference, index_label)] = (tensor_id, index_id)
        tensor_specs.append(
            TensorSpec(
                id=tensor_id,
                name=imported_tensor.name,
                position=CanvasPosition(
                    x=120.0 + (tensor_index - 1) * 240.0,
                    y=160.0,
                ),
                size=TensorSize(),
                indices=index_specs,
                tensor_data=imported_tensor.tensor_data,
            )
        )
    return tensor_specs, index_lookup


def default_connection_name(left_index_name: str, right_index_name: str) -> str:
    """Choose a readable fallback connection name."""
    if left_index_name == right_index_name:
        return left_index_name
    return f"{left_index_name}_{right_index_name}"


def unique_identifier(
    *,
    preferred_identifier: str,
    used_identifiers: set[str],
    fallback_identifier: str,
) -> str:
    """Return a unique, stable identifier derived from source references."""
    base_identifier = sanitize_identifier(preferred_identifier) or sanitize_identifier(
        fallback_identifier
    )
    if not base_identifier:
        base_identifier = "tensor"
    if base_identifier[0].isdigit():
        base_identifier = f"tensor_{base_identifier}"
    candidate = base_identifier
    suffix = 2
    while candidate in used_identifiers:
        candidate = f"{base_identifier}_{suffix}"
        suffix += 1
    used_identifiers.add(candidate)
    return candidate
