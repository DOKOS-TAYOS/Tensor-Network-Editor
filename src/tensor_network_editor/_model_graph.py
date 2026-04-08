"""Core graph data models used by saved network specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self, cast

from ._model_contraction import ContractionPlanSpec
from ._model_geometry import CanvasPosition, TensorSize
from ._payloads import (
    coerce_int,
    coerce_metadata,
    coerce_string,
    new_identifier,
    require_dict,
    require_list,
)
from .types import JSONValue, MetadataDict


class LinearPeriodicCellName(StrEnum):
    """Named cells available in the linear periodic-chain editor mode."""

    INITIAL = "initial"
    PERIODIC = "periodic"
    FINAL = "final"


class LinearPeriodicTensorRole(StrEnum):
    """Special editor-only roles used by virtual boundary tensors."""

    PREVIOUS = "previous"
    NEXT = "next"


@dataclass(slots=True)
class IndexSpec:
    """One named index that belongs to a tensor."""

    id: str = field(default_factory=lambda: new_identifier("index"))
    name: str = "index"
    dimension: int = 2
    offset: CanvasPosition = field(default_factory=lambda: CanvasPosition(x=0.0, y=0.0))
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the index to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "name": self.name,
            "dimension": self.dimension,
            "offset": self.offset.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build an index from a serialized mapping."""
        offset_payload = require_dict(
            payload.get("offset", {"x": 0.0, "y": 0.0}),
            field_name="offset",
        )
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            dimension=coerce_int(payload["dimension"], field_name="dimension"),
            offset=CanvasPosition.from_dict(offset_payload),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class TensorSpec:
    """A tensor node together with its canvas placement and indices."""

    id: str = field(default_factory=lambda: new_identifier("tensor"))
    name: str = "Tensor"
    position: CanvasPosition = field(default_factory=CanvasPosition)
    size: TensorSize = field(default_factory=TensorSize)
    indices: list[IndexSpec] = field(default_factory=list)
    linear_periodic_role: LinearPeriodicTensorRole | None = None
    metadata: MetadataDict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape derived from its index dimensions."""
        return tuple(index.dimension for index in self.indices)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the tensor to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "indices": [index.to_dict() for index in self.indices],
            "linear_periodic_role": (
                self.linear_periodic_role.value
                if self.linear_periodic_role is not None
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a tensor from a serialized mapping."""
        position_payload = require_dict(payload["position"], field_name="position")
        size_payload = require_dict(
            payload.get("size", {"width": 180.0, "height": 108.0}),
            field_name="size",
        )
        indices_payload = require_list(payload.get("indices", []), field_name="indices")
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            position=CanvasPosition.from_dict(position_payload),
            size=TensorSize.from_dict(size_payload),
            indices=[
                IndexSpec.from_dict(require_dict(index, field_name="index"))
                for index in indices_payload
            ],
            linear_periodic_role=_coerce_linear_periodic_tensor_role(
                payload.get("linear_periodic_role"),
                field_name="linear_periodic_role",
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True, frozen=True)
class EdgeEndpointRef:
    """Reference one endpoint of an edge by tensor id and index id."""

    tensor_id: str
    index_id: str

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the endpoint reference to a JSON-compatible mapping."""
        return {"tensor_id": self.tensor_id, "index_id": self.index_id}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build an endpoint reference from a serialized mapping."""
        return cls(
            tensor_id=coerce_string(payload["tensor_id"], field_name="tensor_id"),
            index_id=coerce_string(payload["index_id"], field_name="index_id"),
        )


@dataclass(slots=True)
class EdgeSpec:
    """A pairwise edge connecting two tensor indices."""

    id: str = field(default_factory=lambda: new_identifier("edge"))
    name: str = "edge"
    left: EdgeEndpointRef = field(default_factory=lambda: EdgeEndpointRef("", ""))
    right: EdgeEndpointRef = field(default_factory=lambda: EdgeEndpointRef("", ""))
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the edge to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "name": self.name,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build an edge from a serialized mapping."""
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            left=EdgeEndpointRef.from_dict(
                require_dict(payload["left"], field_name="left")
            ),
            right=EdgeEndpointRef.from_dict(
                require_dict(payload["right"], field_name="right")
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class GroupSpec:
    """A visual grouping of tensor ids in the editor."""

    id: str = field(default_factory=lambda: new_identifier("group"))
    name: str = "Group"
    tensor_ids: list[str] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the group to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "name": self.name,
            "tensor_ids": cast(JSONValue, list(self.tensor_ids)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a group from a serialized mapping."""
        tensor_ids_payload = require_list(
            payload.get("tensor_ids", []), field_name="tensor_ids"
        )
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            tensor_ids=[
                coerce_string(tensor_id, field_name="tensor_id")
                for tensor_id in tensor_ids_payload
            ],
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class CanvasNoteSpec:
    """A free-form text note placed on the editor canvas."""

    id: str = field(default_factory=lambda: new_identifier("note"))
    text: str = "Note"
    position: CanvasPosition = field(default_factory=CanvasPosition)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the note to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "text": self.text,
            "position": self.position.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a note from a serialized mapping."""
        position_payload = require_dict(payload["position"], field_name="position")
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            text=coerce_string(payload["text"], field_name="text"),
            position=CanvasPosition.from_dict(position_payload),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class LinearPeriodicCellSpec:
    """One editable cell inside the linear periodic-chain editor mode."""

    tensors: list[TensorSpec] = field(default_factory=list)
    groups: list[GroupSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    notes: list[CanvasNoteSpec] = field(default_factory=list)
    contraction_plan: ContractionPlanSpec | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the cell to a JSON-compatible mapping."""
        return {
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "groups": [group.to_dict() for group in self.groups],
            "edges": [edge.to_dict() for edge in self.edges],
            "notes": [note.to_dict() for note in self.notes],
            "contraction_plan": (
                self.contraction_plan.to_dict()
                if self.contraction_plan is not None
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a cell from a serialized mapping."""
        tensors_payload = require_list(payload.get("tensors", []), field_name="tensors")
        groups_payload = require_list(payload.get("groups", []), field_name="groups")
        edges_payload = require_list(payload.get("edges", []), field_name="edges")
        notes_payload = require_list(payload.get("notes", []), field_name="notes")
        contraction_plan_payload = payload.get("contraction_plan")
        return cls(
            tensors=[
                TensorSpec.from_dict(require_dict(tensor, field_name="tensor"))
                for tensor in tensors_payload
            ],
            groups=[
                GroupSpec.from_dict(require_dict(group, field_name="group"))
                for group in groups_payload
            ],
            edges=[
                EdgeSpec.from_dict(require_dict(edge, field_name="edge"))
                for edge in edges_payload
            ],
            notes=[
                CanvasNoteSpec.from_dict(require_dict(note, field_name="note"))
                for note in notes_payload
            ],
            contraction_plan=(
                ContractionPlanSpec.from_dict(
                    require_dict(
                        contraction_plan_payload, field_name="contraction_plan"
                    )
                )
                if contraction_plan_payload is not None
                else None
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class LinearPeriodicChainSpec:
    """Typed payload that stores the three-cell linear periodic mode."""

    active_cell: LinearPeriodicCellName = LinearPeriodicCellName.INITIAL
    initial_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    periodic_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    final_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the linear periodic-chain payload."""
        return {
            "active_cell": self.active_cell.value,
            "initial_cell": self.initial_cell.to_dict(),
            "periodic_cell": self.periodic_cell.to_dict(),
            "final_cell": self.final_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build the periodic-chain payload from a serialized mapping."""
        return cls(
            active_cell=_coerce_linear_periodic_cell_name(
                payload.get("active_cell", LinearPeriodicCellName.INITIAL.value),
                field_name="active_cell",
            ),
            initial_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("initial_cell", {}), field_name="initial_cell")
            ),
            periodic_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("periodic_cell", {}), field_name="periodic_cell"
                )
            ),
            final_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("final_cell", {}), field_name="final_cell")
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class NetworkSpec:
    """The root object that stores an abstract tensor-network design."""

    id: str = field(default_factory=lambda: new_identifier("network"))
    name: str = "Tensor Network"
    tensors: list[TensorSpec] = field(default_factory=list)
    groups: list[GroupSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    notes: list[CanvasNoteSpec] = field(default_factory=list)
    contraction_plan: ContractionPlanSpec | None = None
    linear_periodic_chain: LinearPeriodicChainSpec | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def tensor_map(self) -> dict[str, TensorSpec]:
        """Return a mapping from tensor ids to tensor specifications."""
        from ._network_analysis import tensor_map

        return tensor_map(self)

    def index_map(self) -> dict[str, tuple[TensorSpec, IndexSpec]]:
        """Return a mapping from index ids to their owning tensor and index."""
        from ._network_analysis import index_map

        return index_map(self)

    def connected_index_ids(self) -> set[str]:
        """Return the ids of indices that participate in an edge."""
        from ._network_analysis import connected_index_ids

        return connected_index_ids(self)

    def open_indices(self) -> list[tuple[TensorSpec, IndexSpec]]:
        """Return the tensor/index pairs that are not connected by any edge."""
        from ._network_analysis import open_indices

        return open_indices(self)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the network to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "name": self.name,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "groups": [group.to_dict() for group in self.groups],
            "edges": [edge.to_dict() for edge in self.edges],
            "notes": [note.to_dict() for note in self.notes],
            "contraction_plan": (
                self.contraction_plan.to_dict()
                if self.contraction_plan is not None
                else None
            ),
            "linear_periodic_chain": (
                self.linear_periodic_chain.to_dict()
                if self.linear_periodic_chain is not None
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a network from a serialized mapping."""
        tensors_payload = require_list(payload.get("tensors", []), field_name="tensors")
        groups_payload = require_list(payload.get("groups", []), field_name="groups")
        edges_payload = require_list(payload.get("edges", []), field_name="edges")
        notes_payload = require_list(payload.get("notes", []), field_name="notes")
        contraction_plan_payload = payload.get("contraction_plan")
        linear_periodic_chain_payload = payload.get("linear_periodic_chain")
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            tensors=[
                TensorSpec.from_dict(require_dict(tensor, field_name="tensor"))
                for tensor in tensors_payload
            ],
            groups=[
                GroupSpec.from_dict(require_dict(group, field_name="group"))
                for group in groups_payload
            ],
            edges=[
                EdgeSpec.from_dict(require_dict(edge, field_name="edge"))
                for edge in edges_payload
            ],
            notes=[
                CanvasNoteSpec.from_dict(require_dict(note, field_name="note"))
                for note in notes_payload
            ],
            contraction_plan=(
                ContractionPlanSpec.from_dict(
                    require_dict(
                        contraction_plan_payload, field_name="contraction_plan"
                    )
                )
                if contraction_plan_payload is not None
                else None
            ),
            linear_periodic_chain=(
                LinearPeriodicChainSpec.from_dict(
                    require_dict(
                        linear_periodic_chain_payload,
                        field_name="linear_periodic_chain",
                    )
                )
                if linear_periodic_chain_payload is not None
                else None
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


def _coerce_linear_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> LinearPeriodicCellName:
    """Coerce a serialized value to a valid linear periodic cell name."""
    try:
        return LinearPeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid linear periodic cell name."
        ) from exc


def _coerce_linear_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> LinearPeriodicTensorRole | None:
    """Coerce a serialized value to a valid linear periodic tensor role."""
    if value is None:
        return None
    try:
        return LinearPeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid linear periodic tensor role."
        ) from exc
