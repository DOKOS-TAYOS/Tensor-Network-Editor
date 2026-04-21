"""Entity-level graph models shared across network layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self, cast

from ...types import JSONValue, MetadataDict
from ..io._payloads import (
    coerce_int,
    coerce_metadata,
    coerce_string,
    new_identifier,
    require_dict,
    require_list,
)
from ._model_geometry import CanvasPosition, TensorSize
from ._model_periodic_types import (
    GridPeriodicTensorRole,
    LinearPeriodicTensorRole,
    TreePeriodicTensorRole,
    coerce_grid_periodic_tensor_role,
    coerce_linear_periodic_tensor_role,
    coerce_optional_int,
    coerce_tree_periodic_tensor_role,
)
from ._model_tensor_data import TensorDataSpec


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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
    grid_periodic_role: GridPeriodicTensorRole | None = None
    tree_periodic_role: TreePeriodicTensorRole | None = None
    tree_periodic_child_index: int | None = None
    tensor_data: TensorDataSpec | None = None
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
            "grid_periodic_role": (
                self.grid_periodic_role.value
                if self.grid_periodic_role is not None
                else None
            ),
            "tree_periodic_role": (
                self.tree_periodic_role.value
                if self.tree_periodic_role is not None
                else None
            ),
            "tree_periodic_child_index": self.tree_periodic_child_index,
            "tensor_data": (
                self.tensor_data.to_dict() if self.tensor_data is not None else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build a tensor from a serialized mapping."""
        position_payload = require_dict(payload["position"], field_name="position")
        size_payload = require_dict(
            payload.get("size", {"width": 180.0, "height": 108.0}),
            field_name="size",
        )
        indices_payload = require_list(payload.get("indices", []), field_name="indices")
        tensor_data_payload = payload.get("tensor_data")
        return cls(
            id=coerce_string(payload["id"], field_name="id"),
            name=coerce_string(payload["name"], field_name="name"),
            position=CanvasPosition.from_dict(position_payload),
            size=TensorSize.from_dict(size_payload),
            indices=[
                IndexSpec.from_dict(require_dict(index, field_name="index"))
                for index in indices_payload
            ],
            linear_periodic_role=coerce_linear_periodic_tensor_role(
                payload.get("linear_periodic_role"),
                field_name="linear_periodic_role",
            ),
            grid_periodic_role=coerce_grid_periodic_tensor_role(
                payload.get("grid_periodic_role"),
                field_name="grid_periodic_role",
            ),
            tree_periodic_role=coerce_tree_periodic_tensor_role(
                payload.get("tree_periodic_role"),
                field_name="tree_periodic_role",
            ),
            tree_periodic_child_index=coerce_optional_int(
                payload.get("tree_periodic_child_index"),
                field_name="tree_periodic_child_index",
            ),
            tensor_data=(
                TensorDataSpec.from_dict(
                    require_dict(tensor_data_payload, field_name="tensor_data")
                )
                if tensor_data_payload is not None
                else None
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
