"""Core graph data models used by saved network specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
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
from ._model_contraction import ContractionPlanSpec
from ._model_geometry import CanvasPosition, TensorSize


class LinearPeriodicCellName(StrEnum):
    """Named cells available in the linear periodic-chain editor mode."""

    INITIAL = "initial"
    PERIODIC = "periodic"
    FINAL = "final"


class LinearPeriodicTensorRole(StrEnum):
    """Special editor-only roles used by virtual boundary tensors."""

    PREVIOUS = "previous"
    NEXT = "next"


class GridPeriodicCellName(StrEnum):
    """Named cells available in the bidimensional periodic-grid editor mode."""

    TOP_LEFT = "top_left"
    TOP = "top"
    TOP_RIGHT = "top_right"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom_right"


class GridPeriodicTensorRole(StrEnum):
    """Special editor-only roles used by 2D virtual boundary tensors."""

    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class TreePeriodicCellName(StrEnum):
    """Named cells available in the tree periodic editor mode."""

    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"


class TreePeriodicTensorRole(StrEnum):
    """Special editor-only roles used by tree virtual boundary tensors."""

    PARENT = "parent"
    CHILD = "child"


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
            grid_periodic_role=_coerce_grid_periodic_tensor_role(
                payload.get("grid_periodic_role"),
                field_name="grid_periodic_role",
            ),
            tree_periodic_role=_coerce_tree_periodic_tensor_role(
                payload.get("tree_periodic_role"),
                field_name="tree_periodic_role",
            ),
            tree_periodic_child_index=_coerce_optional_int(
                payload.get("tree_periodic_child_index"),
                field_name="tree_periodic_child_index",
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
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
class GridPeriodicGridSpec:
    """Typed payload that stores the nine-cell bidimensional periodic mode."""

    active_cell: GridPeriodicCellName = GridPeriodicCellName.CENTER
    top_left_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    top_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    top_right_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    left_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    center_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    right_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    bottom_left_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    bottom_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    bottom_right_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the grid periodic payload."""
        return {
            "active_cell": self.active_cell.value,
            "top_left_cell": self.top_left_cell.to_dict(),
            "top_cell": self.top_cell.to_dict(),
            "top_right_cell": self.top_right_cell.to_dict(),
            "left_cell": self.left_cell.to_dict(),
            "center_cell": self.center_cell.to_dict(),
            "right_cell": self.right_cell.to_dict(),
            "bottom_left_cell": self.bottom_left_cell.to_dict(),
            "bottom_cell": self.bottom_cell.to_dict(),
            "bottom_right_cell": self.bottom_right_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build the grid periodic payload from a serialized mapping."""
        return cls(
            active_cell=_coerce_grid_periodic_cell_name(
                payload.get("active_cell", GridPeriodicCellName.CENTER.value),
                field_name="active_cell",
            ),
            top_left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("top_left_cell", {}), field_name="top_left_cell"
                )
            ),
            top_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("top_cell", {}), field_name="top_cell")
            ),
            top_right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("top_right_cell", {}),
                    field_name="top_right_cell",
                )
            ),
            left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("left_cell", {}), field_name="left_cell")
            ),
            center_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("center_cell", {}), field_name="center_cell")
            ),
            right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("right_cell", {}), field_name="right_cell")
            ),
            bottom_left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("bottom_left_cell", {}),
                    field_name="bottom_left_cell",
                )
            ),
            bottom_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("bottom_cell", {}), field_name="bottom_cell")
            ),
            bottom_right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("bottom_right_cell", {}),
                    field_name="bottom_right_cell",
                )
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class TreePeriodicTreeSpec:
    """Typed payload that stores the three-cell tree periodic mode."""

    active_cell: TreePeriodicCellName = TreePeriodicCellName.ROOT
    branching_factor: int = 2
    root_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    branch_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    leaf_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the tree periodic payload."""
        return {
            "active_cell": self.active_cell.value,
            "branching_factor": self.branching_factor,
            "root_cell": self.root_cell.to_dict(),
            "branch_cell": self.branch_cell.to_dict(),
            "leaf_cell": self.leaf_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build the tree periodic payload from a serialized mapping."""
        return cls(
            active_cell=_coerce_tree_periodic_cell_name(
                payload.get("active_cell", TreePeriodicCellName.ROOT.value),
                field_name="active_cell",
            ),
            branching_factor=coerce_int(
                payload.get("branching_factor", 2),
                field_name="branching_factor",
            ),
            root_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("root_cell", {}), field_name="root_cell")
            ),
            branch_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("branch_cell", {}), field_name="branch_cell")
            ),
            leaf_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("leaf_cell", {}), field_name="leaf_cell")
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
    grid_periodic_grid: GridPeriodicGridSpec | None = None
    tree_periodic_tree: TreePeriodicTreeSpec | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def tensor_map(self) -> dict[str, TensorSpec]:
        """Return a mapping from tensor ids to tensor specifications."""
        from ..analysis._network_analysis import tensor_map

        return tensor_map(self)

    def index_map(self) -> dict[str, tuple[TensorSpec, IndexSpec]]:
        """Return a mapping from index ids to their owning tensor and index."""
        from ..analysis._network_analysis import index_map

        return index_map(self)

    def connected_index_ids(self) -> set[str]:
        """Return the ids of indices that participate in an edge."""
        from ..analysis._network_analysis import connected_index_ids

        return connected_index_ids(self)

    def open_indices(self) -> list[tuple[TensorSpec, IndexSpec]]:
        """Return the tensor/index pairs that are not connected by any edge."""
        from ..analysis._network_analysis import open_indices

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
            "grid_periodic_grid": (
                self.grid_periodic_grid.to_dict()
                if self.grid_periodic_grid is not None
                else None
            ),
            "tree_periodic_tree": (
                self.tree_periodic_tree.to_dict()
                if self.tree_periodic_tree is not None
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build a network from a serialized mapping."""
        tensors_payload = require_list(payload.get("tensors", []), field_name="tensors")
        groups_payload = require_list(payload.get("groups", []), field_name="groups")
        edges_payload = require_list(payload.get("edges", []), field_name="edges")
        notes_payload = require_list(payload.get("notes", []), field_name="notes")
        contraction_plan_payload = payload.get("contraction_plan")
        linear_periodic_chain_payload = payload.get("linear_periodic_chain")
        grid_periodic_grid_payload = payload.get("grid_periodic_grid")
        tree_periodic_tree_payload = payload.get("tree_periodic_tree")
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
            grid_periodic_grid=(
                GridPeriodicGridSpec.from_dict(
                    require_dict(
                        grid_periodic_grid_payload,
                        field_name="grid_periodic_grid",
                    )
                )
                if grid_periodic_grid_payload is not None
                else None
            ),
            tree_periodic_tree=(
                TreePeriodicTreeSpec.from_dict(
                    require_dict(
                        tree_periodic_tree_payload,
                        field_name="tree_periodic_tree",
                    )
                )
                if tree_periodic_tree_payload is not None
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


def _coerce_grid_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> GridPeriodicCellName:
    """Coerce a serialized value to a valid grid periodic cell name."""
    try:
        return GridPeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid grid periodic cell name."
        ) from exc


def _coerce_grid_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> GridPeriodicTensorRole | None:
    """Coerce a serialized value to a valid grid periodic tensor role."""
    if value is None:
        return None
    try:
        return GridPeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid grid periodic tensor role."
        ) from exc


def _coerce_tree_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> TreePeriodicCellName:
    """Coerce a serialized value to a valid tree periodic cell name."""
    try:
        return TreePeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid tree periodic cell name."
        ) from exc


def _coerce_tree_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> TreePeriodicTensorRole | None:
    """Coerce a serialized value to a valid tree periodic tensor role."""
    if value is None:
        return None
    try:
        return TreePeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid tree periodic tensor role."
        ) from exc


def _coerce_optional_int(value: object, *, field_name: str) -> int | None:
    """Coerce an optional integer payload field."""
    if value is None:
        return None
    return coerce_int(value, field_name=field_name)
