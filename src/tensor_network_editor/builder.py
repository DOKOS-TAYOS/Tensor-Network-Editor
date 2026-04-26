"""Fluent helpers for building normal tensor-network specs from Python."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from .models import (
    CanvasNoteSpec,
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    GroupSpec,
    HyperedgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorDataSpec,
    TensorSize,
    TensorSpec,
)
from .types import JSONValue, MetadataDict
from .validation import ensure_valid_spec

PositionInput: TypeAlias = CanvasPosition | tuple[float, float]
SizeInput: TypeAlias = TensorSize | tuple[float, float]
MetadataInput: TypeAlias = Mapping[str, JSONValue] | None


@dataclass(slots=True, frozen=True)
class IndexHandle:
    """Stable handle for one index created by a ``NetworkBuilder``."""

    _builder: NetworkBuilder
    tensor: TensorSpec
    index: IndexSpec

    @property
    def id(self) -> str:
        """Return the underlying index id."""
        return self.index.id

    @property
    def name(self) -> str:
        """Return the underlying index name."""
        return self.index.name


@dataclass(slots=True, frozen=True)
class TensorHandle:
    """Stable handle for one tensor created by a ``NetworkBuilder``."""

    _builder: NetworkBuilder
    tensor: TensorSpec

    @property
    def id(self) -> str:
        """Return the underlying tensor id."""
        return self.tensor.id

    @property
    def name(self) -> str:
        """Return the underlying tensor name."""
        return self.tensor.name

    def index(
        self,
        name: str = "index",
        dimension: int = 2,
        *,
        id: str | None = None,
        offset: PositionInput | None = None,
        metadata: MetadataInput = None,
    ) -> IndexHandle:
        """Add an index to this tensor and return its handle."""
        return self._builder._add_index(
            self,
            name=name,
            dimension=dimension,
            id=id,
            offset=offset,
            metadata=metadata,
        )

    def __getitem__(self, index_name: str) -> IndexHandle:
        """Return the unique index handle matching ``index_name``."""
        matches = [index for index in self.tensor.indices if index.name == index_name]
        if not matches:
            raise KeyError(index_name)
        if len(matches) > 1:
            raise ValueError(
                f"Tensor '{self.tensor.name}' has more than one index named "
                f"{index_name!r}."
            )
        return IndexHandle(self._builder, self.tensor, matches[0])


class NetworkBuilder:
    """Fluent builder for normal-mode ``NetworkSpec`` objects."""

    def __init__(
        self,
        name: str = "Tensor Network",
        *,
        id: str | None = None,
        metadata: MetadataInput = None,
    ) -> None:
        """Create an empty normal-mode network builder."""
        self._spec = NetworkSpec(name=name, metadata=_metadata_dict(metadata))
        if id is not None:
            self._spec.id = id

    def tensor(
        self,
        name: str = "Tensor",
        *,
        id: str | None = None,
        position: PositionInput | None = None,
        size: SizeInput | None = None,
        tensor_data: TensorDataSpec | None = None,
        metadata: MetadataInput = None,
    ) -> TensorHandle:
        """Add one tensor to the network and return its handle."""
        tensor = TensorSpec(
            name=name,
            position=_position(position),
            size=_size(size),
            tensor_data=tensor_data,
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            tensor.id = id
        self._spec.tensors.append(tensor)
        return TensorHandle(self, tensor)

    def connect(
        self,
        left: IndexHandle,
        right: IndexHandle,
        *,
        id: str | None = None,
        name: str | None = None,
        metadata: MetadataInput = None,
    ) -> EdgeSpec:
        """Connect two index handles with a pairwise edge."""
        left = self._require_index_handle(left)
        right = self._require_index_handle(right)
        edge = EdgeSpec(
            name=name if name is not None else left.index.name,
            left=EdgeEndpointRef(
                tensor_id=left.tensor.id,
                index_id=left.index.id,
            ),
            right=EdgeEndpointRef(
                tensor_id=right.tensor.id,
                index_id=right.index.id,
            ),
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            edge.id = id
        self._spec.edges.append(edge)
        return edge

    def hyperedge(
        self,
        indices: Iterable[IndexHandle],
        *,
        id: str | None = None,
        name: str = "hyperedge",
        hub_offset: PositionInput | None = None,
        metadata: MetadataInput = None,
    ) -> HyperedgeSpec:
        """Connect three or more index handles with one hyperedge."""
        endpoints = [
            EdgeEndpointRef(tensor_id=handle.tensor.id, index_id=handle.index.id)
            for handle in (self._require_index_handle(item) for item in indices)
        ]
        hyperedge = HyperedgeSpec(
            name=name,
            endpoints=endpoints,
            hub_offset=_position(hub_offset),
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            hyperedge.id = id
        self._spec.hyperedges.append(hyperedge)
        return hyperedge

    def group(
        self,
        tensors: Iterable[TensorHandle],
        *,
        id: str | None = None,
        name: str = "Group",
        metadata: MetadataInput = None,
    ) -> GroupSpec:
        """Create a visual group from tensor handles."""
        tensor_ids = [
            self._require_tensor_handle(tensor).tensor.id for tensor in tensors
        ]
        group = GroupSpec(
            name=name,
            tensor_ids=tensor_ids,
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            group.id = id
        self._spec.groups.append(group)
        return group

    def note(
        self,
        text: str = "Note",
        *,
        id: str | None = None,
        position: PositionInput | None = None,
        metadata: MetadataInput = None,
    ) -> CanvasNoteSpec:
        """Add one note to the network canvas."""
        note = CanvasNoteSpec(
            text=text,
            position=_position(position),
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            note.id = id
        self._spec.notes.append(note)
        return note

    def build(self, *, validate: bool = True) -> NetworkSpec:
        """Return the built spec, validating it by default."""
        if validate:
            return ensure_valid_spec(self._spec)
        return self._spec

    def _add_index(
        self,
        tensor: TensorHandle,
        *,
        name: str,
        dimension: int,
        id: str | None,
        offset: PositionInput | None,
        metadata: MetadataInput,
    ) -> IndexHandle:
        """Add one index to a tensor handle owned by this builder."""
        tensor = self._require_tensor_handle(tensor)
        index = IndexSpec(
            name=name,
            dimension=dimension,
            offset=_position(offset),
            metadata=_metadata_dict(metadata),
        )
        if id is not None:
            index.id = id
        tensor.tensor.indices.append(index)
        return IndexHandle(self, tensor.tensor, index)

    def _require_tensor_handle(self, handle: TensorHandle) -> TensorHandle:
        """Return ``handle`` when it belongs to this builder."""
        if handle._builder is not self or handle.tensor not in self._spec.tensors:
            raise ValueError(
                "Cannot use a tensor handle from a different NetworkBuilder."
            )
        return handle

    def _require_index_handle(self, handle: IndexHandle) -> IndexHandle:
        """Return ``handle`` when it belongs to this builder."""
        if (
            handle._builder is not self
            or handle.tensor not in self._spec.tensors
            or handle.index not in handle.tensor.indices
        ):
            raise ValueError(
                "Cannot use an index handle from a different NetworkBuilder."
            )
        return handle


def _position(value: PositionInput | None) -> CanvasPosition:
    """Return a canvas position from either a model object or tuple."""
    if value is None:
        return CanvasPosition()
    if isinstance(value, CanvasPosition):
        return value
    return CanvasPosition(x=float(value[0]), y=float(value[1]))


def _size(value: SizeInput | None) -> TensorSize:
    """Return a tensor size from either a model object or tuple."""
    if value is None:
        return TensorSize()
    if isinstance(value, TensorSize):
        return value
    return TensorSize(width=float(value[0]), height=float(value[1]))


def _metadata_dict(value: MetadataInput) -> MetadataDict:
    """Return a detached metadata dictionary."""
    return dict(value or {})


__all__ = ["IndexHandle", "NetworkBuilder", "TensorHandle"]
