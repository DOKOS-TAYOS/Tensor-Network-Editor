from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self, cast
from uuid import uuid4

from .types import JSONValue, MetadataDict


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be a number.")
    return float(value)


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be an integer.")
    return int(value)


def _require_dict(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object.")
    return value


def _require_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    return value


def _coerce_metadata(value: object, *, field_name: str) -> MetadataDict:
    return cast(MetadataDict, dict(_require_dict(value, field_name=field_name)))


@dataclass(slots=True)
class CanvasPosition:
    x: float = 120.0
    y: float = 120.0

    def to_dict(self) -> dict[str, JSONValue]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        return cls(
            x=_coerce_float(payload["x"], field_name="x"),
            y=_coerce_float(payload["y"], field_name="y"),
        )


@dataclass(slots=True)
class IndexSpec:
    id: str = field(default_factory=lambda: _new_identifier("index"))
    name: str = "index"
    dimension: int = 2
    offset: CanvasPosition = field(
        default_factory=lambda: CanvasPosition(x=0.0, y=0.0)
    )
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "name": self.name,
            "dimension": self.dimension,
            "offset": self.offset.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        offset_payload = _require_dict(
            payload.get("offset", {"x": 0.0, "y": 0.0}),
            field_name="offset",
        )
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            dimension=_coerce_int(payload["dimension"], field_name="dimension"),
            offset=CanvasPosition.from_dict(offset_payload),
            metadata=_coerce_metadata(payload.get("metadata", {}), field_name="metadata"),
        )


@dataclass(slots=True)
class TensorSpec:
    id: str = field(default_factory=lambda: _new_identifier("tensor"))
    name: str = "Tensor"
    position: CanvasPosition = field(default_factory=CanvasPosition)
    indices: list[IndexSpec] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(index.dimension for index in self.indices)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position.to_dict(),
            "indices": [index.to_dict() for index in self.indices],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        position_payload = _require_dict(payload["position"], field_name="position")
        indices_payload = _require_list(payload.get("indices", []), field_name="indices")
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            position=CanvasPosition.from_dict(position_payload),
            indices=[
                IndexSpec.from_dict(_require_dict(index, field_name="index"))
                for index in indices_payload
            ],
            metadata=_coerce_metadata(payload.get("metadata", {}), field_name="metadata"),
        )


@dataclass(slots=True, frozen=True)
class EdgeEndpointRef:
    tensor_id: str
    index_id: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {"tensor_id": self.tensor_id, "index_id": self.index_id}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        return cls(
            tensor_id=str(payload["tensor_id"]), index_id=str(payload["index_id"])
        )


@dataclass(slots=True)
class EdgeSpec:
    id: str = field(default_factory=lambda: _new_identifier("edge"))
    name: str = "edge"
    left: EdgeEndpointRef = field(default_factory=lambda: EdgeEndpointRef("", ""))
    right: EdgeEndpointRef = field(default_factory=lambda: EdgeEndpointRef("", ""))
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "name": self.name,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            left=EdgeEndpointRef.from_dict(
                _require_dict(payload["left"], field_name="left")
            ),
            right=EdgeEndpointRef.from_dict(
                _require_dict(payload["right"], field_name="right")
            ),
            metadata=_coerce_metadata(payload.get("metadata", {}), field_name="metadata"),
        )


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str


class EngineName(StrEnum):
    TENSORNETWORK = "tensornetwork"
    QUIMB = "quimb"
    TENSORKROWCH = "tensorkrowch"
    EINSUM = "einsum"


@dataclass(slots=True)
class CodegenResult:
    engine: EngineName
    code: str
    warnings: list[str] = field(default_factory=list)
    artifacts: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class EditorResult:
    spec: NetworkSpec
    engine: EngineName
    codegen: CodegenResult | None = None
    confirmed: bool = False


@dataclass(slots=True)
class NetworkSpec:
    id: str = field(default_factory=lambda: _new_identifier("network"))
    name: str = "Tensor Network"
    tensors: list[TensorSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=dict)

    def tensor_map(self) -> dict[str, TensorSpec]:
        return {tensor.id: tensor for tensor in self.tensors}

    def index_map(self) -> dict[str, tuple[TensorSpec, IndexSpec]]:
        mapping: dict[str, tuple[TensorSpec, IndexSpec]] = {}
        for tensor in self.tensors:
            for index in tensor.indices:
                mapping[index.id] = (tensor, index)
        return mapping

    def connected_index_ids(self) -> set[str]:
        connected: set[str] = set()
        for edge in self.edges:
            connected.add(edge.left.index_id)
            connected.add(edge.right.index_id)
        return connected

    def open_indices(self) -> list[tuple[TensorSpec, IndexSpec]]:
        connected = self.connected_index_ids()
        return [
            (tensor, index)
            for tensor in self.tensors
            for index in tensor.indices
            if index.id not in connected
        ]

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "name": self.name,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        tensors_payload = _require_list(payload.get("tensors", []), field_name="tensors")
        edges_payload = _require_list(payload.get("edges", []), field_name="edges")
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            tensors=[
                TensorSpec.from_dict(_require_dict(tensor, field_name="tensor"))
                for tensor in tensors_payload
            ],
            edges=[
                EdgeSpec.from_dict(_require_dict(edge, field_name="edge"))
                for edge in edges_payload
            ],
            metadata=_coerce_metadata(payload.get("metadata", {}), field_name="metadata"),
        )
