from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from ._model_geometry import CanvasPosition, TensorSize
from ._payloads import (
    coerce_int,
    coerce_metadata,
    new_identifier,
    require_dict,
    require_list,
)
from .types import JSONValue, MetadataDict


@dataclass(slots=True)
class ContractionStepSpec:
    id: str = field(default_factory=lambda: new_identifier("step"))
    left_operand_id: str = ""
    right_operand_id: str = ""
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "left_operand_id": self.left_operand_id,
            "right_operand_id": self.right_operand_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        return cls(
            id=str(payload["id"]),
            left_operand_id=str(payload["left_operand_id"]),
            right_operand_id=str(payload["right_operand_id"]),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class ContractionOperandLayoutSpec:
    operand_id: str = ""
    position: CanvasPosition = field(default_factory=CanvasPosition)
    size: TensorSize = field(default_factory=TensorSize)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "operand_id": self.operand_id,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        position_payload = require_dict(payload["position"], field_name="position")
        size_payload = require_dict(payload["size"], field_name="size")
        return cls(
            operand_id=str(payload["operand_id"]),
            position=CanvasPosition.from_dict(position_payload),
            size=TensorSize.from_dict(size_payload),
        )


@dataclass(slots=True)
class ContractionViewSnapshotSpec:
    applied_step_count: int = 0
    operand_layouts: list[ContractionOperandLayoutSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "applied_step_count": self.applied_step_count,
            "operand_layouts": [
                operand_layout.to_dict() for operand_layout in self.operand_layouts
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        operand_layouts_payload = require_list(
            payload.get("operand_layouts", []),
            field_name="operand_layouts",
        )
        return cls(
            applied_step_count=coerce_int(
                payload.get("applied_step_count", 0),
                field_name="applied_step_count",
            ),
            operand_layouts=[
                ContractionOperandLayoutSpec.from_dict(
                    require_dict(
                        operand_layout, field_name="contraction_operand_layout"
                    )
                )
                for operand_layout in operand_layouts_payload
            ],
        )


@dataclass(slots=True)
class ContractionPlanSpec:
    id: str = field(default_factory=lambda: new_identifier("plan"))
    name: str = "Manual contraction path"
    steps: list[ContractionStepSpec] = field(default_factory=list)
    view_snapshots: list[ContractionViewSnapshotSpec] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "view_snapshots": [
                view_snapshot.to_dict() for view_snapshot in self.view_snapshots
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        steps_payload = require_list(payload.get("steps", []), field_name="steps")
        view_snapshots_payload = require_list(
            payload.get("view_snapshots", []),
            field_name="view_snapshots",
        )
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            steps=[
                ContractionStepSpec.from_dict(require_dict(step, field_name="step"))
                for step in steps_payload
            ],
            view_snapshots=[
                ContractionViewSnapshotSpec.from_dict(
                    require_dict(view_snapshot, field_name="contraction_view_snapshot")
                )
                for view_snapshot in view_snapshots_payload
            ],
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )
