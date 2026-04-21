"""Root network model composed from graph entities and periodic layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from ...types import JSONValue, MetadataDict
from ..io._payloads import (
    coerce_metadata,
    coerce_string,
    new_identifier,
    require_dict,
    require_list,
)
from ._model_contraction import ContractionPlanSpec
from ._model_entities import (
    CanvasNoteSpec,
    EdgeSpec,
    GroupSpec,
    HyperedgeSpec,
    IndexSpec,
    TensorSpec,
)
from ._model_periodic import (
    GridPeriodicGridSpec,
    LinearPeriodicChainSpec,
    TreePeriodicTreeSpec,
)


@dataclass(slots=True)
class NetworkSpec:
    """The root object that stores an abstract tensor-network design."""

    id: str = field(default_factory=lambda: new_identifier("network"))
    name: str = "Tensor Network"
    tensors: list[TensorSpec] = field(default_factory=list)
    groups: list[GroupSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    hyperedges: list[HyperedgeSpec] = field(default_factory=list)
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
        """Return the ids of indices that participate in any connection."""
        from ..analysis._network_analysis import connected_index_ids

        return connected_index_ids(self)

    def open_indices(self) -> list[tuple[TensorSpec, IndexSpec]]:
        """Return the tensor/index pairs that are not connected at all."""
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
            "hyperedges": [hyperedge.to_dict() for hyperedge in self.hyperedges],
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
        hyperedges_payload = require_list(
            payload.get("hyperedges", []),
            field_name="hyperedges",
        )
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
            hyperedges=[
                HyperedgeSpec.from_dict(require_dict(hyperedge, field_name="hyperedge"))
                for hyperedge in hyperedges_payload
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
