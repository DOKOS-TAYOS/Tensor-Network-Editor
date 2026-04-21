"""Compatibility re-exports for split graph model modules."""

from __future__ import annotations

from ._model_entities import (
    CanvasNoteSpec,
    EdgeEndpointRef,
    EdgeSpec,
    GroupSpec,
    HyperedgeSpec,
    IndexSpec,
    TensorSpec,
)
from ._model_network import NetworkSpec
from ._model_periodic import (
    GridPeriodicGridSpec,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    TreePeriodicTreeSpec,
)
from ._model_periodic_types import (
    GridPeriodicCellName,
    GridPeriodicTensorRole,
    LinearPeriodicCellName,
    LinearPeriodicTensorRole,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
)
from ._model_tensor_data import TensorDataMode, TensorDataSpec

__all__ = [
    "CanvasNoteSpec",
    "EdgeEndpointRef",
    "EdgeSpec",
    "GridPeriodicCellName",
    "GridPeriodicGridSpec",
    "GridPeriodicTensorRole",
    "GroupSpec",
    "HyperedgeSpec",
    "IndexSpec",
    "LinearPeriodicCellName",
    "LinearPeriodicCellSpec",
    "LinearPeriodicChainSpec",
    "LinearPeriodicTensorRole",
    "NetworkSpec",
    "TensorSpec",
    "TensorDataMode",
    "TensorDataSpec",
    "TreePeriodicCellName",
    "TreePeriodicTensorRole",
    "TreePeriodicTreeSpec",
]
