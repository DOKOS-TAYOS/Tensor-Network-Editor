"""Subnetwork extraction helpers for editor routes."""

from __future__ import annotations

from collections.abc import Mapping

from ..internal.io._serialization import deserialize_spec
from ..internal.subnetworks._subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..models import CanvasPosition, NetworkSpec


def extract_serialized_subnetwork(
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
) -> NetworkSpec:
    """Deserialize one payload and extract its selected tensor fragment."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return extract_subnetwork_spec(spec, tensor_ids=tensor_ids)


def prepare_serialized_subnetwork_for_insertion(
    serialized_spec: Mapping[str, object],
    *,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Deserialize one payload and prepare it for editor insertion."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return prepare_subnetwork_for_insertion(spec, target_center=target_center)
