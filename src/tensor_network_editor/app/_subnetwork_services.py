"""Subnetwork extraction helpers for editor routes."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from ..internal._logging import log_branch, log_operation, summarize_spec_counts
from ..internal.io._serialization import deserialize_spec
from ..internal.subnetworks._subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..models import CanvasPosition, NetworkSpec
from ._limits import enforce_spec_api_limits

LOGGER = logging.getLogger(__name__)


def extract_serialized_subnetwork(
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
) -> NetworkSpec:
    """Deserialize one payload and extract its selected tensor fragment."""
    with log_operation(
        LOGGER,
        "Transient subnetwork extraction",
        context={"tensor_id_count": len(tensor_ids)},
    ):
        spec = deserialize_spec(serialized_spec, validate=False)
        enforce_spec_api_limits(spec)
        extracted_spec = extract_subnetwork_spec(spec, tensor_ids=tensor_ids)
        enforce_spec_api_limits(extracted_spec)
        log_branch(
            LOGGER,
            "Extracted transient reusable subnetwork",
            context=summarize_spec_counts(extracted_spec),
        )
        return extracted_spec


def prepare_serialized_subnetwork_for_insertion(
    serialized_spec: Mapping[str, object],
    *,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Deserialize one payload and prepare it for editor insertion."""
    with log_operation(LOGGER, "Transient subnetwork insertion preparation"):
        spec = deserialize_spec(serialized_spec, validate=False)
        enforce_spec_api_limits(spec)
        prepared_spec = prepare_subnetwork_for_insertion(
            spec,
            target_center=target_center,
        )
        enforce_spec_api_limits(prepared_spec)
        log_branch(
            LOGGER,
            "Prepared transient subnetwork for insertion",
            context=summarize_spec_counts(prepared_spec),
        )
        return prepared_spec
