"""Service-layer facade shared by the local editor HTTP routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..internal.analysis._contraction_analysis_types import ContractionAnalysisResult
from ..internal.io._serialization import deserialize_spec
from ..validation import validate_spec
from ._analysis_services import (
    analyze_serialized_contraction as analyze_serialized_contraction_internal,
)
from ._bootstrap_payloads import (
    build_app_metadata_payload,
    build_bootstrap_payload,
    build_template_catalog_payload,
)
from ._protocol import JsonDict
from ._session_requests import complete_session_request, generate_session_request
from ._subnetwork_services import (
    extract_serialized_subnetwork,
    prepare_serialized_subnetwork_for_insertion,
)
from ._template_services import (
    build_template_from_payload,
    delete_session_project_template,
    promote_serialized_subnetwork_to_template,
    rename_session_project_template,
)

if TYPE_CHECKING:
    from .session import EditorSession


__all__ = [
    "EditorSession",
    "JsonDict",
    "analyze_serialized_contraction",
    "build_app_metadata_payload",
    "build_bootstrap_payload",
    "build_template_catalog_payload",
    "build_template_from_payload",
    "complete_session_request",
    "delete_session_project_template",
    "extract_serialized_subnetwork",
    "generate_session_request",
    "prepare_serialized_subnetwork_for_insertion",
    "promote_serialized_subnetwork_to_template",
    "rename_session_project_template",
]


def analyze_serialized_contraction(
    serialized_spec: Mapping[str, object],
) -> ContractionAnalysisResult:
    """Deserialize, validate, and analyze contraction data for one payload."""
    return analyze_serialized_contraction_internal(
        serialized_spec,
        deserialize_spec_fn=lambda payload: deserialize_spec(payload, validate=False),
        validate_spec_fn=validate_spec,
    )
