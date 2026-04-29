"""HTTP route handlers for the local editor server."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Literal, cast

from ..errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
    SpecValidationError,
)
from ..internal._logging import (
    FRONTEND_LOGGER_NAME,
    format_log_message,
    log_branch,
    log_operation,
    summarize_contraction_analysis,
    summarize_spec_counts,
)
from ..internal.analysis._contraction_analysis_types import ContractionAnalysisResult
from ..io import deserialize_spec, serialize_spec
from ..models import CodegenResult, EditorResult, NetworkSpec
from ..rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    render_spec_dot,
    render_spec_mermaid,
    render_spec_pdf,
    render_spec_png,
    render_spec_svg,
    render_spec_tikz,
)
from ..types import JSONValue
from ..validation import validate_spec
from ._drafts import clear_project_draft, load_project_draft, save_project_draft
from ._protocol import (
    JsonDict,
    JsonResponse,
    bad_request_response,
    deserialize_validation_request,
    issues_response,
    ok_response,
    parse_codegen_request,
    parse_subnetwork_library_delete_request,
    parse_subnetwork_library_prepare_insert_request,
    parse_subnetwork_library_rename_request,
    parse_subnetwork_library_save_request,
    parse_subnetwork_prepare_insert_request,
    parse_subnetwork_selection_request,
    parse_template_delete_request,
    parse_template_promote_request,
    parse_template_rename_request,
    require_boolean,
    require_non_empty_string,
    require_serialized_spec,
    serialize_codegen_result,
    serialize_editor_result,
    serialize_spec_payload,
)
from ._services import (
    analyze_serialized_contraction,
    build_bootstrap_payload,
    build_template_from_payload,
    delete_session_project_subnetwork,
    delete_session_project_template,
    extract_serialized_subnetwork,
    prepare_saved_subnetwork_for_insertion,
    prepare_serialized_subnetwork_for_insertion,
    promote_serialized_subnetwork_to_template,
    rename_session_project_subnetwork,
    rename_session_project_template,
    save_serialized_subnetwork_to_library,
)
from .session import EditorSession

LOGGER = logging.getLogger(__name__)
FRONTEND_LOGGER = logging.getLogger(FRONTEND_LOGGER_NAME)
_FRONTEND_CLIENT_LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_MAX_FRONTEND_CLIENT_LOG_EVENTS = 200
_MAX_FRONTEND_CLIENT_LOG_MESSAGE_LENGTH = 400
_MAX_FRONTEND_CLIENT_LOG_CONTEXT_VALUE_LENGTH = 200


@dataclass(slots=True, frozen=True)
class _FrontendClientLogEvent:
    """Validated frontend log event ready for persistence."""

    level: str
    message: str
    context: dict[str, object]


def _route_context(
    session: EditorSession | None,
    route: str,
    **extra: object,
) -> dict[str, object]:
    """Build the shared logging context for one editor route."""
    context: dict[str, object] = {"route": route}
    if session is not None:
        context["session"] = session.session_id
    context.update(extra)
    return context


def handle_bootstrap(session: EditorSession) -> JsonResponse:
    """Return the bootstrap payload used by the browser client."""
    with log_operation(
        LOGGER,
        "Bootstrap route",
        context=_route_context(session, "/api/bootstrap"),
    ) as success_context:
        payload = build_bootstrap_payload(session)
        success_context["template_count"] = len(
            cast(dict[str, object], payload["template_definitions"])
        )
        success_context["warning_count"] = len(
            cast(list[object], payload["template_catalog_warnings"])
        ) + len(cast(list[object], payload["subnetwork_catalog_warnings"]))
        return HTTPStatus.OK, payload


def handle_draft_load(session: EditorSession) -> JsonResponse:
    """Return the active project draft when one has been saved."""
    with log_operation(
        LOGGER,
        "Draft load route",
        context=_route_context(session, "/api/draft", path=session.draft_path),
    ) as success_context:
        try:
            draft = load_project_draft(session.draft_path)
        except ValueError as exc:
            log_branch(
                LOGGER,
                f"Draft load route rejected payload: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        success_context["status"] = "loaded" if draft is not None else "missing"
        return ok_response({"draft": cast(JSONValue, draft)})


def handle_draft_save(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Persist the active browser editor draft for later recovery."""
    with log_operation(
        LOGGER,
        "Draft save route",
        context=_route_context(session, "/api/draft", path=session.draft_path),
    ) as success_context:
        try:
            request = parse_codegen_request(
                payload,
                default_engine=session.default_engine,
                default_collection_format=session.default_collection_format,
            )
            draft = save_project_draft(
                session.draft_path,
                serialized_spec=request.serialized_spec,
                engine=request.engine,
                collection_format=request.collection_format,
            )
        except ValueError as exc:
            log_branch(
                LOGGER,
                f"Draft save route rejected payload: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        success_context["status"] = "saved"
        success_context["engine"] = request.engine
        success_context["format"] = request.collection_format
        return ok_response({"draft": cast(JSONValue, draft)})


def handle_draft_clear(session: EditorSession) -> JsonResponse:
    """Clear the active project draft after an explicit user action."""
    with log_operation(
        LOGGER,
        "Draft clear route",
        context=_route_context(session, "/api/draft/clear", path=session.draft_path),
    ):
        clear_project_draft(session.draft_path)
        return ok_response()


def handle_validate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Validate a serialized spec or supported Python source payload."""
    session_id = session.session_id
    with log_operation(
        LOGGER,
        "Validate route",
        context={"session": session_id, "route": "/api/validate"},
    ):
        try:
            validation_request = deserialize_validation_request(payload)
            spec = validation_request.spec
        except SerializationError as exc:
            log_branch(
                LOGGER,
                f"Validation request contained malformed spec payload: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        except SpecValidationError as exc:
            return issues_response(exc.issues)
        except ValueError:
            log_branch(
                LOGGER,
                "Validation request missing 'spec' or 'python_code' payload.",
                level=logging.WARNING,
            )
            return bad_request_response("Missing 'spec' or 'python_code' payload.")
        issues = validate_spec(spec)
        if issues:
            log_branch(
                LOGGER,
                "Validation completed with issues",
                context={"status": len(issues), **summarize_spec_counts(spec)},
            )
            status, response = issues_response(issues)
            response["spec"] = serialize_spec_payload(spec)
            if validation_request.warnings:
                response["warnings"] = cast(JSONValue, validation_request.warnings)
            return status, response
        log_branch(
            LOGGER,
            "Validation completed without issues",
            context=summarize_spec_counts(spec),
        )
        response_payload: JsonDict = {
            "issues": [],
            "spec": serialize_spec_payload(spec),
        }
        if validation_request.warnings:
            response_payload["warnings"] = cast(JSONValue, validation_request.warnings)
        return ok_response(response_payload)


def handle_generate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Generate preview code for the current editor payload."""
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="generate",
    )
    message = response.get("message")
    if response.get("ok") is False and isinstance(message, str):
        log_branch(
            LOGGER,
            f"Generate route rejected request: {message}",
            level=logging.WARNING,
            context=_route_context(session, "/api/generate"),
        )
    return status, response


def handle_complete(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Finalize an editor session and return the completion payload."""
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="complete",
    )
    message = response.get("message")
    if response.get("ok") is False and isinstance(message, str):
        log_branch(
            LOGGER,
            f"Complete route rejected request: {message}",
            level=logging.WARNING,
            context=_route_context(session, "/api/complete"),
        )
    return status, response


def handle_cancel(session: EditorSession) -> JsonResponse:
    """Cancel the current editor session."""
    with log_operation(
        LOGGER,
        "Cancel route",
        context=_route_context(session, "/api/cancel"),
    ):
        session.cancel()
        return ok_response()


def handle_client_log(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Persist one batch of frontend log events for the current editor session."""
    with log_operation(
        LOGGER,
        "Frontend client log route",
        context=_route_context(session, "/api/client-log"),
    ) as success_context:
        try:
            events = _parse_frontend_client_log_events(
                payload,
                session_id=session.session_id,
            )
        except ValueError as exc:
            log_branch(
                LOGGER,
                f"Frontend client log route rejected payload: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        for event in events:
            FRONTEND_LOGGER.log(
                _FRONTEND_CLIENT_LOG_LEVELS[event.level],
                format_log_message(event.message, context=event.context),
            )
        success_context["event_count"] = len(events)
        return ok_response()


def handle_render(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Render the current editor payload to an academic text format."""
    del session
    with log_operation(
        LOGGER, "Render route", context={"route": "/api/render"}
    ) as success_context:
        try:
            render_format = _resolve_render_format(payload)
            serialized_spec = require_serialized_spec(payload)
            spec = deserialize_spec(serialized_spec, validate=False)
            success_context["format"] = render_format
            success_context.update(summarize_spec_counts(spec))
            svg_options = SvgRenderOptions(
                show_tensor_labels=require_boolean(
                    payload, "show_tensor_names", default=True
                ),
                show_index_labels=require_boolean(
                    payload, "show_index_names", default=True
                ),
                show_edge_labels=require_boolean(
                    payload, "show_bond_names", default=True
                ),
            )
            if render_format == "tikz":
                text = render_spec_tikz(
                    spec,
                    options=TikzRenderOptions(
                        show_tensor_labels=require_boolean(
                            payload, "show_tensor_names", default=True
                        ),
                        show_index_labels=require_boolean(
                            payload, "show_index_names", default=True
                        ),
                        show_edge_labels=require_boolean(
                            payload, "show_bond_names", default=True
                        ),
                    ),
                )
                content_type = "text/x-tex;charset=utf-8"
                response_payload: JsonDict = {
                    "format": render_format,
                    "text": text,
                    "content_type": content_type,
                }
            elif render_format == "dot":
                text = render_spec_dot(
                    spec,
                    options=DotRenderOptions(
                        show_tensor_labels=require_boolean(
                            payload, "show_tensor_names", default=True
                        ),
                        show_index_labels=require_boolean(
                            payload, "show_index_names", default=True
                        ),
                        show_edge_labels=require_boolean(
                            payload, "show_bond_names", default=True
                        ),
                    ),
                )
                content_type = "text/vnd.graphviz;charset=utf-8"
                response_payload = {
                    "format": render_format,
                    "text": text,
                    "content_type": content_type,
                }
            elif render_format == "mermaid":
                text = render_spec_mermaid(
                    spec,
                    options=DotRenderOptions(
                        show_tensor_labels=require_boolean(
                            payload, "show_tensor_names", default=True
                        ),
                        show_index_labels=require_boolean(
                            payload, "show_index_names", default=True
                        ),
                        show_edge_labels=require_boolean(
                            payload, "show_bond_names", default=True
                        ),
                    ),
                )
                content_type = "text/plain;charset=utf-8"
                response_payload = {
                    "format": render_format,
                    "text": text,
                    "content_type": content_type,
                }
            elif render_format == "svg":
                text = render_spec_svg(spec, options=svg_options)
                response_payload = {
                    "format": render_format,
                    "text": text,
                    "content_type": "image/svg+xml;charset=utf-8",
                }
            elif render_format == "png":
                binary = render_spec_png(spec, options=svg_options)
                response_payload = {
                    "format": render_format,
                    "base64": base64.b64encode(binary).decode("ascii"),
                    "content_type": "image/png",
                }
            else:
                binary = render_spec_pdf(spec, options=svg_options)
                response_payload = {
                    "format": render_format,
                    "base64": base64.b64encode(binary).decode("ascii"),
                    "content_type": "application/pdf",
                }
        except ValueError as exc:
            return bad_request_response(str(exc))
        except SerializationError as exc:
            return bad_request_response(str(exc))
        except SpecValidationError as exc:
            return issues_response(exc.issues)
        return ok_response(response_payload)


def handle_template(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Build a template spec from the requested template payload."""
    try:
        template_name = require_non_empty_string(payload, "template")
    except ValueError as exc:
        return bad_request_response(str(exc))
    parameters = payload.get("parameters")
    return _handle_spec_response(
        lambda: build_template_from_payload(
            session,
            template_name,
            parameters,
        ),
        handled_exceptions=(ValueError,),
        operation_name="Template route",
        context=_route_context(
            session,
            "/api/template",
            template_name=template_name,
        ),
    )


def handle_template_promote(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Promote a selected subnetwork fragment to the project template catalog."""
    return _handle_catalog_response(
        lambda: _build_promoted_template_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, SerializationError, TypeError, ValueError),
        operation_name="Template promote route",
        context=_route_context(session, "/api/template/promote"),
    )


def handle_template_rename(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Rename one project-local template entry."""
    return _handle_catalog_response(
        lambda: _build_renamed_template_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, ValueError),
        operation_name="Template rename route",
        context=_route_context(session, "/api/template/rename"),
    )


def handle_template_delete(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Delete one project-local template entry."""
    return _handle_catalog_response(
        lambda: _build_deleted_template_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, ValueError),
        operation_name="Template delete route",
        context=_route_context(session, "/api/template/delete"),
    )


def handle_analyze_contraction(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Analyze contraction information for a validated serialized spec."""
    session_id = session.session_id
    with log_operation(
        LOGGER,
        "Analyze contraction route",
        context={"session": session_id, "route": "/api/analyze-contraction"},
    ) as success_context:
        try:
            serialized_spec = require_serialized_spec(payload)
        except ValueError:
            log_branch(
                LOGGER,
                "Contraction analysis request missing 'spec' payload.",
                level=logging.WARNING,
            )
            return bad_request_response("Missing 'spec' payload.")

        try:
            result = analyze_serialized_contraction(serialized_spec)
        except SerializationError as exc:
            log_branch(
                LOGGER,
                f"Contraction analysis request contained malformed spec: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        except SpecValidationError as exc:
            log_branch(
                LOGGER,
                "Contraction analysis completed with validation issues",
                level=logging.WARNING,
                context={
                    "analysis_status": "issues",
                    "issue_count": len(exc.issues),
                },
            )
            return issues_response(exc.issues)

        success_context.update(
            {
                "memory_dtype": result.memory_dtype,
                **summarize_contraction_analysis(result),
            }
        )
        return ok_response(_serialize_contraction_analysis_result(result))


def handle_subnetwork_extract(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Extract a reusable subnetwork fragment from the current graph."""
    return _handle_spec_response(
        lambda: _extract_subnetwork_spec(payload),
        handled_exceptions=(SerializationError, TypeError, ValueError),
        operation_name="Subnetwork extract route",
        context=_route_context(session, "/api/subnetwork/extract"),
    )


def handle_subnetwork_prepare_insert(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Prepare one saved fragment for insertion into the current design."""
    return _handle_spec_response(
        lambda: _prepare_subnetwork_insert_spec(payload),
        handled_exceptions=(SerializationError, TypeError, ValueError),
        operation_name="Subnetwork prepare-insert route",
        context=_route_context(session, "/api/subnetwork/prepare-insert"),
    )


def handle_subnetwork_library_save(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Save one selected fragment into the reusable-subnetwork catalog."""
    return _handle_catalog_response(
        lambda: _build_saved_subnetwork_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, SerializationError, TypeError, ValueError),
        operation_name="Subnetwork library save route",
        context=_route_context(session, "/api/subnetwork-library/save"),
    )


def handle_subnetwork_library_rename(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Rename one project-local reusable-subnetwork catalog entry."""
    return _handle_catalog_response(
        lambda: _build_renamed_subnetwork_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, ValueError),
        operation_name="Subnetwork library rename route",
        context=_route_context(session, "/api/subnetwork-library/rename"),
    )


def handle_subnetwork_library_delete(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Delete one project-local reusable-subnetwork catalog entry."""
    return _handle_catalog_response(
        lambda: _build_deleted_subnetwork_catalog_payload(session, payload),
        handled_exceptions=(PackageIOError, ValueError),
        operation_name="Subnetwork library delete route",
        context=_route_context(session, "/api/subnetwork-library/delete"),
    )


def handle_subnetwork_library_prepare_insert(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Prepare one saved reusable subnetwork for insertion into the graph."""
    return _handle_spec_response(
        lambda: _prepare_saved_subnetwork_insert_spec(session, payload),
        handled_exceptions=(SerializationError, TypeError, ValueError),
        operation_name="Subnetwork library prepare-insert route",
        context=_route_context(session, "/api/subnetwork-library/prepare-insert"),
    )


def _handle_spec_response(
    build_spec: Callable[[], NetworkSpec],
    *,
    handled_exceptions: tuple[type[Exception], ...],
    operation_name: str,
    context: dict[str, object] | None = None,
) -> JsonResponse:
    """Return a serialized-spec response for one route callback."""
    with log_operation(
        LOGGER, operation_name, context=context or {}
    ) as success_context:
        try:
            spec = build_spec()
        except handled_exceptions as exc:
            log_branch(
                LOGGER,
                f"{operation_name} rejected request: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        success_context.update(summarize_spec_counts(spec))
        return ok_response({"spec": serialize_spec(spec)})


def _handle_catalog_response(
    build_catalog_payload: Callable[[], JsonDict],
    *,
    handled_exceptions: tuple[type[Exception], ...],
    operation_name: str,
    context: dict[str, object] | None = None,
) -> JsonResponse:
    """Return a catalog payload response for one route callback."""
    with log_operation(
        LOGGER, operation_name, context=context or {}
    ) as success_context:
        try:
            catalog_payload = build_catalog_payload()
        except handled_exceptions as exc:
            log_branch(
                LOGGER,
                f"{operation_name} rejected request: {exc}",
                level=logging.WARNING,
            )
            return bad_request_response(str(exc))
        success_context["selected_template"] = catalog_payload.get("selected_template")
        success_context["selected_subnetwork"] = catalog_payload.get(
            "selected_subnetwork"
        )
        return ok_response(catalog_payload)


def _build_promoted_template_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the promoted project-template catalog payload for one request."""
    request = parse_template_promote_request(payload)
    return promote_serialized_subnetwork_to_template(
        session,
        request.serialized_spec,
        tensor_ids=request.tensor_ids,
        template_name=request.template_name,
        overwrite=request.overwrite,
    )


def _build_renamed_template_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the renamed project-template catalog payload for one request."""
    request = parse_template_rename_request(payload)
    return rename_session_project_template(
        session,
        template_name=request.template_name,
        new_template_name=request.new_template_name,
        overwrite=request.overwrite,
    )


def _build_deleted_template_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the deleted project-template catalog payload for one request."""
    request = parse_template_delete_request(payload)
    return delete_session_project_template(
        session,
        template_name=request.template_name,
    )


def _extract_subnetwork_spec(payload: JsonDict) -> NetworkSpec:
    """Return the extracted reusable subnetwork spec for one request."""
    request = parse_subnetwork_selection_request(payload)
    return extract_serialized_subnetwork(
        request.serialized_spec,
        tensor_ids=request.tensor_ids,
    )


def _prepare_subnetwork_insert_spec(payload: JsonDict) -> NetworkSpec:
    """Return the prepared transient subnetwork insertion spec for one request."""
    request = parse_subnetwork_prepare_insert_request(payload)
    return prepare_serialized_subnetwork_for_insertion(
        request.serialized_spec,
        target_center=request.target_center,
    )


def _build_saved_subnetwork_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the saved reusable-subnetwork catalog payload for one request."""
    request = parse_subnetwork_library_save_request(payload)
    return save_serialized_subnetwork_to_library(
        session,
        request.serialized_spec,
        tensor_ids=request.tensor_ids,
        subnetwork_name=request.subnetwork_name,
        tags=request.tags,
        overwrite=request.overwrite,
    )


def _build_renamed_subnetwork_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the renamed reusable-subnetwork catalog payload for one request."""
    request = parse_subnetwork_library_rename_request(payload)
    return rename_session_project_subnetwork(
        session,
        subnetwork_name=request.subnetwork_name,
        new_subnetwork_name=request.new_subnetwork_name,
        overwrite=request.overwrite,
    )


def _build_deleted_subnetwork_catalog_payload(
    session: EditorSession,
    payload: JsonDict,
) -> JsonDict:
    """Return the deleted reusable-subnetwork catalog payload for one request."""
    request = parse_subnetwork_library_delete_request(payload)
    return delete_session_project_subnetwork(
        session,
        subnetwork_name=request.subnetwork_name,
    )


def _prepare_saved_subnetwork_insert_spec(
    session: EditorSession,
    payload: JsonDict,
) -> NetworkSpec:
    """Return the prepared saved-subnetwork insertion spec for one request."""
    request = parse_subnetwork_library_prepare_insert_request(payload)
    return prepare_saved_subnetwork_for_insertion(
        session,
        subnetwork_name=request.subnetwork_name,
        target_center=request.target_center,
    )


def _serialize_generate_result(result: CodegenResult) -> JsonDict:
    """Serialize a generate-route code generation result."""
    return serialize_codegen_result(result)


def _resolve_render_format(
    payload: JsonDict,
) -> Literal["tikz", "dot", "mermaid", "svg", "png", "pdf"]:
    raw_format = payload.get("format")
    if not isinstance(raw_format, str) or not raw_format.strip():
        raise ValueError("Missing 'format' payload.")
    normalized_format = raw_format.strip().lower()
    if normalized_format in {"tikz", "dot", "mermaid", "svg", "png", "pdf"}:
        return cast(
            Literal["tikz", "dot", "mermaid", "svg", "png", "pdf"],
            normalized_format,
        )
    raise ValueError(
        "Unsupported render format "
        f"'{raw_format}'. Expected 'tikz', 'dot', 'mermaid', 'svg', 'png', or 'pdf'."
    )


def _serialize_complete_result(result: EditorResult) -> JsonDict:
    """Serialize a complete-route editor result."""
    return serialize_editor_result(result)


def _handle_session_codegen_request(
    *,
    session: EditorSession,
    payload: JsonDict,
    operation: Literal["generate", "complete"],
) -> JsonResponse:
    """Handle shared generate and complete route behavior."""
    try:
        request = parse_codegen_request(
            payload,
            default_engine=session.default_engine,
            default_collection_format=session.default_collection_format,
        )
    except ValueError as exc:
        return bad_request_response(str(exc))

    try:
        if operation == "generate":
            generate_result = session.generate(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_generate_result(generate_result))
        if operation == "complete":
            complete_result = session.complete(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_complete_result(complete_result))
        raise ValueError(f"Unsupported code generation operation '{operation}'.")
    except SerializationError as exc:
        return bad_request_response(str(exc))
    except CodeGenerationError as exc:
        return bad_request_response(str(exc))
    except PackageIOError as exc:
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)


def _serialize_contraction_analysis_result(
    result: ContractionAnalysisResult,
) -> JsonDict:
    """Serialize a contraction analysis result for the API."""
    return result.to_dict()


def _parse_frontend_client_log_events(
    payload: JsonDict,
    *,
    session_id: str,
) -> list[_FrontendClientLogEvent]:
    """Validate and normalize one frontend log batch payload."""
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Missing 'events' payload.")
    if len(raw_events) > _MAX_FRONTEND_CLIENT_LOG_EVENTS:
        raise ValueError("Frontend log batch exceeds the maximum event count.")

    events: list[_FrontendClientLogEvent] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"Frontend log event {index} must be an object.")
        raw_level = raw_event.get("level")
        if not isinstance(raw_level, str) or raw_level.strip().lower() not in (
            _FRONTEND_CLIENT_LOG_LEVELS
        ):
            raise ValueError(
                f"Frontend log event {index} has an unsupported 'level' value."
            )
        raw_message = raw_event.get("message")
        if not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError(f"Frontend log event {index} is missing a 'message'.")
        raw_context = raw_event.get("context", {})
        if not isinstance(raw_context, dict):
            raise ValueError(f"Frontend log event {index} has an invalid 'context'.")
        context: dict[str, object] = {"session": session_id}
        for key, value in raw_context.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"Frontend log event {index} contains an invalid context key."
                )
            context[key.strip()] = _truncate_frontend_client_log_value(value)
        events.append(
            _FrontendClientLogEvent(
                level=raw_level.strip().lower(),
                message=_truncate_frontend_client_log_message(raw_message),
                context=context,
            )
        )
    return events


def _truncate_frontend_client_log_message(message: str) -> str:
    """Clamp one frontend log message to a readable persistence length."""
    stripped_message = message.strip()
    if len(stripped_message) <= _MAX_FRONTEND_CLIENT_LOG_MESSAGE_LENGTH:
        return stripped_message
    return f"{stripped_message[: _MAX_FRONTEND_CLIENT_LOG_MESSAGE_LENGTH - 3]}..."


def _truncate_frontend_client_log_value(value: object) -> str:
    """Clamp one frontend log context value before persistence."""
    value_text = str(value)
    if len(value_text) <= _MAX_FRONTEND_CLIENT_LOG_CONTEXT_VALUE_LENGTH:
        return value_text
    return f"{value_text[: _MAX_FRONTEND_CLIENT_LOG_CONTEXT_VALUE_LENGTH - 3]}..."
