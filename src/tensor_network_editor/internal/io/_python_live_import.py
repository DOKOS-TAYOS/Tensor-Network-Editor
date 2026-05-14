"""Live Python import support for runtime tensor-network objects."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from ...errors import SerializationError
from ...models import NetworkSpec
from ._python_import_profiles import (
    PythonSourceProfile,
    normalize_python_source_profile,
)
from ._python_import_shared import (
    ExplicitConnection,
    ImportedTensor,
    build_network_from_explicit_connections,
    build_network_from_shared_labels,
)
from ._python_import_shared import (
    default_connection_name as _default_connection_name,
)
from ._python_live_import_runtime import (
    candidate_collection_items as _candidate_collection_items,
)
from ._python_live_import_runtime import (
    coerce_iterable_items as _coerce_iterable_items,
)
from ._python_live_import_runtime import (
    coerce_label_tuple as _coerce_label_tuple,
)
from ._python_live_import_runtime import (
    coerce_optional_axis_position as _coerce_optional_axis_position,
)
from ._python_live_import_runtime import (
    coerce_shape as _coerce_shape,
)
from ._python_live_import_runtime import (
    get_attribute as _get_attribute,
)
from ._python_live_import_runtime import (
    module_name as _module_name,
)
from ._python_live_import_runtime import (
    shape_from_runtime_tensor as _shape_from_runtime_tensor,
)
from ._python_live_import_tensor_data import (
    lower_runtime_tensor_data as _lower_runtime_tensor_data,
)

PythonImportMode = Literal["static", "live"]
LivePythonSourceProfile = Literal["auto", "quimb", "tensornetwork"]
ResolvedLivePythonSourceProfile = Literal["quimb", "tensornetwork"]

_SUPPORTED_PYTHON_IMPORT_MODES = frozenset({"static", "live"})
_SUPPORTED_LIVE_SOURCE_PROFILES = frozenset({"auto", "quimb", "tensornetwork"})
_LIVE_IMPORT_TIMEOUT_SECONDS = 30.0
_SYNTHETIC_LIVE_IMPORT_FILENAME = "tensor_network_editor_live_import.py"


@dataclass(slots=True, frozen=True)
class PythonImportResult:
    """A parsed Python import together with soft warnings."""

    spec: NetworkSpec
    warnings: list[str]


@dataclass(slots=True, frozen=True)
class _LiveImportCandidate:
    """One compatible live object discovered in executed globals."""

    name: str
    backend: ResolvedLivePythonSourceProfile
    value: object


def normalize_python_import_mode(python_import_mode: str) -> PythonImportMode:
    """Normalize and validate one requested Python import mode."""
    normalized_mode = python_import_mode.strip().lower()
    if normalized_mode not in _SUPPORTED_PYTHON_IMPORT_MODES:
        raise ValueError(
            "Unsupported Python import mode "
            f"{python_import_mode!r}. Expected one of {sorted(_SUPPORTED_PYTHON_IMPORT_MODES)!r}."
        )
    return cast(PythonImportMode, normalized_mode)


def normalize_live_source_profile(
    source_profile: str,
) -> LivePythonSourceProfile:
    """Validate the source-profile subset supported by live execution."""
    normalized_profile = normalize_python_source_profile(source_profile)
    if normalized_profile not in _SUPPORTED_LIVE_SOURCE_PROFILES:
        raise SerializationError(
            "The selected source profile is not supported for live import. "
            "Use 'auto', 'quimb', or 'tensornetwork'."
        )
    return cast(LivePythonSourceProfile, normalized_profile)


def import_live_python_source(
    code: str,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_object_name: str | None = None,
    source_path: Path | None = None,
) -> PythonImportResult:
    """Execute Python source in a subprocess and import one live object."""
    normalized_profile = normalize_live_source_profile(source_profile)
    resolved_source_path = source_path.resolve() if source_path is not None else None
    working_directory = (
        resolved_source_path.parent if resolved_source_path is not None else Path.cwd()
    )
    filename = (
        str(resolved_source_path)
        if resolved_source_path is not None
        else str(working_directory / _SYNTHETIC_LIVE_IMPORT_FILENAME)
    )
    request_payload = {
        "code": code,
        "filename": filename,
        "source_profile": normalized_profile,
        "python_object_name": python_object_name,
    }
    runner_path = Path(__file__).with_name("_python_live_import_runner.py")
    try:
        completed_process = subprocess.run(  # noqa: S603, RUF100 - fixed local runner.
            [sys.executable, str(runner_path)],
            input=json.dumps(request_payload),
            capture_output=True,
            check=False,
            cwd=str(working_directory),
            text=True,
            timeout=_LIVE_IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise SerializationError(
            "Live import timed out while executing the provided Python code."
        ) from exc

    response_payload = _parse_live_import_subprocess_response(completed_process)
    network_payload = response_payload.get("network")
    warnings_payload = response_payload.get("warnings", [])
    if not isinstance(network_payload, dict):
        raise SerializationError("Live import returned an invalid network payload.")
    if not isinstance(warnings_payload, list) or any(
        not isinstance(warning, str) for warning in warnings_payload
    ):
        raise SerializationError("Live import returned invalid warning data.")
    try:
        spec = NetworkSpec.from_dict(cast(Mapping[str, object], network_payload))
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError(
            "Live import returned a malformed network payload."
        ) from exc
    return PythonImportResult(spec=spec, warnings=list(warnings_payload))


def build_live_import_result_from_namespace(
    namespace: Mapping[str, object],
    *,
    source_profile: PythonSourceProfile = "auto",
    python_object_name: str | None = None,
) -> PythonImportResult:
    """Build an import result from an executed Python namespace."""
    normalized_profile = normalize_live_source_profile(source_profile)
    resolved_object_name = (
        python_object_name.strip() if python_object_name is not None else None
    )
    candidate = _resolve_live_import_candidate(
        namespace=namespace,
        source_profile=normalized_profile,
        python_object_name=resolved_object_name or None,
    )
    if candidate.backend == "quimb":
        return _build_quimb_import_result(candidate.value)
    if candidate.backend == "tensornetwork":
        return _build_tensornetwork_import_result(candidate.value)
    raise SerializationError("Live import resolved an unsupported backend.")


def _parse_live_import_subprocess_response(
    completed_process: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    """Parse the JSON response emitted by the live-import subprocess."""
    response_payload = _decode_json_object(completed_process.stdout)
    if isinstance(response_payload, dict):
        if response_payload.get("ok") is False:
            message = response_payload.get("message")
            if isinstance(message, str) and message.strip():
                raise SerializationError(message)
            raise SerializationError("Live import failed inside the subprocess.")
        if response_payload.get("ok") is True:
            return response_payload
    if completed_process.returncode != 0:
        detail = completed_process.stderr.strip() or completed_process.stdout.strip()
        if not detail:
            detail = "Unknown subprocess error."
        raise SerializationError(
            f"Live import subprocess failed before returning a network: {detail}"
        )
    raise SerializationError("Live import returned an invalid subprocess response.")


def _decode_json_object(text: str) -> dict[str, object] | None:
    """Decode a JSON object payload when possible."""
    stripped_text = text.strip()
    if not stripped_text:
        return None
    try:
        parsed_payload = json.loads(stripped_text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_payload, dict):
        return cast(dict[str, object], parsed_payload)
    return None


def _resolve_live_import_candidate(
    *,
    namespace: Mapping[str, object],
    source_profile: LivePythonSourceProfile,
    python_object_name: str | None,
) -> _LiveImportCandidate:
    """Resolve the requested live object from one executed namespace."""
    if python_object_name is not None:
        if python_object_name not in namespace:
            raise SerializationError(
                "Live import could not find the requested global "
                f"{python_object_name!r}. Pass a valid python_object_name."
            )
        candidate = _candidate_from_value(
            name=python_object_name,
            value=namespace[python_object_name],
            source_profile=source_profile,
        )
        if candidate is None:
            raise SerializationError(
                "The requested python_object_name does not reference a supported "
                "quimb or TensorNetwork object."
            )
        return candidate

    compatible_candidates = _collect_live_import_candidates(
        namespace=namespace,
        source_profile=source_profile,
    )
    if not compatible_candidates:
        raise SerializationError(
            "Live import could not find a supported tensor network object. "
            "Assign it to a global variable or pass python_object_name."
        )
    if len(compatible_candidates) > 1:
        candidate_names = ", ".join(
            candidate.name for candidate in compatible_candidates
        )
        raise SerializationError(
            "Live import found multiple compatible globals "
            f"({candidate_names}). Pass python_object_name to choose one."
        )
    return compatible_candidates[0]


def _collect_live_import_candidates(
    *,
    namespace: Mapping[str, object],
    source_profile: LivePythonSourceProfile,
) -> list[_LiveImportCandidate]:
    """Collect all compatible live objects from one namespace."""
    candidates_by_identity: dict[int, _LiveImportCandidate] = {}
    for name, value in namespace.items():
        if _should_skip_namespace_value(name, value):
            continue
        candidate = _candidate_from_value(
            name=name,
            value=value,
            source_profile=source_profile,
        )
        if candidate is None:
            continue
        candidates_by_identity.setdefault(id(value), candidate)
    return list(candidates_by_identity.values())


def _should_skip_namespace_value(name: str, value: object) -> bool:
    """Return whether one global value should be ignored during discovery."""
    if name.startswith("__"):
        return True
    if isinstance(value, ModuleType):
        return True
    if callable(value):
        return True
    return False


def _candidate_from_value(
    *,
    name: str,
    value: object,
    source_profile: LivePythonSourceProfile,
) -> _LiveImportCandidate | None:
    """Return one backend candidate when ``value`` matches a supported shape."""
    if source_profile in {"auto", "quimb"} and _is_quimb_candidate(value):
        return _LiveImportCandidate(name=name, backend="quimb", value=value)
    if source_profile in {"auto", "tensornetwork"} and _is_tensornetwork_candidate(
        value
    ):
        return _LiveImportCandidate(name=name, backend="tensornetwork", value=value)
    return None


def _build_quimb_import_result(value: object) -> PythonImportResult:
    """Build a ``NetworkSpec`` from a live ``quimb`` object."""
    tensors = _extract_quimb_tensors(value)
    warnings: list[str] = []
    tensors_by_reference: dict[str, ImportedTensor] = {}
    tensor_order: list[str] = []
    for tensor_index, tensor in enumerate(tensors, start=1):
        reference = f"tensor_{tensor_index}"
        tensor_name = _recover_quimb_tensor_name(tensor, tensor_index)
        shape = _coerce_shape(_get_attribute(tensor, "shape"), context=tensor_name)
        index_labels = _coerce_label_tuple(
            _get_attribute(tensor, "inds"),
            expected_length=len(shape),
            fallback_prefix="index",
            allow_fallback=False,
        )
        tensor_data, warning = _lower_runtime_tensor_data(
            _get_attribute(tensor, "data"),
            shape=shape,
            tensor_name=tensor_name,
        )
        if warning is not None:
            warnings.append(warning)
        tensors_by_reference[reference] = ImportedTensor(
            reference=reference,
            name=tensor_name,
            shape=shape,
            index_labels=index_labels,
            tensor_data=tensor_data,
        )
        tensor_order.append(reference)
    return PythonImportResult(
        spec=build_network_from_shared_labels(
            tensors_by_reference=tensors_by_reference,
            tensor_order=tensor_order,
            allow_hyperedges=True,
        ),
        warnings=warnings,
    )


def _build_tensornetwork_import_result(value: object) -> PythonImportResult:
    """Build a ``NetworkSpec`` from a live TensorNetwork object."""
    nodes = _extract_tensornetwork_nodes(value)
    warnings: list[str] = []
    tensors_by_reference: dict[str, ImportedTensor] = {}
    tensor_order: list[str] = []
    reference_by_node_identity: dict[int, str] = {}
    axis_names_by_reference: dict[str, tuple[str, ...]] = {}
    for node_index, node in enumerate(nodes, start=1):
        reference = f"node_{node_index}"
        tensor_order.append(reference)
        reference_by_node_identity[id(node)] = reference
        tensor_name = _recover_tensornetwork_node_name(node, node_index)
        shape_source = _get_attribute(node, "shape")
        if shape_source is None:
            shape_source = _shape_from_runtime_tensor(_get_attribute(node, "tensor"))
        shape = _coerce_shape(shape_source, context=tensor_name)
        axis_names = _coerce_label_tuple(
            _get_attribute(node, "axis_names"),
            expected_length=len(shape),
            fallback_prefix="axis",
            allow_fallback=True,
        )
        axis_names_by_reference[reference] = axis_names
        tensor_data, warning = _lower_runtime_tensor_data(
            _get_attribute(node, "tensor"),
            shape=shape,
            tensor_name=tensor_name,
        )
        if warning is not None:
            warnings.append(warning)
        tensors_by_reference[reference] = ImportedTensor(
            reference=reference,
            name=tensor_name,
            shape=shape,
            index_labels=axis_names,
            tensor_data=tensor_data,
        )
    explicit_connections = _collect_tensornetwork_connections(
        nodes=nodes,
        reference_by_node_identity=reference_by_node_identity,
        axis_names_by_reference=axis_names_by_reference,
    )
    return PythonImportResult(
        spec=build_network_from_explicit_connections(
            tensors_by_reference=tensors_by_reference,
            tensor_order=tensor_order,
            explicit_connections=explicit_connections,
        ),
        warnings=warnings,
    )


def _collect_tensornetwork_connections(
    *,
    nodes: list[object],
    reference_by_node_identity: dict[int, str],
    axis_names_by_reference: dict[str, tuple[str, ...]],
) -> list[ExplicitConnection]:
    """Collect explicit binary connections from runtime TensorNetwork edges."""
    explicit_connections: list[ExplicitConnection] = []
    seen_edge_ids: set[int] = set()
    for node in nodes:
        for edge in _coerce_iterable_items(_get_attribute(node, "edges")):
            if edge is None:
                continue
            edge_identity = id(edge)
            if edge_identity in seen_edge_ids:
                continue
            seen_edge_ids.add(edge_identity)
            parsed_connection = _parse_tensornetwork_edge(
                edge=edge,
                reference_by_node_identity=reference_by_node_identity,
                axis_names_by_reference=axis_names_by_reference,
            )
            if parsed_connection is None:
                continue
            explicit_connections.append(parsed_connection)
    return explicit_connections


def _parse_tensornetwork_edge(
    *,
    edge: object,
    reference_by_node_identity: dict[int, str],
    axis_names_by_reference: dict[str, tuple[str, ...]],
) -> ExplicitConnection | None:
    """Convert one runtime TensorNetwork edge into an explicit connection."""
    left_node = _get_attribute(edge, "node1")
    right_node = _get_attribute(edge, "node2")
    left_axis = _coerce_optional_axis_position(_get_attribute(edge, "axis1"))
    right_axis = _coerce_optional_axis_position(_get_attribute(edge, "axis2"))
    if (
        left_node is None
        or right_node is None
        or left_axis is None
        or right_axis is None
    ):
        return None
    left_reference = reference_by_node_identity.get(id(left_node))
    right_reference = reference_by_node_identity.get(id(right_node))
    if left_reference is None or right_reference is None:
        return None
    left_axis_names = axis_names_by_reference[left_reference]
    right_axis_names = axis_names_by_reference[right_reference]
    if left_axis >= len(left_axis_names) or right_axis >= len(right_axis_names):
        raise SerializationError(
            "The live TensorNetwork importer found an edge pointing to an unknown axis."
        )
    left_index_name = left_axis_names[left_axis]
    right_index_name = right_axis_names[right_axis]
    return ExplicitConnection(
        name=_recover_tensornetwork_edge_name(
            edge,
            left_index_name=left_index_name,
            right_index_name=right_index_name,
        ),
        left_reference=left_reference,
        left_index_name=left_index_name,
        right_reference=right_reference,
        right_index_name=right_index_name,
    )


def _recover_tensornetwork_edge_name(
    edge: object,
    *,
    left_index_name: str,
    right_index_name: str,
) -> str:
    """Return the runtime edge name or one readable fallback."""
    raw_name = _get_attribute(edge, "name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    return _default_connection_name(left_index_name, right_index_name)


def _recover_quimb_tensor_name(tensor: object, tensor_index: int) -> str:
    """Recover one readable live tensor name from runtime ``quimb`` tags."""
    tags = _coerce_iterable_items(_get_attribute(tensor, "tags"))
    tag_names = sorted(tag for tag in tags if isinstance(tag, str) and tag.strip())
    if tag_names:
        return tag_names[0]
    return f"Tensor {tensor_index}"


def _recover_tensornetwork_node_name(node: object, node_index: int) -> str:
    """Recover one readable live TensorNetwork node name."""
    raw_name = _get_attribute(node, "name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    return f"Node {node_index}"


def _extract_quimb_tensors(value: object) -> list[object]:
    """Extract live quimb tensors from one compatible runtime object."""
    if _is_quimb_tensor(value):
        return [value]
    if _has_quimb_tensor_collection(value):
        return _quimb_tensor_collection_items(value)
    raise SerializationError("Live import expected a supported quimb object.")


def _extract_tensornetwork_nodes(value: object) -> list[object]:
    """Extract live TensorNetwork nodes from one compatible runtime object."""
    if _is_tensornetwork_node(value):
        return [value]
    if _has_tensornetwork_node_collection(value):
        return _tensornetwork_node_collection_items(value)
    raise SerializationError("Live import expected a supported TensorNetwork object.")


def _is_quimb_candidate(value: object) -> bool:
    """Return whether ``value`` matches one supported live quimb shape."""
    return _is_quimb_tensor(value) or _has_quimb_tensor_collection(value)


def _is_tensornetwork_candidate(value: object) -> bool:
    """Return whether ``value`` matches one supported live TensorNetwork shape."""
    return _is_tensornetwork_node(value) or _has_tensornetwork_node_collection(value)


def _is_quimb_tensor(value: object) -> bool:
    """Return whether ``value`` looks like one live ``quimb`` tensor."""
    module_name = _module_name(value)
    return (
        module_name.startswith("quimb")
        and _get_attribute(value, "inds") is not None
        and _get_attribute(value, "shape") is not None
        and _get_attribute(value, "data") is not None
    )


def _is_tensornetwork_node(value: object) -> bool:
    """Return whether ``value`` looks like one live TensorNetwork node."""
    module_name = _module_name(value)
    return (
        module_name.startswith("tensornetwork")
        and _get_attribute(value, "tensor") is not None
        and _get_attribute(value, "edges") is not None
        and (
            _get_attribute(value, "shape") is not None
            or _shape_from_runtime_tensor(_get_attribute(value, "tensor")) is not None
        )
    )


def _has_quimb_tensor_collection(value: object) -> bool:
    """Return whether ``value`` exposes a supported collection of quimb tensors."""
    items = _quimb_tensor_collection_items(value)
    if items:
        return True
    return (
        _module_name(value).startswith("quimb")
        and _get_attribute(value, "tensors") is not None
    )


def _has_tensornetwork_node_collection(value: object) -> bool:
    """Return whether ``value`` exposes a supported collection of TensorNetwork nodes."""
    items = _tensornetwork_node_collection_items(value)
    if items:
        return True
    return (
        _module_name(value).startswith("tensornetwork")
        and _get_attribute(value, "nodes") is not None
    )


def _quimb_tensor_collection_items(value: object) -> list[object]:
    """Return the quimb tensors contained in one supported runtime object."""
    for candidate_items in _candidate_collection_items(value, "tensors"):
        if all(_is_quimb_tensor(item) for item in candidate_items):
            return candidate_items
    return []


def _tensornetwork_node_collection_items(value: object) -> list[object]:
    """Return the TensorNetwork nodes contained in one supported runtime object."""
    for candidate_items in _candidate_collection_items(value, "nodes"):
        if all(_is_tensornetwork_node(item) for item in candidate_items):
            return candidate_items
    return []
