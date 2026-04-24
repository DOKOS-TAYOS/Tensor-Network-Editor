"""Serialize, deserialize, and persist tensor-network specifications."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ...errors import SerializationError
from ...models import NetworkSpec
from ...types import JSONValue, StrPath
from ...validation import ensure_valid_spec
from ._io import read_utf8_text, write_utf8_text
from ._payloads import coerce_int
from ._python_import_profiles import (
    PythonSourceProfile,
    detect_python_source_profile,
    normalize_python_source_profile,
    parse_python_source_by_profile,
)
from ._python_live_import import (
    PythonImportMode,
    import_live_python_source,
    normalize_python_import_mode,
)
from ._python_roundtrip import parse_generated_python_network

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
LOGGER = logging.getLogger(__name__)

PythonReconstructionLevel = Literal["auto", "simple", "best_available"]
ResolvedPythonReconstructionLevel = Literal["simple", "best_available"]

_SUPPORTED_PYTHON_RECONSTRUCTION_LEVELS = frozenset(
    {"auto", "simple", "best_available"}
)


@dataclass(slots=True, frozen=True)
class PythonSpecLoadResult:
    """A loaded network specification together with soft import warnings."""

    spec: NetworkSpec
    warnings: list[str]


def normalize_python_reconstruction_level(
    python_reconstruction_level: str,
) -> PythonReconstructionLevel:
    """Normalize and validate one requested Python reconstruction level."""
    normalized_level = python_reconstruction_level.strip().lower()
    if normalized_level not in _SUPPORTED_PYTHON_RECONSTRUCTION_LEVELS:
        raise ValueError(
            "Unsupported Python reconstruction level "
            f"{python_reconstruction_level!r}. Expected one of "
            f"{sorted(_SUPPORTED_PYTHON_RECONSTRUCTION_LEVELS)!r}."
        )
    return cast(PythonReconstructionLevel, normalized_level)


def resolve_python_reconstruction_level(
    python_reconstruction_level: PythonReconstructionLevel,
    *,
    resolved_source_profile: PythonSourceProfile,
    python_import_mode: PythonImportMode,
) -> ResolvedPythonReconstructionLevel:
    """Resolve the effective reconstruction level for one Python import."""
    if python_reconstruction_level == "auto":
        if python_import_mode == "live":
            return "simple"
        return "best_available" if resolved_source_profile == "generated" else "simple"
    if python_reconstruction_level == "best_available":
        if python_import_mode == "live" or resolved_source_profile != "generated":
            raise SerializationError(
                "Python reconstruction level 'best_available' is only supported "
                "for static imports of the 'generated' source profile. Use "
                "'simple' or 'auto' for external and live imports."
            )
        return "best_available"
    return "simple"


def serialize_spec(spec: NetworkSpec) -> dict[str, JSONValue]:
    """Return the schema-wrapped JSON payload for a specification.

    Args:
        spec: Network specification to validate and serialize.

    Returns:
        A JSON-serializable payload containing the schema wrapper and network
        body.
    """
    ensure_valid_spec(spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }


def deserialize_spec(
    payload: Mapping[str, object], *, validate: bool = True
) -> NetworkSpec:
    """Build a ``NetworkSpec`` from a schema-wrapped JSON payload.

    Args:
        payload: Schema-wrapped network payload.
        validate: Whether to run full validation on the reconstructed spec.

    Returns:
        The reconstructed network specification.

    Raises:
        SerializationError: If the payload shape, schema version, or network
            body is invalid.
    """
    if "schema_version" not in payload:
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        )
    schema_version_raw = payload.get("schema_version")
    if isinstance(schema_version_raw, bool) or not isinstance(
        schema_version_raw, (int, float, str)
    ):
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        )
    try:
        schema_version = coerce_int(
            schema_version_raw,
            field_name="schema_version",
        )
    except TypeError as exc:
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        ) from exc
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SerializationError(
            "Unsupported schema version "
            f"{schema_version}. Expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}."
        )

    network_payload = payload.get("network")
    if not isinstance(network_payload, dict):
        raise SerializationError("Serialized payload must contain a 'network' object.")

    try:
        spec = NetworkSpec.from_dict(network_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError(
            "Serialized payload contains a malformed network object."
        ) from exc
    return ensure_valid_spec(spec) if validate else spec


def save_spec(spec: NetworkSpec, path: StrPath) -> None:
    """Write a specification to disk as formatted UTF-8 JSON.

    Args:
        spec: Network specification to serialize.
        path: Destination path for the JSON file.

    Raises:
        SerializationError: If the specification cannot be serialized to JSON.
    """
    payload = serialize_spec(spec)
    try:
        body = json.dumps(payload, indent=2)
    except TypeError as exc:
        raise SerializationError(
            "Could not serialize the network specification to JSON."
        ) from exc
    write_utf8_text(path, body, description="network specification JSON")


def load_spec(
    path: StrPath,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: PythonReconstructionLevel = "auto",
    python_object_name: str | None = None,
) -> NetworkSpec:
    """Load a saved JSON spec or supported generated Python file from disk.

    Args:
        path: Path to a serialized JSON design or supported generated Python
            export.
        source_profile: Optional supported Python import profile used for ``.py``
            files. ``"auto"`` detects a supported profile from the source.

    Returns:
        The parsed network specification.

    Raises:
        SerializationError: If the file contents cannot be interpreted as a
            supported specification payload.
    """
    return load_spec_result(
        path,
        source_profile=source_profile,
        python_import_mode=python_import_mode,
        python_reconstruction_level=python_reconstruction_level,
        python_object_name=python_object_name,
    ).spec


def load_spec_result(
    path: StrPath,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: PythonReconstructionLevel = "auto",
    python_object_name: str | None = None,
) -> PythonSpecLoadResult:
    """Load one specification from disk together with soft import warnings."""
    source_path = Path(path)
    if source_path.suffix.lower() == ".py":
        body = read_utf8_text(path, description="generated Python code")
        LOGGER.debug("Loaded generated Python code payload from %s", path)
        return deserialize_spec_from_python_code_result(
            body,
            validate=True,
            source_profile=source_profile,
            python_import_mode=python_import_mode,
            python_reconstruction_level=python_reconstruction_level,
            python_object_name=python_object_name,
            source_path=source_path,
        )

    body = read_utf8_text(path, description="network specification JSON")
    LOGGER.debug("Loaded serialized network payload from %s", path)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SerializationError("Could not parse network specification JSON.") from exc
    if not isinstance(payload, dict):
        raise SerializationError("Serialized network must be a JSON object.")
    return PythonSpecLoadResult(spec=deserialize_spec(payload), warnings=[])


def deserialize_spec_from_python_code(
    code: str,
    *,
    validate: bool = True,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: PythonReconstructionLevel = "auto",
    python_object_name: str | None = None,
) -> NetworkSpec:
    """Parse supported generated Python source into a ``NetworkSpec``.

    Args:
        code: Generated Python source emitted by a supported standard network
            export.
        validate: Whether to validate the reconstructed specification.
        source_profile: Optional supported Python import profile. ``"auto"``
            detects a supported profile from the source.

    Returns:
        The reconstructed network specification.

    Raises:
        SerializationError: If the source is unsupported or cannot be parsed.
    """
    return deserialize_spec_from_python_code_result(
        code,
        validate=validate,
        source_profile=source_profile,
        python_import_mode=python_import_mode,
        python_reconstruction_level=python_reconstruction_level,
        python_object_name=python_object_name,
    ).spec


def deserialize_spec_from_python_code_result(
    code: str,
    *,
    validate: bool = True,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: PythonReconstructionLevel = "auto",
    python_object_name: str | None = None,
    source_path: Path | None = None,
) -> PythonSpecLoadResult:
    """Parse Python source into a ``NetworkSpec`` together with warnings."""
    normalized_import_mode = normalize_python_import_mode(python_import_mode)
    normalized_reconstruction_level = normalize_python_reconstruction_level(
        python_reconstruction_level
    )
    if normalized_import_mode == "live":
        resolve_python_reconstruction_level(
            normalized_reconstruction_level,
            resolved_source_profile="generated",
            python_import_mode=normalized_import_mode,
        )
        try:
            result = import_live_python_source(
                code,
                source_profile=source_profile,
                python_object_name=python_object_name,
                source_path=source_path,
            )
        except SerializationError as exc:
            try:
                fallback_result = _deserialize_static_python_code_result(
                    code,
                    validate=validate,
                    source_profile=source_profile,
                    python_reconstruction_level=normalized_reconstruction_level,
                )
            except SerializationError as fallback_exc:
                raise exc from fallback_exc
            return PythonSpecLoadResult(
                spec=fallback_result.spec,
                warnings=[
                    "Live Python import failed "
                    f"({exc}). Loaded the file with the static parser instead.",
                    *fallback_result.warnings,
                ],
            )
        spec = ensure_valid_spec(result.spec) if validate else result.spec
        return PythonSpecLoadResult(spec=spec, warnings=result.warnings)

    return _deserialize_static_python_code_result(
        code,
        validate=validate,
        source_profile=source_profile,
        python_reconstruction_level=normalized_reconstruction_level,
    )


def _deserialize_static_python_code_result(
    code: str,
    *,
    validate: bool,
    source_profile: PythonSourceProfile,
    python_reconstruction_level: PythonReconstructionLevel,
) -> PythonSpecLoadResult:
    """Parse Python source into a ``NetworkSpec`` together with warnings."""
    if "# Tensor Network Editor linear periodic mode" in code:
        raise SerializationError(
            "Loading generated Python from linear periodic mode is not supported."
        )
    if "# Tensor Network Editor grid periodic mode" in code:
        raise SerializationError(
            "Loading generated Python from bidimensional For mode is not supported."
        )
    normalized_profile = normalize_python_source_profile(source_profile)
    resolved_profile = (
        detect_python_source_profile(code)
        if normalized_profile == "auto"
        else normalized_profile
    )
    resolved_reconstruction_level = resolve_python_reconstruction_level(
        python_reconstruction_level,
        resolved_source_profile=resolved_profile,
        python_import_mode="static",
    )
    spec = (
        parse_generated_python_network(
            code,
            include_manual_plan=resolved_reconstruction_level == "best_available",
        )
        if resolved_profile == "generated"
        else parse_python_source_by_profile(code, source_profile=resolved_profile)
    )
    validated_spec = ensure_valid_spec(spec) if validate else spec
    return PythonSpecLoadResult(spec=validated_spec, warnings=[])


def load_spec_from_python_code(
    code: str,
    *,
    source_profile: PythonSourceProfile = "auto",
    python_import_mode: PythonImportMode = "static",
    python_reconstruction_level: PythonReconstructionLevel = "auto",
    python_object_name: str | None = None,
) -> NetworkSpec:
    """Parse and validate supported generated Python source.

    Args:
        code: Generated Python source emitted by a supported standard network
            export.
        source_profile: Optional supported Python import profile. ``"auto"``
            detects a supported profile from the source.

    Returns:
        The parsed and validated network specification.
    """
    return deserialize_spec_from_python_code(
        code,
        validate=True,
        source_profile=source_profile,
        python_import_mode=python_import_mode,
        python_reconstruction_level=python_reconstruction_level,
        python_object_name=python_object_name,
    )
