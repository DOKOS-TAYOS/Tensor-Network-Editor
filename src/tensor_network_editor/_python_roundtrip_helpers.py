from __future__ import annotations

import re

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9a-zA-Z_]+")


def sanitize_identifier(value: str) -> str:
    return _NON_IDENTIFIER_PATTERN.sub("_", value.strip()).strip("_").lower()


def synthetic_data_variable_name(reference: str, fallback_name: str | None) -> str:
    candidate = sanitize_identifier(fallback_name or reference)
    if not candidate:
        candidate = "tensor"
    return f"{candidate}_data"


def default_tensor_name_from_position(position: int) -> str:
    if 0 <= position < 26:
        return chr(ord("A") + position)
    return f"T{position + 1}"


def recover_tensor_name_from_data_variable(
    data_variable_name: str, fallback_name: str | None = None
) -> str:
    base_name = data_variable_name.removesuffix("_data").strip()
    if not base_name and fallback_name:
        base_name = fallback_name.strip()
    if not base_name:
        return "Tensor"

    parts = [part for part in base_name.split("_") if part]
    if not parts:
        return "Tensor"
    return " ".join(
        part.upper() if len(part) == 1 and part.isalpha() else part.capitalize()
        for part in parts
    )


def recover_index_name(
    *,
    label: str,
    tensor_name: str,
    data_variable_name: str,
    connected_edge_label: str | None,
) -> str:
    if connected_edge_label and label == connected_edge_label:
        if "_" in label:
            suffix = label.rsplit("_", maxsplit=1)[-1].strip()
            if suffix:
                return suffix
        return label

    tensor_identifiers = {
        sanitize_identifier(tensor_name),
        sanitize_identifier(recover_tensor_name_from_data_variable(data_variable_name)),
        sanitize_identifier(data_variable_name.removesuffix("_data")),
    }
    for tensor_identifier in tensor_identifiers:
        if tensor_identifier and label.startswith(f"{tensor_identifier}_"):
            candidate = label[len(tensor_identifier) + 1 :].strip("_")
            if candidate:
                return candidate

    if "_" in label:
        suffix = label.rsplit("_", maxsplit=1)[-1].strip()
        if suffix:
            return suffix
    return label
