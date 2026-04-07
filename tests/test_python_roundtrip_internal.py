from __future__ import annotations

from tensor_network_editor._python_roundtrip_helpers import (
    recover_tensor_name_from_data_variable,
    sanitize_identifier,
)


def test_python_roundtrip_internal_helpers_normalize_generated_names() -> None:
    assert recover_tensor_name_from_data_variable("leaf_mid_data") == "Leaf Mid"
    assert recover_tensor_name_from_data_variable("a_data") == "A"
    assert sanitize_identifier(" Leaf Mid ") == "leaf_mid"
