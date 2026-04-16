from __future__ import annotations

import pytest

from tensor_network_editor.app._protocol import (
    parse_subnetwork_prepare_insert_request,
    parse_template_delete_request,
    parse_template_promote_request,
)


def test_parse_template_promote_request_reads_typed_fields() -> None:
    request = parse_template_promote_request(
        {
            "spec": {"schema_version": 4, "network": {"id": "network_demo"}},
            "tensor_ids": ["tensor_a", "tensor_b"],
            "template_name": "project_pair",
            "overwrite": True,
        }
    )

    assert request.serialized_spec["network"]["id"] == "network_demo"
    assert request.tensor_ids == ["tensor_a", "tensor_b"]
    assert request.template_name == "project_pair"
    assert request.overwrite is True


def test_parse_template_delete_request_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="template_name"):
        parse_template_delete_request({"template_name": "   "})


def test_parse_subnetwork_prepare_insert_request_reads_target_center() -> None:
    request = parse_subnetwork_prepare_insert_request(
        {
            "spec": {"schema_version": 4, "network": {"id": "network_demo"}},
            "target_center": {"x": 125.5, "y": 220.0},
        }
    )

    assert request.serialized_spec["network"]["id"] == "network_demo"
    assert request.target_center.x == pytest.approx(125.5)
    assert request.target_center.y == pytest.approx(220.0)
