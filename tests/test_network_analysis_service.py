from __future__ import annotations

from tensor_network_editor._network_analysis import (
    connected_index_ids,
    index_map,
    open_indices,
    tensor_map,
)
from tensor_network_editor.models import NetworkSpec


def test_network_analysis_service_matches_network_spec_helpers(
    sample_spec: NetworkSpec,
) -> None:
    assert tensor_map(sample_spec) == sample_spec.tensor_map()
    assert index_map(sample_spec) == sample_spec.index_map()
    assert connected_index_ids(sample_spec) == sample_spec.connected_index_ids()
    assert open_indices(sample_spec) == sample_spec.open_indices()
