"""Headless helpers for extracting and reusing tensor-network fragments."""

from __future__ import annotations

from .internal.subnetworks._subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)

__all__ = ["extract_subnetwork_spec", "prepare_subnetwork_for_insertion"]
