from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import EngineName, NetworkSpec
from tests.factories import build_sample_spec, serialize_spec_payload


@pytest.fixture
def sample_spec() -> NetworkSpec:
    return build_sample_spec()


@pytest.fixture
def serialized_sample_spec(sample_spec: NetworkSpec) -> dict[str, object]:
    return serialize_spec_payload(sample_spec)


@pytest.fixture
def editor_session(sample_spec: NetworkSpec) -> EditorSession:
    return EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
    )


@pytest.fixture
def editor_server(editor_session: EditorSession) -> Iterator[EditorServer]:
    server = EditorServer(editor_session)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path.cwd() / ".test_output" / f"pytest_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
