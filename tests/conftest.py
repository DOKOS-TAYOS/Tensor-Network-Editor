# ruff: noqa: E402

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CHECKOUT_SRC = (REPO_ROOT / "src").resolve()
if str(CURRENT_CHECKOUT_SRC) in sys.path:
    sys.path.remove(str(CURRENT_CHECKOUT_SRC))
sys.path.insert(0, str(CURRENT_CHECKOUT_SRC))

from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import EngineName, NetworkSpec
from tests.factories import build_sample_spec, serialize_spec_payload


def distribution_for_checkout_import_or_skip(
    imported_package: ModuleType,
    *,
    distribution_name: str = "tensor-network-editor",
    package_relative_path: str = "tensor_network_editor/__init__.py",
) -> importlib.metadata.Distribution:
    """Return installed metadata when it matches the imported checkout package."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "Installed distribution metadata is unavailable in source-only test environments."
        )

    imported_package_file_text = imported_package.__file__
    if imported_package_file_text is None:
        pytest.skip(
            "Imported package has no __file__; cannot match checkout imports to installed metadata."
        )

    imported_package_file = Path(imported_package_file_text).resolve()
    installed_package_file = Path(
        str(distribution.locate_file(package_relative_path))
    ).resolve()
    if (
        imported_package_file.is_relative_to(CURRENT_CHECKOUT_SRC)
        and imported_package_file != installed_package_file
    ):
        pytest.skip(
            "Installed distribution metadata points to a different package installation than the current src checkout."
        )
    return distribution


@pytest.fixture
def sample_spec() -> NetworkSpec:
    return build_sample_spec()


@pytest.fixture
def serialized_sample_spec(sample_spec: NetworkSpec) -> JsonDict:
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
