"""Public helpers for launching the local editor session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._themes import DEFAULT_EDITOR_THEME, EditorThemeName, normalize_editor_theme
from .app.session import launch_editor_session
from .models import (
    EditorResult,
    EngineIdentifier,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from .types import StrPath


@dataclass(slots=True, frozen=True)
class EditorLaunchOptions:
    """Public options for opening the local browser editor."""

    default_engine: EngineIdentifier = EngineName.TENSORKROWCH
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST
    theme: EditorThemeName = DEFAULT_EDITOR_THEME
    open_browser: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    print_code: bool = False
    code_path: StrPath | None = None
    template_catalog_path: StrPath | None = None
    subnetwork_catalog_path: StrPath | None = None
    shared_subnetwork_catalog_path: StrPath | None = None
    draft_path: StrPath | None = None
    _on_server_ready: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate theme names passed at runtime."""
        object.__setattr__(self, "theme", normalize_editor_theme(self.theme))


def open_editor(
    spec: NetworkSpec | None = None,
    *,
    theme: EditorThemeName | None = None,
    options: EditorLaunchOptions | None = None,
) -> EditorResult | None:
    """Launch the local browser editor and wait for the final session result."""
    resolved_options = options or EditorLaunchOptions()
    resolved_theme = (
        normalize_editor_theme(theme) if theme is not None else resolved_options.theme
    )
    return launch_editor_session(
        initial_spec=spec,
        default_engine=resolved_options.default_engine,
        default_collection_format=resolved_options.default_collection_format,
        theme=resolved_theme,
        open_browser=resolved_options.open_browser,
        host=resolved_options.host,
        port=resolved_options.port,
        print_code=resolved_options.print_code,
        code_path=resolved_options.code_path,
        template_catalog_path=resolved_options.template_catalog_path,
        subnetwork_catalog_path=resolved_options.subnetwork_catalog_path,
        shared_subnetwork_catalog_path=resolved_options.shared_subnetwork_catalog_path,
        draft_path=resolved_options.draft_path,
        _on_server_ready=resolved_options._on_server_ready,
    )


__all__ = ["EditorLaunchOptions", "EditorThemeName", "open_editor"]
