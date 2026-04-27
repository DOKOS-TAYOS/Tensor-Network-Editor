"""Shared editor theme names and validation helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias

EditorThemeName: TypeAlias = Literal[
    "dark",
    "light",
    "contrast",
    "colorblind",
    "shiny",
]

DEFAULT_EDITOR_THEME: EditorThemeName = "dark"
SUPPORTED_EDITOR_THEMES: tuple[EditorThemeName, ...] = (
    "dark",
    "light",
    "contrast",
    "colorblind",
    "shiny",
)
_SUPPORTED_EDITOR_THEME_SET = set(SUPPORTED_EDITOR_THEMES)


def normalize_editor_theme(theme: str | None) -> EditorThemeName:
    """Return a normalized editor theme name or raise for unsupported values."""
    if theme is None:
        return DEFAULT_EDITOR_THEME
    normalized = theme.strip().lower()
    if normalized in _SUPPORTED_EDITOR_THEME_SET:
        return normalized
    expected = ", ".join(SUPPORTED_EDITOR_THEMES)
    raise ValueError(
        f"Unsupported editor theme {theme!r}. Expected one of: {expected}."
    )
