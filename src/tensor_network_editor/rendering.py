"""Public helpers for rendering tensor-network specs as static SVG."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from math import ceil
from xml.sax.saxutils import quoteattr

from .internal.io._io import write_utf8_text
from .models import (
    CanvasPosition,
    EdgeSpec,
    HyperedgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from .types import StrPath
from .validation import ensure_valid_spec

_INDEX_RADIUS = 10.0
_GROUP_PADDING = 28.0
_NOTE_WIDTH = 210.0
_NOTE_HEIGHT = 82.0


@dataclass(slots=True, frozen=True)
class SvgRenderOptions:
    """Options for the static SVG renderer."""

    padding: float = 56.0
    show_index_labels: bool = True
    show_edge_labels: bool = True
    background: str = "#171b22"
    tensor_fill: str = "#235f72"
    tensor_stroke: str = "#6fb7ca"
    index_fill: str = "#f0c66a"
    edge_stroke: str = "#9aa8b8"
    hyperedge_stroke: str = "#f08f45"
    group_stroke: str = "#7b8797"
    note_fill: str = "#252b34"
    text_fill: str = "#f2f5f8"
    muted_text_fill: str = "#aeb9c7"
    font_family: str = "Arial, sans-serif"


@dataclass(slots=True, frozen=True)
class _Bounds:
    """Axis-aligned world-space bounds for an SVG export."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return the bounds width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return the bounds height."""
        return self.y2 - self.y1


def render_spec_svg(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a standalone SVG string."""
    resolved_options = options or SvgRenderOptions()
    validated_spec = ensure_valid_spec(spec)
    svg = _SvgRenderer(validated_spec, resolved_options).render()
    if output_path is not None:
        write_utf8_text(output_path, svg, description="SVG network rendering")
    return svg


class _SvgRenderer:
    """Small deterministic SVG renderer for one validated network spec."""

    def __init__(self, spec: NetworkSpec, options: SvgRenderOptions) -> None:
        self._spec = spec
        self._options = options
        self._tensor_by_id = {tensor.id: tensor for tensor in spec.tensors}
        self._index_by_id = {
            index.id: (tensor, index)
            for tensor in spec.tensors
            for index in tensor.indices
        }

    def render(self) -> str:
        """Return the complete SVG document."""
        bounds = self._compute_bounds()
        width = max(240, ceil(bounds.width))
        height = max(180, ceil(bounds.height))
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="{_number(bounds.x1)} {_number(bounds.y1)} '
                f'{_number(width)} {_number(height)}">'
            ),
            f"<title>{_text(self._spec.name)}</title>",
            (
                f"<rect x={_attr(bounds.x1)} y={_attr(bounds.y1)} "
                f"width={_attr(width)} height={_attr(height)} "
                f"fill={_attr(self._options.background)} />"
            ),
        ]
        lines.extend(self._render_groups())
        lines.extend(self._render_edges())
        lines.extend(self._render_hyperedges())
        lines.extend(self._render_tensors())
        lines.extend(self._render_notes())
        lines.append("</svg>")
        return "\n".join(lines)

    def _compute_bounds(self) -> _Bounds:
        points: list[CanvasPosition] = []
        for tensor in self._spec.tensors:
            half_width = tensor.size.width / 2
            half_height = tensor.size.height / 2
            points.extend(
                [
                    CanvasPosition(
                        tensor.position.x - half_width, tensor.position.y - half_height
                    ),
                    CanvasPosition(
                        tensor.position.x + half_width, tensor.position.y + half_height
                    ),
                ]
            )
            points.extend(
                self._index_position(tensor, index) for index in tensor.indices
            )
        for note in self._spec.notes:
            points.extend(
                [
                    note.position,
                    CanvasPosition(
                        note.position.x + _NOTE_WIDTH, note.position.y + _NOTE_HEIGHT
                    ),
                ]
            )
        for hyperedge in self._spec.hyperedges:
            points.append(self._hyperedge_hub_position(hyperedge))
        if not points:
            return _Bounds(
                x1=-self._options.padding,
                y1=-self._options.padding,
                x2=240.0 + self._options.padding,
                y2=180.0 + self._options.padding,
            )
        return _Bounds(
            x1=min(point.x for point in points) - self._options.padding,
            y1=min(point.y for point in points) - self._options.padding,
            x2=max(point.x for point in points) + self._options.padding,
            y2=max(point.y for point in points) + self._options.padding,
        )

    def _render_groups(self) -> list[str]:
        lines: list[str] = []
        for group in self._spec.groups:
            group_tensors = [
                self._tensor_by_id[tensor_id]
                for tensor_id in group.tensor_ids
                if tensor_id in self._tensor_by_id
            ]
            if not group_tensors:
                continue
            bounds = _tensor_collection_bounds(group_tensors, padding=_GROUP_PADDING)
            lines.append(
                f'<rect class="group" x={_attr(bounds.x1)} y={_attr(bounds.y1)} '
                f"width={_attr(bounds.width)} height={_attr(bounds.height)} "
                f'rx="10" ry="10" fill="none" '
                f'stroke={_attr(self._options.group_stroke)} stroke-width="1.5" '
                f'stroke-dasharray="8 6" />'
            )
            lines.append(
                f'<text class="group-label" x={_attr(bounds.x1 + 12)} '
                f"y={_attr(bounds.y1 + 20)} fill={_attr(self._options.muted_text_fill)} "
                f'font-size="12" font-family={_attr(self._options.font_family)}>'
                f"{_text(group.name)}</text>"
            )
        return lines

    def _render_edges(self) -> list[str]:
        lines: list[str] = []
        for edge in self._spec.edges:
            endpoints = self._edge_positions(edge)
            if endpoints is None:
                continue
            source, target = endpoints
            lines.append(
                f'<line class="edge" x1={_attr(source.x)} y1={_attr(source.y)} '
                f"x2={_attr(target.x)} y2={_attr(target.y)} "
                f'stroke={_attr(self._options.edge_stroke)} stroke-width="3" />'
            )
            if self._options.show_edge_labels and edge.name:
                midpoint = _midpoint(source, target)
                lines.append(
                    f'<text class="edge-label" x={_attr(midpoint.x)} '
                    f"y={_attr(midpoint.y - 10)} fill={_attr(self._options.muted_text_fill)} "
                    f'font-size="11" font-family={_attr(self._options.font_family)} '
                    f'text-anchor="middle">{_text(edge.name)}</text>'
                )
        return lines

    def _render_hyperedges(self) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hub = self._hyperedge_hub_position(hyperedge)
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                endpoint_position = self._index_position(*tensor_index)
                lines.append(
                    f'<line class="hyperedge-spoke" x1={_attr(endpoint_position.x)} '
                    f"y1={_attr(endpoint_position.y)} x2={_attr(hub.x)} y2={_attr(hub.y)} "
                    f"stroke={_attr(self._options.hyperedge_stroke)} "
                    f'stroke-width="2.5" stroke-dasharray="5 4" />'
                )
            lines.append(
                f'<circle class="hyperedge-hub" cx={_attr(hub.x)} cy={_attr(hub.y)} '
                f'r="11" fill={_attr(self._options.hyperedge_stroke)} />'
            )
            if self._options.show_edge_labels and hyperedge.name:
                lines.append(
                    f'<text class="hyperedge-label" x={_attr(hub.x)} y={_attr(hub.y - 17)} '
                    f'fill={_attr(self._options.text_fill)} font-size="11" '
                    f'font-family={_attr(self._options.font_family)} text-anchor="middle">'
                    f"{_text(hyperedge.name)}</text>"
                )
        return lines

    def _render_tensors(self) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            x = tensor.position.x - tensor.size.width / 2
            y = tensor.position.y - tensor.size.height / 2
            lines.append(
                f'<rect class="tensor" x={_attr(x)} y={_attr(y)} '
                f"width={_attr(tensor.size.width)} height={_attr(tensor.size.height)} "
                f'rx="8" ry="8" fill={_attr(self._options.tensor_fill)} '
                f'stroke={_attr(self._options.tensor_stroke)} stroke-width="2" />'
            )
            lines.append(
                f'<text class="tensor-label" x={_attr(tensor.position.x)} '
                f"y={_attr(y + 28)} fill={_attr(self._options.text_fill)} "
                f'font-size="18" font-family={_attr(self._options.font_family)} '
                f'text-anchor="middle">{_text(tensor.name)}</text>'
            )
            lines.extend(self._render_indices(tensor))
        return lines

    def _render_indices(self, tensor: TensorSpec) -> list[str]:
        lines: list[str] = []
        for index_position, index in enumerate(tensor.indices, start=1):
            point = self._index_position(tensor, index)
            lines.append(
                f'<circle class="index" cx={_attr(point.x)} cy={_attr(point.y)} '
                f"r={_attr(_INDEX_RADIUS)} fill={_attr(self._options.index_fill)} "
                f'stroke="#47380d" stroke-width="1.5" />'
            )
            lines.append(
                f'<text class="index-number" x={_attr(point.x)} y={_attr(point.y + 4)} '
                f'fill="#1b1b1b" font-size="11" font-weight="700" '
                f'font-family={_attr(self._options.font_family)} text-anchor="middle">'
                f"{index_position}</text>"
            )
            if self._options.show_index_labels:
                lines.append(
                    f'<text class="index-label" x={_attr(point.x)} y={_attr(point.y + 26)} '
                    f'fill={_attr(self._options.muted_text_fill)} font-size="10" '
                    f'font-family={_attr(self._options.font_family)} text-anchor="middle">'
                    f"{_text(index.name)} {_text(str(index.dimension))}</text>"
                )
        return lines

    def _render_notes(self) -> list[str]:
        lines: list[str] = []
        for note in self._spec.notes:
            lines.append(
                f'<rect class="note" x={_attr(note.position.x)} y={_attr(note.position.y)} '
                f'width={_attr(_NOTE_WIDTH)} height={_attr(_NOTE_HEIGHT)} rx="8" ry="8" '
                f"fill={_attr(self._options.note_fill)} stroke={_attr(self._options.group_stroke)} "
                f'stroke-width="1" />'
            )
            for line_index, note_line in enumerate(_wrap_text(note.text, max_chars=32)):
                lines.append(
                    f'<text class="note-text" x={_attr(note.position.x + 12)} '
                    f"y={_attr(note.position.y + 24 + line_index * 16)} "
                    f'fill={_attr(self._options.text_fill)} font-size="12" '
                    f"font-family={_attr(self._options.font_family)}>"
                    f"{_text(note_line)}</text>"
                )
        return lines

    def _edge_positions(
        self, edge: EdgeSpec
    ) -> tuple[CanvasPosition, CanvasPosition] | None:
        left = self._index_by_id.get(edge.left.index_id)
        right = self._index_by_id.get(edge.right.index_id)
        if left is None or right is None:
            return None
        return self._index_position(*left), self._index_position(*right)

    def _hyperedge_hub_position(self, hyperedge: HyperedgeSpec) -> CanvasPosition:
        endpoint_positions = [
            self._index_position(*tensor_index)
            for endpoint in hyperedge.endpoints
            if (tensor_index := self._index_by_id.get(endpoint.index_id)) is not None
        ]
        center = _average_position(endpoint_positions)
        return CanvasPosition(
            x=center.x + hyperedge.hub_offset.x,
            y=center.y + hyperedge.hub_offset.y,
        )

    @staticmethod
    def _index_position(tensor: TensorSpec, index: IndexSpec) -> CanvasPosition:
        return CanvasPosition(
            x=tensor.position.x + index.offset.x,
            y=tensor.position.y + index.offset.y,
        )


def _tensor_collection_bounds(
    tensors: Iterable[TensorSpec], *, padding: float
) -> _Bounds:
    tensor_list = list(tensors)
    return _Bounds(
        x1=min(tensor.position.x - tensor.size.width / 2 for tensor in tensor_list)
        - padding,
        y1=min(tensor.position.y - tensor.size.height / 2 for tensor in tensor_list)
        - padding,
        x2=max(tensor.position.x + tensor.size.width / 2 for tensor in tensor_list)
        + padding,
        y2=max(tensor.position.y + tensor.size.height / 2 for tensor in tensor_list)
        + padding,
    )


def _average_position(points: Iterable[CanvasPosition]) -> CanvasPosition:
    point_list = list(points)
    if not point_list:
        return CanvasPosition()
    return CanvasPosition(
        x=sum(point.x for point in point_list) / len(point_list),
        y=sum(point.y for point in point_list) / len(point_list),
    )


def _midpoint(left: CanvasPosition, right: CanvasPosition) -> CanvasPosition:
    return CanvasPosition(x=(left.x + right.x) / 2, y=(left.y + right.y) / 2)


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current_line = words[0]
    for word in words[1:]:
        if len(current_line) + 1 + len(word) <= max_chars:
            current_line = f"{current_line} {word}"
            continue
        lines.append(current_line)
        current_line = word
    lines.append(current_line)
    return lines[:4]


def _text(value: str) -> str:
    return escape(value, quote=False)


def _attr(value: object) -> str:
    return quoteattr(_number(value) if isinstance(value, (int, float)) else str(value))


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = ["SvgRenderOptions", "render_spec_svg"]
