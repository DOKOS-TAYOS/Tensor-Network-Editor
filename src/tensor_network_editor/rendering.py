"""Public helpers for rendering tensor-network specs as static figures."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from html import escape
from importlib import import_module
from io import BytesIO
from math import ceil, cos, hypot, isfinite, pi, sin
from typing import Any, TypeVar
from xml.sax.saxutils import quoteattr

from .internal._logging import log_operation, summarize_spec_counts
from .internal.io._io import write_binary, write_utf8_text
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

_GROUP_PADDING = 28.0
_NOTE_WIDTH = 210.0
_NOTE_HEIGHT = 82.0
_PARALLEL_EDGE_SPACING = 22.0
_LIGHT_TEXT_FILL = "#f5f9ff"
_DARK_TEXT_FILL = "#091018"
_READABLE_TEXT_LUMINANCE_THRESHOLD = 0.62
_FREE_INDEX_CLEARANCE_MARGIN = 18.0
_FREE_INDEX_DIAGONAL_DIRECTIONS = (
    CanvasPosition(x=1.0, y=1.0),
    CanvasPosition(x=1.0, y=-1.0),
    CanvasPosition(x=-1.0, y=1.0),
    CanvasPosition(x=-1.0, y=-1.0),
)
_FREE_INDEX_CARDINAL_DIRECTIONS = (
    CanvasPosition(x=1.0, y=0.0),
    CanvasPosition(x=-1.0, y=0.0),
    CanvasPosition(x=0.0, y=1.0),
    CanvasPosition(x=0.0, y=-1.0),
)
_FREE_INDEX_DIRECTION_HINTS = {
    "left": CanvasPosition(x=-1.0, y=0.0),
    "right": CanvasPosition(x=1.0, y=0.0),
    "up": CanvasPosition(x=0.0, y=-1.0),
    "down": CanvasPosition(x=0.0, y=1.0),
    "north": CanvasPosition(x=0.0, y=-1.0),
    "south": CanvasPosition(x=0.0, y=1.0),
    "east": CanvasPosition(x=1.0, y=0.0),
    "west": CanvasPosition(x=-1.0, y=0.0),
    "xp": CanvasPosition(x=1.0, y=0.0),
    "xm": CanvasPosition(x=-1.0, y=0.0),
    "yp": CanvasPosition(x=0.0, y=-1.0),
    "ym": CanvasPosition(x=0.0, y=1.0),
}

LOGGER = logging.getLogger(__name__)
_RenderOptionsT = TypeVar("_RenderOptionsT")
_RenderedOutputT = TypeVar("_RenderedOutputT", str, bytes)


@dataclass(slots=True, frozen=True)
class SvgRenderOptions:
    """Options for the academic SVG/PNG/PDF renderers."""

    padding: float = 56.0
    show_tensor_labels: bool = True
    show_index_labels: bool = True
    show_edge_labels: bool = True
    include_groups: bool = True
    include_notes: bool = True
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
class TikzRenderOptions:
    """Options for the static TikZ renderer."""

    scale: float = 1.0
    global_width: str = r"\linewidth"
    show_index_labels: bool = True
    show_edge_labels: bool = True
    include_groups: bool = True
    include_notes: bool = True
    show_tensor_labels: bool = True


@dataclass(slots=True, frozen=True)
class DotRenderOptions:
    """Options for the static Graphviz/DOT renderer."""

    include_open_indices: bool = True
    include_groups: bool = True
    include_notes: bool = True
    include_hyperedges: bool = True
    show_tensor_labels: bool = True
    show_index_labels: bool = True
    show_edge_labels: bool = True


@dataclass(slots=True, frozen=True)
class _Bounds:
    """Axis-aligned world-space bounds for an academic export."""

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


@dataclass(slots=True, frozen=True)
class _RenderedEdge:
    """Geometry and styling data for one pairwise edge render."""

    edge: EdgeSpec
    source: CanvasPosition
    target: CanvasPosition
    control: CanvasPosition | None
    stroke: str


def _render_context(
    spec: NetworkSpec,
    *,
    format_name: str,
    output_path: StrPath | None,
) -> dict[str, object]:
    """Return shared logging context for one static render operation."""
    return {
        "format": format_name,
        "output_path": output_path,
        **summarize_spec_counts(spec),
    }


def _validate_positive_render_scale(
    scale: object,
    *,
    description: str,
) -> float:
    """Normalize one render scale after validating positivity and finiteness."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"{description} must be a positive finite number.")
    resolved_scale = float(scale)
    if not isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError(f"{description} must be a positive finite number.")
    return resolved_scale


def _render_spec_output(
    spec: NetworkSpec,
    *,
    format_name: str,
    options: _RenderOptionsT,
    output_path: StrPath | None,
    description: str,
    render: Callable[[NetworkSpec, _RenderOptionsT], _RenderedOutputT],
    writer: Callable[..., None],
) -> _RenderedOutputT:
    """Validate, render, and optionally persist one static export."""
    with log_operation(
        LOGGER,
        "Render spec",
        context=_render_context(spec, format_name=format_name, output_path=output_path),
    ):
        validated_spec = ensure_valid_spec(spec)
        rendered = render(validated_spec, options)
        if output_path is not None:
            writer(output_path, rendered, description=description)
        return rendered


def render_spec_svg(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a standalone SVG string."""
    resolved_options = options or SvgRenderOptions()

    def _render(validated_spec: NetworkSpec, options: SvgRenderOptions) -> str:
        return _MatplotlibRenderer(validated_spec, options).render_svg()

    return _render_spec_output(
        spec,
        format_name="svg",
        options=resolved_options,
        output_path=output_path,
        description="SVG network rendering",
        render=_render,
        writer=write_utf8_text,
    )


def render_spec_png(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    scale: float = 2.0,
    output_path: StrPath | None = None,
) -> bytes:
    """Render one tensor-network specification as PNG bytes using Matplotlib."""
    resolved_scale = _validate_positive_render_scale(
        scale,
        description="PNG render scale",
    )
    resolved_options = options or SvgRenderOptions()

    def _render(validated_spec: NetworkSpec, options: SvgRenderOptions) -> bytes:
        return _MatplotlibRenderer(
            validated_spec,
            options,
            scale=resolved_scale,
        ).render_png()

    return _render_spec_output(
        spec,
        format_name="png",
        options=resolved_options,
        output_path=output_path,
        description="PNG network rendering",
        render=_render,
        writer=write_binary,
    )


def render_spec_pdf(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    scale: float = 2.0,
    output_path: StrPath | None = None,
) -> bytes:
    """Render one tensor-network specification as PDF bytes using Matplotlib."""
    resolved_scale = _validate_positive_render_scale(
        scale,
        description="PDF render scale",
    )
    resolved_options = options or SvgRenderOptions()

    def _render(validated_spec: NetworkSpec, options: SvgRenderOptions) -> bytes:
        return _MatplotlibRenderer(
            validated_spec,
            options,
            scale=resolved_scale,
        ).render_pdf()

    return _render_spec_output(
        spec,
        format_name="pdf",
        options=resolved_options,
        output_path=output_path,
        description="PDF network rendering",
        render=_render,
        writer=write_binary,
    )


def render_spec_tikz(
    spec: NetworkSpec,
    *,
    options: TikzRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a standalone TikZ picture."""
    resolved_options = options or TikzRenderOptions()
    validated_scale = _validate_positive_render_scale(
        resolved_options.scale,
        description="TikZ render scale",
    )
    resolved_options = TikzRenderOptions(
        scale=validated_scale,
        global_width=resolved_options.global_width,
        show_index_labels=resolved_options.show_index_labels,
        show_edge_labels=resolved_options.show_edge_labels,
        include_groups=resolved_options.include_groups,
        include_notes=resolved_options.include_notes,
        show_tensor_labels=resolved_options.show_tensor_labels,
    )

    def _render(validated_spec: NetworkSpec, options: TikzRenderOptions) -> str:
        return _TikzRenderer(validated_spec, options).render()

    return _render_spec_output(
        spec,
        format_name="tikz",
        options=resolved_options,
        output_path=output_path,
        description="TikZ network rendering",
        render=_render,
        writer=write_utf8_text,
    )


def render_spec_dot(
    spec: NetworkSpec,
    *,
    options: DotRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a Graphviz/DOT graph."""
    resolved_options = options or DotRenderOptions()

    def _render(validated_spec: NetworkSpec, options: DotRenderOptions) -> str:
        return _DotRenderer(validated_spec, options).render()

    return _render_spec_output(
        spec,
        format_name="dot",
        options=resolved_options,
        output_path=output_path,
        description="Graphviz/DOT network rendering",
        render=_render,
        writer=write_utf8_text,
    )


def render_spec_mermaid(
    spec: NetworkSpec,
    *,
    options: DotRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a Mermaid flowchart."""
    resolved_options = options or DotRenderOptions()

    def _render(validated_spec: NetworkSpec, options: DotRenderOptions) -> str:
        return _MermaidRenderer(validated_spec, options).render()

    return _render_spec_output(
        spec,
        format_name="mermaid",
        options=resolved_options,
        output_path=output_path,
        description="Mermaid network rendering",
        render=_render,
        writer=write_utf8_text,
    )


class _SvgRenderer:
    """Shared geometry helper for academic network renderers."""

    def __init__(self, spec: NetworkSpec, options: SvgRenderOptions) -> None:
        self._spec = spec
        self._options = options
        self._tensor_by_id = {tensor.id: tensor for tensor in spec.tensors}
        self._index_by_id = {
            index.id: (tensor, index)
            for tensor in spec.tensors
            for index in tensor.indices
        }
        self._index_order_by_id = {
            index.id: index_position
            for tensor in spec.tensors
            for index_position, index in enumerate(tensor.indices)
        }
        self._connected_index_ids = _connected_index_ids(spec)
        self._free_index_direction_by_id: dict[str, CanvasPosition] = {}
        self._adjacency_by_tensor_id = self._build_tensor_adjacency()
        self._component_tensor_ids_by_tensor_id = self._build_component_tensor_ids()

    def render(self) -> str:
        """Return the complete SVG document."""
        edge_render_infos = self._edge_render_infos()
        bounds = self._compute_bounds(edge_render_infos)
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
        if self._options.include_groups:
            lines.extend(self._render_groups())
        lines.extend(self._render_edges(edge_render_infos))
        lines.extend(self._render_hyperedges())
        lines.extend(self._render_open_indices())
        lines.extend(self._render_tensors())
        lines.extend(self._render_index_labels())
        if self._options.include_notes:
            lines.extend(self._render_notes())
        lines.append("</svg>")
        return "\n".join(lines)

    def _compute_bounds(
        self,
        edge_render_infos: Sequence[_RenderedEdge] | None = None,
    ) -> _Bounds:
        points: list[CanvasPosition] = []
        resolved_edge_render_infos = (
            self._edge_render_infos()
            if edge_render_infos is None
            else edge_render_infos
        )
        for tensor in self._spec.tensors:
            if not self.is_port_tensor(tensor):
                radius = self.tensor_radius(tensor)
                points.extend(
                    [
                        CanvasPosition(
                            tensor.position.x - radius, tensor.position.y - radius
                        ),
                        CanvasPosition(
                            tensor.position.x + radius, tensor.position.y + radius
                        ),
                    ]
                )
            for index in tensor.indices:
                points.append(self.connection_point(tensor, index))
                if not self.is_index_connected(index.id):
                    points.append(self.open_index_endpoint(tensor, index))
                if self._options.show_index_labels:
                    label_point = self.index_label_point(tensor, index)
                    points.extend(
                        [
                            label_point,
                            CanvasPosition(label_point.x + 56.0, label_point.y + 18.0),
                        ]
                    )
        if self._options.include_notes:
            for note in self._spec.notes:
                points.extend(
                    [
                        note.position,
                        CanvasPosition(
                            note.position.x + _NOTE_WIDTH,
                            note.position.y + _NOTE_HEIGHT,
                        ),
                    ]
                )
        for hyperedge in self._spec.hyperedges:
            points.append(self._hyperedge_hub_position(hyperedge))
        for edge_info in resolved_edge_render_infos:
            if edge_info.control is not None:
                points.append(edge_info.control)
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
                f"stroke={_attr(_metadata_color(group.metadata, self._options.group_stroke))} "
                f'stroke-width="1.5" '
                f'stroke-dasharray="8 6" />'
            )
            lines.append(
                f'<text class="group-label" x={_attr(bounds.x1 + 12)} '
                f"y={_attr(bounds.y1 + 20)} fill={_attr(self._options.muted_text_fill)} "
                f'font-size="12" font-family={_attr(self._options.font_family)}>'
                f"{_text(group.name)}</text>"
            )
        return lines

    def _render_edges(self, edge_render_infos: Sequence[_RenderedEdge]) -> list[str]:
        lines: list[str] = []
        for edge_info in edge_render_infos:
            source = edge_info.source
            target = edge_info.target
            if edge_info.control is None:
                path_data = (
                    f"M {_number(source.x)} {_number(source.y)} "
                    f"L {_number(target.x)} {_number(target.y)}"
                )
                label_point = _midpoint(source, target)
            else:
                control = edge_info.control
                path_data = (
                    f"M {_number(source.x)} {_number(source.y)} "
                    f"Q {_number(control.x)} {_number(control.y)} "
                    f"{_number(target.x)} {_number(target.y)}"
                )
                label_point = _quadratic_midpoint(source, control, target)
            lines.append(
                f'<path class="edge" d={_attr(path_data)} fill="none" '
                f'stroke={_attr(edge_info.stroke)} stroke-width="3" />'
            )
            if self._options.show_edge_labels and edge_info.edge.name:
                lines.append(
                    f'<text class="edge-label" x={_attr(label_point.x)} '
                    f"y={_attr(label_point.y - 10)} fill={_attr(self._options.muted_text_fill)} "
                    f'font-size="11" font-family={_attr(self._options.font_family)} '
                    f'text-anchor="middle">{_text(edge_info.edge.name)}</text>'
                )
        return lines

    def _render_hyperedges(self) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hub = self._hyperedge_hub_position(hyperedge)
            hyperedge_stroke = _metadata_color(
                hyperedge.metadata,
                self._options.hyperedge_stroke,
            )
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                tensor, _ = tensor_index
                endpoint_position = tensor.position
                lines.append(
                    f'<line class="hyperedge-spoke" x1={_attr(endpoint_position.x)} '
                    f"y1={_attr(endpoint_position.y)} x2={_attr(hub.x)} y2={_attr(hub.y)} "
                    f"stroke={_attr(hyperedge_stroke)} "
                    f'stroke-width="2.5" stroke-dasharray="5 4" />'
                )
            lines.append(
                f'<circle class="hyperedge-hub" cx={_attr(hub.x)} cy={_attr(hub.y)} '
                f'r="11" fill={_attr(hyperedge_stroke)} />'
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
            if self.is_port_tensor(tensor):
                continue
            radius = self.tensor_radius(tensor)
            tensor_custom_color = _metadata_color_or_none(tensor.metadata)
            tensor_fill = tensor_custom_color or self._options.tensor_fill
            tensor_stroke = (
                _shift_hex_color(tensor_fill, 38, self._options.tensor_stroke)
                if tensor_custom_color
                else self._options.tensor_stroke
            )
            tensor_text_fill = _readable_text_color(tensor_fill)
            lines.append(
                f'<circle class="tensor" cx={_attr(tensor.position.x)} cy={_attr(tensor.position.y)} '
                f"r={_attr(radius)} fill={_attr(tensor_fill)} "
                f"stroke={_attr(tensor_stroke)} "
                f'stroke-width="2" />'
            )
            if self._options.show_tensor_labels:
                lines.append(
                    f'<text class="tensor-label" x={_attr(tensor.position.x)} '
                    f"y={_attr(tensor.position.y + 6)} fill={_attr(tensor_text_fill)} "
                    f'font-size="18" font-family={_attr(self._options.font_family)} '
                    f'text-anchor="middle">{_text(tensor.name)}</text>'
                )
        return lines

    def _render_open_indices(self) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if self.is_index_connected(index.id):
                    continue
                source = self.connection_point(tensor, index)
                target = self.open_index_endpoint(tensor, index)
                lines.append(
                    f'<line class="open-index" x1={_attr(source.x)} y1={_attr(source.y)} '
                    f"x2={_attr(target.x)} y2={_attr(target.y)} "
                    f"stroke={_attr(_metadata_color(index.metadata, self._options.edge_stroke))} "
                    f'stroke-width="3" />'
                )
        return lines

    def _render_index_labels(self) -> list[str]:
        if not self._options.show_index_labels:
            return []
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                label_point = self.index_label_point(tensor, index)
                lines.append(
                    f'<text class="index-label" x={_attr(label_point.x)} y={_attr(label_point.y)} '
                    f'fill={_attr(self._options.muted_text_fill)} font-size="10" '
                    f"font-family={_attr(self._options.font_family)} text-anchor={_attr(self._svg_text_anchor(tensor, index))}>"
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
        return left[0].position, right[0].position

    def _edge_render_infos(self) -> list[_RenderedEdge]:
        grouped_edges: dict[
            tuple[str, str], list[tuple[EdgeSpec, CanvasPosition, CanvasPosition]]
        ] = {}
        for edge in self._spec.edges:
            endpoints = self._edge_positions(edge)
            if endpoints is None:
                continue
            source, target = endpoints
            left_tensor_id = edge.left.tensor_id
            right_tensor_id = edge.right.tensor_id
            key: tuple[str, str] = (
                (left_tensor_id, right_tensor_id)
                if left_tensor_id <= right_tensor_id
                else (right_tensor_id, left_tensor_id)
            )
            grouped_edges.setdefault(key, []).append((edge, source, target))

        render_infos: list[_RenderedEdge] = []
        for edge_entries in grouped_edges.values():
            entry_count = len(edge_entries)
            for edge_position, (edge, source, target) in enumerate(edge_entries):
                control = (
                    _parallel_edge_control_point(
                        source,
                        target,
                        edge_position=edge_position,
                        edge_count=entry_count,
                    )
                    if entry_count > 1
                    else None
                )
                render_infos.append(
                    _RenderedEdge(
                        edge=edge,
                        source=source,
                        target=target,
                        control=control,
                        stroke=_metadata_color(
                            edge.metadata, self._options.edge_stroke
                        ),
                    )
                )
        return render_infos

    def _hyperedge_hub_position(self, hyperedge: HyperedgeSpec) -> CanvasPosition:
        endpoint_positions = [
            self.connection_point(*tensor_index)
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

    def tensor_radius(self, tensor: TensorSpec) -> float:
        return max(24.0, min(tensor.size.width, tensor.size.height) / 2.0)

    def is_port_tensor(self, tensor: TensorSpec) -> bool:
        return (
            tensor.linear_periodic_role is not None
            or tensor.grid_periodic_role is not None
            or tensor.tree_periodic_role is not None
        )

    def is_index_connected(self, index_id: str) -> bool:
        return index_id in self._connected_index_ids

    def connection_point(self, tensor: TensorSpec, index: IndexSpec) -> CanvasPosition:
        direction = self._index_direction(tensor, index)
        radius = 0.0 if self.is_port_tensor(tensor) else self.tensor_radius(tensor)
        return CanvasPosition(
            x=tensor.position.x + direction.x * radius,
            y=tensor.position.y + direction.y * radius,
        )

    def open_index_endpoint(
        self,
        tensor: TensorSpec,
        index: IndexSpec,
        *,
        port_length: float | None = None,
    ) -> CanvasPosition:
        direction = self._index_direction(tensor, index)
        source = self.connection_point(tensor, index)
        resolved_port_length = (
            self.open_index_port_length(tensor) if port_length is None else port_length
        )
        return CanvasPosition(
            x=source.x + direction.x * resolved_port_length,
            y=source.y + direction.y * resolved_port_length,
        )

    def index_label_point(self, tensor: TensorSpec, index: IndexSpec) -> CanvasPosition:
        direction = self._index_direction(tensor, index)
        source = self.connection_point(tensor, index)
        label_distance = (
            18.0
            if self.is_index_connected(index.id)
            else self.open_index_port_length(tensor) + 14.0
        )
        return CanvasPosition(
            x=source.x + direction.x * label_distance,
            y=source.y + direction.y * label_distance + 4.0,
        )

    def _svg_text_anchor(self, tensor: TensorSpec, index: IndexSpec) -> str:
        direction = self._index_direction(tensor, index)
        if direction.x >= 0.4:
            return "start"
        if direction.x <= -0.4:
            return "end"
        return "middle"

    def _index_direction(self, tensor: TensorSpec, index: IndexSpec) -> CanvasPosition:
        if not self.is_index_connected(index.id):
            return self._free_index_direction(tensor, index)
        return self._stored_index_direction(tensor, index)

    def open_index_port_length(self, tensor: TensorSpec) -> float:
        return 2.0 * self.tensor_radius(tensor)

    def _stored_index_direction(
        self, tensor: TensorSpec, index: IndexSpec
    ) -> CanvasPosition:
        magnitude = hypot(index.offset.x, index.offset.y)
        if magnitude > 1e-6:
            return CanvasPosition(
                x=index.offset.x / magnitude, y=index.offset.y / magnitude
            )
        index_order = self._index_order_by_id.get(index.id, 0)
        index_count = max(1, len(tensor.indices))
        angle = -pi / 2 + (2 * pi * index_order / index_count)
        return CanvasPosition(x=cos(angle), y=sin(angle))

    def _free_index_direction(
        self, tensor: TensorSpec, index: IndexSpec
    ) -> CanvasPosition:
        cached_direction = self._free_index_direction_by_id.get(index.id)
        if cached_direction is not None:
            return cached_direction
        self._assign_free_index_directions(tensor)
        cached_direction = self._free_index_direction_by_id.get(index.id)
        if cached_direction is not None:
            return cached_direction
        return self._stored_index_direction(tensor, index)

    def _assign_free_index_directions(self, tensor: TensorSpec) -> None:
        free_indices = [
            index for index in tensor.indices if not self.is_index_connected(index.id)
        ]
        if not free_indices:
            return
        component_tensors = self._component_tensors_for_tensor(tensor.id)
        occupied_directions = self._occupied_directions(tensor)
        assigned_directions: list[CanvasPosition] = []
        for index in free_indices:
            candidate_directions = self._candidate_directions_for_free_index(
                tensor,
                index,
                component_tensors,
            )
            direction = self._choose_best_free_index_direction(
                tensor,
                component_tensors,
                candidate_directions,
                [*occupied_directions, *assigned_directions],
            )
            self._free_index_direction_by_id[index.id] = direction
            assigned_directions.append(direction)

    def _occupied_directions(self, tensor: TensorSpec) -> list[CanvasPosition]:
        directions: list[CanvasPosition] = []
        for index in tensor.indices:
            if not self.is_index_connected(index.id):
                continue
            direction = self._connected_index_direction(tensor, index)
            if direction is not None:
                directions.append(direction)
        return directions

    def _connected_index_direction(
        self, tensor: TensorSpec, index: IndexSpec
    ) -> CanvasPosition | None:
        for edge in self._spec.edges:
            if (
                edge.left.index_id == index.id
                and edge.right.tensor_id in self._tensor_by_id
            ):
                return _normalize_direction(
                    CanvasPosition(
                        x=self._tensor_by_id[edge.right.tensor_id].position.x
                        - tensor.position.x,
                        y=self._tensor_by_id[edge.right.tensor_id].position.y
                        - tensor.position.y,
                    )
                )
            if (
                edge.right.index_id == index.id
                and edge.left.tensor_id in self._tensor_by_id
            ):
                return _normalize_direction(
                    CanvasPosition(
                        x=self._tensor_by_id[edge.left.tensor_id].position.x
                        - tensor.position.x,
                        y=self._tensor_by_id[edge.left.tensor_id].position.y
                        - tensor.position.y,
                    )
                )
        for hyperedge in self._spec.hyperedges:
            endpoint_ids = {endpoint.index_id for endpoint in hyperedge.endpoints}
            if index.id not in endpoint_ids:
                continue
            hub = self._hyperedge_hub_position(hyperedge)
            return _normalize_direction(
                CanvasPosition(x=hub.x - tensor.position.x, y=hub.y - tensor.position.y)
            )
        return None

    def _candidate_directions_for_free_index(
        self,
        tensor: TensorSpec,
        index: IndexSpec,
        component_tensors: Sequence[TensorSpec],
    ) -> list[CanvasPosition]:
        directional_hint = self._directional_index_hint(index)
        candidates: list[CanvasPosition] = []
        component_kind = self._classify_component_shape(component_tensors)
        if component_kind == "linear":
            candidates.extend(self._linear_component_candidates(component_tensors))
        elif component_kind == "circular":
            candidates.extend(
                self._circular_component_candidates(tensor, component_tensors)
            )
        elif component_kind == "grid2d":
            candidates.extend(
                self._grid_component_candidates(tensor, component_tensors)
            )
        if directional_hint is not None:
            if component_kind == "generic":
                candidates.insert(0, directional_hint)
            else:
                candidates.append(directional_hint)
        candidates.extend(self._generic_component_candidates(tensor, component_tensors))
        candidates.extend(_FREE_INDEX_CARDINAL_DIRECTIONS)
        candidates.extend(_FREE_INDEX_DIAGONAL_DIRECTIONS)
        return _deduplicate_directions(candidates)

    def _choose_best_free_index_direction(
        self,
        tensor: TensorSpec,
        component_tensors: Sequence[TensorSpec],
        candidate_directions: Sequence[CanvasPosition],
        occupied_directions: Sequence[CanvasPosition],
    ) -> CanvasPosition:
        best_direction: CanvasPosition | None = None
        best_score = float("-inf")
        for candidate_index, candidate_direction in enumerate(candidate_directions):
            score = 1000.0 - candidate_index * 10.0
            score -= self._occupied_direction_penalty(
                candidate_direction, occupied_directions
            )
            score -= self._tensor_clearance_penalty(
                tensor, component_tensors, candidate_direction
            )
            if score > best_score:
                best_score = score
                best_direction = candidate_direction
        return best_direction or CanvasPosition(x=0.0, y=-1.0)

    def _occupied_direction_penalty(
        self,
        candidate_direction: CanvasPosition,
        occupied_directions: Sequence[CanvasPosition],
    ) -> float:
        if not occupied_directions:
            return 0.0
        closest_alignment = max(
            _dot_product(candidate_direction, occupied_direction)
            for occupied_direction in occupied_directions
        )
        if closest_alignment <= 0.2:
            return 0.0
        return (closest_alignment - 0.2) * 500.0

    def _tensor_clearance_penalty(
        self,
        tensor: TensorSpec,
        component_tensors: Sequence[TensorSpec],
        candidate_direction: CanvasPosition,
    ) -> float:
        radius = 0.0 if self.is_port_tensor(tensor) else self.tensor_radius(tensor)
        segment_start = CanvasPosition(
            x=tensor.position.x + candidate_direction.x * radius,
            y=tensor.position.y + candidate_direction.y * radius,
        )
        segment_end = CanvasPosition(
            x=segment_start.x
            + candidate_direction.x * self.open_index_port_length(tensor),
            y=segment_start.y
            + candidate_direction.y * self.open_index_port_length(tensor),
        )
        penalty = 0.0
        for other_tensor in component_tensors:
            if other_tensor.id == tensor.id:
                continue
            distance = _point_to_segment_distance(
                other_tensor.position, segment_start, segment_end
            )
            clearance = distance - self.tensor_radius(other_tensor)
            if clearance < _FREE_INDEX_CLEARANCE_MARGIN:
                penalty += (_FREE_INDEX_CLEARANCE_MARGIN - clearance) * 8.0
        return penalty

    def _directional_index_hint(self, index: IndexSpec) -> CanvasPosition | None:
        normalized_name = index.name.strip().lower()
        return _FREE_INDEX_DIRECTION_HINTS.get(normalized_name)

    def _linear_component_candidates(
        self, component_tensors: Sequence[TensorSpec]
    ) -> list[CanvasPosition]:
        axis_direction = self._component_primary_axis(component_tensors)
        perpendicular_direction = CanvasPosition(
            x=-axis_direction.y, y=axis_direction.x
        )
        return [
            perpendicular_direction,
            CanvasPosition(x=-perpendicular_direction.x, y=-perpendicular_direction.y),
            axis_direction,
            CanvasPosition(x=-axis_direction.x, y=-axis_direction.y),
        ]

    def _circular_component_candidates(
        self, tensor: TensorSpec, component_tensors: Sequence[TensorSpec]
    ) -> list[CanvasPosition]:
        center = _average_position(
            [component_tensor.position for component_tensor in component_tensors]
        )
        radial_direction = _normalize_direction(
            CanvasPosition(
                x=tensor.position.x - center.x,
                y=tensor.position.y - center.y,
            )
        )
        perpendicular_direction = CanvasPosition(
            x=-radial_direction.y, y=radial_direction.x
        )
        return [
            radial_direction,
            perpendicular_direction,
            CanvasPosition(x=-perpendicular_direction.x, y=-perpendicular_direction.y),
            CanvasPosition(x=-radial_direction.x, y=-radial_direction.y),
        ]

    def _grid_component_candidates(
        self, tensor: TensorSpec, component_tensors: Sequence[TensorSpec]
    ) -> list[CanvasPosition]:
        basis_directions = self._grid_component_basis(component_tensors)
        if basis_directions is None:
            return []
        primary_axis, secondary_axis = basis_directions
        primary_projections = sorted(
            {
                round(
                    _dot_product(component_tensor.position, primary_axis),
                    6,
                )
                for component_tensor in component_tensors
            }
        )
        secondary_projections = sorted(
            {
                round(
                    _dot_product(component_tensor.position, secondary_axis),
                    6,
                )
                for component_tensor in component_tensors
            }
        )
        primary_projection = round(_dot_product(tensor.position, primary_axis), 6)
        secondary_projection = round(_dot_product(tensor.position, secondary_axis), 6)
        candidates: list[CanvasPosition] = []
        if (
            secondary_projections
            and abs(secondary_projection - secondary_projections[0]) < 1e-6
        ):
            candidates.append(CanvasPosition(x=-secondary_axis.x, y=-secondary_axis.y))
        if (
            secondary_projections
            and abs(secondary_projection - secondary_projections[-1]) < 1e-6
        ):
            candidates.append(secondary_axis)
        if (
            primary_projections
            and abs(primary_projection - primary_projections[0]) < 1e-6
        ):
            candidates.append(CanvasPosition(x=-primary_axis.x, y=-primary_axis.y))
        if (
            primary_projections
            and abs(primary_projection - primary_projections[-1]) < 1e-6
        ):
            candidates.append(primary_axis)
        if len(candidates) >= 2:
            diagonal = _normalize_direction(
                CanvasPosition(
                    x=sum(candidate.x for candidate in candidates),
                    y=sum(candidate.y for candidate in candidates),
                )
            )
            candidates.insert(0, diagonal)
        return candidates

    def _generic_component_candidates(
        self, tensor: TensorSpec, component_tensors: Sequence[TensorSpec]
    ) -> list[CanvasPosition]:
        neighbor_tensor_ids = self._adjacency_by_tensor_id.get(tensor.id, set())
        neighbor_tensors = [
            self._tensor_by_id[neighbor_tensor_id]
            for neighbor_tensor_id in neighbor_tensor_ids
            if neighbor_tensor_id in self._tensor_by_id
        ]
        candidates: list[CanvasPosition] = []
        if neighbor_tensors:
            neighbor_center = _average_position(
                [neighbor_tensor.position for neighbor_tensor in neighbor_tensors]
            )
            candidates.append(
                _normalize_direction(
                    CanvasPosition(
                        x=tensor.position.x - neighbor_center.x,
                        y=tensor.position.y - neighbor_center.y,
                    )
                )
            )
        component_center = _average_position(
            [component_tensor.position for component_tensor in component_tensors]
        )
        if (
            abs(component_center.x - tensor.position.x) > 1e-6
            or abs(component_center.y - tensor.position.y) > 1e-6
        ):
            candidates.append(
                _normalize_direction(
                    CanvasPosition(
                        x=tensor.position.x - component_center.x,
                        y=tensor.position.y - component_center.y,
                    )
                )
            )
        candidates.extend(
            sorted(
                [*_FREE_INDEX_CARDINAL_DIRECTIONS, *_FREE_INDEX_DIAGONAL_DIRECTIONS],
                key=lambda direction: self._generic_direction_sort_key(
                    tensor,
                    component_tensors,
                    direction,
                ),
            )
        )
        return candidates

    def _generic_direction_sort_key(
        self,
        tensor: TensorSpec,
        component_tensors: Sequence[TensorSpec],
        direction: CanvasPosition,
    ) -> tuple[float, float]:
        penalty = self._tensor_clearance_penalty(tensor, component_tensors, direction)
        return (penalty, -abs(direction.x) - abs(direction.y))

    def _classify_component_shape(self, component_tensors: Sequence[TensorSpec]) -> str:
        tensor_ids = {tensor.id for tensor in component_tensors}
        if len(tensor_ids) >= 2 and self._is_linear_component(tensor_ids):
            return "linear"
        if len(tensor_ids) >= 3 and self._is_circular_component(tensor_ids):
            return "circular"
        if len(tensor_ids) >= 4 and self._is_grid_component(component_tensors):
            return "grid2d"
        return "generic"

    def _is_linear_component(self, tensor_ids: set[str]) -> bool:
        component_tensors = [
            self._tensor_by_id[tensor_id]
            for tensor_id in tensor_ids
            if tensor_id in self._tensor_by_id
        ]
        degree_by_tensor_id = {
            tensor_id: len(self._adjacency_by_tensor_id.get(tensor_id, set()))
            for tensor_id in tensor_ids
        }
        edge_count = sum(degree_by_tensor_id.values()) // 2
        degree_one_count = sum(
            1 for degree in degree_by_tensor_id.values() if degree == 1
        )
        axis_direction = self._component_primary_axis(component_tensors)
        axis_origin = _average_position(
            [component_tensor.position for component_tensor in component_tensors]
        )
        projections = [
            _dot_product(
                CanvasPosition(
                    x=component_tensor.position.x - axis_origin.x,
                    y=component_tensor.position.y - axis_origin.y,
                ),
                axis_direction,
            )
            for component_tensor in component_tensors
        ]
        dominant_span = max(projections, default=0.0) - min(projections, default=0.0)
        minor_span = max(
            (
                abs(
                    _cross_product(
                        axis_direction,
                        CanvasPosition(
                            x=component_tensor.position.x - axis_origin.x,
                            y=component_tensor.position.y - axis_origin.y,
                        ),
                    )
                )
                for component_tensor in component_tensors
            ),
            default=0.0,
        )
        return (
            edge_count == len(tensor_ids) - 1
            and max(degree_by_tensor_id.values(), default=0) <= 2
            and degree_one_count == 2
            and (dominant_span <= 1e-6 or minor_span <= dominant_span * 0.18)
        )

    def _is_circular_component(self, tensor_ids: set[str]) -> bool:
        degree_by_tensor_id = {
            tensor_id: len(self._adjacency_by_tensor_id.get(tensor_id, set()))
            for tensor_id in tensor_ids
        }
        edge_count = sum(degree_by_tensor_id.values()) // 2
        return edge_count == len(tensor_ids) and all(
            degree == 2 for degree in degree_by_tensor_id.values()
        )

    def _is_grid_component(self, component_tensors: Sequence[TensorSpec]) -> bool:
        basis_directions = self._grid_component_basis(component_tensors)
        if basis_directions is None:
            return False
        primary_axis, secondary_axis = basis_directions
        primary_projections = {
            round(_dot_product(tensor.position, primary_axis), 6)
            for tensor in component_tensors
        }
        secondary_projections = {
            round(_dot_product(tensor.position, secondary_axis), 6)
            for tensor in component_tensors
        }
        if len(primary_projections) < 2 or len(secondary_projections) < 2:
            return False
        tensor_ids = {tensor.id for tensor in component_tensors}
        for edge in self._spec.edges:
            if (
                edge.left.tensor_id not in tensor_ids
                or edge.right.tensor_id not in tensor_ids
            ):
                continue
            left_tensor = self._tensor_by_id[edge.left.tensor_id]
            right_tensor = self._tensor_by_id[edge.right.tensor_id]
            edge_direction = _normalize_direction(
                CanvasPosition(
                    x=right_tensor.position.x - left_tensor.position.x,
                    y=right_tensor.position.y - left_tensor.position.y,
                )
            )
            if (
                abs(_dot_product(edge_direction, primary_axis)) < 0.94
                and abs(_dot_product(edge_direction, secondary_axis)) < 0.94
            ):
                return False
        return True

    def _component_primary_axis(
        self, component_tensors: Sequence[TensorSpec]
    ) -> CanvasPosition:
        if len(component_tensors) < 2:
            return CanvasPosition(x=0.0, y=-1.0)
        widest_pair: tuple[TensorSpec, TensorSpec] | None = None
        widest_distance = float("-inf")
        for left_tensor in component_tensors:
            for right_tensor in component_tensors:
                if left_tensor.id == right_tensor.id:
                    continue
                distance = hypot(
                    right_tensor.position.x - left_tensor.position.x,
                    right_tensor.position.y - left_tensor.position.y,
                )
                if distance > widest_distance:
                    widest_distance = distance
                    widest_pair = (left_tensor, right_tensor)
        if widest_pair is None:
            return CanvasPosition(x=0.0, y=-1.0)
        return _normalize_direction(
            CanvasPosition(
                x=widest_pair[1].position.x - widest_pair[0].position.x,
                y=widest_pair[1].position.y - widest_pair[0].position.y,
            )
        )

    def _grid_component_basis(
        self, component_tensors: Sequence[TensorSpec]
    ) -> tuple[CanvasPosition, CanvasPosition] | None:
        tensor_ids = {tensor.id for tensor in component_tensors}
        basis_directions: list[CanvasPosition] = []
        for edge in self._spec.edges:
            if (
                edge.left.tensor_id not in tensor_ids
                or edge.right.tensor_id not in tensor_ids
            ):
                continue
            left_tensor = self._tensor_by_id[edge.left.tensor_id]
            right_tensor = self._tensor_by_id[edge.right.tensor_id]
            direction = _canonicalize_undirected_direction(
                _normalize_direction(
                    CanvasPosition(
                        x=right_tensor.position.x - left_tensor.position.x,
                        y=right_tensor.position.y - left_tensor.position.y,
                    )
                )
            )
            if all(
                abs(_dot_product(direction, basis)) < 0.94 for basis in basis_directions
            ):
                basis_directions.append(direction)
        if len(basis_directions) != 2:
            return None
        if abs(_dot_product(basis_directions[0], basis_directions[1])) > 0.3:
            return None
        return basis_directions[0], basis_directions[1]

    def _build_tensor_adjacency(self) -> dict[str, set[str]]:
        adjacency_by_tensor_id: dict[str, set[str]] = {
            tensor.id: set() for tensor in self._spec.tensors
        }
        for edge in self._spec.edges:
            adjacency_by_tensor_id.setdefault(edge.left.tensor_id, set()).add(
                edge.right.tensor_id
            )
            adjacency_by_tensor_id.setdefault(edge.right.tensor_id, set()).add(
                edge.left.tensor_id
            )
        for hyperedge in self._spec.hyperedges:
            endpoint_tensor_ids = [
                endpoint.tensor_id
                for endpoint in hyperedge.endpoints
                if endpoint.tensor_id in self._tensor_by_id
            ]
            for left_tensor_id in endpoint_tensor_ids:
                for right_tensor_id in endpoint_tensor_ids:
                    if left_tensor_id == right_tensor_id:
                        continue
                    adjacency_by_tensor_id.setdefault(left_tensor_id, set()).add(
                        right_tensor_id
                    )
        return adjacency_by_tensor_id

    def _build_component_tensor_ids(self) -> dict[str, list[str]]:
        component_tensor_ids_by_tensor_id: dict[str, list[str]] = {}
        visited_tensor_ids: set[str] = set()
        for tensor in self._spec.tensors:
            if tensor.id in visited_tensor_ids:
                continue
            queue = [tensor.id]
            component_tensor_ids: list[str] = []
            visited_tensor_ids.add(tensor.id)
            while queue:
                current_tensor_id = queue.pop(0)
                component_tensor_ids.append(current_tensor_id)
                for neighbor_tensor_id in self._adjacency_by_tensor_id.get(
                    current_tensor_id, set()
                ):
                    if neighbor_tensor_id in visited_tensor_ids:
                        continue
                    visited_tensor_ids.add(neighbor_tensor_id)
                    queue.append(neighbor_tensor_id)
            for component_tensor_id in component_tensor_ids:
                component_tensor_ids_by_tensor_id[component_tensor_id] = (
                    component_tensor_ids
                )
        return component_tensor_ids_by_tensor_id

    def _component_tensors_for_tensor(self, tensor_id: str) -> list[TensorSpec]:
        component_tensor_ids = self._component_tensor_ids_by_tensor_id.get(
            tensor_id, [tensor_id]
        )
        return [
            self._tensor_by_id[component_tensor_id]
            for component_tensor_id in component_tensor_ids
            if component_tensor_id in self._tensor_by_id
        ]


class _TikzRenderer:
    """Small deterministic TikZ renderer for one validated network spec."""

    def __init__(self, spec: NetworkSpec, options: TikzRenderOptions) -> None:
        self._spec = spec
        self._options = options
        self._geometry = _SvgRenderer(spec, SvgRenderOptions())
        self._index_by_id = self._geometry._index_by_id

    def render(self) -> str:
        """Return the complete TikZ picture."""
        bounds = self._geometry._compute_bounds()
        world_width = max(bounds.width, 1.0)
        lines = [
            rf"\def\tneGlobalWidth{{{self._options.global_width}}}",
            rf"\pgfmathsetlengthmacro{{\tneUnit}}{{{_number(self._options.scale)}*\tneGlobalWidth/{_number(world_width)}}}",
            r"\pgfmathsetlengthmacro{\tneLineWidth}{0.6*\tneUnit}",
            r"\pgfmathsetlengthmacro{\tneHyperedgeSize}{9*\tneUnit}",
            *self._color_definitions(),
            r"\begin{tikzpicture}[x=\tneUnit, y=\tneUnit, line width=\tneLineWidth]",
            r"\tikzset{",
            r"  tne tensor/.style={draw, circle, fill=blue!12, align=center, inner sep=3*\tneUnit, font=\scriptsize},",
            r"  tne open index/.style={draw},",
            r"  tne index label/.style={font=\fontsize{4}{4.8}\selectfont, align=center},",
            r"  tne edge/.style={draw},",
            r"  tne edge label/.style={font=\fontsize{4}{4.8}\selectfont, fill=white, inner sep=1*\tneUnit},",
            r"  tne hyperedge/.style={draw, circle, fill=orange!70, inner sep=1.5*\tneUnit, minimum size=\tneHyperedgeSize},",
            r"  tne hyperedge spoke/.style={draw, dashed, orange!80, line width=0.5*\tneUnit},",
            r"  tne group/.style={draw, dashed, rounded corners=2*\tneUnit, gray},",
            r"  tne group label/.style={font=\fontsize{4}{4.8}\selectfont, gray, anchor=west},",
            r"  tne note/.style={draw, rounded corners=2*\tneUnit, fill=gray!10, align=left, anchor=north west, font=\fontsize{4.5}{5.4}\selectfont},",
            r"}",
        ]
        if self._options.include_groups:
            lines.extend(self._render_groups(bounds))
        lines.extend(self._render_edges(bounds))
        lines.extend(self._render_hyperedges(bounds))
        lines.extend(self._render_open_indices(bounds))
        lines.extend(self._render_tensors(bounds))
        lines.extend(self._render_index_labels(bounds))
        if self._options.include_notes:
            lines.extend(self._render_notes(bounds))
        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)

    def _render_groups(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        tensor_by_id = self._geometry._tensor_by_id
        for group in self._spec.groups:
            group_tensors = [
                tensor_by_id[tensor_id]
                for tensor_id in group.tensor_ids
                if tensor_id in tensor_by_id
            ]
            if not group_tensors:
                continue
            group_bounds = _tensor_collection_bounds(
                group_tensors, padding=_GROUP_PADDING
            )
            group_color = _metadata_color_or_none(group.metadata)
            group_style = "tne group"
            if group_color:
                group_style += f", draw={_tikz_color_name(group_color)}"
            lines.append(
                rf"\draw[{group_style}] "
                rf"{self._point(CanvasPosition(group_bounds.x1, group_bounds.y1), bounds)} "
                rf"rectangle {self._point(CanvasPosition(group_bounds.x2, group_bounds.y2), bounds)};"
            )
            lines.append(
                rf"\node[tne group label] at {self._point(CanvasPosition(group_bounds.x1 + 8.0, group_bounds.y1 + 14.0), bounds)} "
                rf"{{{_latex_text(group.name)}}};"
            )
        return lines

    def _render_edges(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for edge_info in self._geometry._edge_render_infos():
            draw_style = "tne edge"
            if _metadata_color_or_none(edge_info.edge.metadata):
                draw_style += f", draw={_tikz_color_name(edge_info.stroke)}"
            if edge_info.control is None:
                line = (
                    rf"\draw[{draw_style}] {self._point(edge_info.source, bounds)} "
                    rf"-- {self._point(edge_info.target, bounds)}"
                )
            else:
                line = (
                    rf"\draw[{draw_style}] {self._point(edge_info.source, bounds)} "
                    rf".. controls {self._point(edge_info.control, bounds)} .. "
                    rf"{self._point(edge_info.target, bounds)}"
                )
            if self._options.show_edge_labels and edge_info.edge.name:
                line += rf" node[midway, above, tne edge label] {{{_latex_text(edge_info.edge.name)}}}"
            lines.append(line + ";")
        return lines

    def _render_hyperedges(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hub_node_id = _tikz_node_id("hyperedge", hyperedge.id)
            hub_coordinate_id = _tikz_node_id("hyperedge_coord", hyperedge.id)
            hub = self._geometry._hyperedge_hub_position(hyperedge)
            hyperedge_color = _metadata_color(hyperedge.metadata, "#f08f45")
            hyperedge_custom_color = _metadata_color_or_none(hyperedge.metadata)
            hyperedge_style = "tne hyperedge"
            hyperedge_spoke_style = "tne hyperedge spoke"
            if hyperedge_custom_color:
                hyperedge_style += (
                    f", fill={_tikz_color_name(hyperedge_color)}, "
                    f"draw={_tikz_color_name(hyperedge_color)}"
                )
                hyperedge_spoke_style += f", draw={_tikz_color_name(hyperedge_color)}"
            spoke_lines: list[str] = []
            lines.append(
                rf"\coordinate ({hub_coordinate_id}) at {self._point(hub, bounds)};"
            )
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                tensor, _ = tensor_index
                endpoint_position = tensor.position
                spoke_lines.append(
                    rf"\draw[{hyperedge_spoke_style}] "
                    rf"{self._point(endpoint_position, bounds)} -- ({hub_coordinate_id});"
                )
            lines.extend(spoke_lines)
            lines.append(
                rf"\node[{hyperedge_style}] "
                rf"({hub_node_id}) at ({hub_coordinate_id}) {{}};"
            )
            if self._options.show_edge_labels and hyperedge.name:
                label_position = CanvasPosition(hub.x, hub.y - 17.0)
                lines.append(
                    rf"\node[tne edge label] at {self._point(label_position, bounds)} {{{_latex_text(hyperedge.name)}}};"
                )
        return lines

    def _render_tensors(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            if self._geometry.is_port_tensor(tensor):
                continue
            label = _latex_text(tensor.name) if self._options.show_tensor_labels else ""
            tensor_size = self._geometry.tensor_radius(tensor) * 2.0
            tensor_custom_color = _metadata_color_or_none(tensor.metadata)
            tensor_fill = tensor_custom_color or "#235f72"
            tensor_stroke = (
                _shift_hex_color(tensor_fill, 38, "#6fb7ca")
                if tensor_custom_color
                else "#6fb7ca"
            )
            tensor_text_fill = _readable_text_color(tensor_fill)
            lines.append(
                rf"\node[tne tensor, minimum size={self._length(tensor_size)}, "
                rf"fill={_tikz_color_name(tensor_fill)}, draw={_tikz_color_name(tensor_stroke)}, "
                rf"text={_tikz_color_name(tensor_text_fill)}] "
                rf"({_tikz_node_id('tensor', tensor.id)}) at {self._point(tensor.position, bounds)} "
                rf"{{{label}}};"
            )
        return lines

    def _render_open_indices(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if self._geometry.is_index_connected(index.id):
                    continue
                source = self._geometry.connection_point(tensor, index)
                target = self._geometry.open_index_endpoint(tensor, index)
                open_style = "tne open index"
                if index_color := _metadata_color_or_none(index.metadata):
                    open_style += f", draw={_tikz_color_name(index_color)}"
                lines.append(
                    rf"\draw[{open_style}] "
                    rf"{self._point(source, bounds)} -- {self._point(target, bounds)};"
                )
        return lines

    def _render_index_labels(self, bounds: _Bounds) -> list[str]:
        if not self._options.show_index_labels:
            return []
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                label_point = self._geometry.index_label_point(tensor, index)
                lines.append(
                    rf"\node[tne index label] at {self._point(label_point, bounds)} "
                    rf"{{{_latex_text(index.name)} {_latex_text(str(index.dimension))}}};"
                )
        return lines

    def _render_notes(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for note in self._spec.notes:
            lines.append(
                rf"\node[tne note, text width={self._length(_NOTE_WIDTH)}] "
                rf"({_tikz_node_id('note', note.id)}) at {self._point(note.position, bounds)} "
                rf"{{{_latex_text(note.text)}}};"
            )
        return lines

    def _point(self, point: CanvasPosition, bounds: _Bounds) -> str:
        return f"({_number(point.x - bounds.x1)}, {_number(bounds.y2 - point.y)})"

    def _length(self, value: float) -> str:
        return f"{_number(value)}*\\tneUnit"

    def _color_definitions(self) -> list[str]:
        colors: set[str] = set()
        for tensor in self._spec.tensors:
            if self._geometry.is_port_tensor(tensor):
                continue
            tensor_custom_color = _metadata_color_or_none(tensor.metadata)
            tensor_fill = tensor_custom_color or "#235f72"
            colors.add(tensor_fill)
            colors.add(
                _shift_hex_color(tensor_fill, 38, "#6fb7ca")
                if tensor_custom_color
                else "#6fb7ca"
            )
            colors.add(_readable_text_color(tensor_fill))
            for index in tensor.indices:
                if index_color := _metadata_color_or_none(index.metadata):
                    colors.add(index_color)
        for edge in self._spec.edges:
            if edge_color := _metadata_color_or_none(edge.metadata):
                colors.add(edge_color)
        for hyperedge in self._spec.hyperedges:
            if hyperedge_color := _metadata_color_or_none(hyperedge.metadata):
                colors.add(hyperedge_color)
        for group in self._spec.groups:
            if group_color := _metadata_color_or_none(group.metadata):
                colors.add(group_color)
        return [
            rf"\definecolor{{{_tikz_color_name(color)}}}{{HTML}}{{{color.removeprefix('#')}}}"
            for color in sorted(colors)
        ]


class _DotRenderer:
    """Small deterministic Graphviz/DOT renderer for one validated network spec."""

    def __init__(self, spec: NetworkSpec, options: DotRenderOptions) -> None:
        self._spec = spec
        self._options = options
        self._geometry = _SvgRenderer(spec, SvgRenderOptions())
        self._tensor_by_id = self._geometry._tensor_by_id
        self._index_by_id = self._geometry._index_by_id
        self._connected_index_ids = _connected_index_ids(spec)

    def render(self) -> str:
        """Return the complete DOT graph."""
        lines = [
            f"graph {_dot_string(self._spec.name or self._spec.id)} {{",
            '  graph [rankdir="LR"];',
            "  graph [splines=true];",
            '  node [fontname="Arial"];',
            '  edge [fontname="Arial"];',
        ]
        lines.extend(self._render_tensors())
        if self._options.include_open_indices:
            lines.extend(self._render_open_indices())
        lines.extend(self._render_edges())
        if self._options.include_hyperedges:
            lines.extend(self._render_hyperedges())
        if self._options.include_groups:
            lines.extend(self._render_groups())
        if self._options.include_notes:
            lines.extend(self._render_notes())
        lines.append("}")
        return "\n".join(lines)

    def _render_tensors(self) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            tensor_color = _metadata_color_or_none(tensor.metadata)
            attributes = _dot_attributes(
                label=_dot_tensor_label(tensor, self._options),
                shape="circle",
                style="filled" if tensor_color else None,
                fillcolor=tensor_color,
                color=(
                    _shift_hex_color(tensor_color, 38, "#6fb7ca")
                    if tensor_color
                    else None
                ),
                fontcolor=_readable_text_color(tensor_color) if tensor_color else None,
            )
            lines.append(f"  {_dot_string(tensor.id)}{attributes};")
        return lines

    def _render_open_indices(self) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if index.id in self._connected_index_ids:
                    continue
                open_node_id = _dot_open_index_id(index.id)
                lines.append(
                    f"  {_dot_string(open_node_id)} "
                    f'[label={_dot_string(_dot_index_label(index, self._options))}, shape="circle"];'
                )
                open_color = _metadata_color_or_none(index.metadata)
                open_attributes = _dot_attributes(
                    style="dotted",
                    color=open_color,
                )
                lines.append(
                    f"  {_dot_string(tensor.id)} -- {_dot_string(open_node_id)}"
                    f"{open_attributes};"
                )
        return lines

    def _render_edges(self) -> list[str]:
        lines: list[str] = []
        for edge in self._spec.edges:
            left_entry = self._index_by_id.get(edge.left.index_id)
            right_entry = self._index_by_id.get(edge.right.index_id)
            if left_entry is None or right_entry is None:
                continue
            left_tensor, left_index = left_entry
            right_tensor, _ = right_entry
            edge_label = _dot_edge_label(edge, left_index, self._options)
            edge_attributes = _dot_attributes(
                label=edge_label or None,
                color=_metadata_color_or_none(edge.metadata),
            )
            lines.append(
                f"  {_dot_string(left_tensor.id)} -- {_dot_string(right_tensor.id)}"
                f"{edge_attributes};"
            )
        return lines

    def _render_hyperedges(self) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hyperedge_color = _metadata_color_or_none(hyperedge.metadata)
            hyperedge_attributes = _dot_attributes(
                label=_dot_hyperedge_label(hyperedge, self._options),
                shape="point",
                color=hyperedge_color,
                fillcolor=hyperedge_color,
            )
            lines.append(f"  {_dot_string(hyperedge.id)}{hyperedge_attributes};")
            for endpoint in hyperedge.endpoints:
                endpoint_entry = self._index_by_id.get(endpoint.index_id)
                if endpoint_entry is None:
                    continue
                tensor, index = endpoint_entry
                endpoint_label = _dot_hyperedge_endpoint_label(index, self._options)
                edge_attributes = _dot_attributes(
                    label=endpoint_label or None,
                    color=hyperedge_color,
                )
                lines.append(
                    f"  {_dot_string(tensor.id)} -- {_dot_string(hyperedge.id)}"
                    f"{edge_attributes};"
                )
        return lines

    def _render_groups(self) -> list[str]:
        lines: list[str] = []
        for group in self._spec.groups:
            cluster_id = _dot_cluster_id(group.id)
            lines.append(f"  subgraph {cluster_id} {{")
            lines.append(f"    label={_dot_string(group.name)};")
            if group_color := _metadata_color_or_none(group.metadata):
                lines.append(f"    color={_dot_string(group_color)};")
            for tensor_id in group.tensor_ids:
                if tensor_id in self._tensor_by_id:
                    lines.append(f"    {_dot_string(tensor_id)};")
            lines.append("  }")
        return lines

    def _render_notes(self) -> list[str]:
        return [
            f'  {_dot_string(note.id)} [label={_dot_string(note.text)}, shape="note"];'
            for note in self._spec.notes
        ]


class _MermaidRenderer:
    """Small deterministic Mermaid renderer for one validated network spec."""

    def __init__(self, spec: NetworkSpec, options: DotRenderOptions) -> None:
        self._spec = spec
        self._options = options
        self._tensor_by_id = {tensor.id: tensor for tensor in spec.tensors}
        self._index_by_id = {
            index.id: (tensor, index)
            for tensor in spec.tensors
            for index in tensor.indices
        }
        self._connected_index_ids = _connected_index_ids(spec)

    def render(self) -> str:
        """Return the complete Mermaid flowchart."""
        lines = ["flowchart LR"]
        if self._options.include_notes:
            lines.extend(self._render_notes())
        lines.extend(self._render_tensors())
        if self._options.include_open_indices:
            lines.extend(self._render_open_indices())
        lines.extend(self._render_edges())
        if self._options.include_hyperedges:
            lines.extend(self._render_hyperedges())
        return "\n".join(lines)

    def _render_tensors(self) -> list[str]:
        grouped_tensor_ids: set[str] = set()
        lines: list[str] = []
        if self._options.include_groups:
            for group in self._spec.groups:
                group_tensor_ids = [
                    tensor_id
                    for tensor_id in group.tensor_ids
                    if tensor_id in self._tensor_by_id
                    and tensor_id not in grouped_tensor_ids
                ]
                if not group_tensor_ids:
                    continue
                lines.append(
                    f'    subgraph {_mermaid_group_id(group.id)}["{_mermaid_label(group.name)}"]'
                )
                for tensor_id in group_tensor_ids:
                    grouped_tensor_ids.add(tensor_id)
                    lines.append(
                        f"        {self._tensor_line(self._tensor_by_id[tensor_id])}"
                    )
                lines.append("    end")
        for tensor in self._spec.tensors:
            if tensor.id in grouped_tensor_ids:
                continue
            lines.append(f"    {self._tensor_line(tensor)}")
        return lines

    def _render_open_indices(self) -> list[str]:
        lines: list[str] = []
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if index.id in self._connected_index_ids:
                    continue
                open_node_id = _mermaid_open_index_id(index.id)
                lines.append(f'    {open_node_id}((" "))')
                lines.append(
                    f"    {_mermaid_tensor_id(tensor.id)}"
                    f"{_mermaid_edge_suffix(self._open_index_edge_label(index))}"
                    f"{open_node_id}"
                )
        return lines

    def _render_edges(self) -> list[str]:
        lines: list[str] = []
        for edge in self._spec.edges:
            left_entry = self._index_by_id.get(edge.left.index_id)
            right_entry = self._index_by_id.get(edge.right.index_id)
            if left_entry is None or right_entry is None:
                continue
            left_tensor, left_index = left_entry
            right_tensor, _ = right_entry
            lines.append(
                f"    {_mermaid_tensor_id(left_tensor.id)}"
                f"{_mermaid_edge_suffix(_dot_edge_label(edge, left_index, self._options))}"
                f"{_mermaid_tensor_id(right_tensor.id)}"
            )
        return lines

    def _render_hyperedges(self) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hyperedge_node_id = _mermaid_hyperedge_id(hyperedge.id)
            lines.append(
                f'    {hyperedge_node_id}["{_mermaid_label(self._hyperedge_label(hyperedge))}"]'
            )
            for endpoint in hyperedge.endpoints:
                endpoint_entry = self._index_by_id.get(endpoint.index_id)
                if endpoint_entry is None:
                    continue
                tensor, index = endpoint_entry
                lines.append(
                    f"    {_mermaid_tensor_id(tensor.id)}"
                    f"{_mermaid_edge_suffix(_dot_hyperedge_endpoint_label(index, self._options))}"
                    f"{hyperedge_node_id}"
                )
        return lines

    def _render_notes(self) -> list[str]:
        return [
            f"    %% Note: {_mermaid_comment_text(note.text)}"
            for note in self._spec.notes
        ]

    def _tensor_line(self, tensor: TensorSpec) -> str:
        return (
            f'{_mermaid_tensor_id(tensor.id)}["'
            f"{_mermaid_label(self._tensor_label(tensor))}"
            '"]'
        )

    def _tensor_label(self, tensor: TensorSpec) -> str:
        if self._options.show_tensor_labels:
            return tensor.name
        return tensor.id

    def _open_index_edge_label(self, index: IndexSpec) -> str:
        return _dot_index_label(index, self._options)

    def _hyperedge_label(self, hyperedge: HyperedgeSpec) -> str:
        dot_label = _dot_hyperedge_label(hyperedge, self._options)
        if dot_label:
            return dot_label
        return hyperedge.id


class _MatplotlibRenderer:
    """Matplotlib-backed academic renderer for one validated network spec."""

    _BASE_DPI = 100.0

    def __init__(
        self,
        spec: NetworkSpec,
        options: SvgRenderOptions,
        *,
        scale: float = 1.0,
    ) -> None:
        self._spec = spec
        self._options = options
        self._scale = scale
        self._geometry = _SvgRenderer(spec, options)
        self._tensor_by_id = self._geometry._tensor_by_id
        self._index_by_id = self._geometry._index_by_id
        self._font_families = _font_family_list(options.font_family)

    def render_svg(self) -> str:
        """Return the complete SVG document."""
        svg = self._render_document(file_format="svg").decode("utf-8")
        return _normalize_svg_document(svg)

    def render_png(self) -> bytes:
        """Return the complete PNG document as bytes."""
        return self._render_document(file_format="png")

    def render_pdf(self) -> bytes:
        """Return the complete vector PDF document as bytes."""
        return self._render_document(file_format="pdf")

    def _render_document(self, *, file_format: str) -> bytes:
        matplotlib_module, pyplot_module, patches_module, path_module = (
            _load_matplotlib_modules()
        )
        figure: Any | None = None
        with matplotlib_module.rc_context(self._rc_params()):
            try:
                (
                    figure,
                    axes,
                    edge_render_infos,
                    bounds,
                    canvas_width,
                    canvas_height,
                ) = self._build_figure(pyplot_module)
                self._draw_scene(
                    axes,
                    patches_module,
                    path_module,
                    edge_render_infos,
                    bounds,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                )
                buffer = BytesIO()
                assert figure is not None
                figure.savefig(buffer, **self._savefig_kwargs(file_format=file_format))
                return buffer.getvalue()
            finally:
                if figure is not None:
                    pyplot_module.close(figure)

    def _rc_params(self) -> dict[str, Any]:
        return {
            "font.family": list(self._font_families),
            "font.sans-serif": list(self._font_families),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }

    def _build_figure(
        self,
        pyplot_module: Any,
    ) -> tuple[Any, Any, list[_RenderedEdge], _Bounds, int, int]:
        edge_render_infos = self._geometry._edge_render_infos()
        bounds = self._geometry._compute_bounds(edge_render_infos)
        canvas_width = max(240, ceil(bounds.width))
        canvas_height = max(180, ceil(bounds.height))
        figure = pyplot_module.figure(
            figsize=(
                canvas_width / self._BASE_DPI,
                canvas_height / self._BASE_DPI,
            ),
            dpi=self._BASE_DPI,
            facecolor=self._options.background,
        )
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_facecolor(self._options.background)
        axes.set_xlim(bounds.x1, bounds.x1 + canvas_width)
        axes.set_ylim(bounds.y1 + canvas_height, bounds.y1)
        axes.set_aspect("equal", adjustable="box")
        axes.set_axis_off()
        return (
            figure,
            axes,
            edge_render_infos,
            bounds,
            canvas_width,
            canvas_height,
        )

    def _savefig_kwargs(self, *, file_format: str) -> dict[str, Any]:
        save_kwargs: dict[str, Any] = {
            "format": file_format,
            "bbox_inches": None,
            "pad_inches": 0.0,
            "facecolor": self._options.background,
            "edgecolor": self._options.background,
            "transparent": False,
            "metadata": self._render_metadata(file_format=file_format),
        }
        if file_format == "png":
            save_kwargs["dpi"] = self._BASE_DPI * self._scale
        return save_kwargs

    def _render_metadata(self, *, file_format: str) -> dict[str, Any]:
        if file_format == "svg":
            return {
                "Creator": "tensor-network-editor",
                "Date": None,
                "Title": self._spec.name,
            }
        if file_format == "png":
            return {
                "Software": "tensor-network-editor",
                "Title": self._spec.name,
            }
        return {"Creator": "tensor-network-editor", "Title": self._spec.name}

    def _draw_scene(
        self,
        axes: Any,
        patches_module: Any,
        path_module: Any,
        edge_render_infos: Sequence[_RenderedEdge],
        bounds: _Bounds,
        *,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        background = patches_module.Rectangle(
            (bounds.x1, bounds.y1),
            canvas_width,
            canvas_height,
            facecolor=self._options.background,
            edgecolor="none",
            zorder=-100,
        )
        axes.add_patch(background)
        if self._options.include_groups:
            self._render_groups(axes, patches_module)
        self._render_edges(axes, patches_module, path_module, edge_render_infos)
        self._render_hyperedges(axes, patches_module)
        self._render_open_indices(axes)
        self._render_tensors(axes, patches_module)
        self._render_index_labels(axes)
        if self._options.include_notes:
            self._render_notes(axes, patches_module)

    def _render_groups(self, axes: Any, patches_module: Any) -> None:
        for group in self._spec.groups:
            group_tensors = [
                self._tensor_by_id[tensor_id]
                for tensor_id in group.tensor_ids
                if tensor_id in self._tensor_by_id
            ]
            if not group_tensors:
                continue
            group_bounds = _tensor_collection_bounds(
                group_tensors,
                padding=_GROUP_PADDING,
            )
            axes.add_patch(
                patches_module.FancyBboxPatch(
                    (group_bounds.x1, group_bounds.y1),
                    group_bounds.width,
                    group_bounds.height,
                    boxstyle="round,pad=0,rounding_size=10",
                    facecolor="none",
                    edgecolor=_metadata_color(
                        group.metadata, self._options.group_stroke
                    ),
                    linewidth=1.5,
                    linestyle=(0, (8, 6)),
                    zorder=0.5,
                )
            )
            axes.text(
                group_bounds.x1 + 12,
                group_bounds.y1 + 20,
                group.name,
                color=self._options.muted_text_fill,
                fontsize=12,
                fontfamily=list(self._font_families),
                ha="left",
                va="center",
                zorder=3.0,
            )

    def _render_edges(
        self,
        axes: Any,
        patches_module: Any,
        path_module: Any,
        edge_render_infos: Sequence[_RenderedEdge],
    ) -> None:
        for edge_info in edge_render_infos:
            if edge_info.control is None:
                vertices = [
                    (edge_info.source.x, edge_info.source.y),
                    (edge_info.target.x, edge_info.target.y),
                ]
                codes = [path_module.Path.MOVETO, path_module.Path.LINETO]
                label_point = _midpoint(edge_info.source, edge_info.target)
            else:
                vertices = [
                    (edge_info.source.x, edge_info.source.y),
                    (edge_info.control.x, edge_info.control.y),
                    (edge_info.target.x, edge_info.target.y),
                ]
                codes = [
                    path_module.Path.MOVETO,
                    path_module.Path.CURVE3,
                    path_module.Path.CURVE3,
                ]
                label_point = _quadratic_midpoint(
                    edge_info.source,
                    edge_info.control,
                    edge_info.target,
                )
            axes.add_patch(
                patches_module.PathPatch(
                    path_module.Path(vertices, codes),
                    facecolor="none",
                    edgecolor=edge_info.stroke,
                    linewidth=2.8,
                    capstyle="round",
                    joinstyle="round",
                    zorder=1.0,
                )
            )
            if self._options.show_edge_labels and edge_info.edge.name:
                axes.text(
                    label_point.x,
                    label_point.y - 10,
                    edge_info.edge.name,
                    color=self._options.muted_text_fill,
                    fontsize=11,
                    fontfamily=list(self._font_families),
                    ha="center",
                    va="center",
                    zorder=3.0,
                )

    def _render_hyperedges(self, axes: Any, patches_module: Any) -> None:
        for hyperedge in self._spec.hyperedges:
            hub = self._geometry._hyperedge_hub_position(hyperedge)
            hyperedge_stroke = _metadata_color(
                hyperedge.metadata,
                self._options.hyperedge_stroke,
            )
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                tensor, _ = tensor_index
                endpoint_position = tensor.position
                axes.plot(
                    [endpoint_position.x, hub.x],
                    [endpoint_position.y, hub.y],
                    color=hyperedge_stroke,
                    linewidth=2.3,
                    linestyle=(0, (5, 4)),
                    solid_capstyle="round",
                    zorder=1.1,
                )
            axes.add_patch(
                patches_module.Circle(
                    (hub.x, hub.y),
                    radius=11.0,
                    facecolor=hyperedge_stroke,
                    edgecolor="none",
                    zorder=1.2,
                )
            )
            if self._options.show_edge_labels and hyperedge.name:
                axes.text(
                    hub.x,
                    hub.y - 17,
                    hyperedge.name,
                    color=self._options.text_fill,
                    fontsize=11,
                    fontfamily=list(self._font_families),
                    ha="center",
                    va="center",
                    zorder=3.0,
                )

    def _render_tensors(self, axes: Any, patches_module: Any) -> None:
        for tensor in self._spec.tensors:
            if self._geometry.is_port_tensor(tensor):
                continue
            radius = self._geometry.tensor_radius(tensor)
            tensor_custom_color = _metadata_color_or_none(tensor.metadata)
            tensor_fill = tensor_custom_color or self._options.tensor_fill
            tensor_stroke = (
                _shift_hex_color(tensor_fill, 38, self._options.tensor_stroke)
                if tensor_custom_color
                else self._options.tensor_stroke
            )
            tensor_text_fill = _readable_text_color(tensor_fill)
            axes.add_patch(
                patches_module.Circle(
                    (tensor.position.x, tensor.position.y),
                    radius=radius,
                    facecolor=tensor_fill,
                    edgecolor=tensor_stroke,
                    linewidth=2.0,
                    zorder=2.0,
                )
            )
            if self._options.show_tensor_labels:
                axes.text(
                    tensor.position.x,
                    tensor.position.y,
                    tensor.name,
                    color=tensor_text_fill,
                    fontsize=18,
                    fontfamily=list(self._font_families),
                    ha="center",
                    va="center",
                    zorder=3.0,
                )

    def _render_open_indices(self, axes: Any) -> None:
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if self._geometry.is_index_connected(index.id):
                    continue
                source = self._geometry.connection_point(tensor, index)
                target = self._geometry.open_index_endpoint(tensor, index)
                axes.plot(
                    [source.x, target.x],
                    [source.y, target.y],
                    color=_metadata_color(index.metadata, self._options.edge_stroke),
                    linewidth=2.8,
                    solid_capstyle="round",
                    zorder=1.0,
                )

    def _render_index_labels(self, axes: Any) -> None:
        if not self._options.show_index_labels:
            return
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                label_point = self._geometry.index_label_point(tensor, index)
                axes.text(
                    label_point.x,
                    label_point.y,
                    f"{index.name} {index.dimension}",
                    color=self._options.muted_text_fill,
                    fontsize=10,
                    fontfamily=list(self._font_families),
                    ha=_horizontal_alignment(
                        self._geometry._svg_text_anchor(tensor, index)
                    ),
                    va="center",
                    zorder=3.0,
                )

    def _render_notes(self, axes: Any, patches_module: Any) -> None:
        for note in self._spec.notes:
            axes.add_patch(
                patches_module.FancyBboxPatch(
                    (note.position.x, note.position.y),
                    _NOTE_WIDTH,
                    _NOTE_HEIGHT,
                    boxstyle="round,pad=0,rounding_size=8",
                    facecolor=_metadata_color(note.metadata, self._options.note_fill),
                    edgecolor=_metadata_color(
                        note.metadata, self._options.group_stroke
                    ),
                    linewidth=1.0,
                    zorder=0.8,
                )
            )
            for line_index, note_line in enumerate(_wrap_text(note.text, max_chars=32)):
                axes.text(
                    note.position.x + 12,
                    note.position.y + 24 + line_index * 16,
                    note_line,
                    color=self._options.text_fill,
                    fontsize=12,
                    fontfamily=list(self._font_families),
                    ha="left",
                    va="center",
                    zorder=3.0,
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


def _quadratic_midpoint(
    left: CanvasPosition,
    control: CanvasPosition,
    right: CanvasPosition,
) -> CanvasPosition:
    return CanvasPosition(
        x=(left.x + 2 * control.x + right.x) / 4,
        y=(left.y + 2 * control.y + right.y) / 4,
    )


def _parallel_edge_control_point(
    source: CanvasPosition,
    target: CanvasPosition,
    *,
    edge_position: int,
    edge_count: int,
) -> CanvasPosition:
    midpoint = _midpoint(source, target)
    dx = target.x - source.x
    dy = target.y - source.y
    length = hypot(dx, dy)
    if length <= 1e-6:
        angle = (2 * pi * edge_position) / max(1, edge_count)
        return CanvasPosition(
            x=midpoint.x + cos(angle) * _PARALLEL_EDGE_SPACING,
            y=midpoint.y + sin(angle) * _PARALLEL_EDGE_SPACING,
        )
    spacing = max(
        _PARALLEL_EDGE_SPACING,
        min(length * 0.24, 72.0),
    )
    offset = (edge_position - (edge_count - 1) / 2) * spacing
    return CanvasPosition(
        x=midpoint.x + (-dy / length) * offset,
        y=midpoint.y + (dx / length) * offset,
    )


def _normalize_direction(direction: CanvasPosition) -> CanvasPosition:
    magnitude = hypot(direction.x, direction.y)
    if magnitude <= 1e-6:
        return CanvasPosition(x=0.0, y=-1.0)
    return CanvasPosition(x=direction.x / magnitude, y=direction.y / magnitude)


def _dot_product(left: CanvasPosition, right: CanvasPosition) -> float:
    return left.x * right.x + left.y * right.y


def _cross_product(left: CanvasPosition, right: CanvasPosition) -> float:
    return left.x * right.y - left.y * right.x


def _canonicalize_undirected_direction(direction: CanvasPosition) -> CanvasPosition:
    if direction.x < -1e-6:
        return CanvasPosition(x=-direction.x, y=-direction.y)
    if abs(direction.x) <= 1e-6 and direction.y < -1e-6:
        return CanvasPosition(x=-direction.x, y=-direction.y)
    return direction


def _deduplicate_directions(
    directions: Iterable[CanvasPosition],
) -> list[CanvasPosition]:
    unique_directions: list[CanvasPosition] = []
    seen_keys: set[tuple[int, int]] = set()
    for direction in directions:
        normalized_direction = _normalize_direction(direction)
        direction_key = (
            round(normalized_direction.x * 1000),
            round(normalized_direction.y * 1000),
        )
        if direction_key in seen_keys:
            continue
        seen_keys.add(direction_key)
        unique_directions.append(normalized_direction)
    return unique_directions


def _point_to_segment_distance(
    point: CanvasPosition,
    segment_start: CanvasPosition,
    segment_end: CanvasPosition,
) -> float:
    segment_dx = segment_end.x - segment_start.x
    segment_dy = segment_end.y - segment_start.y
    segment_length_squared = segment_dx * segment_dx + segment_dy * segment_dy
    if segment_length_squared <= 1e-6:
        return hypot(point.x - segment_start.x, point.y - segment_start.y)
    projection = (
        ((point.x - segment_start.x) * segment_dx)
        + ((point.y - segment_start.y) * segment_dy)
    ) / segment_length_squared
    clamped_projection = min(1.0, max(0.0, projection))
    closest_point = CanvasPosition(
        x=segment_start.x + segment_dx * clamped_projection,
        y=segment_start.y + segment_dy * clamped_projection,
    )
    return hypot(point.x - closest_point.x, point.y - closest_point.y)


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


def _latex_text(value: str) -> str:
    normalized = " ".join(value.splitlines())
    escaped_characters: list[str] = []
    opening_quote = True
    for character in normalized:
        if character == "\\":
            escaped_characters.append(r"\textbackslash{}")
        elif character == "{":
            escaped_characters.append(r"\{")
        elif character == "}":
            escaped_characters.append(r"\}")
        elif character == "$":
            escaped_characters.append(r"\$")
        elif character == "&":
            escaped_characters.append(r"\&")
        elif character == "%":
            escaped_characters.append(r"\%")
        elif character == "#":
            escaped_characters.append(r"\#")
        elif character == "_":
            escaped_characters.append(r"\_")
        elif character == "~":
            escaped_characters.append(r"\textasciitilde{}")
        elif character == "^":
            escaped_characters.append(r"\textasciicircum{}")
        elif character == '"':
            escaped_characters.append("``" if opening_quote else "''")
            opening_quote = not opening_quote
        else:
            escaped_characters.append(character)
    return "".join(escaped_characters)


def _tikz_node_id(prefix: str, raw_id: str) -> str:
    return f"{prefix}_{_safe_identifier(raw_id)}"


def _safe_identifier(raw_id: str) -> str:
    characters = [
        character if character.isalnum() else "_" for character in raw_id.strip()
    ]
    identifier = "".join(characters).strip("_")
    return identifier or "item"


def _dot_string(value: str) -> str:
    return (
        '"'
        + value.replace("\\", r"\\")
        .replace('"', r"\"")
        .replace("\r\n", r"\n")
        .replace("\n", r"\n")
        .replace("\r", r"\n")
        + '"'
    )


def _dot_open_index_id(index_id: str) -> str:
    return f"open_{index_id}"


def _dot_cluster_id(group_id: str) -> str:
    return f"cluster_{_safe_identifier(group_id)}"


def _mermaid_tensor_id(tensor_id: str) -> str:
    return f"tensor_{_safe_identifier(tensor_id)}"


def _mermaid_open_index_id(index_id: str) -> str:
    return f"open_{_safe_identifier(index_id)}"


def _mermaid_hyperedge_id(hyperedge_id: str) -> str:
    return f"hyperedge_{_safe_identifier(hyperedge_id)}"


def _mermaid_group_id(group_id: str) -> str:
    return f"group_{_safe_identifier(group_id)}"


def _mermaid_label(value: str) -> str:
    return " ".join(value.splitlines()).replace("\\", "\\\\").replace('"', "&quot;")


def _mermaid_edge_suffix(label: str) -> str:
    if not label:
        return " --- "
    return f" ---|{_mermaid_label(label)}| "


def _mermaid_comment_text(value: str) -> str:
    return " ".join(value.splitlines())


def _dot_tensor_label(tensor: TensorSpec, options: DotRenderOptions) -> str:
    return tensor.name if options.show_tensor_labels else ""


def _dot_index_label(index: IndexSpec, options: DotRenderOptions) -> str:
    if not options.show_index_labels:
        return ""
    return f"{index.name} ({index.dimension})"


def _dot_edge_label(
    edge: EdgeSpec,
    left_index: IndexSpec,
    options: DotRenderOptions,
) -> str:
    labels: list[str] = []
    if options.show_edge_labels and edge.name:
        labels.append(edge.name)
    if options.show_index_labels:
        labels.append(f"{left_index.name}={left_index.dimension}")
    return " / ".join(labels)


def _dot_hyperedge_label(
    hyperedge: HyperedgeSpec,
    options: DotRenderOptions,
) -> str:
    if not options.show_edge_labels:
        return ""
    return hyperedge.name


def _dot_hyperedge_endpoint_label(
    index: IndexSpec,
    options: DotRenderOptions,
) -> str:
    if not options.show_index_labels:
        return ""
    return f"{index.name}={index.dimension}"


def _dot_attributes(
    *,
    label: str | None = None,
    shape: str | None = None,
    style: str | None = None,
    fillcolor: str | None = None,
    color: str | None = None,
    fontcolor: str | None = None,
) -> str:
    attributes: list[str] = []
    if label is not None:
        attributes.append(f"label={_dot_string(label)}")
    if shape:
        attributes.append(f"shape={_dot_string(shape)}")
    if style:
        attributes.append(f"style={_dot_string(style)}")
    if fillcolor:
        attributes.append(f"fillcolor={_dot_string(fillcolor)}")
    if color:
        attributes.append(f"color={_dot_string(color)}")
    if fontcolor:
        attributes.append(f"fontcolor={_dot_string(fontcolor)}")
    if not attributes:
        return ""
    return f" [{', '.join(attributes)}]"


def _connected_index_ids(spec: NetworkSpec) -> set[str]:
    connected_index_ids: set[str] = set()
    for edge in spec.edges:
        connected_index_ids.add(edge.left.index_id)
        connected_index_ids.add(edge.right.index_id)
    for hyperedge in spec.hyperedges:
        for endpoint in hyperedge.endpoints:
            connected_index_ids.add(endpoint.index_id)
    return connected_index_ids


def _metadata_color(metadata: dict[str, Any], fallback: str) -> str:
    return _metadata_color_or_none(metadata) or fallback


def _metadata_color_or_none(metadata: dict[str, Any]) -> str | None:
    candidate = metadata.get("color") if isinstance(metadata, dict) else None
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip().lower()
    if (
        len(normalized) == 7
        and normalized.startswith("#")
        and all(character in "0123456789abcdef" for character in normalized[1:])
    ):
        return normalized
    return None


def _shift_hex_color(hex_color: str, amount: int, fallback: str) -> str:
    normalized = _metadata_color({"color": hex_color}, fallback)
    try:
        red = int(normalized[1:3], 16)
        green = int(normalized[3:5], 16)
        blue = int(normalized[5:7], 16)
    except ValueError:
        return fallback
    return "#" + "".join(
        f"{max(0, min(255, component + amount)):02x}"
        for component in (red, green, blue)
    )


def _readable_text_color(hex_color: str) -> str:
    normalized = _metadata_color({"color": hex_color}, "#000000")
    try:
        red = int(normalized[1:3], 16)
        green = int(normalized[3:5], 16)
        blue = int(normalized[5:7], 16)
    except ValueError:
        return _LIGHT_TEXT_FILL
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    if luminance > _READABLE_TEXT_LUMINANCE_THRESHOLD:
        return _DARK_TEXT_FILL
    return _LIGHT_TEXT_FILL


def _tikz_color_name(hex_color: str) -> str:
    normalized = _metadata_color({"color": hex_color}, "#000000")
    return f"tneColor{normalized.removeprefix('#')}"


def _attr(value: object) -> str:
    return quoteattr(_number(value) if isinstance(value, (int, float)) else str(value))


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _font_family_list(font_family: str) -> tuple[str, ...]:
    families = tuple(item.strip() for item in font_family.split(",") if item.strip())
    if families:
        return families
    return ("DejaVu Sans", "sans-serif")


def _horizontal_alignment(svg_anchor: str) -> str:
    if svg_anchor == "start":
        return "left"
    if svg_anchor == "end":
        return "right"
    return "center"


def _load_matplotlib_modules() -> tuple[Any, Any, Any, Any]:
    """Import Matplotlib lazily for academic SVG/PNG/PDF exports."""
    try:
        matplotlib_module = import_module("matplotlib")
        matplotlib_module.use("Agg", force=True)
        return (
            matplotlib_module,
            import_module("matplotlib.pyplot"),
            import_module("matplotlib.patches"),
            import_module("matplotlib.path"),
        )
    except ImportError as exc:
        raise RuntimeError(
            "PNG/SVG/PDF rendering requires Matplotlib. "
            "Reinstall the package or add Matplotlib to the current environment."
        ) from exc


def _normalize_svg_document(svg: str) -> str:
    normalized = svg.lstrip("\ufeff")
    if normalized.startswith("<?xml"):
        first_line, separator, remainder = normalized.partition("\n")
        if separator:
            return "\n".join(['<?xml version="1.0" encoding="UTF-8"?>', remainder])
        return '<?xml version="1.0" encoding="UTF-8"?>'
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + normalized


__all__ = [
    "DotRenderOptions",
    "SvgRenderOptions",
    "TikzRenderOptions",
    "render_spec_dot",
    "render_spec_mermaid",
    "render_spec_pdf",
    "render_spec_png",
    "render_spec_svg",
    "render_spec_tikz",
]
