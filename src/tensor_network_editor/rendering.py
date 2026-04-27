"""Public helpers for rendering tensor-network specs as static SVG."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from importlib import import_module
from io import BytesIO
from math import ceil, cos, hypot, isfinite, pi, sin
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

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

_INDEX_RADIUS = 10.0
_GROUP_PADDING = 28.0
_NOTE_WIDTH = 210.0
_NOTE_HEIGHT = 82.0


@dataclass(slots=True, frozen=True)
class SvgRenderOptions:
    """Options for the static SVG renderer."""

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


def render_spec_png(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    scale: float = 2.0,
    output_path: StrPath | None = None,
) -> bytes:
    """Render one tensor-network specification as PNG bytes using Pillow."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("PNG render scale must be a positive finite number.")
    if not isfinite(float(scale)) or scale <= 0:
        raise ValueError("PNG render scale must be a positive finite number.")
    resolved_options = options or SvgRenderOptions()
    validated_spec = ensure_valid_spec(spec)
    png = _PngRenderer(validated_spec, resolved_options, scale=float(scale)).render()
    if output_path is not None:
        write_binary(output_path, png, description="PNG network rendering")
    return png


def render_spec_pdf(
    spec: NetworkSpec,
    *,
    options: SvgRenderOptions | None = None,
    scale: float = 2.0,
    output_path: StrPath | None = None,
) -> bytes:
    """Render one tensor-network specification as PDF bytes using Pillow."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("PDF render scale must be a positive finite number.")
    if not isfinite(float(scale)) or scale <= 0:
        raise ValueError("PDF render scale must be a positive finite number.")
    resolved_options = options or SvgRenderOptions()
    validated_spec = ensure_valid_spec(spec)
    pdf = _PngRenderer(
        validated_spec, resolved_options, scale=float(scale)
    ).render_pdf()
    if output_path is not None:
        write_binary(output_path, pdf, description="PDF network rendering")
    return pdf


def render_spec_tikz(
    spec: NetworkSpec,
    *,
    options: TikzRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a standalone TikZ picture."""
    resolved_options = options or TikzRenderOptions()
    if (
        isinstance(resolved_options.scale, bool)
        or not isinstance(resolved_options.scale, (int, float))
        or not isfinite(float(resolved_options.scale))
        or resolved_options.scale <= 0
    ):
        raise ValueError("TikZ render scale must be a positive finite number.")
    validated_spec = ensure_valid_spec(spec)
    tikz = _TikzRenderer(validated_spec, resolved_options).render()
    if output_path is not None:
        write_utf8_text(output_path, tikz, description="TikZ network rendering")
    return tikz


def render_spec_dot(
    spec: NetworkSpec,
    *,
    options: DotRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    """Render one tensor-network specification as a Graphviz/DOT graph."""
    resolved_options = options or DotRenderOptions()
    validated_spec = ensure_valid_spec(spec)
    dot = _DotRenderer(validated_spec, resolved_options).render()
    if output_path is not None:
        write_utf8_text(output_path, dot, description="Graphviz/DOT network rendering")
    return dot


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
        self._index_order_by_id = {
            index.id: index_position
            for tensor in spec.tensors
            for index_position, index in enumerate(tensor.indices)
        }
        self._connected_index_ids = _connected_index_ids(spec)

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
        if self._options.include_groups:
            lines.extend(self._render_groups())
        lines.extend(self._render_edges())
        lines.extend(self._render_hyperedges())
        lines.extend(self._render_open_indices())
        lines.extend(self._render_tensors())
        lines.extend(self._render_index_labels())
        if self._options.include_notes:
            lines.extend(self._render_notes())
        lines.append("</svg>")
        return "\n".join(lines)

    def _compute_bounds(self) -> _Bounds:
        points: list[CanvasPosition] = []
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
                endpoint_position = self.connection_point(*tensor_index)
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
            if self.is_port_tensor(tensor):
                continue
            radius = self.tensor_radius(tensor)
            lines.append(
                f'<circle class="tensor" cx={_attr(tensor.position.x)} cy={_attr(tensor.position.y)} '
                f"r={_attr(radius)} fill={_attr(self._options.tensor_fill)} "
                f'stroke={_attr(self._options.tensor_stroke)} stroke-width="2" />'
            )
            if self._options.show_tensor_labels:
                lines.append(
                    f'<text class="tensor-label" x={_attr(tensor.position.x)} '
                    f"y={_attr(tensor.position.y + 6)} fill={_attr(self._options.text_fill)} "
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
                    f'stroke={_attr(self._options.edge_stroke)} stroke-width="3" />'
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
        return self.connection_point(*left), self.connection_point(*right)

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
        port_length: float = 34.0,
    ) -> CanvasPosition:
        direction = self._index_direction(tensor, index)
        source = self.connection_point(tensor, index)
        return CanvasPosition(
            x=source.x + direction.x * port_length,
            y=source.y + direction.y * port_length,
        )

    def index_label_point(self, tensor: TensorSpec, index: IndexSpec) -> CanvasPosition:
        direction = self._index_direction(tensor, index)
        source = self.connection_point(tensor, index)
        label_distance = 18.0 if self.is_index_connected(index.id) else 24.0
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
        magnitude = hypot(index.offset.x, index.offset.y)
        if magnitude > 1e-6:
            return CanvasPosition(
                x=index.offset.x / magnitude, y=index.offset.y / magnitude
            )
        index_order = self._index_order_by_id.get(index.id, 0)
        index_count = max(1, len(tensor.indices))
        angle = -pi / 2 + (2 * pi * index_order / index_count)
        return CanvasPosition(x=cos(angle), y=sin(angle))


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
        lines.extend(self._render_tensors(bounds))
        lines.extend(self._render_edges(bounds))
        lines.extend(self._render_hyperedges(bounds))
        lines.extend(self._render_open_indices(bounds))
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
            lines.append(
                rf"\draw[tne group] {self._point(CanvasPosition(group_bounds.x1, group_bounds.y1), bounds)} "
                rf"rectangle {self._point(CanvasPosition(group_bounds.x2, group_bounds.y2), bounds)};"
            )
            lines.append(
                rf"\node[tne group label] at {self._point(CanvasPosition(group_bounds.x1 + 8.0, group_bounds.y1 + 14.0), bounds)} "
                rf"{{{_latex_text(group.name)}}};"
            )
        return lines

    def _render_edges(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for edge in self._spec.edges:
            left_entry = self._index_by_id.get(edge.left.index_id)
            right_entry = self._index_by_id.get(edge.right.index_id)
            if left_entry is None or right_entry is None:
                continue
            source = self._geometry.connection_point(*left_entry)
            target = self._geometry.connection_point(*right_entry)
            line = rf"\draw[tne edge] {self._point(source, bounds)} -- {self._point(target, bounds)}"
            if self._options.show_edge_labels and edge.name:
                line += rf" node[midway, above, tne edge label] {{{_latex_text(edge.name)}}}"
            lines.append(line + ";")
        return lines

    def _render_hyperedges(self, bounds: _Bounds) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            hub_node_id = _tikz_node_id("hyperedge", hyperedge.id)
            hub = self._geometry._hyperedge_hub_position(hyperedge)
            lines.append(
                rf"\node[tne hyperedge] ({hub_node_id}) at {self._point(hub, bounds)} {{}};"
            )
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                endpoint_position = self._geometry.connection_point(*tensor_index)
                lines.append(
                    rf"\draw[tne hyperedge spoke] {self._point(endpoint_position, bounds)} -- ({hub_node_id});"
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
            lines.append(
                rf"\node[tne tensor, minimum size={self._length(tensor_size)}] "
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
                lines.append(
                    rf"\draw[tne open index] {self._point(source, bounds)} -- {self._point(target, bounds)};"
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
        return [
            f"  {_dot_string(tensor.id)} "
            f'[label={_dot_string(_dot_tensor_label(tensor, self._options))}, shape="circle"];'
            for tensor in self._spec.tensors
        ]

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
                lines.append(
                    f'  {_dot_string(tensor.id)} -- {_dot_string(open_node_id)} [style="dotted"];'
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
            edge_attributes = _dot_attributes(
                label=_dot_edge_label(edge, left_index, self._options)
            )
            lines.append(
                f"  {_dot_string(left_tensor.id)} -- {_dot_string(right_tensor.id)}"
                f"{edge_attributes};"
            )
        return lines

    def _render_hyperedges(self) -> list[str]:
        lines: list[str] = []
        for hyperedge in self._spec.hyperedges:
            lines.append(
                f"  {_dot_string(hyperedge.id)} "
                f'[label={_dot_string(_dot_hyperedge_label(hyperedge, self._options))}, shape="point"];'
            )
            for endpoint in hyperedge.endpoints:
                endpoint_entry = self._index_by_id.get(endpoint.index_id)
                if endpoint_entry is None:
                    continue
                tensor, index = endpoint_entry
                edge_attributes = _dot_attributes(
                    label=_dot_hyperedge_endpoint_label(index, self._options)
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


class _PngRenderer:
    """Small deterministic Pillow renderer for one validated network spec."""

    def __init__(
        self,
        spec: NetworkSpec,
        options: SvgRenderOptions,
        *,
        scale: float,
    ) -> None:
        self._spec = spec
        self._options = options
        self._scale = scale
        self._geometry = _SvgRenderer(spec, options)
        self._tensor_by_id = self._geometry._tensor_by_id
        self._index_by_id = self._geometry._index_by_id

    def render(self) -> bytes:
        """Return the complete PNG document as bytes."""
        image = self._render_image()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def render_pdf(self) -> bytes:
        """Return the complete PDF document as bytes."""
        image = self._render_image()
        buffer = BytesIO()
        image.save(buffer, format="PDF", resolution=72.0 * self._scale)
        return buffer.getvalue()

    def _render_image(self) -> Any:
        """Return the rendered Pillow image."""
        image_module, draw_module, font_module = _load_pillow_modules()
        bounds = self._geometry._compute_bounds()
        width = max(240, ceil(bounds.width))
        height = max(180, ceil(bounds.height))
        pixel_size = (ceil(width * self._scale), ceil(height * self._scale))
        image = image_module.new("RGB", pixel_size, self._options.background)
        draw = draw_module.Draw(image)
        fonts = _PillowFontBundle(font_module, scale=self._scale)
        if self._options.include_groups:
            self._render_groups(draw, bounds, fonts.group)
        self._render_edges(draw, bounds, fonts.edge)
        self._render_hyperedges(draw, bounds, fonts.edge)
        self._render_open_indices(draw, bounds)
        self._render_tensors(draw, bounds, fonts.tensor)
        self._render_index_labels(draw, bounds, fonts.index)
        if self._options.include_notes:
            self._render_notes(draw, bounds, fonts.note)
        return image

    def _render_groups(self, draw: Any, bounds: _Bounds, font: Any) -> None:
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
            draw.rounded_rectangle(
                self._box_to_pixels(group_bounds, bounds),
                radius=self._pixels(10.0),
                outline=self._options.group_stroke,
                width=self._pixels(2.0),
            )
            self._draw_text(
                draw,
                self._point_to_pixels(
                    CanvasPosition(group_bounds.x1 + 12, group_bounds.y1 + 14),
                    bounds,
                ),
                group.name,
                fill=self._options.muted_text_fill,
                font=font,
                anchor="lm",
            )

    def _render_edges(self, draw: Any, bounds: _Bounds, font: Any) -> None:
        for edge in self._spec.edges:
            endpoints = self._geometry._edge_positions(edge)
            if endpoints is None:
                continue
            source, target = endpoints
            draw.line(
                [
                    self._point_to_pixels(source, bounds),
                    self._point_to_pixels(target, bounds),
                ],
                fill=self._options.edge_stroke,
                width=self._pixels(3.0),
            )
            if self._options.show_edge_labels and edge.name:
                midpoint = _midpoint(source, target)
                self._draw_text(
                    draw,
                    self._point_to_pixels(
                        CanvasPosition(midpoint.x, midpoint.y - 10),
                        bounds,
                    ),
                    edge.name,
                    fill=self._options.muted_text_fill,
                    font=font,
                )

    def _render_hyperedges(self, draw: Any, bounds: _Bounds, font: Any) -> None:
        for hyperedge in self._spec.hyperedges:
            hub = self._geometry._hyperedge_hub_position(hyperedge)
            hub_point = self._point_to_pixels(hub, bounds)
            for endpoint in hyperedge.endpoints:
                tensor_index = self._index_by_id.get(endpoint.index_id)
                if tensor_index is None:
                    continue
                endpoint_position = self._geometry.connection_point(*tensor_index)
                draw.line(
                    [
                        self._point_to_pixels(endpoint_position, bounds),
                        hub_point,
                    ],
                    fill=self._options.hyperedge_stroke,
                    width=self._pixels(2.5),
                )
            hub_radius = self._pixels(11.0)
            draw.ellipse(
                _pixel_circle_box(hub_point, hub_radius),
                fill=self._options.hyperedge_stroke,
            )
            if self._options.show_edge_labels and hyperedge.name:
                self._draw_text(
                    draw,
                    self._point_to_pixels(CanvasPosition(hub.x, hub.y - 17), bounds),
                    hyperedge.name,
                    fill=self._options.text_fill,
                    font=font,
                )

    def _render_tensors(self, draw: Any, bounds: _Bounds, font: Any) -> None:
        for tensor in self._spec.tensors:
            if self._geometry.is_port_tensor(tensor):
                continue
            radius = self._geometry.tensor_radius(tensor)
            pixel_point = self._point_to_pixels(tensor.position, bounds)
            pixel_radius = self._pixels(radius)
            draw.ellipse(
                _pixel_circle_box(pixel_point, pixel_radius),
                fill=self._options.tensor_fill,
                outline=self._options.tensor_stroke,
                width=self._pixels(2.0),
            )
            if self._options.show_tensor_labels:
                self._draw_text(
                    draw,
                    pixel_point,
                    tensor.name,
                    fill=self._options.text_fill,
                    font=font,
                )

    def _render_open_indices(self, draw: Any, bounds: _Bounds) -> None:
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                if self._geometry.is_index_connected(index.id):
                    continue
                source = self._geometry.connection_point(tensor, index)
                target = self._geometry.open_index_endpoint(tensor, index)
                draw.line(
                    [
                        self._point_to_pixels(source, bounds),
                        self._point_to_pixels(target, bounds),
                    ],
                    fill=self._options.edge_stroke,
                    width=self._pixels(3.0),
                )

    def _render_index_labels(self, draw: Any, bounds: _Bounds, font: Any) -> None:
        if not self._options.show_index_labels:
            return
        for tensor in self._spec.tensors:
            for index in tensor.indices:
                self._draw_text(
                    draw,
                    self._point_to_pixels(
                        self._geometry.index_label_point(tensor, index),
                        bounds,
                    ),
                    f"{index.name} {index.dimension}",
                    fill=self._options.muted_text_fill,
                    font=font,
                )

    def _render_notes(self, draw: Any, bounds: _Bounds, font: Any) -> None:
        for note in self._spec.notes:
            note_bounds = _Bounds(
                x1=note.position.x,
                y1=note.position.y,
                x2=note.position.x + _NOTE_WIDTH,
                y2=note.position.y + _NOTE_HEIGHT,
            )
            draw.rounded_rectangle(
                self._box_to_pixels(note_bounds, bounds),
                radius=self._pixels(8.0),
                fill=self._options.note_fill,
                outline=self._options.group_stroke,
                width=self._pixels(1.0),
            )
            for line_index, note_line in enumerate(_wrap_text(note.text, max_chars=32)):
                self._draw_text(
                    draw,
                    self._point_to_pixels(
                        CanvasPosition(
                            note.position.x + 12,
                            note.position.y + 20 + line_index * 16,
                        ),
                        bounds,
                    ),
                    note_line,
                    fill=self._options.text_fill,
                    font=font,
                    anchor="lm",
                )

    def _point_to_pixels(
        self,
        point: CanvasPosition,
        bounds: _Bounds,
    ) -> tuple[int, int]:
        return (
            self._pixels(point.x - bounds.x1),
            self._pixels(point.y - bounds.y1),
        )

    def _box_to_pixels(
        self,
        box: _Bounds,
        bounds: _Bounds,
    ) -> tuple[int, int, int, int]:
        return (
            self._pixels(box.x1 - bounds.x1),
            self._pixels(box.y1 - bounds.y1),
            self._pixels(box.x2 - bounds.x1),
            self._pixels(box.y2 - bounds.y1),
        )

    def _pixels(self, value: float) -> int:
        return max(1, int(round(value * self._scale)))

    @staticmethod
    def _draw_text(
        draw: Any,
        xy: tuple[int, int],
        text: str,
        *,
        fill: str,
        font: Any,
        anchor: str = "mm",
    ) -> None:
        try:
            draw.text(xy, text, fill=fill, font=font, anchor=anchor)
        except TypeError:
            draw.text(xy, text, fill=fill, font=font)


@dataclass(slots=True, frozen=True)
class _PillowFontBundle:
    """Scaled Pillow fonts for the academic raster renderers."""

    tensor: Any
    index: Any
    edge: Any
    group: Any
    note: Any

    def __init__(self, font_module: Any, *, scale: float) -> None:
        object.__setattr__(
            self,
            "tensor",
            _load_pillow_font(font_module, pixel_size=max(22, int(round(26 * scale)))),
        )
        object.__setattr__(
            self,
            "index",
            _load_pillow_font(font_module, pixel_size=max(16, int(round(18 * scale)))),
        )
        object.__setattr__(
            self,
            "edge",
            _load_pillow_font(font_module, pixel_size=max(16, int(round(18 * scale)))),
        )
        object.__setattr__(
            self,
            "group",
            _load_pillow_font(font_module, pixel_size=max(16, int(round(18 * scale)))),
        )
        object.__setattr__(
            self,
            "note",
            _load_pillow_font(font_module, pixel_size=max(16, int(round(18 * scale)))),
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


def _dot_attributes(*, label: str) -> str:
    if not label:
        return ""
    return f" [label={_dot_string(label)}]"


def _connected_index_ids(spec: NetworkSpec) -> set[str]:
    connected_index_ids: set[str] = set()
    for edge in spec.edges:
        connected_index_ids.add(edge.left.index_id)
        connected_index_ids.add(edge.right.index_id)
    for hyperedge in spec.hyperedges:
        for endpoint in hyperedge.endpoints:
            connected_index_ids.add(endpoint.index_id)
    return connected_index_ids


def _attr(value: object) -> str:
    return quoteattr(_number(value) if isinstance(value, (int, float)) else str(value))


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _pixel_circle_box(
    center: tuple[int, int],
    radius: int,
) -> tuple[int, int, int, int]:
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def _load_pillow_modules() -> tuple[Any, Any, Any]:
    """Import Pillow lazily so SVG rendering keeps zero dependencies."""
    try:
        return (
            import_module("PIL.Image"),
            import_module("PIL.ImageDraw"),
            import_module("PIL.ImageFont"),
        )
    except ImportError as exc:
        raise RuntimeError(
            "PNG/PDF rendering requires Pillow. "
            "Reinstall the package or add Pillow to the current environment."
        ) from exc


def _load_pillow_font(font_module: Any, *, pixel_size: int) -> Any:
    """Load a scalable Pillow font, falling back gracefully when unavailable."""
    package_font_path = (
        Path(font_module.__file__).resolve().parent / "Fonts" / "DejaVuSans.ttf"
    )
    for font_name in (
        str(package_font_path),
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    ):
        try:
            return font_module.truetype(font_name, pixel_size)
        except (AttributeError, OSError):
            continue
    try:
        return font_module.load_default(size=pixel_size)
    except TypeError:
        return font_module.load_default()


__all__ = [
    "DotRenderOptions",
    "SvgRenderOptions",
    "TikzRenderOptions",
    "render_spec_dot",
    "render_spec_pdf",
    "render_spec_png",
    "render_spec_svg",
    "render_spec_tikz",
]
