export function registerExportMinimap(ctx) {
  const state = ctx.state;
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  } = ctx.constants;
  const {
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportPyButton,
    exportPngButton,
    exportSvgButton,
    templateSelect,
    insertTemplateButton,
    createGroupButton,
    helpButton,
    helpModal,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    groupLayer,
    resizeLayer,
    selectionBox,
    minimapCanvas,
  } = ctx.dom;
  const { apiGet, apiPost, window, document, cytoscape } = ctx;

  function handleMinimapMouseDown(event) {
    event.preventDefault();
    event.stopPropagation();
    if (!state.minimapTransform) {
      return;
    }
    state.minimapDrag = { active: true };
    minimapCanvas.classList.add("is-dragging");
    updateViewportFromMinimapClientPoint(event.clientX, event.clientY);
  }

  function updateViewportFromMinimapClientPoint(clientX, clientY) {
    if (!state.minimapTransform || !state.cy) {
      return;
    }
    const rect = minimapCanvas.getBoundingClientRect();
    const localX = ctx.clamp(clientX - rect.left, 0, rect.width);
    const localY = ctx.clamp(clientY - rect.top, 0, rect.height);
    const transform = state.minimapTransform;
    const modelPoint = {
      x: (localX - transform.offsetX) / transform.scale + transform.bounds.x1,
      y: (localY - transform.offsetY) / transform.scale + transform.bounds.y1,
    };
    centerViewportAt(modelPoint);
  }

  function centerViewportAt(point) {
    if (!state.cy) {
      return;
    }
    const zoom = state.cy.zoom();
    state.cy.pan({
      x: state.cy.width() / 2 - point.x * zoom,
      y: state.cy.height() / 2 - point.y * zoom,
    });
    renderMinimap();
  }

  function renderMinimap() {
    const context = minimapCanvas.getContext("2d");
    if (!context) {
      return;
    }
    const canvasWidth = minimapCanvas.width;
    const canvasHeight = minimapCanvas.height;
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    context.fillStyle = "#0d121b";
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    if (!state.spec || !state.spec.tensors.length) {
      context.fillStyle = "#95a3b8";
      context.font = '12px "Segoe UI", "Helvetica Neue", sans-serif';
      context.textAlign = "center";
      context.fillText("Minimap will appear here.", canvasWidth / 2, canvasHeight / 2);
      state.minimapTransform = null;
      return;
    }

    const worldBounds = ctx.computeDesignBounds(48);
    const innerWidth = canvasWidth - 16;
    const innerHeight = canvasHeight - 16;
    const scale = Math.min(innerWidth / Math.max(1, worldBounds.x2 - worldBounds.x1), innerHeight / Math.max(1, worldBounds.y2 - worldBounds.y1));
    const drawnWidth = (worldBounds.x2 - worldBounds.x1) * scale;
    const drawnHeight = (worldBounds.y2 - worldBounds.y1) * scale;
    const offsetX = (canvasWidth - drawnWidth) / 2;
    const offsetY = (canvasHeight - drawnHeight) / 2;

    state.minimapTransform = {
      bounds: worldBounds,
      scale,
      offsetX,
      offsetY,
    };

    context.save();
    context.translate(offsetX, offsetY);
    context.scale(scale, scale);

    state.spec.edges.forEach((edge) => {
      const left = ctx.findIndexOwner(edge.left.index_id);
      const right = ctx.findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = ctx.indexAbsolutePosition(left.tensor, left.index);
      const target = ctx.indexAbsolutePosition(right.tensor, right.index);
      const curve = ctx.buildQuadraticCurve(source, target);
      context.beginPath();
      context.strokeStyle = state.selectionIds.includes(edge.id) ? "#8bc2ff" : ctx.getMetadataColor(edge.metadata, "#8da1c3");
      context.lineWidth = 3 / scale;
      context.moveTo(source.x - worldBounds.x1, source.y - worldBounds.y1);
      context.quadraticCurveTo(
        curve.control.x - worldBounds.x1,
        curve.control.y - worldBounds.y1,
        target.x - worldBounds.x1,
        target.y - worldBounds.y1
      );
      context.stroke();
    });

    state.spec.tensors.forEach((tensor) => {
      const tensorColor = ctx.getMetadataColor(tensor.metadata, "#18212c");
      const left = tensor.position.x - ctx.tensorWidth(tensor) / 2 - worldBounds.x1;
      const top = tensor.position.y - ctx.tensorHeight(tensor) / 2 - worldBounds.y1;
      ctx.drawRoundRectPath(context, left, top, ctx.tensorWidth(tensor), ctx.tensorHeight(tensor), 22);
      context.fillStyle = tensorColor;
      context.fill();
      context.lineWidth = (state.selectionIds.includes(tensor.id) ? 3 : 2) / scale;
      context.strokeStyle = state.selectionIds.includes(tensor.id) ? "#8bc2ff" : ctx.shiftColor(tensorColor, 26);
      context.stroke();

      tensor.indices.forEach((index) => {
        const absolutePosition = ctx.indexAbsolutePosition(tensor, index);
        const indexColor = ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(index.id)));
        context.beginPath();
        context.fillStyle = indexColor;
        context.strokeStyle = state.selectionIds.includes(index.id) ? "#8bc2ff" : ctx.shiftColor(indexColor, 34);
        context.lineWidth = (state.selectionIds.includes(index.id) ? 3 : 1.5) / scale;
        context.arc(absolutePosition.x - worldBounds.x1, absolutePosition.y - worldBounds.y1, INDEX_RADIUS, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      });
    });

    context.restore();

    if (state.cy) {
      const extent = state.cy.extent();
      const topLeft = worldToMinimapPoint({ x: extent.x1, y: extent.y1 });
      const bottomRight = worldToMinimapPoint({ x: extent.x2, y: extent.y2 });
      context.fillStyle = "rgba(97, 168, 255, 0.12)";
      context.strokeStyle = "#8bc2ff";
      context.lineWidth = 2;
      context.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
      context.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    }
  }

  function worldToMinimapPoint(point) {
    return {
      x: state.minimapTransform.offsetX + (point.x - state.minimapTransform.bounds.x1) * state.minimapTransform.scale,
      y: state.minimapTransform.offsetY + (point.y - state.minimapTransform.bounds.y1) * state.minimapTransform.scale,
    };
  }

  function downloadPngExport() {
    if (!state.cy || !state.spec) {
      return;
    }
    try {
      const pngDataUrl = withSelectionSuppressed(() =>
        state.cy.png({
          full: true,
          scale: 2,
          bg: "#0b0f14",
        })
      );
      ctx.downloadDataUrl(`${ctx.sanitizeFilename(state.spec.name || "tensor-network")}.png`, pngDataUrl);
      ctx.setStatus("Exported a PNG image of the current design.", "success");
    } catch (error) {
      ctx.setStatus(`Could not export PNG: ${error.message}`, "error");
    }
  }

  function downloadSvgExport() {
    if (!state.spec) {
      return;
    }
    try {
      const svgText = buildSvgExport();
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      ctx.downloadBlob(`${ctx.sanitizeFilename(state.spec.name || "tensor-network")}.svg`, blob);
      ctx.setStatus("Exported an SVG image of the current design.", "success");
    } catch (error) {
      ctx.setStatus(`Could not export SVG: ${error.message}`, "error");
    }
  }

  function buildSvgExport() {
    const bounds = ctx.computeDesignBounds(56);
    const width = Math.max(240, Math.ceil(bounds.x2 - bounds.x1));
    const height = Math.max(180, Math.ceil(bounds.y2 - bounds.y1));
    const lines = [];

    lines.push('<?xml version="1.0" encoding="UTF-8"?>');
    lines.push(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${bounds.x1} ${bounds.y1} ${width} ${height}">`
    );
    lines.push(`<rect x="${bounds.x1}" y="${bounds.y1}" width="${width}" height="${height}" fill="#0b0f14" />`);

    state.spec.edges.forEach((edge) => {
      const left = ctx.findIndexOwner(edge.left.index_id);
      const right = ctx.findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = ctx.indexAbsolutePosition(left.tensor, left.index);
      const target = ctx.indexAbsolutePosition(right.tensor, right.index);
      const curve = ctx.buildQuadraticCurve(source, target);
      const edgeColor = ctx.getMetadataColor(edge.metadata, "#8da1c3");
      const labelPosition = ctx.quadraticPointAt(source, curve.control, target, 0.5);
      lines.push(
        `<path d="M ${source.x} ${source.y} Q ${curve.control.x} ${curve.control.y} ${target.x} ${target.y}" fill="none" stroke="${edgeColor}" stroke-width="3" />`
      );
      lines.push(
        `<text x="${labelPosition.x}" y="${labelPosition.y - 10}" fill="${ctx.shiftColor(edgeColor, 72)}" font-size="11" font-family="Segoe UI, Helvetica Neue, sans-serif" text-anchor="middle">${ctx.escapeSvgText(edge.name)}</text>`
      );
    });

    state.spec.tensors.forEach((tensor) => {
      const tensorColor = ctx.getMetadataColor(tensor.metadata, "#18212c");
      const borderColor = ctx.shiftColor(tensorColor, 26);
      lines.push(
        `<rect x="${tensor.position.x - ctx.tensorWidth(tensor) / 2}" y="${tensor.position.y - ctx.tensorHeight(tensor) / 2}" width="${ctx.tensorWidth(tensor)}" height="${ctx.tensorHeight(tensor)}" rx="22" ry="22" fill="${tensorColor}" stroke="${borderColor}" stroke-width="2" />`
      );
      lines.push(
        `<text x="${tensor.position.x}" y="${tensor.position.y - ctx.tensorHeight(tensor) / 2 + 26}" fill="${ctx.readableTextColor(tensorColor)}" font-size="18" font-family="Georgia, Times New Roman, serif" text-anchor="middle">${ctx.escapeSvgText(tensor.name)}</text>`
      );

      tensor.indices.forEach((index, indexPosition) => {
        const absolutePosition = ctx.indexAbsolutePosition(tensor, index);
        const indexColor = ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(index.id)));
        lines.push(
          `<circle cx="${absolutePosition.x}" cy="${absolutePosition.y}" r="${INDEX_RADIUS}" fill="${indexColor}" stroke="${ctx.shiftColor(indexColor, 34)}" stroke-width="2" />`
        );
        lines.push(
          `<text x="${absolutePosition.x}" y="${absolutePosition.y + 4}" fill="${ctx.readableTextColor(indexColor)}" font-size="12" font-family="Segoe UI, Helvetica Neue, sans-serif" font-weight="700" text-anchor="middle">${indexPosition + 1}</text>`
        );
        lines.push(
          `<text x="${absolutePosition.x}" y="${absolutePosition.y + 28}" fill="${ctx.shiftColor(indexColor, 64)}" font-size="10" font-family="Segoe UI, Helvetica Neue, sans-serif" text-anchor="middle">${ctx.escapeSvgText(`${index.name} · ${index.dimension}`)}</text>`
        );
      });
    });

    lines.push("</svg>");
    return lines.join("\n");
  }

  function withSelectionSuppressed(action) {
    if (!state.cy) {
      return action();
    }
    const selectionIds = [...state.selectionIds];
    state.cy.$(":selected").unselect();
    try {
      return action();
    } finally {
      state.selectionIds = selectionIds;
      ctx.syncSelectedElementState();
      ctx.syncCySelection();
    }
  }

  Object.assign(ctx, {
    handleMinimapMouseDown,
    updateViewportFromMinimapClientPoint,
    centerViewportAt,
    renderMinimap,
    worldToMinimapPoint,
    downloadPngExport,
    downloadSvgExport,
    buildSvgExport,
    withSelectionSuppressed
  });
}
