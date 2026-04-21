import { createCodeHighlightingSupport } from "../core/codeHighlighting.js";
import { GRAPH_THEME } from "../core/theme.js";

export function createUtilityBaseBindings({ ctx, state, dom }) {
  const { window, document } = ctx;
  const { canvasShell } = dom;
  const codeHighlightingSupport = createCodeHighlightingSupport({
    windowRef: window,
    documentRef: document,
  });

  function sanitizeFilename(value) {
    const sanitized = value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    return sanitized.replace(/^-+|-+$/g, "") || "tensor-network";
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function escapeSvgText(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&apos;");
  }

  function isIndexNode(element) {
    return element.isNode() && element.data("kind") === "index";
  }

  function isTextInput(element) {
    return Boolean(element) && ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
  }

  function isAdditiveSelectionModifier(event) {
    return Boolean(event && (event.shiftKey || event.ctrlKey || event.metaKey));
  }

  function deepClone(value) {
    if (typeof window.structuredClone === "function") {
      return window.structuredClone(value);
    }
    const serializedValue = JSON.stringify(value);
    return serializedValue === undefined ? undefined : JSON.parse(serializedValue);
  }

  function clientPointToCanvasPoint(clientX, clientY) {
    const rect = canvasShell.getBoundingClientRect();
    return {
      x: clamp(clientX - rect.left, 0, rect.width),
      y: clamp(clientY - rect.top, 0, rect.height),
    };
  }

  function clientPointToWorldPoint(clientX, clientY) {
    const canvasPoint = clientPointToCanvasPoint(clientX, clientY);
    if (!state.cy) {
      return canvasPoint;
    }
    const zoom = state.cy.zoom();
    const pan = state.cy.pan();
    return {
      x: (canvasPoint.x - pan.x) / zoom,
      y: (canvasPoint.y - pan.y) / zoom,
    };
  }

  function worldToCanvasPoint(point) {
    if (!state.cy) {
      return point;
    }
    const zoom = state.cy.zoom();
    const pan = state.cy.pan();
    return {
      x: point.x * zoom + pan.x,
      y: point.y * zoom + pan.y,
    };
  }

  function normalizedBox(startPoint, currentPoint) {
    const left = Math.min(startPoint.x, currentPoint.x);
    const top = Math.min(startPoint.y, currentPoint.y);
    const width = Math.abs(currentPoint.x - startPoint.x);
    const height = Math.abs(currentPoint.y - startPoint.y);
    return { left, top, width, height };
  }

  function boxesIntersect(leftBox, rightBox) {
    return !(
      leftBox.left + leftBox.width < rightBox.x1 ||
      leftBox.left > rightBox.x2 ||
      leftBox.top + leftBox.height < rightBox.y1 ||
      leftBox.top > rightBox.y2
    );
  }

  function indexLabelNodeId(indexId) {
    return `${indexId}__label`;
  }

  function indexLabelPosition(indexPositionAbsolute) {
    return {
      x: indexPositionAbsolute.x,
      y: indexPositionAbsolute.y + 28,
    };
  }

  function getMetadataColor(metadata, fallback) {
    const candidate = metadata && typeof metadata.color === "string" ? metadata.color.trim() : "";
    return /^#[0-9a-fA-F]{6}$/.test(candidate) ? candidate.toLowerCase() : fallback;
  }

  function getIndexColor(index, isConnected) {
    return getMetadataColor(
      index.metadata,
      isConnected ? GRAPH_THEME.indexConnected : GRAPH_THEME.indexOpen
    );
  }

  function shiftColor(hexColor, amount) {
    const { red, green, blue } = parseHexColor(hexColor);
    return formatColorHex({
      red: clamp(Math.round(red + amount), 0, 255),
      green: clamp(Math.round(green + amount), 0, 255),
      blue: clamp(Math.round(blue + amount), 0, 255),
    });
  }

  function readableTextColor(hexColor) {
    const { red, green, blue } = parseHexColor(hexColor);
    const luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255;
    return luminance > 0.62 ? "#091018" : "#f5f9ff";
  }

  function parseHexColor(hexColor) {
    const normalized = getMetadataColor({ color: hexColor }, "#000000");
    return {
      red: Number.parseInt(normalized.slice(1, 3), 16),
      green: Number.parseInt(normalized.slice(3, 5), 16),
      blue: Number.parseInt(normalized.slice(5, 7), 16),
    };
  }

  function formatColorHex({ red, green, blue }) {
    return `#${[red, green, blue]
      .map((component) => component.toString(16).padStart(2, "0"))
      .join("")}`;
  }

  function downloadDataUrl(filename, dataUrl) {
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = filename;
    anchor.click();
  }

  function downloadBlob(filename, blob) {
    const anchor = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(objectUrl);
  }

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function asFiniteNumber(value, fallback) {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : fallback;
  }

  function makeId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}_${window.crypto.randomUUID().replace(/-/g, "").slice(0, 10)}`;
    }
    return `${prefix}_${Math.random().toString(16).slice(2, 12)}`;
  }

  function nextName(prefix, existingNames) {
    let counter = 1;
    while (existingNames.includes(`${prefix}${counter}`)) {
      counter += 1;
    }
    return `${prefix}${counter}`;
  }

  function tensorIndexNameExists(tensor, candidateName, excludedIndexId = null) {
    const normalizedCandidate = candidateName.trim();
    return tensor.indices.some(
      (index) =>
        index.id !== excludedIndexId &&
        typeof index.name === "string" &&
        index.name.trim() === normalizedCandidate
    );
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  return {
    sanitizeFilename,
    escapeHtml,
    escapeSvgText,
    isIndexNode,
    isTextInput,
    isAdditiveSelectionModifier,
    deepClone,
    clientPointToCanvasPoint,
    clientPointToWorldPoint,
    worldToCanvasPoint,
    normalizedBox,
    boxesIntersect,
    indexLabelNodeId,
    indexLabelPosition,
    getIndexColor,
    getMetadataColor,
    shiftColor,
    readableTextColor,
    parseHexColor,
    formatColorHex,
    downloadDataUrl,
    downloadBlob,
    isObject,
    asFiniteNumber,
    makeId,
    nextName,
    tensorIndexNameExists,
    loadCodeHighlighter: codeHighlightingSupport.loadPrismHighlighter,
    highlightCodeElement: codeHighlightingSupport.highlightElement,
    clamp,
  };
}
