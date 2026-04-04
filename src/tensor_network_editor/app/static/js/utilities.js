import {
  DEFAULT_INDEX_SLOTS,
  INDEX_PADDING,
  INDEX_RADIUS,
  TENSOR_HEIGHT,
  TENSOR_WIDTH,
  refs,
  state,
} from "./core.js";

const { canvasShell, engineSelect, redoButton, statusMessage, undoButton } = refs;

function populateEngineOptions(engines) {
    engineSelect.innerHTML = "";
    engines.forEach((engineName) => {
      const option = document.createElement("option");
      option.value = engineName;
      option.textContent = engineName;
      if (engineName === state.selectedEngine) {
        option.selected = true;
      }
      engineSelect.appendChild(option);
    });
  }

  function serializeCurrentSpec() {
    return {
      schema_version: 1,
      network: state.spec,
    };
  }

  function stripImportLines(code) {
    const keptLines = code
      .split(/\r?\n/)
      .filter((line) => !/^\s*(import|from)\s+/.test(line));
    while (keptLines.length && keptLines[0].trim() === "") {
      keptLines.shift();
    }
    while (keptLines.length && keptLines[keptLines.length - 1].trim() === "") {
      keptLines.pop();
    }
    return keptLines.join("\n");
  }

  function moveIndex(tensorId, indexPosition, direction) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const targetPosition = indexPosition + direction;
    if (targetPosition < 0 || targetPosition >= tensor.indices.length) {
      return;
    }
    const [index] = tensor.indices.splice(indexPosition, 1);
    tensor.indices.splice(targetPosition, 0, index);
  }

  function removeTensor(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const tensorIndexIds = new Set(tensor.indices.map((index) => index.id));
    state.spec.edges = state.spec.edges.filter(
      (edge) => !tensorIndexIds.has(edge.left.index_id) && !tensorIndexIds.has(edge.right.index_id)
    );
    state.spec.tensors = state.spec.tensors.filter((candidate) => candidate.id !== tensorId);
    state.tensorOrder = state.tensorOrder.filter((candidateId) => candidateId !== tensorId);
  }

  function removeIndex(tensorId, indexId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    state.spec.edges = state.spec.edges.filter(
      (edge) => edge.left.index_id !== indexId && edge.right.index_id !== indexId
    );
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function removeEdge(edgeId) {
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== edgeId);
  }

  function findTensorById(tensorId) {
    return state.spec.tensors.find((tensor) => tensor.id === tensorId) || null;
  }

  function findEdgeById(edgeId) {
    return state.spec.edges.find((edge) => edge.id === edgeId) || null;
  }

  function findIndexOwner(indexId) {
    for (const tensor of state.spec.tensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return null;
  }

  function findEdgeByIndexId(indexId) {
    return (
      state.spec.edges.find(
        (edge) => edge.left.index_id === indexId || edge.right.index_id === indexId
      ) || null
    );
  }

  function createTensor(x, y) {
    const tensor = {
      id: makeId("tensor"),
      name: nextName("T", state.spec.tensors.map((tensor) => tensor.name)),
      position: { x, y },
      indices: [],
      metadata: {},
    };
    tensor.indices.push(createIndex(tensor, 0));
    tensor.indices.push(createIndex(tensor, 1));
    return tensor;
  }

  function createIndex(tensor, indexPosition) {
    return {
      id: makeId("index"),
      name: nextName("i", tensor.indices.map((index) => index.name)),
      dimension: 2,
      offset: defaultIndexOffsetForOrder(indexPosition),
      metadata: {},
    };
  }

  function normalizeSpec(spec) {
    const normalized = deepClone(spec || {});
    normalized.metadata = isObject(normalized.metadata) ? normalized.metadata : {};
    normalized.tensors = Array.isArray(normalized.tensors) ? normalized.tensors : [];
    normalized.edges = Array.isArray(normalized.edges) ? normalized.edges : [];

    normalized.tensors.forEach((tensor) => {
      tensor.metadata = isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.position = {
        x: asFiniteNumber(tensor.position && tensor.position.x, 120),
        y: asFiniteNumber(tensor.position && tensor.position.y, 120),
      };
      tensor.indices = Array.isArray(tensor.indices) ? tensor.indices : [];
      tensor.indices.forEach((index, indexPosition) => {
        index.metadata = isObject(index.metadata) ? index.metadata : {};
        index.dimension = Math.max(1, Math.round(asFiniteNumber(index.dimension, 2)));
        index.offset = {
          x: asFiniteNumber(index.offset && index.offset.x, 0),
          y: asFiniteNumber(index.offset && index.offset.y, 0),
        };
        if (!index.name) {
          index.name = nextName("i", tensor.indices.slice(0, indexPosition).map((candidate) => candidate.name));
        }
      });
      ensureTensorIndexOffsets(tensor);
    });

    return normalized;
  }

  function ensureTensorIndexOffsets(tensor) {
    const needsAutoLayout =
      tensor.indices.length > 0 &&
      tensor.indices.every((index) => Math.abs(index.offset.x) < 0.001 && Math.abs(index.offset.y) < 0.001);

    tensor.indices.forEach((index, indexPosition) => {
      if (needsAutoLayout) {
        index.offset = defaultIndexOffsetForOrder(indexPosition);
      } else {
        index.offset = clampIndexOffset(index.offset);
      }
    });
  }

  function defaultIndexOffsetForOrder(indexPosition) {
    const slot = DEFAULT_INDEX_SLOTS[indexPosition];
    if (slot) {
      return clampIndexOffset(slot);
    }
    return clampIndexOffset({
      x: indexPosition % 2 === 0 ? -58 : 58,
      y: -30 + Math.floor(indexPosition / 2) * 18,
    });
  }

  function clampIndexOffset(offset) {
    return {
      x: clamp(
        asFiniteNumber(offset.x, 0),
        -TENSOR_WIDTH / 2 + INDEX_RADIUS + INDEX_PADDING,
        TENSOR_WIDTH / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
      y: clamp(
        asFiniteNumber(offset.y, 0),
        -TENSOR_HEIGHT / 2 + INDEX_RADIUS + INDEX_PADDING,
        TENSOR_HEIGHT / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
    };
  }

  function indexAbsolutePosition(tensor, index) {
    const offset = clampIndexOffset(index.offset);
    index.offset = offset;
    return {
      x: tensor.position.x + offset.x,
      y: tensor.position.y + offset.y,
    };
  }

  function syncIndexNodePositions(tensor) {
    runWithIndexSync(() => {
      tensor.indices.forEach((index) => {
        syncSingleIndexNodePosition(tensor, index);
      });
    });
  }

  function syncSingleIndexNodePosition(tensor, index) {
    if (!state.cy) {
      return;
    }
    const indexElement = state.cy.getElementById(index.id);
    const absolutePosition = indexAbsolutePosition(tensor, index);
    if (indexElement && indexElement.length) {
      indexElement.position(absolutePosition);
    }
    syncIndexLabelNodePosition(index, absolutePosition);
    if (!indexElement || !indexElement.length) {
      return;
    }
  }

  function syncIndexLabelNodePosition(index, absolutePosition) {
    if (!state.cy) {
      return;
    }
    const labelElement = state.cy.getElementById(indexLabelNodeId(index.id));
    if (!labelElement || !labelElement.length) {
      return;
    }
    labelElement.position(indexLabelPosition(absolutePosition));
    labelElement.data("label", `${index.name} · ${index.dimension}`);
    labelElement.data(
      "textColor",
      shiftColor(getIndexColor(index, Boolean(findEdgeByIndexId(index.id))), 64)
    );
  }

  function runWithIndexSync(action) {
    state.syncingIndexPositions = true;
    try {
      action();
    } finally {
      state.syncingIndexPositions = false;
    }
  }

  function reconcileTensorOrder() {
    const tensorIds = state.spec ? state.spec.tensors.map((tensor) => tensor.id) : [];
    const nextOrder = (Array.isArray(state.tensorOrder) ? state.tensorOrder : []).filter((tensorId) =>
      tensorIds.includes(tensorId)
    );
    tensorIds.forEach((tensorId) => {
      if (!nextOrder.includes(tensorId)) {
        nextOrder.push(tensorId);
      }
    });
    state.tensorOrder = nextOrder;
  }

  function tensorLayerRank(tensorId) {
    const index = state.tensorOrder.indexOf(tensorId);
    return index >= 0 ? index : 0;
  }

  function bringTensorToFront(tensorId) {
    if (!tensorId) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder = state.tensorOrder.filter((id) => id !== tensorId);
    state.tensorOrder.push(tensorId);
    applyTensorLayerData();
  }

  function applyTensorLayerData() {
    if (!state.cy) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder.forEach((tensorId, tensorRank) => {
      const tensorElement = state.cy.getElementById(tensorId);
      if (tensorElement && tensorElement.length) {
        const tensor = findTensorById(tensorId);
        const tensorColor = tensor ? getMetadataColor(tensor.metadata, "#18212c") : "#18212c";
        tensorElement.data("zIndex", 100 + tensorRank * 20);
        tensorElement.data("backgroundColor", tensorColor);
        tensorElement.data("borderColor", shiftColor(tensorColor, 26));
        tensorElement.data("textColor", readableTextColor(tensorColor));
      }
      const tensor = findTensorById(tensorId);
      if (!tensor) {
        return;
      }
      tensor.indices.forEach((index, indexPosition) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        const indexElement = state.cy.getElementById(index.id);
        if (indexElement && indexElement.length) {
          const indexColor = getIndexColor(index, Boolean(findEdgeByIndexId(index.id)));
          indexElement.data("zIndex", 300 + tensorRank * 20 + indexPosition);
          indexElement.data("orderLabel", String(indexPosition + 1));
          indexElement.data("backgroundColor", indexColor);
          indexElement.data("borderColor", shiftColor(indexColor, 34));
          indexElement.data("textColor", readableTextColor(indexColor));
          indexElement.position(absolutePosition);
        }
        syncIndexLabelNodePosition(index, absolutePosition);
        const labelElement = state.cy.getElementById(indexLabelNodeId(index.id));
        if (labelElement && labelElement.length) {
          labelElement.data("zIndex", 310 + tensorRank * 20 + indexPosition);
        }
      });
    });
    state.cy.edges().forEach((edgeElement) => {
      const edge = findEdgeById(edgeElement.id());
      const edgeColor = edge ? getMetadataColor(edge.metadata, "#8da1c3") : "#8da1c3";
      edgeElement.data("zIndex", 220);
      edgeElement.data("lineColor", edgeColor);
      edgeElement.data("textColor", shiftColor(edgeColor, 72));
    });
  }

  function applyColorToSelection(colorValue) {
    getSelectedEntries().forEach((entry) => {
      if (entry.kind === "tensor") {
        entry.tensor.metadata.color = colorValue;
      } else if (entry.kind === "index") {
        entry.located.index.metadata.color = colorValue;
      } else if (entry.kind === "edge") {
        entry.edge.metadata.color = colorValue;
      }
    });
  }

  function getBatchColorValue(selectedEntries) {
    if (!selectedEntries.length) {
      return "#61a8ff";
    }
    return getEntryColor(selectedEntries[0]);
  }

  function getEntryColor(entry) {
    if (entry.kind === "tensor") {
      return getMetadataColor(entry.tensor.metadata, "#18212c");
    }
    if (entry.kind === "index") {
      return getMetadataColor(
        entry.located.index.metadata,
        getIndexColor(entry.located.index, Boolean(findEdgeByIndexId(entry.id)))
      );
    }
    return getMetadataColor(entry.edge.metadata, "#8da1c3");
  }

  function buildQuadraticCurve(source, target) {
    const midpoint = {
      x: (source.x + target.x) / 2,
      y: (source.y + target.y) / 2,
    };
    const deltaX = target.x - source.x;
    const deltaY = target.y - source.y;
    const distance = Math.max(1, Math.sqrt(deltaX * deltaX + deltaY * deltaY));
    const normal = { x: -deltaY / distance, y: deltaX / distance };
    const bend = clamp(distance * 0.18, 18, 60);
    return {
      control: {
        x: midpoint.x + normal.x * bend,
        y: midpoint.y + normal.y * bend,
      },
    };
  }

  function quadraticPointAt(source, control, target, t) {
    const inverse = 1 - t;
    return {
      x: inverse * inverse * source.x + 2 * inverse * t * control.x + t * t * target.x,
      y: inverse * inverse * source.y + 2 * inverse * t * control.y + t * t * target.y,
    };
  }

  function drawRoundRectPath(context, x, y, width, height, radius) {
    const effectiveRadius = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + effectiveRadius, y);
    context.lineTo(x + width - effectiveRadius, y);
    context.quadraticCurveTo(x + width, y, x + width, y + effectiveRadius);
    context.lineTo(x + width, y + height - effectiveRadius);
    context.quadraticCurveTo(x + width, y + height, x + width - effectiveRadius, y + height);
    context.lineTo(x + effectiveRadius, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - effectiveRadius);
    context.lineTo(x, y + effectiveRadius);
    context.quadraticCurveTo(x, y, x + effectiveRadius, y);
    context.closePath();
  }

  function computeDesignBounds(padding) {
    const bounds = {
      x1: Number.POSITIVE_INFINITY,
      y1: Number.POSITIVE_INFINITY,
      x2: Number.NEGATIVE_INFINITY,
      y2: Number.NEGATIVE_INFINITY,
    };

    state.spec.tensors.forEach((tensor) => {
      expandBounds(bounds, tensor.position.x - TENSOR_WIDTH / 2, tensor.position.y - TENSOR_HEIGHT / 2);
      expandBounds(bounds, tensor.position.x + TENSOR_WIDTH / 2, tensor.position.y + TENSOR_HEIGHT / 2);
      tensor.indices.forEach((index) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        expandBounds(bounds, absolutePosition.x - INDEX_RADIUS, absolutePosition.y - INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + INDEX_RADIUS, absolutePosition.y + INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + 50, absolutePosition.y + 42);
      });
    });

    state.spec.edges.forEach((edge) => {
      const left = findIndexOwner(edge.left.index_id);
      const right = findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = indexAbsolutePosition(left.tensor, left.index);
      const target = indexAbsolutePosition(right.tensor, right.index);
      const curve = buildQuadraticCurve(source, target);
      expandBounds(bounds, source.x, source.y);
      expandBounds(bounds, target.x, target.y);
      expandBounds(bounds, curve.control.x, curve.control.y);
    });

    if (!Number.isFinite(bounds.x1)) {
      if (state.cy) {
        const extent = state.cy.extent();
        return {
          x1: extent.x1 - padding,
          y1: extent.y1 - padding,
          x2: extent.x2 + padding,
          y2: extent.y2 + padding,
        };
      }
      return {
        x1: -padding,
        y1: -padding,
        x2: 240 + padding,
        y2: 200 + padding,
      };
    }

    return {
      x1: bounds.x1 - padding,
      y1: bounds.y1 - padding,
      x2: bounds.x2 + padding,
      y2: bounds.y2 + padding,
    };
  }

  function expandBounds(bounds, x, y) {
    bounds.x1 = Math.min(bounds.x1, x);
    bounds.y1 = Math.min(bounds.y1, y);
    bounds.x2 = Math.max(bounds.x2, x);
    bounds.y2 = Math.max(bounds.y2, y);
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

  async function apiGet(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  }

  async function apiPost(path, payload) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.message || "Request failed.");
    }
    return body;
  }

  function updateToolbarState() {
    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
  }

  function formatIssues(issues) {
    if (!issues || !issues.length) {
      return "The design is not valid yet.";
    }
    return issues
      .slice(0, 3)
      .map((issue) => issue.message)
      .join(" ");
  }

  function setStatus(message, kind = "info") {
    if (!statusMessage) {
      return;
    }
    statusMessage.textContent = message;
    statusMessage.classList.remove("status-error", "status-success");
    if (kind === "error") {
      statusMessage.classList.add("status-error");
    }
    if (kind === "success") {
      statusMessage.classList.add("status-success");
    }
  }

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

  function deepClone(value) {
    if (typeof structuredClone === "function") {
      return structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  }

  function createSnapshotSignature(spec, tensorOrder) {
    return JSON.stringify({
      spec,
      tensorOrder: Array.isArray(tensorOrder) ? tensorOrder : [],
    });
  }

  function clientPointToCanvasPoint(clientX, clientY) {
    const rect = canvasShell.getBoundingClientRect();
    return {
      x: clamp(clientX - rect.left, 0, rect.width),
      y: clamp(clientY - rect.top, 0, rect.height),
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

  function getIndexColor(index, isConnected) {
    return getMetadataColor(index.metadata, isConnected ? "#61a8ff" : "#e0b566");
  }

  function getMetadataColor(metadata, fallback) {
    const candidate = metadata && typeof metadata.color === "string" ? metadata.color.trim() : "";
    return /^#[0-9a-fA-F]{6}$/.test(candidate) ? candidate.toLowerCase() : fallback;
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

export {
  apiGet,
  apiPost,
  applyColorToSelection,
  applyTensorLayerData,
  asFiniteNumber,
  boxesIntersect,
  bringTensorToFront,
  buildQuadraticCurve,
  clamp,
  clampIndexOffset,
  clientPointToCanvasPoint,
  computeDesignBounds,
  createIndex,
  createSnapshotSignature,
  createTensor,
  deepClone,
  defaultIndexOffsetForOrder,
  downloadBlob,
  downloadDataUrl,
  drawRoundRectPath,
  escapeHtml,
  escapeSvgText,
  expandBounds,
  findEdgeById,
  findEdgeByIndexId,
  findIndexOwner,
  findTensorById,
  formatColorHex,
  formatIssues,
  getBatchColorValue,
  getEntryColor,
  getIndexColor,
  getMetadataColor,
  indexAbsolutePosition,
  indexLabelNodeId,
  indexLabelPosition,
  isIndexNode,
  isObject,
  isTextInput,
  makeId,
  moveIndex,
  nextName,
  normalizeSpec,
  parseHexColor,
  populateEngineOptions,
  quadraticPointAt,
  readableTextColor,
  reconcileTensorOrder,
  removeEdge,
  removeIndex,
  removeTensor,
  runWithIndexSync,
  sanitizeFilename,
  serializeCurrentSpec,
  setStatus,
  shiftColor,
  stripImportLines,
  syncIndexLabelNodePosition,
  syncIndexNodePositions,
  syncSingleIndexNodePosition,
  tensorIndexNameExists,
  tensorLayerRank,
  updateToolbarState,
};
