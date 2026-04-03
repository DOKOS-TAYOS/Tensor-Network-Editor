(function () {
  "use strict";

  const TENSOR_WIDTH = 180;
  const TENSOR_HEIGHT = 108;
  const INDEX_RADIUS = 15;
  const INDEX_PADDING = 8;
  const DEFAULT_INDEX_SLOTS = [
    { x: -58, y: -20 },
    { x: 58, y: -20 },
    { x: -58, y: 20 },
    { x: 58, y: 20 },
    { x: 0, y: -28 },
    { x: 0, y: 28 },
    { x: -24, y: -34 },
    { x: 24, y: 34 },
  ];

  const state = {
    spec: null,
    selectedEngine: null,
    selectedElement: null,
    generatedCode: "",
    connectMode: false,
    pendingIndexId: null,
    cy: null,
    hasFitCanvas: false,
    editorFinished: false,
    tensorOrder: [],
    syncingIndexPositions: false,
  };

  const statusMessage = document.getElementById("status-message");
  const propertiesPanel = document.getElementById("properties-panel");
  const generatedCode = document.getElementById("generated-code");
  const engineSelect = document.getElementById("engine-select");
  const connectButton = document.getElementById("connect-button");
  const loadInput = document.getElementById("load-input");

  document.addEventListener("DOMContentLoaded", () => {
    attachToolbarHandlers();
    bootstrap().catch((error) => {
      setStatus(`Failed to load the editor: ${error.message}`, "error");
    });
  });

  async function bootstrap() {
    const payload = await apiGet("/api/bootstrap");
    state.spec = normalizeSpec(payload.spec.network);
    state.selectedEngine = payload.default_engine;
    populateEngineOptions(payload.engines);
    initGraph();
    render();
    setStatus("Editor ready. Add tensors, move ports, and generate code.", "success");
  }

  function attachToolbarHandlers() {
    document.getElementById("new-design-button").addEventListener("click", handleNewDesign);
    document.getElementById("add-tensor-button").addEventListener("click", addTensorAtCenter);
    document.getElementById("connect-button").addEventListener("click", toggleConnectMode);
    document.getElementById("delete-button").addEventListener("click", deleteSelection);
    document.getElementById("save-button").addEventListener("click", saveDesign);
    document.getElementById("load-button").addEventListener("click", () => loadInput.click());
    document.getElementById("validate-button").addEventListener("click", validateDesign);
    document.getElementById("generate-button").addEventListener("click", generateCode);
    document.getElementById("done-button").addEventListener("click", completeEditor);
    document.getElementById("cancel-button").addEventListener("click", cancelEditor);
    document.getElementById("copy-code-button").addEventListener("click", copyGeneratedCode);
    engineSelect.addEventListener("change", (event) => {
      state.selectedEngine = event.target.value;
    });
    loadInput.addEventListener("change", loadDesignFromFile);
    window.addEventListener("keydown", handleKeydown);
    window.addEventListener("beforeunload", sendCancelBeacon);
    window.addEventListener("pagehide", sendCancelBeacon);
    window.addEventListener("resize", handleWindowResize);
  }

  function handleWindowResize() {
    if (state.cy) {
      state.cy.resize();
    }
  }

  function sendCancelBeacon() {
    if (state.editorFinished || !navigator.sendBeacon) {
      return;
    }
    const payload = new Blob([JSON.stringify({})], { type: "application/json" });
    navigator.sendBeacon("/api/cancel", payload);
  }

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

  function initGraph() {
    state.cy = cytoscape({
      container: document.getElementById("canvas"),
      layout: { name: "preset" },
      minZoom: 0.3,
      maxZoom: 2.5,
      selectionType: "single",
      wheelSensitivity: 0.18,
      style: [
        {
          selector: "node, edge",
          style: {
            "z-index-compare": "manual",
          },
        },
        {
          selector: "node[kind = 'tensor']",
          style: {
            shape: "round-rectangle",
            width: TENSOR_WIDTH,
            height: TENSOR_HEIGHT,
            "background-color": "data(backgroundColor)",
            "border-width": 2,
            "border-color": "data(borderColor)",
            color: "data(textColor)",
            label: "data(label)",
            "font-size": 18,
            "font-family": "Georgia",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-y": 20,
            "padding-top": 42,
            "padding-bottom": 18,
            "padding-left": 24,
            "padding-right": 24,
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: "node[kind = 'index']",
          style: {
            width: INDEX_RADIUS * 2,
            height: INDEX_RADIUS * 2,
            label: "data(orderLabel)",
            "font-size": 12,
            "font-weight": 700,
            color: "data(textColor)",
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 2,
            "border-color": "data(borderColor)",
            "background-color": "data(backgroundColor)",
            "overlay-opacity": 0,
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: "node[kind = 'index-label']",
          style: {
            width: 1,
            height: 1,
            shape: "round-rectangle",
            label: "data(label)",
            color: "data(textColor)",
            "background-opacity": 0,
            "border-opacity": 0,
            "overlay-opacity": 0,
            "font-size": 10,
            "text-wrap": "wrap",
            "text-max-width": 90,
            "text-valign": "top",
            "text-halign": "center",
            "z-index": "data(zIndex)",
            events: "no",
          },
        },
        {
          selector: "node.index-open",
          style: {
            "background-color": "data(backgroundColor)",
            "border-color": "data(borderColor)",
            color: "data(textColor)",
          },
        },
        {
          selector: "node.index-connected",
          style: {
            "background-color": "data(backgroundColor)",
            "border-color": "data(borderColor)",
            color: "data(textColor)",
          },
        },
        {
          selector: "edge",
          style: {
            width: 3,
            "line-color": "data(lineColor)",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 11,
            color: "data(textColor)",
            "text-background-color": "#101720",
            "text-background-opacity": 0.92,
            "text-background-padding": 4,
            "text-rotation": "autorotate",
            "target-arrow-shape": "none",
            "source-arrow-shape": "none",
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: ":selected",
          style: {
            "overlay-color": "#61a8ff",
            "overlay-opacity": 0.14,
            "border-color": "#8bc2ff",
            "line-color": "#8bc2ff",
          },
        },
      ],
    });

    state.cy.on("tap", "node, edge", (event) => {
      const element = event.target;
      if (state.connectMode && isIndexNode(element)) {
        handleConnectClick(element.id());
        return;
      }

      const kind = element.data("kind");
      if (kind === "tensor") {
        bringTensorToFront(element.id());
      } else if (kind === "index") {
        const located = findIndexOwner(element.id());
        if (located) {
          bringTensorToFront(located.tensor.id);
        }
      }

      setSelectedElement(kind, element.id());
    });

    state.cy.on("tap", (event) => {
      if (event.target === state.cy) {
        clearSelection();
      }
    });

    state.cy.on("grab", "node[kind = 'tensor']", (event) => {
      bringTensorToFront(event.target.id());
    });

    state.cy.on("position", "node[kind = 'tensor']", (event) => {
      const tensor = findTensorById(event.target.id());
      if (!tensor) {
        return;
      }
      tensor.position.x = Math.round(event.target.position("x"));
      tensor.position.y = Math.round(event.target.position("y"));
      syncIndexNodePositions(tensor);
    });

    state.cy.on("dragfree", "node[kind = 'tensor']", (event) => {
      const tensor = findTensorById(event.target.id());
      if (!tensor) {
        return;
      }
      syncIndexNodePositions(tensor);
      renderProperties();
    });

    state.cy.on("grab", "node[kind = 'index']", (event) => {
      const located = findIndexOwner(event.target.id());
      if (located) {
        bringTensorToFront(located.tensor.id);
      }
    });

    state.cy.on("position", "node[kind = 'index']", (event) => {
      if (state.syncingIndexPositions) {
        return;
      }

      const located = findIndexOwner(event.target.id());
      if (!located) {
        return;
      }

      located.index.offset = clampIndexOffset({
        x: event.target.position("x") - located.tensor.position.x,
        y: event.target.position("y") - located.tensor.position.y,
      });
      const absolutePosition = indexAbsolutePosition(located.tensor, located.index);
      syncIndexLabelNodePosition(located.index, absolutePosition);
      if (
        Math.abs(absolutePosition.x - event.target.position("x")) > 0.5 ||
        Math.abs(absolutePosition.y - event.target.position("y")) > 0.5
      ) {
        runWithIndexSync(() => {
          event.target.position(absolutePosition);
        });
      }
    });

    state.cy.on("dragfree", "node[kind = 'index']", (event) => {
      const located = findIndexOwner(event.target.id());
      if (!located) {
        return;
      }
      located.index.offset = clampIndexOffset(located.index.offset);
      syncSingleIndexNodePosition(located.tensor, located.index);
      renderProperties();
    });
  }

  function render() {
    renderGraph();
    renderProperties();
    generatedCode.value = state.generatedCode;
    connectButton.classList.toggle("is-active", state.connectMode);
  }

  function renderGraph() {
    if (!state.cy || !state.spec) {
      return;
    }

    const previousSelection = state.selectedElement ? state.selectedElement.id : null;
    reconcileTensorOrder();
    const elements = buildGraphElements();

    state.cy.batch(() => {
      state.cy.elements().remove();
      state.cy.add(elements);
    });

    applyTensorLayerData();

    if (!state.hasFitCanvas) {
      if (state.spec.tensors.length) {
        state.cy.fit(undefined, 40);
      } else {
        state.cy.center();
      }
      state.hasFitCanvas = true;
    }

    if (previousSelection && state.cy.getElementById(previousSelection).length) {
      state.cy.getElementById(previousSelection).select();
    }
  }

  function buildGraphElements() {
    const tensorElements = [];
    const edgeElements = [];
    const indexElements = [];
    const indexLabelElements = [];
    const connectedIndexIds = new Set();

    state.spec.edges.forEach((edge) => {
      connectedIndexIds.add(edge.left.index_id);
      connectedIndexIds.add(edge.right.index_id);
    });

    state.spec.tensors.forEach((tensor) => {
      ensureTensorIndexOffsets(tensor);
      const tensorRank = tensorLayerRank(tensor.id);
      const tensorColor = getMetadataColor(tensor.metadata, "#18212c");
      tensorElements.push({
        group: "nodes",
        data: {
          id: tensor.id,
          label: tensor.name,
          kind: "tensor",
          backgroundColor: tensorColor,
          borderColor: shiftColor(tensorColor, 26),
          textColor: readableTextColor(tensorColor),
          zIndex: 100 + tensorRank * 20,
        },
        position: { x: tensor.position.x, y: tensor.position.y },
      });

      tensor.indices.forEach((index, indexPosition) => {
        const indexColor = getIndexColor(index, connectedIndexIds.has(index.id));
        const indexPositionAbsolute = indexAbsolutePosition(tensor, index);
        indexElements.push({
          group: "nodes",
          data: {
            id: index.id,
            kind: "index",
            tensor_id: tensor.id,
            orderLabel: String(indexPosition + 1),
            backgroundColor: indexColor,
            borderColor: shiftColor(indexColor, 34),
            textColor: readableTextColor(indexColor),
            zIndex: 300 + tensorRank * 20 + indexPosition,
          },
          classes: connectedIndexIds.has(index.id) ? "index-connected" : "index-open",
          position: indexPositionAbsolute,
          grabbable: true,
          selectable: true,
        });

        indexLabelElements.push({
          group: "nodes",
          data: {
            id: indexLabelNodeId(index.id),
            kind: "index-label",
            label: `${index.name} · ${index.dimension}`,
            textColor: shiftColor(indexColor, 64),
            zIndex: 310 + tensorRank * 20 + indexPosition,
          },
          position: indexLabelPosition(indexPositionAbsolute),
          grabbable: false,
          selectable: false,
        });
      });
    });

    state.spec.edges.forEach((edge) => {
      const edgeColor = getMetadataColor(edge.metadata, "#8da1c3");
      edgeElements.push({
        group: "edges",
        data: {
          id: edge.id,
          source: edge.left.index_id,
          target: edge.right.index_id,
          label: edge.name,
          kind: "edge",
          lineColor: edgeColor,
          textColor: shiftColor(edgeColor, 72),
          zIndex: 220,
        },
      });
    });

    return [...tensorElements, ...edgeElements, ...indexElements, ...indexLabelElements];
  }

  function renderProperties() {
    if (!state.selectedElement) {
      renderNetworkProperties();
      return;
    }
    if (state.selectedElement.kind === "tensor") {
      renderTensorProperties(state.selectedElement.id);
      return;
    }
    if (state.selectedElement.kind === "index") {
      renderIndexProperties(state.selectedElement.id);
      return;
    }
    if (state.selectedElement.kind === "edge") {
      renderEdgeProperties(state.selectedElement.id);
      return;
    }
    renderNetworkProperties();
  }

  function renderNetworkProperties() {
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="network-name-input">Design name</label>
        <input id="network-name-input" value="${escapeHtml(state.spec.name)}" />
      </div>
      <div class="properties-chip">
        <span>Tensors</span>
        <strong>${state.spec.tensors.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Connections</span>
        <strong>${state.spec.edges.length}</strong>
      </div>
      <p class="property-meta">
        Drag tensors directly on the canvas. Ports can also be moved inside each tensor.
      </p>
    `;

    document.getElementById("network-name-input").addEventListener("input", (event) => {
      state.spec.name = event.target.value;
    });
  }

  function renderTensorProperties(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      clearSelection();
      return;
    }

    const indexList = tensor.indices
      .map(
        (index, indexPosition) => `
          <button type="button" class="properties-chip index-select-button" data-index-id="${index.id}">
            <span>${indexPosition + 1}. ${escapeHtml(index.name)}</span>
            <strong>${index.dimension}</strong>
          </button>
        `
      )
      .join("");

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="tensor-name-input">Tensor name</label>
        <input id="tensor-name-input" value="${escapeHtml(tensor.name)}" />
      </div>
      <div class="button-row">
        <button id="add-index-button" type="button">Add Index</button>
        <button id="center-tensor-button" type="button">Center in View</button>
        <label class="control-inline-color" for="tensor-color-input">
          <input id="tensor-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(tensor.metadata, "#18212c"))}" />
        </label>
      </div>
      <div class="button-row">
        <button id="delete-tensor-button" type="button" class="danger">Delete Tensor</button>
        <button id="apply-tensor-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
      <div class="properties-list">${indexList || "<p class='property-meta'>This tensor has no indices yet.</p>"}</div>
    `;

    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-tensor-button"),
      inputElements: [tensorNameInput, tensorColorInput],
      isDirty: () =>
        tensorNameInput.value !== tensor.name ||
        tensorColorInput.value !== getMetadataColor(tensor.metadata, "#18212c"),
      onApply: () => {
        tensor.name = tensorNameInput.value;
        tensor.metadata.color = tensorColorInput.value;
        renderGraph();
        renderProperties();
        setStatus(`Updated tensor ${tensor.name}.`, "success");
      },
    });

    document.getElementById("add-index-button").addEventListener("click", () => {
      const nextIndex = createIndex(tensor, tensor.indices.length);
      tensor.indices.push(nextIndex);
      render();
      setSelectedElement("index", nextIndex.id);
    });
    document.getElementById("center-tensor-button").addEventListener("click", () => {
      centerTensor(tensor.id);
    });
    document.getElementById("delete-tensor-button").addEventListener("click", () => {
      removeTensor(tensor.id);
      render();
    });
    document.querySelectorAll(".index-select-button").forEach((button) => {
      button.addEventListener("click", () => {
        setSelectedElement("index", button.dataset.indexId);
      });
    });
  }

  function renderIndexProperties(indexId) {
    const located = findIndexOwner(indexId);
    if (!located) {
      clearSelection();
      return;
    }

    const { tensor, index, indexPosition } = located;
    propertiesPanel.innerHTML = `
      <div class="field-row">
        <div class="field-group">
          <label for="index-name-input">Index name</label>
          <input id="index-name-input" value="${escapeHtml(index.name)}" />
        </div>
        <div class="field-group compact-number-field">
          <label for="index-dimension-input">Dimension</label>
          <input id="index-dimension-input" type="number" min="1" step="1" value="${index.dimension}" />
        </div>
      </div>
      <div class="button-row">
        <button id="move-index-up-button" type="button">Move Earlier</button>
        <button id="move-index-down-button" type="button">Move Later</button>
        <label class="control-inline-color" for="index-color-input">
          <input id="index-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(index.metadata, getIndexColor(index, Boolean(findEdgeByIndexId(indexId)))))}" />
        </label>
        <button id="delete-index-button" type="button" class="danger">Delete Index</button>
        <button id="apply-index-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const indexNameInput = document.getElementById("index-name-input");
    const indexDimensionInput = document.getElementById("index-dimension-input");
    const indexColorInput = document.getElementById("index-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-index-button"),
      inputElements: [indexNameInput, indexDimensionInput, indexColorInput],
      isDirty: () =>
        indexNameInput.value !== index.name ||
        indexDimensionInput.value !== String(index.dimension) ||
        indexColorInput.value !== getMetadataColor(index.metadata, getIndexColor(index, Boolean(findEdgeByIndexId(indexId)))),
      onApply: () => {
        const proposedName = indexNameInput.value.trim();
        const parsed = Number.parseInt(indexDimensionInput.value, 10);
        if (!proposedName) {
          setStatus("Index name cannot be empty.", "error");
          return;
        }
        if (!Number.isFinite(parsed) || parsed <= 0) {
          setStatus("Index dimension must be a positive integer.", "error");
          return;
        }
        if (tensorIndexNameExists(tensor, proposedName, index.id)) {
          setStatus(
            `Tensor ${tensor.name} already has an index named ${proposedName}.`,
            "error"
          );
          return;
        }
        index.name = proposedName;
        index.dimension = parsed;
        index.metadata.color = indexColorInput.value;
        render();
        setSelectedElement("index", index.id);
        setStatus(`Updated index ${index.name}.`, "success");
      },
    });

    document.getElementById("move-index-up-button").addEventListener("click", () => {
      moveIndex(tensor.id, indexPosition, -1);
    });
    document.getElementById("move-index-down-button").addEventListener("click", () => {
      moveIndex(tensor.id, indexPosition, 1);
    });
    document.getElementById("delete-index-button").addEventListener("click", () => {
      removeIndex(tensor.id, index.id);
      render();
    });
  }

  function renderEdgeProperties(edgeId) {
    const edge = findEdgeById(edgeId);
    if (!edge) {
      clearSelection();
      return;
    }
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="edge-name-input">Edge name</label>
        <input id="edge-name-input" value="${escapeHtml(edge.name)}" />
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="edge-color-input">
          <input id="edge-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(edge.metadata, "#8da1c3"))}" />
        </label>
        <button id="delete-edge-button" type="button" class="danger">Delete Connection</button>
        <button id="apply-edge-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const edgeNameInput = document.getElementById("edge-name-input");
    const edgeColorInput = document.getElementById("edge-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-edge-button"),
      inputElements: [edgeNameInput, edgeColorInput],
      isDirty: () =>
        edgeNameInput.value !== edge.name ||
        edgeColorInput.value !== getMetadataColor(edge.metadata, "#8da1c3"),
      onApply: () => {
        edge.name = edgeNameInput.value;
        edge.metadata.color = edgeColorInput.value;
        renderGraph();
        renderProperties();
        setStatus(`Updated connection ${edge.name}.`, "success");
      },
    });

    document.getElementById("delete-edge-button").addEventListener("click", () => {
      removeEdge(edge.id);
      render();
    });
  }

  function installDirtyApply({ buttonElement, inputElements, isDirty, onApply }) {
    const refreshVisibility = () => {
      buttonElement.classList.toggle("is-hidden", !isDirty());
    };
    inputElements.forEach((element) => {
      element.addEventListener("input", refreshVisibility);
    });
    buttonElement.addEventListener("click", onApply);
    refreshVisibility();
  }

  function setSelectedElement(kind, id) {
    if (kind === "tensor") {
      bringTensorToFront(id);
    } else if (kind === "index") {
      const located = findIndexOwner(id);
      if (located) {
        bringTensorToFront(located.tensor.id);
      }
    }

    state.selectedElement = { kind, id };
    if (state.cy && state.cy.getElementById(id).length) {
      state.cy.$(":selected").unselect();
      state.cy.getElementById(id).select();
    }
    renderProperties();
  }

  function clearSelection() {
    state.selectedElement = null;
    state.pendingIndexId = null;
    if (state.cy) {
      state.cy.$(":selected").unselect();
    }
    renderProperties();
  }

  function handleNewDesign() {
    if (!window.confirm("Start a new design? Unsaved changes in this browser tab will be lost.")) {
      return;
    }

    state.spec = normalizeSpec({
      id: makeId("network"),
      name: "Untitled Network",
      tensors: [],
      edges: [],
      metadata: {},
    });
    state.generatedCode = "";
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.hasFitCanvas = false;
    render();
    setStatus("Started a new empty design.");
  }

  function addTensorAtCenter() {
    const center = viewportCenterPosition();
    const suggestedPosition = suggestTensorPosition(center);
    const tensor = createTensor(suggestedPosition.x, suggestedPosition.y);
    state.spec.tensors.push(tensor);
    reconcileTensorOrder();
    bringTensorToFront(tensor.id);
    render();
    setSelectedElement("tensor", tensor.id);
    setStatus(`Added tensor ${tensor.name}.`);
  }

  function suggestTensorPosition(center) {
    const offsets = [
      { x: 0, y: 0 },
      { x: 220, y: 0 },
      { x: -220, y: 0 },
      { x: 0, y: 170 },
      { x: 0, y: -170 },
      { x: 220, y: 170 },
      { x: 220, y: -170 },
      { x: -220, y: 170 },
      { x: -220, y: -170 },
      { x: 440, y: 0 },
      { x: -440, y: 0 },
      { x: 0, y: 340 },
      { x: 0, y: -340 },
    ];

    for (const offset of offsets) {
      const candidate = { x: center.x + offset.x, y: center.y + offset.y };
      if (!isTensorPositionOccupied(candidate)) {
        return candidate;
      }
    }

    return {
      x: center.x + state.spec.tensors.length * 36,
      y: center.y + state.spec.tensors.length * 28,
    };
  }

  function isTensorPositionOccupied(candidate) {
    return state.spec.tensors.some((tensor) => {
      return (
        Math.abs(tensor.position.x - candidate.x) < 170 &&
        Math.abs(tensor.position.y - candidate.y) < 120
      );
    });
  }

  function viewportCenterPosition() {
    if (!state.cy) {
      return { x: 240, y: 200 };
    }
    const zoom = state.cy.zoom();
    const pan = state.cy.pan();
    return {
      x: Math.round((state.cy.width() / 2 - pan.x) / zoom),
      y: Math.round((state.cy.height() / 2 - pan.y) / zoom),
    };
  }

  function centerTensor(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const center = viewportCenterPosition();
    tensor.position.x = center.x;
    tensor.position.y = center.y;
    render();
    setSelectedElement("tensor", tensorId);
    setStatus(`Centered tensor ${tensor.name} in the current view.`, "success");
  }

  function toggleConnectMode() {
    state.connectMode = !state.connectMode;
    state.pendingIndexId = null;
    render();
    setStatus(
      state.connectMode
        ? "Connect mode active. Click two open indices with the same dimension."
        : "Connect mode disabled."
    );
  }

  function handleConnectClick(indexId) {
    const located = findIndexOwner(indexId);
    if (!located) {
      return;
    }
    if (findEdgeByIndexId(indexId)) {
      setStatus("This index is already connected. Delete the connection first.", "error");
      return;
    }

    if (!state.pendingIndexId) {
      state.pendingIndexId = indexId;
      setSelectedElement("index", indexId);
      setStatus("First index selected. Click another compatible open index to connect.");
      return;
    }

    if (state.pendingIndexId === indexId) {
      state.pendingIndexId = null;
      setStatus("Connection cancelled.");
      return;
    }

    const left = findIndexOwner(state.pendingIndexId);
    if (!left) {
      state.pendingIndexId = null;
      return;
    }
    if (left.index.dimension !== located.index.dimension) {
      setStatus("Connected indices must have the same dimension.", "error");
      return;
    }

    state.spec.edges.push({
      id: makeId("edge"),
      name: nextName("bond", state.spec.edges.map((edge) => edge.name)),
      left: { tensor_id: left.tensor.id, index_id: left.index.id },
      right: { tensor_id: located.tensor.id, index_id: located.index.id },
      metadata: {},
    });
    state.pendingIndexId = null;
    render();
    setStatus("Connection created.", "success");
  }

  function deleteSelection() {
    if (!state.selectedElement) {
      setStatus("Nothing is selected to delete.");
      return;
    }

    if (state.selectedElement.kind === "tensor") {
      removeTensor(state.selectedElement.id);
    } else if (state.selectedElement.kind === "index") {
      const located = findIndexOwner(state.selectedElement.id);
      if (located) {
        removeIndex(located.tensor.id, located.index.id);
      }
    } else if (state.selectedElement.kind === "edge") {
      removeEdge(state.selectedElement.id);
    }

    state.selectedElement = null;
    render();
    setStatus("Selection deleted.");
  }

  async function validateDesign() {
    try {
      const payload = await apiPost("/api/validate", { spec: serializeCurrentSpec() });
      if (payload.ok) {
        setStatus("Design is valid and ready for code generation.", "success");
        return;
      }
      setStatus(formatIssues(payload.issues), "error");
    } catch (error) {
      setStatus(`Validation failed: ${error.message}`, "error");
    }
  }

  async function generateCode() {
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        spec: serializeCurrentSpec(),
      });
      if (!payload.ok) {
        setStatus(formatIssues(payload.issues), "error");
        return;
      }
      state.generatedCode = stripImportLines(payload.code);
      generatedCode.value = state.generatedCode;
      setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      setStatus(`Code generation failed: ${error.message}`, "error");
    }
  }

  async function completeEditor() {
    try {
      const payload = await apiPost("/api/complete", {
        engine: state.selectedEngine,
        spec: serializeCurrentSpec(),
      });
      if (!payload.ok) {
        setStatus(formatIssues(payload.issues), "error");
        return;
      }
      state.editorFinished = true;
      setStatus("Returning the design to Python. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      setStatus(`Could not finish the editor session: ${error.message}`, "error");
    }
  }

  async function cancelEditor() {
    try {
      state.editorFinished = true;
      await apiPost("/api/cancel", {});
      setStatus("Editor cancelled. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      setStatus(`Could not cancel the editor session: ${error.message}`, "error");
    }
  }

  function saveDesign() {
    const blob = new Blob([JSON.stringify(serializeCurrentSpec(), null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${sanitizeFilename(state.spec.name || "tensor-network")}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    setStatus("Design downloaded as JSON.");
  }

  function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result);
        state.spec = normalizeSpec(payload.network ? payload.network : payload);
        state.selectedElement = null;
        state.pendingIndexId = null;
        state.generatedCode = "";
        generatedCode.value = "";
        render();
        setStatus(`Loaded design from ${file.name}.`, "success");
      } catch (error) {
        setStatus(`Could not load ${file.name}: ${error.message}`, "error");
      } finally {
        loadInput.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  }

  async function copyGeneratedCode() {
    const codeToCopy = stripImportLines(generatedCode.value);
    if (!codeToCopy.trim()) {
      setStatus("There is no generated code to copy yet.");
      return;
    }
    try {
      await navigator.clipboard.writeText(codeToCopy);
      setStatus("Generated code copied to the clipboard without import lines.", "success");
    } catch (error) {
      setStatus(`Could not copy the generated code: ${error.message}`, "error");
    }
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

  function handleKeydown(event) {
    if (event.key === "Delete" || event.key === "Backspace") {
      if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        return;
      }
      deleteSelection();
    }
    if (event.key === "Escape") {
      if (state.connectMode) {
        state.pendingIndexId = null;
        state.connectMode = false;
        render();
        setStatus("Connect mode cancelled.");
      } else {
        clearSelection();
      }
    }
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
    render();
    setSelectedElement("index", index.id);
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
    reconcileTensorOrder();
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
    const normalized = spec || {};
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

    reconcileTensorOrder();
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

  function serializeCurrentSpec() {
    return {
      schema_version: 1,
      network: state.spec,
    };
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
    return value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function isIndexNode(element) {
    return element.isNode() && element.data("kind") === "index";
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

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }
})();
