export function registerUtilities(ctx) {
  const state = ctx.state;
  const TENSOR_BASE_Z_INDEX = 10;
  const EDGE_Z_INDEX = 100;
  const PORT_BASE_Z_INDEX = 200;
  const INDEX_LABEL_BASE_Z_INDEX = 230;
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    NOTE_WIDTH,
    NOTE_HEIGHT,
    NOTE_MIN_WIDTH,
    NOTE_MIN_HEIGHT,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  } = ctx.constants;
  const {
    workspace,
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    collectionFormatSelect,
    addNoteButton,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportPyButton,
    exportPngButton,
    exportSvgButton,
    templateSelect,
    templateParameterPanel,
    templateGraphSizeLabel,
    templateGraphSizeInput,
    templateBondDimensionInput,
    templatePhysicalDimensionInput,
    insertTemplateButton,
    createGroupButton,
    helpButton,
    helpModal,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    groupLayer,
    resizeLayer,
    notesLayer,
    selectionBox,
    minimapCanvas,
    sidebar,
    plannerPanel,
  } = ctx.dom;
  const { apiGet, apiPost, window, document, cytoscape } = ctx;
  const ENGINE_LABELS = {
    tensornetwork: "TensorNetwork",
    quimb: "Quimb",
    tensorkrowch: "TensorKrowch",
    einsum_numpy: "NumPy einsum",
    einsum_torch: "PyTorch einsum",
  };
  const COLLECTION_FORMAT_LABELS = {
    list: "List",
    matrix: "Matrix",
    dict: "Dictionary",
  };

  function formatEngineLabel(engineName) {
    return Object.prototype.hasOwnProperty.call(ENGINE_LABELS, engineName)
      ? ENGINE_LABELS[engineName]
      : engineName;
  }

  function populateEngineOptions(engines) {
    engineSelect.innerHTML = "";
    engines.forEach((engineName) => {
      const option = document.createElement("option");
      option.value = engineName;
      option.textContent = formatEngineLabel(engineName);
      if (engineName === state.selectedEngine) {
        option.selected = true;
      }
      engineSelect.appendChild(option);
    });
  }

  function formatCollectionFormatLabel(collectionFormat) {
    return Object.prototype.hasOwnProperty.call(
      COLLECTION_FORMAT_LABELS,
      collectionFormat
    )
      ? COLLECTION_FORMAT_LABELS[collectionFormat]
      : collectionFormat;
  }

  function populateCollectionFormatOptions(collectionFormats) {
    if (!collectionFormatSelect) {
      return;
    }
    collectionFormatSelect.innerHTML = "";
    collectionFormats.forEach((collectionFormat) => {
      const option = document.createElement("option");
      option.value = collectionFormat;
      option.textContent = formatCollectionFormatLabel(collectionFormat);
      if (collectionFormat === state.selectedCollectionFormat) {
        option.selected = true;
      }
      collectionFormatSelect.appendChild(option);
    });
  }

  function populateTemplateOptions(templateNames) {
    templateSelect.innerHTML = "";
    templateNames.forEach((templateName) => {
      const option = document.createElement("option");
      option.value = templateName;
      option.textContent = formatTemplateLabel(templateName);
      templateSelect.appendChild(option);
    });
    if (templateNames.length && !templateSelect.value) {
      templateSelect.value = templateNames[0];
    }
  }

  function formatTemplateLabel(templateName) {
    const definition = getTemplateDefinition(templateName);
    if (definition && typeof definition.display_name === "string" && definition.display_name) {
      return definition.display_name;
    }
    return templateName.replaceAll("_", " ");
  }

  function getTemplateDefinition(templateName = templateSelect.value) {
    if (!templateName || !state.templateDefinitions || typeof state.templateDefinitions !== "object") {
      return null;
    }
    return state.templateDefinitions[templateName] || null;
  }

  function buildTemplateParameterState(templateNames, templateDefinitions) {
    return Object.fromEntries(
      templateNames.map((templateName) => {
        const definition = templateDefinitions && templateDefinitions[templateName]
          ? templateDefinitions[templateName]
          : null;
        const defaults = definition && definition.defaults ? definition.defaults : {};
        return [
          templateName,
          {
            graph_size: sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
            bond_dimension: sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
            physical_dimension: sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
          },
        ];
      })
    );
  }

  function sanitizeTemplateIntegerValue(value, fallback, minimum) {
    const numericValue = Number(value);
    if (!Number.isInteger(numericValue)) {
      return Math.max(minimum, fallback);
    }
    return Math.max(minimum, numericValue);
  }

  function syncTemplateParameterControls(templateName = templateSelect.value) {
    if (!templateParameterPanel) {
      return;
    }
    const definition = getTemplateDefinition(templateName);
    if (!definition) {
      templateParameterPanel.hidden = true;
      return;
    }
    templateParameterPanel.hidden = false;
    const minimums = definition.minimums || {};
    const defaults = definition.defaults || {};
    const parameters = state.templateParametersByTemplate[templateName]
      || buildTemplateParameterState([templateName], { [templateName]: definition })[templateName];
    templateGraphSizeLabel.textContent = `Graph size (${definition.graph_size_label || "Graph size"})`;
    templateGraphSizeInput.min = String(sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1));
    templateBondDimensionInput.min = String(sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1));
    templatePhysicalDimensionInput.min = String(sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1));
    templateGraphSizeInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.graph_size,
        sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
        sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1)
      )
    );
    templateBondDimensionInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.bond_dimension,
        sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
        sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1)
      )
    );
    templatePhysicalDimensionInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.physical_dimension,
        sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
        sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1)
      )
    );
  }

  function readTemplateParametersFromControls() {
    const definition = getTemplateDefinition();
    if (!definition) {
      return {
        graph_size: 2,
        bond_dimension: 3,
        physical_dimension: 2,
      };
    }
    const minimums = definition.minimums || {};
    const defaults = definition.defaults || {};
    const parameters = {
      graph_size: sanitizeTemplateIntegerValue(
        templateGraphSizeInput.value,
        sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
        sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1)
      ),
      bond_dimension: sanitizeTemplateIntegerValue(
        templateBondDimensionInput.value,
        sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
        sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1)
      ),
      physical_dimension: sanitizeTemplateIntegerValue(
        templatePhysicalDimensionInput.value,
        sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
        sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1)
      ),
    };
    templateGraphSizeInput.value = String(parameters.graph_size);
    templateBondDimensionInput.value = String(parameters.bond_dimension);
    templatePhysicalDimensionInput.value = String(parameters.physical_dimension);
    return parameters;
  }

  function persistTemplateParametersFromControls() {
    const templateName = templateSelect.value;
    if (!templateName) {
      return null;
    }
    const parameters = readTemplateParametersFromControls();
    state.templateParametersByTemplate[templateName] = { ...parameters };
    return parameters;
  }

  function handleTemplateSelectionChange(event) {
    if (!event || !event.target) {
      return;
    }
    syncTemplateParameterControls(event.target.value);
    updateToolbarState();
  }

  function handleTemplateParameterInput() {
    persistTemplateParametersFromControls();
  }

  function serializeCurrentSpec() {
    return {
      schema_version: state.schemaVersion,
      network: state.spec,
    };
  }

  function captureEditableFocus() {
    const activeElement = document.activeElement;
    if (!activeElement || !(activeElement instanceof HTMLElement)) {
      return null;
    }
    const focusKey = activeElement.dataset ? activeElement.dataset.focusKey : "";
    if (!focusKey) {
      return null;
    }
    const focusState = {
      key: focusKey,
      selectionStart: null,
      selectionEnd: null,
    };
    if (
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement
    ) {
      focusState.selectionStart = activeElement.selectionStart;
      focusState.selectionEnd = activeElement.selectionEnd;
    }
    return focusState;
  }

  function restoreEditableFocus(focusState) {
    if (!focusState) {
      return;
    }
    const target = Array.from(
      document.querySelectorAll("[data-focus-key]")
    ).find((element) => element.dataset.focusKey === focusState.key);
    if (!(target instanceof HTMLElement)) {
      return;
    }
    target.focus({ preventScroll: true });
    if (
      typeof focusState.selectionStart === "number" &&
      typeof focusState.selectionEnd === "number" &&
      (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)
    ) {
      target.setSelectionRange(
        focusState.selectionStart,
        focusState.selectionEnd
      );
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
    state.spec.groups = state.spec.groups
      .map((group) => ({
        ...group,
        tensor_ids: group.tensor_ids.filter((candidateId) => candidateId !== tensorId),
      }))
      .filter((group) => group.tensor_ids.length > 0);
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

  function findGroupById(groupId) {
    return state.spec.groups.find((group) => group.id === groupId) || null;
  }

  function findGroupsByTensorId(tensorId) {
    return state.spec.groups.filter((group) => group.tensor_ids.includes(tensorId));
  }

  function findEdgeById(edgeId) {
    return state.spec.edges.find((edge) => edge.id === edgeId) || null;
  }

  function findVisibleIndexOwner(indexId) {
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : [];
    for (const tensor of visibleTensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return null;
  }

  function findIndexOwner(indexId) {
    for (const tensor of state.spec.tensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return findVisibleIndexOwner(indexId);
  }

  function findEdgeByIndexId(indexId) {
    const baseEdge = state.spec.edges.find(
      (edge) => edge.left.index_id === indexId || edge.right.index_id === indexId
    );
    if (baseEdge) {
      return baseEdge;
    }
    const visibleEdges =
      typeof ctx.getVisibleEdges === "function" ? ctx.getVisibleEdges() : [];
    return (
      visibleEdges.find(
        (edge) => edge.leftIndexId === indexId || edge.rightIndexId === indexId
      ) || null
    );
  }

  function createTensor(x, y) {
    const tensor = {
      id: makeId("tensor"),
      name: nextName("T", state.spec.tensors.map((tensor) => tensor.name)),
      position: { x, y },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
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
      offset: defaultIndexOffsetForOrder(indexPosition, tensor),
      metadata: {},
    };
  }

  function normalizeSpec(spec) {
    const normalized = deepClone(spec || {});
    normalized.metadata = isObject(normalized.metadata) ? normalized.metadata : {};
    normalized.tensors = Array.isArray(normalized.tensors) ? normalized.tensors : [];
    normalized.groups = Array.isArray(normalized.groups) ? normalized.groups : [];
    normalized.edges = Array.isArray(normalized.edges) ? normalized.edges : [];
    normalized.notes = Array.isArray(normalized.notes) ? normalized.notes : [];
    normalized.contraction_plan = isObject(normalized.contraction_plan)
      ? normalized.contraction_plan
      : null;

    normalized.tensors.forEach((tensor) => {
      tensor.metadata = isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.position = {
        x: asFiniteNumber(tensor.position && tensor.position.x, 120),
        y: asFiniteNumber(tensor.position && tensor.position.y, 120),
      };
      tensor.size = {
        width: Math.max(MIN_TENSOR_WIDTH, asFiniteNumber(tensor.size && tensor.size.width, TENSOR_WIDTH)),
        height: Math.max(MIN_TENSOR_HEIGHT, asFiniteNumber(tensor.size && tensor.size.height, TENSOR_HEIGHT)),
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

    normalized.groups.forEach((group, groupPosition) => {
      group.metadata = isObject(group.metadata) ? group.metadata : {};
      group.tensor_ids = Array.isArray(group.tensor_ids) ? group.tensor_ids.map((tensorId) => String(tensorId)) : [];
      if (!group.id) {
        group.id = makeId("group");
      }
      if (!group.name) {
        group.name = `Group ${groupPosition + 1}`;
      }
    });

    normalized.notes.forEach((note) => {
      note.metadata = isObject(note.metadata) ? note.metadata : {};
      note.position = {
        x: asFiniteNumber(note.position && note.position.x, 120),
        y: asFiniteNumber(note.position && note.position.y, 120),
      };
      note.size = {
        width: Math.max(1, asFiniteNumber(note.size && note.size.width, NOTE_WIDTH)),
        height: Math.max(1, asFiniteNumber(note.size && note.size.height, NOTE_HEIGHT)),
      };
      note.text = typeof note.text === "string" && note.text.trim() ? note.text : "Note";
      if (!note.id) {
        note.id = makeId("note");
      }
    });

    if (normalized.contraction_plan) {
      normalized.contraction_plan.metadata = isObject(normalized.contraction_plan.metadata)
        ? normalized.contraction_plan.metadata
        : {};
      normalized.contraction_plan.steps = Array.isArray(normalized.contraction_plan.steps)
        ? normalized.contraction_plan.steps
        : [];
      normalized.contraction_plan.view_snapshots = Array.isArray(normalized.contraction_plan.view_snapshots)
        ? normalized.contraction_plan.view_snapshots
        : [];
      if (!normalized.contraction_plan.id) {
        normalized.contraction_plan.id = makeId("plan");
      }
      if (!normalized.contraction_plan.name) {
        normalized.contraction_plan.name = "Manual path";
      }
      normalized.contraction_plan.steps.forEach((step) => {
        step.metadata = isObject(step.metadata) ? step.metadata : {};
        if (!step.id) {
          step.id = makeId("step");
        }
        step.left_operand_id = String(step.left_operand_id || "");
        step.right_operand_id = String(step.right_operand_id || "");
      });
      normalized.contraction_plan.view_snapshots.forEach((snapshot, snapshotIndex) => {
        snapshot.applied_step_count = Math.max(
          0,
          Math.round(asFiniteNumber(snapshot.applied_step_count, snapshotIndex))
        );
        snapshot.operand_layouts = Array.isArray(snapshot.operand_layouts)
          ? snapshot.operand_layouts
          : [];
        snapshot.operand_layouts.forEach((layout) => {
          layout.operand_id = String(layout.operand_id || "");
          layout.position = {
            x: asFiniteNumber(layout.position && layout.position.x, 120),
            y: asFiniteNumber(layout.position && layout.position.y, 120),
          };
          layout.size = {
            width: Math.max(
              MIN_TENSOR_WIDTH,
              asFiniteNumber(layout.size && layout.size.width, TENSOR_WIDTH)
            ),
            height: Math.max(
              MIN_TENSOR_HEIGHT,
              asFiniteNumber(layout.size && layout.size.height, TENSOR_HEIGHT)
            ),
          };
        });
      });
    }

    return normalized;
  }

  function ensureTensorIndexOffsets(tensor) {
    const needsAutoLayout =
      tensor.indices.length > 0 &&
      tensor.indices.every((index) => Math.abs(index.offset.x) < 0.001 && Math.abs(index.offset.y) < 0.001);

    tensor.indices.forEach((index, indexPosition) => {
      if (needsAutoLayout) {
        index.offset = defaultIndexOffsetForOrder(indexPosition, tensor);
      } else {
        index.offset = clampIndexOffset(index.offset, tensor);
      }
    });
  }

  function defaultIndexOffsetForOrder(indexPosition, tensor) {
    const scaleX = ctx.tensorWidth(tensor) / TENSOR_WIDTH;
    const scaleY = ctx.tensorHeight(tensor) / TENSOR_HEIGHT;
    const slot = DEFAULT_INDEX_SLOTS[indexPosition];
    if (slot) {
      return clampIndexOffset(
        { x: slot.x * scaleX, y: slot.y * scaleY },
        tensor
      );
    }
    return clampIndexOffset(
      {
        x: (indexPosition % 2 === 0 ? -58 : 58) * scaleX,
        y: (-30 + Math.floor(indexPosition / 2) * 18) * scaleY,
      },
      tensor
    );
  }

  function clampIndexOffset(offset, tensor) {
    return {
      x: clamp(
        asFiniteNumber(offset.x, 0),
        -ctx.tensorWidth(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING,
        ctx.tensorWidth(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
      y: clamp(
        asFiniteNumber(offset.y, 0),
        -ctx.tensorHeight(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING,
        ctx.tensorHeight(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
    };
  }

  function indexAbsolutePosition(tensor, index) {
    const offset = clampIndexOffset(index.offset, tensor);
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

  function runWithTensorSync(action) {
    state.syncingTensorPositions = true;
    try {
      action();
    } finally {
      state.syncingTensorPositions = false;
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
        tensorElement.data("zIndex", TENSOR_BASE_Z_INDEX + tensorRank);
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
          indexElement.data("zIndex", PORT_BASE_Z_INDEX + tensorRank * 10 + indexPosition);
          indexElement.data("orderLabel", String(indexPosition + 1));
          indexElement.data("backgroundColor", indexColor);
          indexElement.data("borderColor", shiftColor(indexColor, 34));
          indexElement.data("textColor", readableTextColor(indexColor));
          indexElement.position(absolutePosition);
        }
        syncIndexLabelNodePosition(index, absolutePosition);
        const labelElement = state.cy.getElementById(indexLabelNodeId(index.id));
        if (labelElement && labelElement.length) {
          labelElement.data("zIndex", INDEX_LABEL_BASE_Z_INDEX + tensorRank * 10 + indexPosition);
        }
      });
    });
    state.cy.edges().forEach((edgeElement) => {
      const edge = findEdgeById(edgeElement.id());
      const edgeColor = edge ? getMetadataColor(edge.metadata, "#8da1c3") : "#8da1c3";
      edgeElement.data("zIndex", EDGE_Z_INDEX);
      edgeElement.data("lineColor", edgeColor);
      edgeElement.data("textColor", shiftColor(edgeColor, 72));
    });
  }

  function applyColorToSelection(colorValue) {
    ctx.getSelectedEntries().forEach((entry) => {
      if (entry.kind === "tensor") {
        entry.tensor.metadata.color = colorValue;
      } else if (entry.kind === "index") {
        entry.located.index.metadata.color = colorValue;
      } else if (entry.kind === "edge") {
        entry.edge.metadata.color = colorValue;
      } else if (entry.kind === "group") {
        entry.group.metadata.color = colorValue;
      } else if (entry.kind === "note") {
        entry.note.metadata.color = colorValue;
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
    if (entry.kind === "group") {
      return getMetadataColor(entry.group.metadata, "#61a8ff");
    }
    if (entry.kind === "note") {
      return getMetadataColor(entry.note.metadata, "#5f95ff");
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
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : state.spec.tensors;
    const visibleEdges =
      typeof ctx.getVisibleEdges === "function" ? ctx.getVisibleEdges() : state.spec.edges;

    visibleTensors.forEach((tensor) => {
      expandBounds(bounds, tensor.position.x - ctx.tensorWidth(tensor) / 2, tensor.position.y - ctx.tensorHeight(tensor) / 2);
      expandBounds(bounds, tensor.position.x + ctx.tensorWidth(tensor) / 2, tensor.position.y + ctx.tensorHeight(tensor) / 2);
      tensor.indices.forEach((index) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        expandBounds(bounds, absolutePosition.x - INDEX_RADIUS, absolutePosition.y - INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + INDEX_RADIUS, absolutePosition.y + INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + 50, absolutePosition.y + 42);
      });
    });

    visibleEdges.forEach((edge) => {
      const left = findIndexOwner(edge.leftIndexId || edge.left.index_id);
      const right = findIndexOwner(edge.rightIndexId || edge.right.index_id);
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


  function updateToolbarState() {
    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
    exportPyButton.disabled = !state.spec || !state.selectedEngine;
    insertTemplateButton.disabled = !templateSelect.value;
    createGroupButton.disabled = ctx.getSelectedIdsByKind("tensor").length < 2;
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
    return JSON.parse(JSON.stringify(value));
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

  Object.assign(ctx, {
    populateEngineOptions,
    formatEngineLabel,
    populateCollectionFormatOptions,
    formatCollectionFormatLabel,
    populateTemplateOptions,
    formatTemplateLabel,
    getTemplateDefinition,
    buildTemplateParameterState,
    syncTemplateParameterControls,
    readTemplateParametersFromControls,
    persistTemplateParametersFromControls,
    handleTemplateSelectionChange,
    handleTemplateParameterInput,
    serializeCurrentSpec,
    captureEditableFocus,
    restoreEditableFocus,
    stripImportLines,
    moveIndex,
    removeTensor,
    removeIndex,
    removeEdge,
    findTensorById,
    findGroupById,
    findGroupsByTensorId,
    findEdgeById,
    findVisibleIndexOwner,
    findIndexOwner,
    findEdgeByIndexId,
    createTensor,
    createIndex,
    normalizeSpec,
    ensureTensorIndexOffsets,
    defaultIndexOffsetForOrder,
    clampIndexOffset,
    indexAbsolutePosition,
    syncIndexNodePositions,
    syncSingleIndexNodePosition,
    syncIndexLabelNodePosition,
    runWithIndexSync,
    runWithTensorSync,
    reconcileTensorOrder,
    tensorLayerRank,
    bringTensorToFront,
    applyTensorLayerData,
    applyColorToSelection,
    getBatchColorValue,
    getEntryColor,
    buildQuadraticCurve,
    quadraticPointAt,
    drawRoundRectPath,
    computeDesignBounds,
    expandBounds,
    downloadDataUrl,
    downloadBlob,
    updateToolbarState,
    formatIssues,
    setStatus,
    sanitizeFilename,
    escapeHtml,
    escapeSvgText,
    isIndexNode,
    isTextInput,
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
    isObject,
    asFiniteNumber,
    makeId,
    nextName,
    tensorIndexNameExists,
    clamp
  });
}
