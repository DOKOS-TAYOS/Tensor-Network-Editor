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
    exportFormatSelect,
    addNoteButton,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportButton,
    toggleLinearPeriodicButton,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    linearPeriodicNextCellButton,
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
    generateButton,
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
  const LINEAR_PERIODIC_SUPPORTED_ENGINES = new Set([
    "tensornetwork",
    "tensorkrowch",
  ]);
  const LINEAR_PERIODIC_CELL_ORDER = ["initial", "periodic", "final"];
  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };
  const LINEAR_PERIODIC_BOUNDARY_SETTINGS = {
    previous: {
      name: "Previous cell",
      color: "#456cbf",
    },
    next: {
      name: "Next cell",
      color: "#2f9b8f",
    },
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
    enforceLinearPeriodicEngineSupport();
  }

  function formatCollectionFormatLabel(collectionFormat) {
    return Object.prototype.hasOwnProperty.call(
      COLLECTION_FORMAT_LABELS,
      collectionFormat
    )
      ? COLLECTION_FORMAT_LABELS[collectionFormat]
      : collectionFormat;
  }

  function buildEmptyGraphSection() {
    return {
      tensors: [],
      groups: [],
      edges: [],
      notes: [],
      contraction_plan: null,
      metadata: {},
    };
  }

  function clearGraphSectionOnSpec(spec) {
    spec.tensors = [];
    spec.groups = [];
    spec.edges = [];
    spec.notes = [];
    spec.contraction_plan = null;
  }

  function buildGraphSectionFromSpec(spec, existingCell = null) {
    return {
      tensors: deepClone(Array.isArray(spec && spec.tensors) ? spec.tensors : []),
      groups: deepClone(Array.isArray(spec && spec.groups) ? spec.groups : []),
      edges: deepClone(Array.isArray(spec && spec.edges) ? spec.edges : []),
      notes: deepClone(Array.isArray(spec && spec.notes) ? spec.notes : []),
      contraction_plan: deepClone(
        spec && isObject(spec.contraction_plan) ? spec.contraction_plan : null
      ),
      metadata: deepClone(
        existingCell && isObject(existingCell.metadata) ? existingCell.metadata : {}
      ),
    };
  }

  function replaceGraphSectionOnSpec(spec, graphSection) {
    const nextSection = normalizeGraphSectionInPlace(
      deepClone(graphSection || buildEmptyGraphSection())
    );
    spec.tensors = nextSection.tensors;
    spec.groups = nextSection.groups;
    spec.edges = nextSection.edges;
    spec.notes = nextSection.notes;
    spec.contraction_plan = nextSection.contraction_plan;
  }

  function normalizeGraphSectionInPlace(graphSection) {
    graphSection.metadata = isObject(graphSection.metadata)
      ? graphSection.metadata
      : {};
    graphSection.tensors = Array.isArray(graphSection.tensors)
      ? graphSection.tensors
      : [];
    graphSection.groups = Array.isArray(graphSection.groups)
      ? graphSection.groups
      : [];
    graphSection.edges = Array.isArray(graphSection.edges) ? graphSection.edges : [];
    graphSection.notes = Array.isArray(graphSection.notes) ? graphSection.notes : [];
    graphSection.contraction_plan = isObject(graphSection.contraction_plan)
      ? graphSection.contraction_plan
      : null;

    graphSection.tensors.forEach((tensor) => {
      tensor.metadata = isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.position = {
        x: asFiniteNumber(tensor.position && tensor.position.x, 120),
        y: asFiniteNumber(tensor.position && tensor.position.y, 120),
      };
      tensor.size = {
        width: Math.max(
          MIN_TENSOR_WIDTH,
          asFiniteNumber(tensor.size && tensor.size.width, TENSOR_WIDTH)
        ),
        height: Math.max(
          MIN_TENSOR_HEIGHT,
          asFiniteNumber(tensor.size && tensor.size.height, TENSOR_HEIGHT)
        ),
      };
      tensor.linear_periodic_role =
        tensor.linear_periodic_role === "previous" ||
        tensor.linear_periodic_role === "next"
          ? tensor.linear_periodic_role
          : null;
      tensor.indices = Array.isArray(tensor.indices) ? tensor.indices : [];
      tensor.indices.forEach((index, indexPosition) => {
        index.metadata = isObject(index.metadata) ? index.metadata : {};
        index.dimension = Math.max(1, Math.round(asFiniteNumber(index.dimension, 2)));
        index.offset = {
          x: asFiniteNumber(index.offset && index.offset.x, 0),
          y: asFiniteNumber(index.offset && index.offset.y, 0),
        };
        if (!index.id) {
          index.id = makeId("index");
        }
        if (!index.name) {
          index.name = nextName(
            "i",
            tensor.indices
              .slice(0, indexPosition)
              .map((candidate) => candidate.name)
          );
        }
      });
      if (!tensor.id) {
        tensor.id = makeId("tensor");
      }
      if (!tensor.name) {
        tensor.name = nextName(
          "T",
          graphSection.tensors
            .slice(0, graphSection.tensors.indexOf(tensor))
            .map((candidate) => candidate.name)
        );
      }
      ensureTensorIndexOffsets(tensor);
    });

    graphSection.groups.forEach((group, groupPosition) => {
      group.metadata = isObject(group.metadata) ? group.metadata : {};
      group.tensor_ids = Array.isArray(group.tensor_ids)
        ? group.tensor_ids.map((tensorId) => String(tensorId))
        : [];
      if (!group.id) {
        group.id = makeId("group");
      }
      if (!group.name) {
        group.name = `Group ${groupPosition + 1}`;
      }
    });

    graphSection.edges.forEach((edge, edgePosition) => {
      edge.metadata = isObject(edge.metadata) ? edge.metadata : {};
      if (!edge.id) {
        edge.id = makeId("edge");
      }
      if (!edge.name) {
        edge.name = `bond_${edgePosition + 1}`;
      }
      edge.left = isObject(edge.left) ? edge.left : {};
      edge.right = isObject(edge.right) ? edge.right : {};
      edge.left.tensor_id = String(edge.left.tensor_id || "");
      edge.left.index_id = String(edge.left.index_id || "");
      edge.right.tensor_id = String(edge.right.tensor_id || "");
      edge.right.index_id = String(edge.right.index_id || "");
    });

    graphSection.notes.forEach((note) => {
      note.metadata = isObject(note.metadata) ? note.metadata : {};
      note.position = {
        x: asFiniteNumber(note.position && note.position.x, 120),
        y: asFiniteNumber(note.position && note.position.y, 120),
      };
      note.size = {
        width: Math.max(1, asFiniteNumber(note.size && note.size.width, NOTE_WIDTH)),
        height: Math.max(
          1,
          asFiniteNumber(note.size && note.size.height, NOTE_HEIGHT)
        ),
      };
      note.text =
        typeof note.text === "string" && note.text.trim() ? note.text : "Note";
      if (!note.id) {
        note.id = makeId("note");
      }
    });

    if (graphSection.contraction_plan) {
      graphSection.contraction_plan.metadata = isObject(
        graphSection.contraction_plan.metadata
      )
        ? graphSection.contraction_plan.metadata
        : {};
      graphSection.contraction_plan.steps = Array.isArray(
        graphSection.contraction_plan.steps
      )
        ? graphSection.contraction_plan.steps
        : [];
      graphSection.contraction_plan.view_snapshots = Array.isArray(
        graphSection.contraction_plan.view_snapshots
      )
        ? graphSection.contraction_plan.view_snapshots
        : [];
      if (!graphSection.contraction_plan.id) {
        graphSection.contraction_plan.id = makeId("plan");
      }
      if (!graphSection.contraction_plan.name) {
        graphSection.contraction_plan.name = "Manual path";
      }
      graphSection.contraction_plan.steps.forEach((step) => {
        step.metadata = isObject(step.metadata) ? step.metadata : {};
        if (!step.id) {
          step.id = makeId("step");
        }
        step.left_operand_id = String(step.left_operand_id || "");
        step.right_operand_id = String(step.right_operand_id || "");
      });
      graphSection.contraction_plan.view_snapshots.forEach(
        (snapshot, snapshotIndex) => {
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
        }
      );
    }

    return graphSection;
  }

  function normalizeLinearPeriodicChainInPlace(chain) {
    chain.metadata = isObject(chain.metadata) ? chain.metadata : {};
    chain.active_cell = LINEAR_PERIODIC_CELL_ORDER.includes(chain.active_cell)
      ? chain.active_cell
      : "initial";
    chain.initial_cell = normalizeGraphSectionInPlace(
      deepClone(isObject(chain.initial_cell) ? chain.initial_cell : buildEmptyGraphSection())
    );
    chain.periodic_cell = normalizeGraphSectionInPlace(
      deepClone(
        isObject(chain.periodic_cell) ? chain.periodic_cell : buildEmptyGraphSection()
      )
    );
    chain.final_cell = normalizeGraphSectionInPlace(
      deepClone(isObject(chain.final_cell) ? chain.final_cell : buildEmptyGraphSection())
    );
    return chain;
  }

  function getLinearPeriodicChain(spec = state.spec) {
    return spec && isObject(spec.linear_periodic_chain)
      ? spec.linear_periodic_chain
      : null;
  }

  function isLinearPeriodicMode(spec = state.spec) {
    return Boolean(getLinearPeriodicChain(spec));
  }

  function getActiveLinearPeriodicCellName(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    return chain ? chain.active_cell : null;
  }

  function getLinearPeriodicCell(spec = state.spec, cellName = getActiveLinearPeriodicCellName(spec)) {
    const chain = getLinearPeriodicChain(spec);
    if (!chain || !cellName) {
      return null;
    }
    return chain[`${cellName}_cell`] || null;
  }

  function isLinearPeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.linear_periodic_role === "previous" ||
          tensor.linear_periodic_role === "next")
    );
  }

  function getContractibleTensors(spec = state.spec) {
    return Array.isArray(spec && spec.tensors)
      ? spec.tensors.filter((tensor) => !isLinearPeriodicBoundaryTensor(tensor))
      : [];
  }

  function getContractibleEdges(spec = state.spec) {
    const tensorById = Object.fromEntries(
      (Array.isArray(spec && spec.tensors) ? spec.tensors : []).map((tensor) => [
        tensor.id,
        tensor,
      ])
    );
    return Array.isArray(spec && spec.edges)
      ? spec.edges.filter((edge) => {
          const leftTensor = tensorById[edge.left && edge.left.tensor_id];
          const rightTensor = tensorById[edge.right && edge.right.tensor_id];
          return (
            leftTensor &&
            rightTensor &&
            !isLinearPeriodicBoundaryTensor(leftTensor) &&
            !isLinearPeriodicBoundaryTensor(rightTensor)
          );
        })
      : [];
  }

  function getExpectedLinearPeriodicRoles(cellName) {
    if (cellName === "initial") {
      return ["next"];
    }
    if (cellName === "periodic") {
      return ["previous", "next"];
    }
    if (cellName === "final") {
      return ["previous"];
    }
    return [];
  }

  function getRealTensorBounds(spec = state.spec) {
    const tensors = getContractibleTensors(spec);
    if (!tensors.length) {
      return {
        minX: 120,
        maxX: 320,
        centerY: 140,
      };
    }
    const leftEdges = tensors.map((tensor) => tensor.position.x - tensorWidth(tensor) / 2);
    const rightEdges = tensors.map(
      (tensor) => tensor.position.x + tensorWidth(tensor) / 2
    );
    const centersY = tensors.map((tensor) => tensor.position.y);
    return {
      minX: Math.min(...leftEdges),
      maxX: Math.max(...rightEdges),
      centerY: centersY.reduce((sum, value) => sum + value, 0) / centersY.length,
    };
  }

  function positionLinearPeriodicBoundaryTensor(tensor, role, spec = state.spec) {
    const bounds = getRealTensorBounds(spec);
    tensor.position = {
      x: role === "previous" ? bounds.minX - 220 : bounds.maxX + 220,
      y: bounds.centerY,
    };
  }

  function createLinearPeriodicBoundaryTensor(role, spec = state.spec) {
    const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[role];
    const tensor = {
      id: makeId("tensor"),
      name: settings.name,
      position: { x: 0, y: 0 },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      linear_periodic_role: role,
      metadata: { color: settings.color },
    };
    positionLinearPeriodicBoundaryTensor(tensor, role, spec);
    return tensor;
  }

  function ensureActiveLinearPeriodicBoundaryTensors(spec = state.spec) {
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!activeCellName) {
      return;
    }
    const expectedRoles = new Set(getExpectedLinearPeriodicRoles(activeCellName));
    const nextTensors = [];
    const seenRoles = new Set();
    (Array.isArray(spec.tensors) ? spec.tensors : []).forEach((tensor) => {
      if (!isLinearPeriodicBoundaryTensor(tensor)) {
        nextTensors.push(tensor);
        return;
      }
      if (!expectedRoles.has(tensor.linear_periodic_role) || seenRoles.has(tensor.linear_periodic_role)) {
        return;
      }
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[tensor.linear_periodic_role];
      tensor.linear_periodic_role = tensor.linear_periodic_role;
      tensor.name = settings.name;
      tensor.metadata = isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.metadata.color = getMetadataColor(tensor.metadata, settings.color);
      seenRoles.add(tensor.linear_periodic_role);
      nextTensors.push(tensor);
    });
    getExpectedLinearPeriodicRoles(activeCellName).forEach((role) => {
      if (!seenRoles.has(role)) {
        nextTensors.push(createLinearPeriodicBoundaryTensor(role, spec));
      }
    });
    spec.tensors = nextTensors;
    const allowedBoundaryIds = new Set(
      spec.tensors.filter((tensor) => isLinearPeriodicBoundaryTensor(tensor)).map((tensor) => tensor.id)
    );
    spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter((edge) => {
      const touchesBoundary =
        allowedBoundaryIds.has(edge.left && edge.left.tensor_id) ||
        allowedBoundaryIds.has(edge.right && edge.right.tensor_id);
      if (!touchesBoundary) {
        return true;
      }
      return (
        allowedBoundaryIds.has(edge.left && edge.left.tensor_id) ||
        allowedBoundaryIds.has(edge.right && edge.right.tensor_id)
      );
    });
  }

  function getLinearPeriodicCandidateOwners(spec = state.spec) {
    const tensorById = Object.fromEntries(
      (Array.isArray(spec.tensors) ? spec.tensors : []).map((tensor) => [tensor.id, tensor])
    );
    const internallyConnectedIndexIds = new Set();
    (Array.isArray(spec.edges) ? spec.edges : []).forEach((edge) => {
      const leftTensor = tensorById[edge.left && edge.left.tensor_id];
      const rightTensor = tensorById[edge.right && edge.right.tensor_id];
      if (
        leftTensor &&
        rightTensor &&
        !isLinearPeriodicBoundaryTensor(leftTensor) &&
        !isLinearPeriodicBoundaryTensor(rightTensor)
      ) {
        internallyConnectedIndexIds.add(edge.left.index_id);
        internallyConnectedIndexIds.add(edge.right.index_id);
      }
    });
    const owners = [];
    getContractibleTensors(spec).forEach((tensor) => {
      tensor.indices.forEach((index, indexPosition) => {
        if (!internallyConnectedIndexIds.has(index.id)) {
          owners.push({ tensor, index, indexPosition });
        }
      });
    });
    return owners;
  }

  function getBoundaryInterfaceDimensions(
    graphSection,
    preferredRole = "next"
  ) {
    const boundaryTensors = (Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : [])
      .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor));
    const preferredTensor = boundaryTensors.find(
      (tensor) =>
        tensor.linear_periodic_role === preferredRole &&
        Array.isArray(tensor.indices) &&
        tensor.indices.length
    );
    const fallbackTensor =
      preferredTensor ||
      boundaryTensors.find(
        (tensor) => Array.isArray(tensor.indices) && tensor.indices.length
      );
    return fallbackTensor
      ? fallbackTensor.indices.map((index) => index.dimension)
      : [];
  }

  function getCanonicalLinearPeriodicInterfaceDimensions(spec = state.spec) {
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!activeCellName) {
      return [];
    }
    if (activeCellName === "initial") {
      return getLinearPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
    }
    const activePreferredRole = activeCellName === "final" ? "previous" : "next";
    const activeDimensions = getBoundaryInterfaceDimensions(
      spec,
      activePreferredRole
    );
    if (activeDimensions.length) {
      return activeDimensions;
    }
    const chain = getLinearPeriodicChain(spec);
    if (!chain) {
      return getLinearPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
    }
    const initialDimensions = getBoundaryInterfaceDimensions(
      chain.initial_cell,
      "next"
    );
    return initialDimensions.length
      ? initialDimensions
      : getLinearPeriodicCandidateOwners(spec).map(
          (owner) => owner.index.dimension
        );
  }

  function syncLinearPeriodicBoundaryTensors(
    spec = state.spec,
    interfaceDimensions = null
  ) {
    if (!isLinearPeriodicMode(spec)) {
      return;
    }
    ensureActiveLinearPeriodicBoundaryTensors(spec);
    const resolvedInterfaceDimensions = Array.isArray(interfaceDimensions)
      ? interfaceDimensions
      : getCanonicalLinearPeriodicInterfaceDimensions(spec);
    const boundaryTensors = (Array.isArray(spec.tensors) ? spec.tensors : []).filter((tensor) =>
      isLinearPeriodicBoundaryTensor(tensor)
    );
    boundaryTensors.forEach((boundaryTensor) => {
      const existingIndices = Array.isArray(boundaryTensor.indices)
        ? boundaryTensor.indices
        : [];
      const keptIndices = existingIndices.slice(0, resolvedInterfaceDimensions.length);
      const removedIndexIds = new Set(
        existingIndices.slice(resolvedInterfaceDimensions.length).map((index) => index.id)
      );
      if (removedIndexIds.size) {
        spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
          (edge) =>
            !removedIndexIds.has(edge.left && edge.left.index_id) &&
            !removedIndexIds.has(edge.right && edge.right.index_id)
        );
      }
      boundaryTensor.indices = resolvedInterfaceDimensions.map(
        (dimension, indexPosition) => {
        const existingIndex = keptIndices[indexPosition];
        return {
          id: existingIndex && existingIndex.id ? existingIndex.id : makeId("index"),
          name: `slot_${indexPosition + 1}`,
          dimension,
          offset:
            existingIndex && existingIndex.offset
              ? existingIndex.offset
              : defaultIndexOffsetForOrder(indexPosition, boundaryTensor),
          metadata:
            existingIndex && isObject(existingIndex.metadata)
              ? existingIndex.metadata
              : {},
        };
        }
      );
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[boundaryTensor.linear_periodic_role];
      boundaryTensor.name = settings.name;
      boundaryTensor.metadata = isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = getMetadataColor(
        boundaryTensor.metadata,
        settings.color
      );
      ensureTensorIndexOffsets(boundaryTensor);
    });
  }

  function syncLinearPeriodicChainInterfaceDimensions(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!chain || !activeCellName) {
      return spec;
    }
    const activeCell = getLinearPeriodicCell(spec, activeCellName);
    const interfaceDimensions = getCanonicalLinearPeriodicInterfaceDimensions(spec);

    chain[`${activeCellName}_cell`] = buildGraphSectionFromSpec(spec, activeCell);

    LINEAR_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellSpec = normalizeGraphSectionInPlace(
        deepClone(getLinearPeriodicCell(spec, cellName) || buildEmptyGraphSection())
      );
      chain[`${cellName}_cell`] = seedLinearPeriodicCell(
        cellName,
        cellSpec,
        interfaceDimensions
      );
    });

    replaceGraphSectionOnSpec(
      spec,
      getLinearPeriodicCell(spec, activeCellName) || buildEmptyGraphSection()
    );

    return spec;
  }

  function syncCurrentGraphIntoLinearPeriodicChain(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    if (!chain) {
      return spec;
    }
    syncLinearPeriodicBoundaryTensors(spec);
    syncLinearPeriodicChainInterfaceDimensions(spec);
    return spec;
  }

  function hydrateActiveLinearPeriodicCell(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    if (!chain) {
      return spec;
    }
    chain.active_cell = LINEAR_PERIODIC_CELL_ORDER.includes(chain.active_cell)
      ? chain.active_cell
      : "initial";
    const activeCell = getLinearPeriodicCell(spec, chain.active_cell);
    replaceGraphSectionOnSpec(spec, activeCell || buildEmptyGraphSection());
    syncLinearPeriodicBoundaryTensors(spec);
    return spec;
  }

  function stripLinearPeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = normalizeGraphSectionInPlace(
      deepClone(graphSection || buildEmptyGraphSection())
    );
    const boundaryTensorIds = new Set(
      stripped.tensors
        .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
        .map((tensor) => tensor.id)
    );
    stripped.tensors = stripped.tensors.filter(
      (tensor) => !boundaryTensorIds.has(tensor.id)
    );
    stripped.edges = stripped.edges.filter(
      (edge) =>
        !boundaryTensorIds.has(edge.left && edge.left.tensor_id) &&
        !boundaryTensorIds.has(edge.right && edge.right.tensor_id)
    );
    stripped.groups = stripped.groups
      .map((group) => ({
        ...group,
        tensor_ids: group.tensor_ids.filter((tensorId) => !boundaryTensorIds.has(tensorId)),
      }))
      .filter((group) => group.tensor_ids.length > 0);
    return stripped;
  }

  function seedLinearPeriodicCell(
    cellName,
    graphSection,
    interfaceDimensions = null
  ) {
    const runtimeSpec = normalizeGraphSectionInPlace(
      deepClone(graphSection || buildEmptyGraphSection())
    );
    runtimeSpec.linear_periodic_chain = { active_cell: cellName };
    ensureActiveLinearPeriodicBoundaryTensors(runtimeSpec);
    syncLinearPeriodicBoundaryTensors(runtimeSpec, interfaceDimensions);
    return buildGraphSectionFromSpec(runtimeSpec);
  }

  function buildHistorySnapshotSpec(spec = state.spec) {
    const snapshotSpec = deepClone(spec || {});
    if (isLinearPeriodicMode(snapshotSpec)) {
      syncCurrentGraphIntoLinearPeriodicChain(snapshotSpec);
    }
    return snapshotSpec;
  }

  function buildSerializedSpec(spec = state.spec) {
    const serializedSpec = deepClone(spec || {});
    if (isLinearPeriodicMode(serializedSpec)) {
      syncCurrentGraphIntoLinearPeriodicChain(serializedSpec);
      clearGraphSectionOnSpec(serializedSpec);
    }
    return serializedSpec;
  }

  function resetTransientEditorStateForCellSwitch() {
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerInspectionStepCount = null;
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    state.activeTensorDrag = null;
    state.activeIndexDrag = null;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.boxSelection = null;
    state.connectMode = false;
  }

  function switchLinearPeriodicCell(direction) {
    if (!isLinearPeriodicMode()) {
      return;
    }
    const activeCellName = getActiveLinearPeriodicCellName();
    const activeIndex = LINEAR_PERIODIC_CELL_ORDER.indexOf(activeCellName);
    const nextIndex = clamp(
      activeIndex + direction,
      0,
      LINEAR_PERIODIC_CELL_ORDER.length - 1
    );
    if (nextIndex === activeIndex) {
      return;
    }
    syncCurrentGraphIntoLinearPeriodicChain();
    state.spec.linear_periodic_chain.active_cell =
      LINEAR_PERIODIC_CELL_ORDER[nextIndex];
    hydrateActiveLinearPeriodicCell();
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    ctx.reconcileTensorOrder();
    resetTransientEditorStateForCellSwitch();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.setStatus(
      `Editing ${LINEAR_PERIODIC_CELL_LABELS[state.spec.linear_periodic_chain.active_cell].toLowerCase()}.`,
      "success"
    );
  }

  function enforceLinearPeriodicEngineSupport() {
    if (!engineSelect) {
      return false;
    }
    const linearPeriodicMode = isLinearPeriodicMode();
    Array.from(engineSelect.options).forEach((option) => {
      option.disabled =
        linearPeriodicMode &&
        !LINEAR_PERIODIC_SUPPORTED_ENGINES.has(option.value);
    });
    if (
      linearPeriodicMode &&
      !LINEAR_PERIODIC_SUPPORTED_ENGINES.has(state.selectedEngine)
    ) {
      state.selectedEngine = "tensornetwork";
      engineSelect.value = state.selectedEngine;
      return true;
    }
    return false;
  }

  function toggleLinearPeriodicMode() {
    if (!state.spec) {
      return;
    }
    if (isLinearPeriodicMode()) {
      if (
        !window.confirm(
          "Leave For mode and keep only the initial cell? The periodic and final cells will be discarded."
        )
      ) {
        return;
      }
      syncCurrentGraphIntoLinearPeriodicChain();
      const plainInitialCell = stripLinearPeriodicBoundaryTensorsFromGraphSection(
        state.spec.linear_periodic_chain.initial_cell
      );
      state.spec.linear_periodic_chain = null;
      replaceGraphSectionOnSpec(state.spec, plainInitialCell);
      if (typeof ctx.bumpSpecRevision === "function") {
        ctx.bumpSpecRevision();
      }
      ctx.reconcileTensorOrder();
      resetTransientEditorStateForCellSwitch();
      if (typeof ctx.clearGeneratedCodePreview === "function") {
        ctx.clearGeneratedCodePreview();
      }
      ctx.render();
      if (typeof ctx.refreshContractionAnalysis === "function") {
        ctx.refreshContractionAnalysis();
      }
      ctx.setStatus("For mode disabled. Restored the initial cell as a normal network.", "success");
      return;
    }

    const initialCell = seedLinearPeriodicCell("initial", buildGraphSectionFromSpec(state.spec));
    const periodicCell = seedLinearPeriodicCell("periodic", buildEmptyGraphSection());
    const finalCell = seedLinearPeriodicCell("final", buildEmptyGraphSection());
    state.spec.linear_periodic_chain = {
      active_cell: "initial",
      initial_cell: initialCell,
      periodic_cell: periodicCell,
      final_cell: finalCell,
      metadata: {},
    };
    hydrateActiveLinearPeriodicCell();
    if (enforceLinearPeriodicEngineSupport()) {
      ctx.setStatus(
        "For mode enabled. TensorNetwork is selected because this mode currently supports TensorNetwork and TensorKrowch.",
        "success"
      );
    } else {
      ctx.setStatus("For mode enabled. You are editing the initial cell.", "success");
    }
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    ctx.reconcileTensorOrder();
    resetTransientEditorStateForCellSwitch();
    if (typeof ctx.clearGeneratedCodePreview === "function") {
      ctx.clearGeneratedCodePreview();
    }
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
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

  function bumpSpecRevision() {
    state.specRevision += 1;
    state.lookupRevision = -1;
  }

  function ensureSpecLookups() {
    if (!state.spec) {
      state.lookupRevision = state.specRevision;
      state.tensorById = {};
      state.edgeById = {};
      state.edgeByIndexId = {};
      state.groupById = {};
      state.indexOwnerById = {};
      state.groupsByTensorId = {};
      state.noteById = {};
      return;
    }
    if (state.lookupRevision === state.specRevision) {
      return;
    }

    const tensorById = {};
    const edgeById = {};
    const edgeByIndexId = {};
    const groupById = {};
    const indexOwnerById = {};
    const groupsByTensorId = {};
    const noteById = {};

    state.spec.tensors.forEach((tensor) => {
      tensorById[tensor.id] = tensor;
      tensor.indices.forEach((index, indexPosition) => {
        indexOwnerById[index.id] = { tensor, index, indexPosition };
      });
    });
    state.spec.edges.forEach((edge) => {
      edgeById[edge.id] = edge;
      edgeByIndexId[edge.left.index_id] = edge;
      edgeByIndexId[edge.right.index_id] = edge;
    });
    state.spec.groups.forEach((group) => {
      groupById[group.id] = group;
      group.tensor_ids.forEach((tensorId) => {
        if (!Array.isArray(groupsByTensorId[tensorId])) {
          groupsByTensorId[tensorId] = [];
        }
        groupsByTensorId[tensorId].push(group);
      });
    });
    state.spec.notes.forEach((note) => {
      noteById[note.id] = note;
    });

    state.tensorById = tensorById;
    state.edgeById = edgeById;
    state.edgeByIndexId = edgeByIndexId;
    state.groupById = groupById;
    state.indexOwnerById = indexOwnerById;
    state.groupsByTensorId = groupsByTensorId;
    state.noteById = noteById;
    state.lookupRevision = state.specRevision;
  }

  function serializeCurrentSpec(options = {}) {
    const { persistViewSnapshots = false } = options;
    if (
      persistViewSnapshots &&
      state.spec &&
      state.spec.contraction_plan &&
      typeof ctx.ensureContractionViewSnapshots === "function"
    ) {
      ctx.ensureContractionViewSnapshots();
    }
    return {
      schema_version: state.schemaVersion,
      network: buildSerializedSpec(),
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
    if (!tensor || isLinearPeriodicBoundaryTensor(tensor)) {
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

  function findBaseIndexOwner(indexId) {
    ensureSpecLookups();
    return state.indexOwnerById[indexId] || null;
  }

  function removeIndex(tensorId, indexId) {
    const tensor = findTensorById(tensorId);
    if (!tensor || isLinearPeriodicBoundaryTensor(tensor)) {
      return;
    }
    state.spec.edges = state.spec.edges.filter(
      (edge) => edge.left.index_id !== indexId && edge.right.index_id !== indexId
    );
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function resolveBaseEdgeId(edgeId) {
    if (!edgeId) {
      return null;
    }
    ensureSpecLookups();
    const baseEdge = state.edgeById[edgeId];
    if (baseEdge) {
      return baseEdge.id;
    }
    const visibleEdge =
      typeof ctx.findVisibleEdgeById === "function"
        ? ctx.findVisibleEdgeById(edgeId)
        : null;
    if (
      visibleEdge &&
      typeof visibleEdge.baseEdgeId === "string" &&
      visibleEdge.baseEdgeId
    ) {
      return visibleEdge.baseEdgeId;
    }
    return null;
  }

  function removeEdge(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId) || edgeId;
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== resolvedEdgeId);
  }

  function findTensorById(tensorId) {
    ensureSpecLookups();
    return state.tensorById[tensorId] || null;
  }

  function findGroupById(groupId) {
    ensureSpecLookups();
    return state.groupById[groupId] || null;
  }

  function findGroupsByTensorId(tensorId) {
    ensureSpecLookups();
    return state.groupsByTensorId[tensorId] || [];
  }

  function findEdgeById(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId);
    if (!resolvedEdgeId) {
      return null;
    }
    ensureSpecLookups();
    return state.edgeById[resolvedEdgeId] || null;
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
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    return findVisibleIndexOwner(indexId);
  }

  function resolveConnectableIndexOwner(indexId) {
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    const visibleOwner = findVisibleIndexOwner(indexId);
    if (
      !visibleOwner ||
      typeof visibleOwner.index.sourceIndexId !== "string" ||
      !visibleOwner.index.sourceIndexId
    ) {
      return null;
    }
    return findBaseIndexOwner(visibleOwner.index.sourceIndexId);
  }

  function findEdgeByIndexId(indexId) {
    ensureSpecLookups();
    const baseEdge = state.edgeByIndexId[indexId];
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

  function syncConnectedIndexDimension(indexId, nextDimension) {
    const connectedEdge = findEdgeByIndexId(indexId);
    if (!connectedEdge) {
      return;
    }
    const connectedIndexId =
      connectedEdge.left && connectedEdge.left.index_id === indexId
        ? connectedEdge.right && connectedEdge.right.index_id
        : connectedEdge.left && connectedEdge.left.index_id;
    if (!connectedIndexId) {
      return;
    }
    const connectedOwner = findIndexOwner(connectedIndexId);
    if (!connectedOwner || !connectedOwner.index) {
      return;
    }
    connectedOwner.index.dimension = nextDimension;
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
    normalizeGraphSectionInPlace(normalized);
    normalized.linear_periodic_chain = isObject(normalized.linear_periodic_chain)
      ? normalizeLinearPeriodicChainInPlace(normalized.linear_periodic_chain)
      : null;
    if (normalized.linear_periodic_chain) {
      hydrateActiveLinearPeriodicCell(normalized);
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
    const linearPeriodicMode = isLinearPeriodicMode();
    const activeLinearPeriodicCell = getActiveLinearPeriodicCellName();
    enforceLinearPeriodicEngineSupport();
    const unsupportedLinearPeriodicEngine =
      linearPeriodicMode &&
      !LINEAR_PERIODIC_SUPPORTED_ENGINES.has(state.selectedEngine);
    const selectedExportFormat = exportFormatSelect ? exportFormatSelect.value : "py";
    const exportNeedsEngine = selectedExportFormat === "py";

    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
    if (exportButton) {
      exportButton.disabled =
        !state.spec ||
        (exportNeedsEngine && (!state.selectedEngine || unsupportedLinearPeriodicEngine));
    }
    if (generateButton) {
      generateButton.disabled =
        !state.spec || !state.selectedEngine || unsupportedLinearPeriodicEngine;
    }
    insertTemplateButton.disabled = !templateSelect.value;
    createGroupButton.disabled = ctx.getSelectedIdsByKind("tensor").length < 2;
    if (toggleLinearPeriodicButton) {
      toggleLinearPeriodicButton.classList.toggle("is-active", linearPeriodicMode);
    }
    if (linearPeriodicCellLabel) {
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : "Normal";
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "initial";
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "final";
    }
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
    bumpSpecRevision,
    ensureSpecLookups,
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
    resolveConnectableIndexOwner,
    findEdgeByIndexId,
    syncConnectedIndexDimension,
    resolveBaseEdgeId,
    getLinearPeriodicChain,
    isLinearPeriodicMode,
    getActiveLinearPeriodicCellName,
    getLinearPeriodicCell,
    isLinearPeriodicBoundaryTensor,
    getContractibleTensors,
    getContractibleEdges,
    syncCurrentGraphIntoLinearPeriodicChain,
    hydrateActiveLinearPeriodicCell,
    stripLinearPeriodicBoundaryTensorsFromGraphSection,
    syncLinearPeriodicChainInterfaceDimensions,
    buildHistorySnapshotSpec,
    buildSerializedSpec,
    switchLinearPeriodicCell,
    toggleLinearPeriodicMode,
    syncLinearPeriodicBoundaryTensors,
    enforceLinearPeriodicEngineSupport,
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
