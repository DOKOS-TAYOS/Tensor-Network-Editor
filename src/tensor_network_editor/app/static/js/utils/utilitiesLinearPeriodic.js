export function createUtilityLinearPeriodicBindings({
  ctx,
  state,
  constants,
  dom,
  runtime,
}) {
  const { TENSOR_WIDTH, TENSOR_HEIGHT } = constants;
  const { engineSelect } = dom;
  const { window } = ctx;

  const LINEAR_PERIODIC_CELL_ORDER = ["initial", "periodic", "final"];
  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };
  const LINEAR_PERIODIC_PREVIOUS_OPERAND_ID = "__linear_previous__";
  const LINEAR_PERIODIC_NEXT_OPERAND_ID = "__linear_next__";
  const LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE = {
    previous: LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    next: LINEAR_PERIODIC_NEXT_OPERAND_ID,
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

  function getLinearPeriodicReservedOperandId(role) {
    return Object.prototype.hasOwnProperty.call(
      LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE,
      role
    )
      ? LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE[role]
      : null;
  }

  function isLinearPeriodicReservedOperandId(operandId) {
    return (
      operandId === LINEAR_PERIODIC_PREVIOUS_OPERAND_ID ||
      operandId === LINEAR_PERIODIC_NEXT_OPERAND_ID
    );
  }

  function normalizeLinearPeriodicChainInPlace(chain) {
    chain.metadata = runtime.isObject(chain.metadata) ? chain.metadata : {};
    chain.active_cell = LINEAR_PERIODIC_CELL_ORDER.includes(chain.active_cell)
      ? chain.active_cell
      : "initial";
    chain.initial_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.initial_cell)
          ? chain.initial_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    chain.periodic_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.periodic_cell)
          ? chain.periodic_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    chain.final_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.final_cell)
          ? chain.final_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    return chain;
  }

  function getLinearPeriodicChain(spec = state.spec) {
    return spec && runtime.isObject(spec.linear_periodic_chain)
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

  function getLinearPeriodicCell(
    spec = state.spec,
    cellName = getActiveLinearPeriodicCellName(spec)
  ) {
    const chain = getLinearPeriodicChain(spec);
    if (!chain || !cellName) {
      return null;
    }
    return chain[`${cellName}_cell`] || null;
  }

  function getLinearPeriodicBoundaryTensorByRole(role, spec = state.spec) {
    return (
      Array.isArray(spec && spec.tensors) ? spec.tensors : []
    ).find((tensor) => tensor.linear_periodic_role === role) || null;
  }

  function isLinearPeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.linear_periodic_role === "previous" ||
          tensor.linear_periodic_role === "next")
    );
  }

  function getLinearPeriodicReservedOperandIdForTensor(
    tensorOrId,
    spec = state.spec
  ) {
    const tensor =
      typeof tensorOrId === "string"
        ? (
            Array.isArray(spec && spec.tensors) ? spec.tensors : []
          ).find((candidate) => candidate.id === tensorOrId) || null
        : tensorOrId;
    if (!isLinearPeriodicBoundaryTensor(tensor)) {
      return null;
    }
    return getLinearPeriodicReservedOperandId(tensor.linear_periodic_role);
  }

  function isContractibleBoundaryTensor(tensor) {
    return (
      isLinearPeriodicBoundaryTensor(tensor) ||
      (typeof runtime.isTreePeriodicBoundaryTensor === "function" &&
        runtime.isTreePeriodicBoundaryTensor(tensor)) ||
      (typeof runtime.isGridPeriodicBoundaryTensor === "function" &&
        runtime.isGridPeriodicBoundaryTensor(tensor))
    );
  }

  function buildContractibleCollections(spec) {
    const tensors = Array.isArray(spec && spec.tensors) ? spec.tensors : [];
    const edges = Array.isArray(spec && spec.edges) ? spec.edges : [];
    const contractibleTensors = tensors.filter(
      (tensor) => !isContractibleBoundaryTensor(tensor)
    );
    if (!edges.length || !contractibleTensors.length) {
      return {
        tensors: contractibleTensors,
        edges: [],
      };
    }

    let tensorById = {};
    if (
      spec === state.spec &&
      typeof ctx.ensureSpecLookups === "function"
    ) {
      ctx.ensureSpecLookups();
      tensorById = state.tensorById;
    } else {
      tensorById = Object.fromEntries(tensors.map((tensor) => [tensor.id, tensor]));
    }

    return {
      tensors: contractibleTensors,
      edges: edges.filter((edge) => {
        const leftTensor = tensorById[edge.left && edge.left.tensor_id];
        const rightTensor = tensorById[edge.right && edge.right.tensor_id];
        return (
          leftTensor &&
          rightTensor &&
          !isContractibleBoundaryTensor(leftTensor) &&
          !isContractibleBoundaryTensor(rightTensor)
        );
      }),
    };
  }

  function ensureContractibleCollections() {
    if (!state.spec) {
      state.contractibleCacheRevision = state.specRevision;
      state.contractibleCacheTensorRef = null;
      state.contractibleCacheTensorCount = 0;
      state.contractibleCacheEdgeRef = null;
      state.contractibleCacheEdgeCount = 0;
      state.contractibleTensorsCache = [];
      state.contractibleEdgesCache = [];
      state.contractibleCacheToken += 1;
      return;
    }

    const tensors = Array.isArray(state.spec.tensors) ? state.spec.tensors : [];
    const edges = Array.isArray(state.spec.edges) ? state.spec.edges : [];
    const cacheIsFresh =
      state.contractibleCacheRevision === state.specRevision &&
      state.contractibleCacheTensorRef === tensors &&
      state.contractibleCacheTensorCount === tensors.length &&
      state.contractibleCacheEdgeRef === edges &&
      state.contractibleCacheEdgeCount === edges.length;
    if (cacheIsFresh) {
      return;
    }

    const collections = buildContractibleCollections(state.spec);
    state.contractibleCacheRevision = state.specRevision;
    state.contractibleCacheTensorRef = tensors;
    state.contractibleCacheTensorCount = tensors.length;
    state.contractibleCacheEdgeRef = edges;
    state.contractibleCacheEdgeCount = edges.length;
    state.contractibleTensorsCache = collections.tensors;
    state.contractibleEdgesCache = collections.edges;
    state.contractibleCacheToken += 1;
  }

  function getContractibleTensors(spec = state.spec) {
    if (spec !== state.spec) {
      return buildContractibleCollections(spec).tensors;
    }
    ensureContractibleCollections();
    return state.contractibleTensorsCache;
  }

  function getContractibleEdges(spec = state.spec) {
    if (spec !== state.spec) {
      return buildContractibleCollections(spec).edges;
    }
    ensureContractibleCollections();
    return state.contractibleEdgesCache;
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
    const leftEdges = tensors.map(
      (tensor) => tensor.position.x - runtime.tensorWidth(tensor) / 2
    );
    const rightEdges = tensors.map(
      (tensor) => tensor.position.x + runtime.tensorWidth(tensor) / 2
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
      id: runtime.makeId("tensor"),
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
      if (
        !expectedRoles.has(tensor.linear_periodic_role) ||
        seenRoles.has(tensor.linear_periodic_role)
      ) {
        return;
      }
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[tensor.linear_periodic_role];
      tensor.name = settings.name;
      tensor.metadata = runtime.isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.metadata.color = runtime.getMetadataColor(tensor.metadata, settings.color);
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
      spec.tensors
        .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
        .map((tensor) => tensor.id)
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
    const isActiveSpec = spec === state.spec;
    const tensors = Array.isArray(spec && spec.tensors) ? spec.tensors : [];
    const edges = Array.isArray(spec && spec.edges) ? spec.edges : [];
    if (isActiveSpec && typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    const tensorById = isActiveSpec
      ? state.tensorById
      : Object.fromEntries(tensors.map((tensor) => [tensor.id, tensor]));
    const internallyConnectedIndexIds = new Set();
    edges.forEach((edge) => {
      const leftTensor = tensorById[edge.left && edge.left.tensor_id];
      const rightTensor = tensorById[edge.right && edge.right.tensor_id];
      if (
        leftTensor &&
        rightTensor &&
        !isContractibleBoundaryTensor(leftTensor) &&
        !isContractibleBoundaryTensor(rightTensor)
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

  function getBoundaryInterfaceDimensions(graphSection, preferredRole = "next") {
    const boundaryTensors = (
      Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : []
    ).filter((tensor) => isLinearPeriodicBoundaryTensor(tensor));
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
    const activeDimensions = getBoundaryInterfaceDimensions(spec, activePreferredRole);
    if (activeDimensions.length) {
      return activeDimensions;
    }
    const chain = getLinearPeriodicChain(spec);
    if (!chain) {
      return getLinearPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
    }
    const initialDimensions = getBoundaryInterfaceDimensions(chain.initial_cell, "next");
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
            id:
              existingIndex && existingIndex.id
                ? existingIndex.id
                : runtime.makeId("index"),
            name: `slot_${indexPosition + 1}`,
            dimension,
            offset:
              existingIndex && existingIndex.offset
                ? existingIndex.offset
                : runtime.defaultIndexOffsetForOrder(indexPosition, boundaryTensor),
            metadata:
              existingIndex && runtime.isObject(existingIndex.metadata)
                ? existingIndex.metadata
                : {},
          };
        }
      );
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[boundaryTensor.linear_periodic_role];
      boundaryTensor.name = settings.name;
      boundaryTensor.metadata = runtime.isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = runtime.getMetadataColor(
        boundaryTensor.metadata,
        settings.color
      );
      runtime.ensureTensorIndexOffsets(boundaryTensor);
    });
  }

  function seedLinearPeriodicCell(
    cellName,
    graphSection,
    interfaceDimensions = null
  ) {
    const runtimeSpec = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    runtimeSpec.linear_periodic_chain = { active_cell: cellName };
    ensureActiveLinearPeriodicBoundaryTensors(runtimeSpec);
    syncLinearPeriodicBoundaryTensors(runtimeSpec, interfaceDimensions);
    return runtime.buildGraphSectionFromSpec(runtimeSpec);
  }

  function syncLinearPeriodicChainInterfaceDimensions(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!chain || !activeCellName) {
      return spec;
    }
    const activeCell = getLinearPeriodicCell(spec, activeCellName);
    const interfaceDimensions = getCanonicalLinearPeriodicInterfaceDimensions(spec);

    chain[`${activeCellName}_cell`] = runtime.buildGraphSectionFromSpec(spec, activeCell);

    LINEAR_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellSpec = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          getLinearPeriodicCell(spec, cellName) || runtime.buildEmptyGraphSection()
        )
      );
      chain[`${cellName}_cell`] = seedLinearPeriodicCell(
        cellName,
        cellSpec,
        interfaceDimensions
      );
    });

    runtime.replaceGraphSectionOnSpec(
      spec,
      getLinearPeriodicCell(spec, activeCellName) || runtime.buildEmptyGraphSection()
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

  function invalidateActiveLinearPeriodicLookups(spec = state.spec) {
    if (spec !== state.spec) {
      return;
    }
    state.lookupRevision = -1;
    if (typeof ctx.resetDerivedStateCaches === "function") {
      ctx.resetDerivedStateCaches();
    }
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
    runtime.replaceGraphSectionOnSpec(
      spec,
      activeCell || runtime.buildEmptyGraphSection()
    );
    invalidateActiveLinearPeriodicLookups(spec);
    syncLinearPeriodicBoundaryTensors(spec);
    return spec;
  }

  function stripLinearPeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
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
    state.plannerPreviewBadgeDisclosure = {};
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
    const nextIndex = runtime.clamp(
      activeIndex + direction,
      0,
      LINEAR_PERIODIC_CELL_ORDER.length - 1
    );
    if (nextIndex === activeIndex) {
      return;
    }
    syncCurrentGraphIntoLinearPeriodicChain();
    state.spec.linear_periodic_chain.active_cell = LINEAR_PERIODIC_CELL_ORDER[nextIndex];
    hydrateActiveLinearPeriodicCell();
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    runtime.reconcileTensorOrder();
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
    const hasSelectedEngine = Array.from(engineSelect.options).some(
      (option) => option.value === state.selectedEngine
    );
    if (hasSelectedEngine) {
      engineSelect.value = state.selectedEngine;
    }
    return false;
  }

  function toggleLinearPeriodicMode() {
    if (!state.spec) {
      return;
    }
    if (
      !isLinearPeriodicMode() &&
      typeof runtime.isBenchmarkMode === "function" &&
      runtime.isBenchmarkMode()
    ) {
      ctx.setStatus(
        "Leave Benchmark mode before enabling For unidimensional mode.",
        "error"
      );
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
      runtime.replaceGraphSectionOnSpec(state.spec, plainInitialCell);
      if (typeof ctx.bumpSpecRevision === "function") {
        ctx.bumpSpecRevision();
      }
      runtime.reconcileTensorOrder();
      resetTransientEditorStateForCellSwitch();
      if (typeof ctx.clearGeneratedCodePreview === "function") {
        ctx.clearGeneratedCodePreview();
      }
      ctx.render();
      if (typeof ctx.refreshContractionAnalysis === "function") {
        ctx.refreshContractionAnalysis();
      }
      ctx.setStatus(
        "For mode disabled. Restored the initial cell as a normal network.",
        "success"
      );
      return;
    }

    let seedGraphSection = runtime.buildGraphSectionFromSpec(state.spec);
    if (
      typeof runtime.isGridPeriodicMode === "function" &&
      runtime.isGridPeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoGridPeriodicGrid();
      seedGraphSection =
        typeof runtime.stripGridPeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripGridPeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.grid_periodic_grid = null;
      runtime.replaceGraphSectionOnSpec(state.spec, seedGraphSection);
    } else if (
      typeof runtime.isTreePeriodicMode === "function" &&
      runtime.isTreePeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoTreePeriodicTree();
      seedGraphSection =
        typeof runtime.stripTreePeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripTreePeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.tree_periodic_tree = null;
      runtime.replaceGraphSectionOnSpec(state.spec, seedGraphSection);
    }
    const initialCell = seedLinearPeriodicCell(
      "initial",
      seedGraphSection
    );
    const periodicCell = seedLinearPeriodicCell(
      "periodic",
      runtime.buildEmptyGraphSection()
    );
    const finalCell = seedLinearPeriodicCell(
      "final",
      runtime.buildEmptyGraphSection()
    );
    state.spec.linear_periodic_chain = {
      active_cell: "initial",
      initial_cell: initialCell,
      periodic_cell: periodicCell,
      final_cell: finalCell,
      metadata: {},
    };
    state.spec.grid_periodic_grid = null;
    state.spec.tree_periodic_tree = null;
    hydrateActiveLinearPeriodicCell();
    enforceLinearPeriodicEngineSupport();
    ctx.setStatus("For mode enabled. You are editing the initial cell.", "success");
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    runtime.reconcileTensorOrder();
    resetTransientEditorStateForCellSwitch();
    if (typeof ctx.clearGeneratedCodePreview === "function") {
      ctx.clearGeneratedCodePreview();
    }
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
  }

  function setLinearPeriodicMode(enabled) {
    const shouldEnable = Boolean(enabled);
    if (shouldEnable === isLinearPeriodicMode()) {
      return;
    }
    toggleLinearPeriodicMode();
  }

  return {
    getLinearPeriodicReservedOperandId,
    isLinearPeriodicReservedOperandId,
    normalizeLinearPeriodicChainInPlace,
    getLinearPeriodicChain,
    isLinearPeriodicMode,
    getActiveLinearPeriodicCellName,
    getLinearPeriodicCell,
    getLinearPeriodicBoundaryTensorByRole,
    isLinearPeriodicBoundaryTensor,
    getLinearPeriodicReservedOperandIdForTensor,
    getContractibleTensors,
    getContractibleEdges,
    syncCurrentGraphIntoLinearPeriodicChain,
    hydrateActiveLinearPeriodicCell,
    stripLinearPeriodicBoundaryTensorsFromGraphSection,
    syncLinearPeriodicChainInterfaceDimensions,
    buildHistorySnapshotSpec: runtime.buildHistorySnapshotSpec,
    buildSerializedSpec: runtime.buildSerializedSpec,
    switchLinearPeriodicCell,
    toggleLinearPeriodicMode,
    setLinearPeriodicMode,
    syncLinearPeriodicBoundaryTensors,
    enforceLinearPeriodicEngineSupport,
  };
}
