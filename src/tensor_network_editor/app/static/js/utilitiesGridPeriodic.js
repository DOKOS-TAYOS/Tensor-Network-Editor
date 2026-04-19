export function createUtilityGridPeriodicBindings({
  ctx,
  state,
  constants,
  runtime,
}) {
  const { TENSOR_WIDTH, TENSOR_HEIGHT } = constants;
  const { window } = ctx;

  const GRID_PERIODIC_CELL_ORDER = [
    "top_left",
    "top",
    "top_right",
    "left",
    "center",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
  ];
  const GRID_PERIODIC_CELL_LABELS = {
    top_left: "Top-left cell",
    top: "Top cell",
    top_right: "Top-right cell",
    left: "Left cell",
    center: "Center cell",
    right: "Right cell",
    bottom_left: "Bottom-left cell",
    bottom: "Bottom cell",
    bottom_right: "Bottom-right cell",
  };
  const GRID_PERIODIC_NAVIGATION = {
    top_left: { up: null, right: "top", down: "left", left: null },
    top: { up: null, right: "top_right", down: "center", left: "top_left" },
    top_right: { up: null, right: null, down: "right", left: "top" },
    left: { up: "top_left", right: "center", down: "bottom_left", left: null },
    center: { up: "top", right: "right", down: "bottom", left: "left" },
    right: { up: "top_right", right: null, down: "bottom_right", left: "center" },
    bottom_left: { up: "left", right: "bottom", down: null, left: null },
    bottom: { up: "center", right: "bottom_right", down: null, left: "bottom_left" },
    bottom_right: { up: "right", right: null, down: null, left: "bottom" },
  };
  const GRID_PERIODIC_BOUNDARY_SETTINGS = {
    up: {
      name: "Upper cell",
      color: "#456cbf",
    },
    right: {
      name: "Right cell",
      color: "#2f9b8f",
    },
    down: {
      name: "Lower cell",
      color: "#d38a37",
    },
    left: {
      name: "Left cell",
      color: "#8e5bcc",
    },
  };
  const GRID_PERIODIC_CELL_KEYS = Object.fromEntries(
    GRID_PERIODIC_CELL_ORDER.map((cellName) => [cellName, `${cellName}_cell`])
  );
  const GRID_PERIODIC_FAMILY_BY_CELL_ROLE = {
    top_left: {
      right: "row_top",
      down: "column_left",
    },
    top: {
      left: "row_top",
      right: "row_top",
      down: "column_center",
    },
    top_right: {
      left: "row_top",
      down: "column_right",
    },
    left: {
      up: "column_left",
      right: "row_middle",
      down: "column_left",
    },
    center: {
      up: "column_center",
      right: "row_middle",
      down: "column_center",
      left: "row_middle",
    },
    right: {
      up: "column_right",
      down: "column_right",
      left: "row_middle",
    },
    bottom_left: {
      up: "column_left",
      right: "row_bottom",
    },
    bottom: {
      up: "column_center",
      left: "row_bottom",
      right: "row_bottom",
    },
    bottom_right: {
      up: "column_right",
      left: "row_bottom",
    },
  };
  const GRID_PERIODIC_EXPECTED_ROLES = {
    top_left: ["right", "down"],
    top: ["left", "right", "down"],
    top_right: ["left", "down"],
    left: ["up", "right", "down"],
    center: ["up", "right", "down", "left"],
    right: ["up", "down", "left"],
    bottom_left: ["up", "right"],
    bottom: ["up", "left", "right"],
    bottom_right: ["up", "left"],
  };

  function getGridPeriodicCellKey(cellName) {
    return Object.prototype.hasOwnProperty.call(GRID_PERIODIC_CELL_KEYS, cellName)
      ? GRID_PERIODIC_CELL_KEYS[cellName]
      : null;
  }

  function normalizeGridPeriodicGridInPlace(grid) {
    grid.metadata = runtime.isObject(grid.metadata) ? grid.metadata : {};
    grid.active_cell = GRID_PERIODIC_CELL_ORDER.includes(grid.active_cell)
      ? grid.active_cell
      : "center";
    GRID_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellKey = getGridPeriodicCellKey(cellName);
      grid[cellKey] = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          runtime.isObject(grid[cellKey])
            ? grid[cellKey]
            : runtime.buildEmptyGraphSection()
        )
      );
      grid[cellKey].contraction_plan = null;
    });
    return grid;
  }

  function getGridPeriodicGrid(spec = state.spec) {
    return spec && runtime.isObject(spec.grid_periodic_grid)
      ? spec.grid_periodic_grid
      : null;
  }

  function isGridPeriodicMode(spec = state.spec) {
    return Boolean(getGridPeriodicGrid(spec));
  }

  function isForMode(spec = state.spec) {
    return (
      (typeof runtime.isLinearPeriodicMode === "function" &&
        runtime.isLinearPeriodicMode(spec)) ||
      isGridPeriodicMode(spec)
    );
  }

  function getActiveGridPeriodicCellName(spec = state.spec) {
    const grid = getGridPeriodicGrid(spec);
    return grid ? grid.active_cell : null;
  }

  function getGridPeriodicCell(
    spec = state.spec,
    cellName = getActiveGridPeriodicCellName(spec)
  ) {
    const grid = getGridPeriodicGrid(spec);
    const cellKey = getGridPeriodicCellKey(cellName);
    if (!grid || !cellKey) {
      return null;
    }
    return grid[cellKey] || null;
  }

  function getGridPeriodicCellLabel(cellName) {
    return GRID_PERIODIC_CELL_LABELS[cellName] || "Grid cell";
  }

  function getGridPeriodicNeighborCellName(cellName, direction) {
    return (
      (GRID_PERIODIC_NAVIGATION[cellName] &&
        GRID_PERIODIC_NAVIGATION[cellName][direction]) ||
      null
    );
  }

  function canSwitchGridPeriodicCell(direction, spec = state.spec) {
    const activeCellName = getActiveGridPeriodicCellName(spec);
    return Boolean(
      activeCellName && getGridPeriodicNeighborCellName(activeCellName, direction)
    );
  }

  function getGridPeriodicBoundaryTensorByRole(role, spec = state.spec) {
    return (
      Array.isArray(spec && spec.tensors) ? spec.tensors : []
    ).find((tensor) => tensor.grid_periodic_role === role) || null;
  }

  function isGridPeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.grid_periodic_role === "up" ||
          tensor.grid_periodic_role === "right" ||
          tensor.grid_periodic_role === "down" ||
          tensor.grid_periodic_role === "left")
    );
  }

  function isForBoundaryTensor(tensor) {
    return (
      isGridPeriodicBoundaryTensor(tensor) ||
      (typeof runtime.isLinearPeriodicBoundaryTensor === "function" &&
        runtime.isLinearPeriodicBoundaryTensor(tensor))
    );
  }

  function getExpectedGridPeriodicRoles(cellName) {
    return Object.prototype.hasOwnProperty.call(
      GRID_PERIODIC_EXPECTED_ROLES,
      cellName
    )
      ? [...GRID_PERIODIC_EXPECTED_ROLES[cellName]]
      : [];
  }

  function getRealTensorBounds(spec = state.spec) {
    const tensors = (Array.isArray(spec && spec.tensors) ? spec.tensors : []).filter(
      (tensor) => !isForBoundaryTensor(tensor)
    );
    if (!tensors.length) {
      return {
        minX: 120,
        maxX: 320,
        minY: 120,
        maxY: 260,
        centerX: 220,
        centerY: 190,
      };
    }
    const leftEdges = tensors.map(
      (tensor) => tensor.position.x - runtime.tensorWidth(tensor) / 2
    );
    const rightEdges = tensors.map(
      (tensor) => tensor.position.x + runtime.tensorWidth(tensor) / 2
    );
    const topEdges = tensors.map(
      (tensor) => tensor.position.y - runtime.tensorHeight(tensor) / 2
    );
    const bottomEdges = tensors.map(
      (tensor) => tensor.position.y + runtime.tensorHeight(tensor) / 2
    );
    const centersX = tensors.map((tensor) => tensor.position.x);
    const centersY = tensors.map((tensor) => tensor.position.y);
    return {
      minX: Math.min(...leftEdges),
      maxX: Math.max(...rightEdges),
      minY: Math.min(...topEdges),
      maxY: Math.max(...bottomEdges),
      centerX: centersX.reduce((sum, value) => sum + value, 0) / centersX.length,
      centerY: centersY.reduce((sum, value) => sum + value, 0) / centersY.length,
    };
  }

  function positionGridPeriodicBoundaryTensor(tensor, role, spec = state.spec) {
    const bounds = getRealTensorBounds(spec);
    if (role === "up") {
      tensor.position = { x: bounds.centerX, y: bounds.minY - 180 };
      return;
    }
    if (role === "right") {
      tensor.position = { x: bounds.maxX + 220, y: bounds.centerY };
      return;
    }
    if (role === "down") {
      tensor.position = { x: bounds.centerX, y: bounds.maxY + 180 };
      return;
    }
    tensor.position = { x: bounds.minX - 220, y: bounds.centerY };
  }

  function createGridPeriodicBoundaryTensor(role, spec = state.spec) {
    const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[role];
    const tensor = {
      id: runtime.makeId("tensor"),
      name: settings.name,
      position: { x: 0, y: 0 },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      grid_periodic_role: role,
      metadata: { color: settings.color },
    };
    positionGridPeriodicBoundaryTensor(tensor, role, spec);
    return tensor;
  }

  function ensureActiveGridPeriodicBoundaryTensors(spec = state.spec) {
    const activeCellName = getActiveGridPeriodicCellName(spec);
    if (!activeCellName) {
      return;
    }
    const expectedRoles = new Set(getExpectedGridPeriodicRoles(activeCellName));
    const nextTensors = [];
    const removedBoundaryIds = new Set();
    const seenRoles = new Set();
    (Array.isArray(spec.tensors) ? spec.tensors : []).forEach((tensor) => {
      if (!isGridPeriodicBoundaryTensor(tensor)) {
        nextTensors.push(tensor);
        return;
      }
      if (
        !expectedRoles.has(tensor.grid_periodic_role) ||
        seenRoles.has(tensor.grid_periodic_role)
      ) {
        removedBoundaryIds.add(tensor.id);
        return;
      }
      const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[tensor.grid_periodic_role];
      tensor.name = settings.name;
      tensor.metadata = runtime.isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.metadata.color = runtime.getMetadataColor(tensor.metadata, settings.color);
      seenRoles.add(tensor.grid_periodic_role);
      positionGridPeriodicBoundaryTensor(tensor, tensor.grid_periodic_role, spec);
      nextTensors.push(tensor);
    });
    getExpectedGridPeriodicRoles(activeCellName).forEach((role) => {
      if (!seenRoles.has(role)) {
        nextTensors.push(createGridPeriodicBoundaryTensor(role, spec));
      }
    });
    spec.tensors = nextTensors;
    if (removedBoundaryIds.size) {
      spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
        (edge) =>
          !removedBoundaryIds.has(edge.left && edge.left.tensor_id) &&
          !removedBoundaryIds.has(edge.right && edge.right.tensor_id)
      );
    }
  }

  function getGridPeriodicCandidateOwners(spec = state.spec) {
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
        !isForBoundaryTensor(leftTensor) &&
        !isForBoundaryTensor(rightTensor)
      ) {
        internallyConnectedIndexIds.add(edge.left.index_id);
        internallyConnectedIndexIds.add(edge.right.index_id);
      }
    });
    const owners = [];
    tensors.forEach((tensor) => {
      if (isForBoundaryTensor(tensor)) {
        return;
      }
      tensor.indices.forEach((index, indexPosition) => {
        if (!internallyConnectedIndexIds.has(index.id)) {
          owners.push({ tensor, index, indexPosition });
        }
      });
    });
    return owners;
  }

  function getGridPeriodicBoundaryInterfaceDimensions(
    graphSection,
    preferredRole = "right"
  ) {
    const boundaryTensors = (
      Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : []
    ).filter((tensor) => isGridPeriodicBoundaryTensor(tensor));
    const preferredTensor = boundaryTensors.find(
      (tensor) =>
        tensor.grid_periodic_role === preferredRole &&
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

  function buildEmptyGridPeriodicFamilyDimensions() {
    return {
      row_top: [],
      row_middle: [],
      row_bottom: [],
      column_left: [],
      column_center: [],
      column_right: [],
    };
  }

  function getGridPeriodicRoleFamily(cellName, role) {
    if (
      !Object.prototype.hasOwnProperty.call(GRID_PERIODIC_FAMILY_BY_CELL_ROLE, cellName)
    ) {
      return null;
    }
    const roleFamilies = GRID_PERIODIC_FAMILY_BY_CELL_ROLE[cellName];
    return Object.prototype.hasOwnProperty.call(roleFamilies, role)
      ? roleFamilies[role]
      : null;
  }

  function assignGridPeriodicFamilyDimensionsFromCell(
    familyDimensions,
    cellName,
    graphSection
  ) {
    getExpectedGridPeriodicRoles(cellName).forEach((role) => {
      const familyKey = getGridPeriodicRoleFamily(cellName, role);
      if (!familyKey || familyDimensions[familyKey].length) {
        return;
      }
      const candidateDimensions = getGridPeriodicBoundaryInterfaceDimensions(
        graphSection,
        role
      );
      if (candidateDimensions.length) {
        familyDimensions[familyKey] = [...candidateDimensions];
      }
    });
  }

  function getCanonicalGridPeriodicFamilyDimensions(spec = state.spec) {
    const familyDimensions = buildEmptyGridPeriodicFamilyDimensions();
    const activeCellName = getActiveGridPeriodicCellName(spec);
    if (activeCellName) {
      assignGridPeriodicFamilyDimensionsFromCell(familyDimensions, activeCellName, spec);
    }
    const grid = getGridPeriodicGrid(spec);
    if (grid) {
      GRID_PERIODIC_CELL_ORDER.forEach((cellName) => {
        assignGridPeriodicFamilyDimensionsFromCell(
          familyDimensions,
          cellName,
          getGridPeriodicCell(spec, cellName)
        );
      });
    }
    if (
      !Object.values(familyDimensions).some((dimensions) => dimensions.length) &&
      activeCellName
    ) {
      const ownerDimensions = getGridPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
      const activeRoles = getExpectedGridPeriodicRoles(activeCellName);
      activeRoles.forEach((role) => {
        const familyKey = getGridPeriodicRoleFamily(activeCellName, role);
        if (familyKey && !familyDimensions[familyKey].length) {
          familyDimensions[familyKey] = [...ownerDimensions];
        }
      });
    }
    return familyDimensions;
  }

  function syncGridPeriodicBoundaryTensors(
    spec = state.spec,
    familyDimensions = null
  ) {
    if (!isGridPeriodicMode(spec)) {
      return;
    }
    ensureActiveGridPeriodicBoundaryTensors(spec);
    const activeCellName = getActiveGridPeriodicCellName(spec);
    const resolvedFamilyDimensions = runtime.isObject(familyDimensions)
      ? familyDimensions
      : getCanonicalGridPeriodicFamilyDimensions(spec);
    const boundaryTensors = (Array.isArray(spec.tensors) ? spec.tensors : []).filter(
      (tensor) => isGridPeriodicBoundaryTensor(tensor)
    );
    boundaryTensors.forEach((boundaryTensor) => {
      const familyKey = getGridPeriodicRoleFamily(
        activeCellName,
        boundaryTensor.grid_periodic_role
      );
      const resolvedInterfaceDimensions =
        familyKey && Array.isArray(resolvedFamilyDimensions[familyKey])
          ? resolvedFamilyDimensions[familyKey]
          : [];
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
      const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[boundaryTensor.grid_periodic_role];
      boundaryTensor.name = settings.name;
      boundaryTensor.metadata = runtime.isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = runtime.getMetadataColor(
        boundaryTensor.metadata,
        settings.color
      );
      runtime.ensureTensorIndexOffsets(boundaryTensor);
      positionGridPeriodicBoundaryTensor(
        boundaryTensor,
        boundaryTensor.grid_periodic_role,
        spec
      );
    });
  }

  function seedGridPeriodicCell(
    cellName,
    graphSection,
    familyDimensions = null
  ) {
    const runtimeSpec = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    runtimeSpec.contraction_plan = null;
    runtimeSpec.grid_periodic_grid = { active_cell: cellName };
    ensureActiveGridPeriodicBoundaryTensors(runtimeSpec);
    syncGridPeriodicBoundaryTensors(runtimeSpec, familyDimensions);
    return runtime.buildGraphSectionFromSpec(runtimeSpec);
  }

  function buildActiveGridPeriodicCellSnapshot(spec = state.spec) {
    const graphSection = runtime.buildGraphSectionFromSpec(spec);
    graphSection.contraction_plan = null;
    return graphSection;
  }

  function syncGridPeriodicGridInterfaceDimensions(spec = state.spec) {
    const grid = getGridPeriodicGrid(spec);
    const activeCellName = getActiveGridPeriodicCellName(spec);
    if (!grid || !activeCellName) {
      return spec;
    }
    const familyDimensions = getCanonicalGridPeriodicFamilyDimensions(spec);
    const activeCellKey = getGridPeriodicCellKey(activeCellName);
    grid[activeCellKey] = seedGridPeriodicCell(
      activeCellName,
      buildActiveGridPeriodicCellSnapshot(spec),
      familyDimensions
    );

    GRID_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellKey = getGridPeriodicCellKey(cellName);
      const cellSpec = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(getGridPeriodicCell(spec, cellName) || runtime.buildEmptyGraphSection())
      );
      cellSpec.contraction_plan = null;
      grid[cellKey] = seedGridPeriodicCell(cellName, cellSpec, familyDimensions);
    });

    runtime.replaceGraphSectionOnSpec(
      spec,
      getGridPeriodicCell(spec, activeCellName) || runtime.buildEmptyGraphSection()
    );
    spec.contraction_plan = null;
    return spec;
  }

  function syncCurrentGraphIntoGridPeriodicGrid(spec = state.spec) {
    const grid = getGridPeriodicGrid(spec);
    if (!grid) {
      return spec;
    }
    spec.contraction_plan = null;
    syncGridPeriodicBoundaryTensors(spec);
    syncGridPeriodicGridInterfaceDimensions(spec);
    return spec;
  }

  function invalidateActiveGridPeriodicLookups(spec = state.spec) {
    if (spec !== state.spec) {
      return;
    }
    state.lookupRevision = -1;
    if (typeof ctx.resetDerivedStateCaches === "function") {
      ctx.resetDerivedStateCaches();
    }
  }

  function hydrateActiveGridPeriodicCell(spec = state.spec) {
    const grid = getGridPeriodicGrid(spec);
    if (!grid) {
      return spec;
    }
    grid.active_cell = GRID_PERIODIC_CELL_ORDER.includes(grid.active_cell)
      ? grid.active_cell
      : "center";
    const activeCell = getGridPeriodicCell(spec, grid.active_cell);
    runtime.replaceGraphSectionOnSpec(
      spec,
      activeCell || runtime.buildEmptyGraphSection()
    );
    spec.contraction_plan = null;
    invalidateActiveGridPeriodicLookups(spec);
    syncGridPeriodicBoundaryTensors(spec);
    return spec;
  }

  function stripGridPeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    stripped.contraction_plan = null;
    const boundaryTensorIds = new Set(
      stripped.tensors
        .filter((tensor) => isGridPeriodicBoundaryTensor(tensor))
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
    state.activeTensorDrag = null;
    state.activeIndexDrag = null;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.boxSelection = null;
    state.connectMode = false;
  }

  function switchGridPeriodicCell(direction) {
    if (!isGridPeriodicMode()) {
      return;
    }
    const activeCellName = getActiveGridPeriodicCellName();
    const nextCellName = getGridPeriodicNeighborCellName(activeCellName, direction);
    if (!nextCellName) {
      return;
    }
    syncCurrentGraphIntoGridPeriodicGrid();
    state.spec.grid_periodic_grid.active_cell = nextCellName;
    hydrateActiveGridPeriodicCell();
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
      `Editing ${getGridPeriodicCellLabel(nextCellName).toLowerCase()}.`,
      "success"
    );
  }

  function buildGridPeriodicSeedGraphSection() {
    if (
      typeof runtime.isLinearPeriodicMode === "function" &&
      runtime.isLinearPeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain();
      const strippedLinearCell =
        typeof runtime.stripLinearPeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripLinearPeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.linear_periodic_chain = null;
      runtime.replaceGraphSectionOnSpec(state.spec, strippedLinearCell);
      return strippedLinearCell;
    }
    return runtime.buildGraphSectionFromSpec(state.spec);
  }

  function toggleGridPeriodicMode() {
    if (!state.spec) {
      return;
    }
    if (
      !isGridPeriodicMode() &&
      typeof runtime.isBenchmarkMode === "function" &&
      runtime.isBenchmarkMode()
    ) {
      ctx.setStatus(
        "Leave Benchmark mode before enabling For bidimensional mode.",
        "error"
      );
      return;
    }
    if (isGridPeriodicMode()) {
      if (
        !window.confirm(
          "Leave For bidimensional mode and keep the current cell as a normal network?"
        )
      ) {
        return;
      }
      syncCurrentGraphIntoGridPeriodicGrid();
      const plainActiveCell = stripGridPeriodicBoundaryTensorsFromGraphSection(
        runtime.buildGraphSectionFromSpec(state.spec)
      );
      state.spec.grid_periodic_grid = null;
      runtime.replaceGraphSectionOnSpec(state.spec, plainActiveCell);
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
        "For bidimensional mode disabled. Restored the active cell as a normal network.",
        "success"
      );
      return;
    }

    const seedGraphSection = buildGridPeriodicSeedGraphSection();
    state.spec.linear_periodic_chain = null;
    state.spec.grid_periodic_grid = {
      active_cell: "center",
      top_left_cell: seedGridPeriodicCell("top_left", runtime.buildEmptyGraphSection()),
      top_cell: seedGridPeriodicCell("top", runtime.buildEmptyGraphSection()),
      top_right_cell: seedGridPeriodicCell("top_right", runtime.buildEmptyGraphSection()),
      left_cell: seedGridPeriodicCell("left", runtime.buildEmptyGraphSection()),
      center_cell: seedGridPeriodicCell("center", seedGraphSection),
      right_cell: seedGridPeriodicCell("right", runtime.buildEmptyGraphSection()),
      bottom_left_cell: seedGridPeriodicCell("bottom_left", runtime.buildEmptyGraphSection()),
      bottom_cell: seedGridPeriodicCell("bottom", runtime.buildEmptyGraphSection()),
      bottom_right_cell: seedGridPeriodicCell(
        "bottom_right",
        runtime.buildEmptyGraphSection()
      ),
      metadata: {},
    };
    hydrateActiveGridPeriodicCell();
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
      "For bidimensional mode enabled. You are editing the center cell.",
      "success"
    );
  }

  function setGridPeriodicMode(enabled) {
    const shouldEnable = Boolean(enabled);
    if (shouldEnable === isGridPeriodicMode()) {
      return;
    }
    toggleGridPeriodicMode();
  }

  return {
    normalizeGridPeriodicGridInPlace,
    getGridPeriodicGrid,
    isGridPeriodicMode,
    isForMode,
    getActiveGridPeriodicCellName,
    getGridPeriodicCell,
    getGridPeriodicCellLabel,
    getGridPeriodicNeighborCellName,
    canSwitchGridPeriodicCell,
    getGridPeriodicBoundaryTensorByRole,
    isGridPeriodicBoundaryTensor,
    isForBoundaryTensor,
    getExpectedGridPeriodicRoles,
    syncGridPeriodicBoundaryTensors,
    syncGridPeriodicGridInterfaceDimensions,
    syncCurrentGraphIntoGridPeriodicGrid,
    hydrateActiveGridPeriodicCell,
    stripGridPeriodicBoundaryTensorsFromGraphSection,
    switchGridPeriodicCell,
    toggleGridPeriodicMode,
    setGridPeriodicMode,
  };
}
