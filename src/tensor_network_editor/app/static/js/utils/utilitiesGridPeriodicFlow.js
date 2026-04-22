import { GRID_PERIODIC_CELL_ORDER } from "./utilitiesGridPeriodicState.js";

export function createGridPeriodicFlowSupport({
  ctx,
  state,
  runtime,
  gridState,
  gridBoundaries,
}) {
  const { window } = ctx;
  const {
    getGridPeriodicGrid,
    isGridPeriodicMode,
    getActiveGridPeriodicCellName,
    getGridPeriodicCellKey,
    getGridPeriodicCell,
    getGridPeriodicCellLabel,
    getGridPeriodicNeighborCellName,
  } = gridState;
  const {
    getCanonicalGridPeriodicFamilyDimensions,
    syncGridPeriodicBoundaryTensors,
    seedGridPeriodicCell,
    buildActiveGridPeriodicCellSnapshot,
  } = gridBoundaries;

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
        runtime.deepClone(
          getGridPeriodicCell(spec, cellName) || runtime.buildEmptyGraphSection()
        )
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
        .filter((tensor) => gridState.isGridPeriodicBoundaryTensor(tensor))
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
    if (
      typeof runtime.isTreePeriodicMode === "function" &&
      runtime.isTreePeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoTreePeriodicTree();
      const strippedTreeCell =
        typeof runtime.stripTreePeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripTreePeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.tree_periodic_tree = null;
      runtime.replaceGraphSectionOnSpec(state.spec, strippedTreeCell);
      return strippedTreeCell;
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
    if (
      !isGridPeriodicMode() &&
      Array.isArray(state.spec?.hyperedges) &&
      state.spec.hyperedges.length
    ) {
      ctx.setStatus(
        "For bidimensional mode does not support hyperedges yet. Remove them first or stay in normal mode.",
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
    state.spec.tree_periodic_tree = null;
    state.spec.grid_periodic_grid = {
      active_cell: "center",
      top_left_cell: seedGridPeriodicCell("top_left", runtime.buildEmptyGraphSection()),
      top_cell: seedGridPeriodicCell("top", runtime.buildEmptyGraphSection()),
      top_right_cell: seedGridPeriodicCell("top_right", runtime.buildEmptyGraphSection()),
      left_cell: seedGridPeriodicCell("left", runtime.buildEmptyGraphSection()),
      center_cell: seedGridPeriodicCell("center", seedGraphSection),
      right_cell: seedGridPeriodicCell("right", runtime.buildEmptyGraphSection()),
      bottom_left_cell: seedGridPeriodicCell(
        "bottom_left",
        runtime.buildEmptyGraphSection()
      ),
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
    syncGridPeriodicGridInterfaceDimensions,
    syncCurrentGraphIntoGridPeriodicGrid,
    invalidateActiveGridPeriodicLookups,
    hydrateActiveGridPeriodicCell,
    stripGridPeriodicBoundaryTensorsFromGraphSection,
    switchGridPeriodicCell,
    toggleGridPeriodicMode,
    setGridPeriodicMode,
  };
}
