import { TREE_PERIODIC_CELL_ORDER } from "./utilitiesTreePeriodicState.js";

export function createTreePeriodicFlowSupport({
  ctx,
  state,
  runtime,
  treeState,
  treeBoundaries,
}) {
  const { window } = ctx;
  const {
    getTreePeriodicTree,
    isTreePeriodicMode,
    getTreePeriodicBranchingFactor,
    getActiveTreePeriodicCellName,
    getTreePeriodicCellKey,
    getTreePeriodicCell,
    getTreePeriodicCellLabel,
    getTreePeriodicNeighborCellName,
  } = treeState;
  const {
    getTreePeriodicBoundaryInterfaceDimensions,
    getCanonicalTreePeriodicInterfaceDimensions,
    syncTreePeriodicBoundaryTensors,
    seedTreePeriodicCell,
  } = treeBoundaries;

  function syncTreePeriodicTreeInterfaceDimensions(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    const activeCellName = getActiveTreePeriodicCellName(spec);
    const cellKey = getTreePeriodicCellKey(activeCellName);
    if (!tree || !activeCellName || !cellKey) {
      return spec;
    }
    const branchingFactor = getTreePeriodicBranchingFactor(spec);
    const activeCell = getTreePeriodicCell(spec, activeCellName);
    const interfaceDimensions = getCanonicalTreePeriodicInterfaceDimensions(spec);

    tree[cellKey] = runtime.buildGraphSectionFromSpec(spec, activeCell);

    TREE_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellSpec = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          getTreePeriodicCell(spec, cellName) || runtime.buildEmptyGraphSection()
        )
      );
      tree[getTreePeriodicCellKey(cellName)] = seedTreePeriodicCell(
        cellName,
        cellSpec,
        branchingFactor,
        interfaceDimensions
      );
    });

    runtime.replaceGraphSectionOnSpec(
      spec,
      getTreePeriodicCell(spec, activeCellName) || runtime.buildEmptyGraphSection()
    );

    return spec;
  }

  function syncCurrentGraphIntoTreePeriodicTree(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    if (!tree) {
      return spec;
    }
    syncTreePeriodicBoundaryTensors(spec);
    syncTreePeriodicTreeInterfaceDimensions(spec);
    return spec;
  }

  function invalidateActiveTreePeriodicLookups(spec = state.spec) {
    if (spec !== state.spec) {
      return;
    }
    state.lookupRevision = -1;
    if (typeof ctx.resetDerivedStateCaches === "function") {
      ctx.resetDerivedStateCaches();
    }
  }

  function hydrateActiveTreePeriodicCell(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    if (!tree) {
      return spec;
    }
    tree.active_cell = TREE_PERIODIC_CELL_ORDER.includes(tree.active_cell)
      ? tree.active_cell
      : "root";
    const activeCell = getTreePeriodicCell(spec, tree.active_cell);
    runtime.replaceGraphSectionOnSpec(
      spec,
      activeCell || runtime.buildEmptyGraphSection()
    );
    invalidateActiveTreePeriodicLookups(spec);
    syncTreePeriodicBoundaryTensors(spec);
    return spec;
  }

  function stripTreePeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    stripped.contraction_plan = null;
    const boundaryTensorIds = new Set(
      stripped.tensors
        .filter((tensor) => treeState.isTreePeriodicBoundaryTensor(tensor))
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
        tensor_ids: group.tensor_ids.filter(
          (tensorId) => !boundaryTensorIds.has(tensorId)
        ),
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

  function switchTreePeriodicCell(direction) {
    if (!isTreePeriodicMode()) {
      return;
    }
    const activeCellName = getActiveTreePeriodicCellName();
    const nextCellName = getTreePeriodicNeighborCellName(activeCellName, direction);
    if (!nextCellName) {
      return;
    }
    syncCurrentGraphIntoTreePeriodicTree();
    state.spec.tree_periodic_tree.active_cell = nextCellName;
    hydrateActiveTreePeriodicCell();
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
      `Editing ${getTreePeriodicCellLabel(nextCellName).toLowerCase()}.`,
      "success"
    );
  }

  function buildTreePeriodicSeedGraphSection() {
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
      typeof runtime.isGridPeriodicMode === "function" &&
      runtime.isGridPeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoGridPeriodicGrid();
      const strippedGridCell =
        typeof runtime.stripGridPeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripGridPeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.grid_periodic_grid = null;
      runtime.replaceGraphSectionOnSpec(state.spec, strippedGridCell);
      return strippedGridCell;
    }
    return runtime.buildGraphSectionFromSpec(state.spec);
  }

  function readTreePeriodicBranchingFactor() {
    if (!window || typeof window.prompt !== "function") {
      return null;
    }
    const response = window.prompt(
      "How many child branches should each tree node have?",
      "2"
    );
    if (response === null) {
      return null;
    }
    const trimmedResponse = String(response).trim();
    const branchingFactor = Number(trimmedResponse);
    if (!Number.isInteger(branchingFactor) || branchingFactor < 2) {
      ctx.setStatus(
        "Enter an integer branching factor of 2 or more to enable For Tree mode.",
        "error"
      );
      return null;
    }
    return branchingFactor;
  }

  function toggleTreePeriodicMode() {
    if (!state.spec) {
      return;
    }
    if (
      !isTreePeriodicMode() &&
      typeof runtime.isBenchmarkMode === "function" &&
      runtime.isBenchmarkMode()
    ) {
      ctx.setStatus(
        "Leave Benchmark mode before enabling For Tree mode.",
        "error"
      );
      return;
    }
    if (
      !isTreePeriodicMode() &&
      Array.isArray(state.spec?.hyperedges) &&
      state.spec.hyperedges.length
    ) {
      ctx.setStatus(
        "For Tree mode does not support hyperedges yet. Remove them first or stay in normal mode.",
        "error"
      );
      return;
    }
    if (isTreePeriodicMode()) {
      if (
        !window.confirm(
          "Leave For Tree mode and keep the current cell as a normal network?"
        )
      ) {
        return;
      }
      syncCurrentGraphIntoTreePeriodicTree();
      const plainActiveCell = stripTreePeriodicBoundaryTensorsFromGraphSection(
        runtime.buildGraphSectionFromSpec(state.spec)
      );
      state.spec.tree_periodic_tree = null;
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
        "For Tree mode disabled. Restored the active cell as a normal network.",
        "success"
      );
      return;
    }

    const branchingFactor = readTreePeriodicBranchingFactor();
    if (branchingFactor === null) {
      return;
    }
    const rootCell = seedTreePeriodicCell(
      "root",
      buildTreePeriodicSeedGraphSection(),
      branchingFactor
    );
    const interfaceDimensions = {
      rootChildDimensions: getTreePeriodicBoundaryInterfaceDimensions(
        rootCell,
        "child",
        0
      ),
      branchChildDimensions: [],
    };
    const branchCell = seedTreePeriodicCell(
      "branch",
      runtime.buildEmptyGraphSection(),
      branchingFactor,
      interfaceDimensions
    );
    const leafCell = seedTreePeriodicCell(
      "leaf",
      runtime.buildEmptyGraphSection(),
      branchingFactor,
      interfaceDimensions
    );
    state.spec.linear_periodic_chain = null;
    state.spec.grid_periodic_grid = null;
    state.spec.tree_periodic_tree = {
      active_cell: "root",
      branching_factor: branchingFactor,
      root_cell: rootCell,
      branch_cell: branchCell,
      leaf_cell: leafCell,
      metadata: {},
    };
    hydrateActiveTreePeriodicCell();
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
      "For Tree mode enabled. You are editing the root cell.",
      "success"
    );
  }

  function setTreePeriodicMode(enabled) {
    const shouldEnable = Boolean(enabled);
    if (shouldEnable === isTreePeriodicMode()) {
      return;
    }
    toggleTreePeriodicMode();
  }

  return {
    syncTreePeriodicTreeInterfaceDimensions,
    syncCurrentGraphIntoTreePeriodicTree,
    invalidateActiveTreePeriodicLookups,
    hydrateActiveTreePeriodicCell,
    stripTreePeriodicBoundaryTensorsFromGraphSection,
    switchTreePeriodicCell,
    toggleTreePeriodicMode,
    setTreePeriodicMode,
  };
}
