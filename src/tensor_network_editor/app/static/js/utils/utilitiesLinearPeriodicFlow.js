import {
  LINEAR_PERIODIC_CELL_LABELS,
  LINEAR_PERIODIC_CELL_ORDER,
} from "./utilitiesLinearPeriodicState.js";

export function createLinearPeriodicFlowSupport({
  ctx,
  state,
  dom,
  runtime,
  linearState,
  linearBoundaries,
}) {
  const { window } = ctx;
  const { engineSelect } = dom;
  const {
    getLinearPeriodicChain,
    isLinearPeriodicMode,
    getActiveLinearPeriodicCellName,
    getLinearPeriodicCell,
  } = linearState;
  const {
    getCanonicalLinearPeriodicInterfaceDimensions,
    syncLinearPeriodicBoundaryTensors,
    stripLinearPeriodicBoundaryTensorsFromGraphSection,
  } = linearBoundaries;

  function seedLinearPeriodicCell(
    cellName,
    graphSection,
    interfaceDimensions = null
  ) {
    const runtimeSpec = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    runtimeSpec.linear_periodic_chain = { active_cell: cellName };
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

  function buildLinearPeriodicSeedGraphSection() {
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
    return seedGraphSection;
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
    if (
      !isLinearPeriodicMode() &&
      Array.isArray(state.spec?.hyperedges) &&
      state.spec.hyperedges.length
    ) {
      ctx.setStatus(
        "For unidimensional mode does not support hyperedges yet. Remove them first or stay in normal mode.",
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

    const initialCell = seedLinearPeriodicCell(
      "initial",
      buildLinearPeriodicSeedGraphSection()
    );
    const periodicCell = seedLinearPeriodicCell(
      "periodic",
      runtime.buildEmptyGraphSection()
    );
    const finalCell = seedLinearPeriodicCell("final", runtime.buildEmptyGraphSection());
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
    seedLinearPeriodicCell,
    syncLinearPeriodicChainInterfaceDimensions,
    syncCurrentGraphIntoLinearPeriodicChain,
    invalidateActiveLinearPeriodicLookups,
    hydrateActiveLinearPeriodicCell,
    resetTransientEditorStateForCellSwitch,
    switchLinearPeriodicCell,
    enforceLinearPeriodicEngineSupport,
    toggleLinearPeriodicMode,
    setLinearPeriodicMode,
  };
}
