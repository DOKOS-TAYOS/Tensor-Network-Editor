import { createGridPeriodicBoundarySupport } from "./utilitiesGridPeriodicBoundaries.js";
import { createGridPeriodicFlowSupport } from "./utilitiesGridPeriodicFlow.js";
import { createGridPeriodicStateSupport } from "./utilitiesGridPeriodicState.js";

export function createUtilityGridPeriodicBindings({
  ctx,
  state,
  constants,
  runtime,
}) {
  const gridState = createGridPeriodicStateSupport({ state, runtime });
  const gridBoundaries = createGridPeriodicBoundarySupport({
    ctx,
    state,
    constants,
    runtime,
    gridState,
  });
  const gridFlow = createGridPeriodicFlowSupport({
    ctx,
    state,
    runtime,
    gridState,
    gridBoundaries,
  });

  return {
    normalizeGridPeriodicGridInPlace: gridState.normalizeGridPeriodicGridInPlace,
    getGridPeriodicGrid: gridState.getGridPeriodicGrid,
    isGridPeriodicMode: gridState.isGridPeriodicMode,
    isForMode: gridState.isForMode,
    getActiveGridPeriodicCellName: gridState.getActiveGridPeriodicCellName,
    getGridPeriodicCell: gridState.getGridPeriodicCell,
    getGridPeriodicCellLabel: gridState.getGridPeriodicCellLabel,
    getGridPeriodicNeighborCellName: gridState.getGridPeriodicNeighborCellName,
    canSwitchGridPeriodicCell: gridState.canSwitchGridPeriodicCell,
    getGridPeriodicBoundaryTensorByRole:
      gridState.getGridPeriodicBoundaryTensorByRole,
    isGridPeriodicBoundaryTensor: gridState.isGridPeriodicBoundaryTensor,
    isForBoundaryTensor: gridState.isForBoundaryTensor,
    getExpectedGridPeriodicRoles: gridState.getExpectedGridPeriodicRoles,
    syncGridPeriodicBoundaryTensors: gridBoundaries.syncGridPeriodicBoundaryTensors,
    syncGridPeriodicGridInterfaceDimensions:
      gridFlow.syncGridPeriodicGridInterfaceDimensions,
    syncCurrentGraphIntoGridPeriodicGrid:
      gridFlow.syncCurrentGraphIntoGridPeriodicGrid,
    hydrateActiveGridPeriodicCell: gridFlow.hydrateActiveGridPeriodicCell,
    stripGridPeriodicBoundaryTensorsFromGraphSection:
      gridFlow.stripGridPeriodicBoundaryTensorsFromGraphSection,
    switchGridPeriodicCell: gridFlow.switchGridPeriodicCell,
    toggleGridPeriodicMode: gridFlow.toggleGridPeriodicMode,
    setGridPeriodicMode: gridFlow.setGridPeriodicMode,
  };
}
