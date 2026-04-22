import { createTreePeriodicBoundarySupport } from "./utilitiesTreePeriodicBoundaries.js";
import { createTreePeriodicFlowSupport } from "./utilitiesTreePeriodicFlow.js";
import { createTreePeriodicStateSupport } from "./utilitiesTreePeriodicState.js";

export function createUtilityTreePeriodicBindings({
  ctx,
  state,
  constants,
  runtime,
}) {
  const treeState = createTreePeriodicStateSupport({ state, runtime });
  const treeBoundaries = createTreePeriodicBoundarySupport({
    state,
    constants,
    runtime,
    treeState,
  });
  const treeFlow = createTreePeriodicFlowSupport({
    ctx,
    state,
    runtime,
    treeState,
    treeBoundaries,
  });

  return {
    normalizeTreePeriodicTreeInPlace: treeState.normalizeTreePeriodicTreeInPlace,
    getTreePeriodicTree: treeState.getTreePeriodicTree,
    isTreePeriodicMode: treeState.isTreePeriodicMode,
    getTreePeriodicBranchingFactor: treeState.getTreePeriodicBranchingFactor,
    getActiveTreePeriodicCellName: treeState.getActiveTreePeriodicCellName,
    getTreePeriodicCellKey: treeState.getTreePeriodicCellKey,
    getTreePeriodicCell: treeState.getTreePeriodicCell,
    getTreePeriodicCellLabel: treeState.getTreePeriodicCellLabel,
    getTreePeriodicNeighborCellName: treeState.getTreePeriodicNeighborCellName,
    canSwitchTreePeriodicCell: treeState.canSwitchTreePeriodicCell,
    isTreePeriodicBoundaryTensor: treeState.isTreePeriodicBoundaryTensor,
    syncTreePeriodicBoundaryTensors: treeBoundaries.syncTreePeriodicBoundaryTensors,
    syncCurrentGraphIntoTreePeriodicTree:
      treeFlow.syncCurrentGraphIntoTreePeriodicTree,
    hydrateActiveTreePeriodicCell: treeFlow.hydrateActiveTreePeriodicCell,
    stripTreePeriodicBoundaryTensorsFromGraphSection:
      treeFlow.stripTreePeriodicBoundaryTensorsFromGraphSection,
    switchTreePeriodicCell: treeFlow.switchTreePeriodicCell,
    toggleTreePeriodicMode: treeFlow.toggleTreePeriodicMode,
    setTreePeriodicMode: treeFlow.setTreePeriodicMode,
  };
}
