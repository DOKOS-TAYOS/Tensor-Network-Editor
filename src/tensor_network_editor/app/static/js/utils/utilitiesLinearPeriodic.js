import { createLinearPeriodicBoundarySupport } from "./utilitiesLinearPeriodicBoundaries.js";
import { createLinearPeriodicFlowSupport } from "./utilitiesLinearPeriodicFlow.js";
import { createLinearPeriodicStateSupport } from "./utilitiesLinearPeriodicState.js";

export function createUtilityLinearPeriodicBindings({
  ctx,
  state,
  constants,
  dom,
  runtime,
}) {
  const linearState = createLinearPeriodicStateSupport({
    ctx,
    state,
    runtime,
  });
  const linearBoundaries = createLinearPeriodicBoundarySupport({
    ctx,
    state,
    constants,
    runtime,
    linearState,
  });
  const linearFlow = createLinearPeriodicFlowSupport({
    ctx,
    state,
    dom,
    runtime,
    linearState,
    linearBoundaries,
  });

  return {
    getLinearPeriodicReservedOperandId:
      linearState.getLinearPeriodicReservedOperandId,
    isLinearPeriodicReservedOperandId:
      linearState.isLinearPeriodicReservedOperandId,
    normalizeLinearPeriodicChainInPlace:
      linearState.normalizeLinearPeriodicChainInPlace,
    getLinearPeriodicChain: linearState.getLinearPeriodicChain,
    isLinearPeriodicMode: linearState.isLinearPeriodicMode,
    getActiveLinearPeriodicCellName: linearState.getActiveLinearPeriodicCellName,
    getLinearPeriodicCell: linearState.getLinearPeriodicCell,
    getLinearPeriodicBoundaryTensorByRole:
      linearState.getLinearPeriodicBoundaryTensorByRole,
    isLinearPeriodicBoundaryTensor: linearState.isLinearPeriodicBoundaryTensor,
    getLinearPeriodicReservedOperandIdForTensor:
      linearState.getLinearPeriodicReservedOperandIdForTensor,
    getContractibleTensors: linearState.getContractibleTensors,
    getContractibleEdges: linearState.getContractibleEdges,
    syncCurrentGraphIntoLinearPeriodicChain:
      linearFlow.syncCurrentGraphIntoLinearPeriodicChain,
    hydrateActiveLinearPeriodicCell: linearFlow.hydrateActiveLinearPeriodicCell,
    stripLinearPeriodicBoundaryTensorsFromGraphSection:
      linearBoundaries.stripLinearPeriodicBoundaryTensorsFromGraphSection,
    syncLinearPeriodicChainInterfaceDimensions:
      linearFlow.syncLinearPeriodicChainInterfaceDimensions,
    buildHistorySnapshotSpec: runtime.buildHistorySnapshotSpec,
    buildSerializedSpec: runtime.buildSerializedSpec,
    switchLinearPeriodicCell: linearFlow.switchLinearPeriodicCell,
    toggleLinearPeriodicMode: linearFlow.toggleLinearPeriodicMode,
    setLinearPeriodicMode: linearFlow.setLinearPeriodicMode,
    syncLinearPeriodicBoundaryTensors:
      linearBoundaries.syncLinearPeriodicBoundaryTensors,
    enforceLinearPeriodicEngineSupport:
      linearFlow.enforceLinearPeriodicEngineSupport,
  };
}
