import { createContractionSceneCacheSupport } from "./contractionSceneCache.js";
import { createContractionSceneEditingSupport } from "./contractionSceneEditing.js";
import { createContractionSceneOperandsSupport } from "./contractionSceneOperands.js";

export function registerContractionScene(ctx) {
  const state = ctx.state;

  const operandsSupport = createContractionSceneOperandsSupport({
    state,
    ensureSpecLookups: () => ctx.ensureSpecLookups(),
    getContractibleEdges: () => ctx.getContractibleEdges(),
    getContractibleTensors: () => ctx.getContractibleTensors(),
    getLinearPeriodicReservedOperandId:
      typeof ctx.getLinearPeriodicReservedOperandId === "function"
        ? (role) => ctx.getLinearPeriodicReservedOperandId(role)
        : null,
    getLinearPeriodicReservedOperandIdForTensor:
      typeof ctx.getLinearPeriodicReservedOperandIdForTensor === "function"
        ? (tensor) => ctx.getLinearPeriodicReservedOperandIdForTensor(tensor)
        : null,
    isLinearPeriodicBoundaryTensor:
      typeof ctx.isLinearPeriodicBoundaryTensor === "function"
        ? (tensor) => ctx.isLinearPeriodicBoundaryTensor(tensor)
        : null,
    isLinearPeriodicMode:
      typeof ctx.isLinearPeriodicMode === "function"
        ? () => ctx.isLinearPeriodicMode()
        : null,
  });
  const {
    buildContractionOperandProgression,
    buildContractionOperandState,
    getContractionPlan,
    getPlanSteps,
    getTensorKrowchManualPlanIssue,
    getTensorKrowchManualPlanIssueMessage,
    isNextOperandId,
    isPreviousOperandId,
  } = operandsSupport;
  const cacheSupport = createContractionSceneCacheSupport({
    state,
    asFiniteNumber: (value, fallbackValue) => ctx.asFiniteNumber(value, fallbackValue),
    buildContractionOperandProgression,
    constants: ctx.constants,
    defaultIndexOffsetForOrder: (indexPosition, tensor) =>
      ctx.defaultIndexOffsetForOrder(indexPosition, tensor),
    ensureSpecLookups: () => ctx.ensureSpecLookups(),
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    getContractionPlan,
    isNextOperandId,
    isPreviousOperandId,
    tensorHeight: (tensor) => ctx.tensorHeight(tensor),
    tensorWidth: (tensor) => ctx.tensorWidth(tensor),
  });
  const {
    buildContractionScene,
    buildLayoutMapFromSnapshot,
    captureVisibleOperandLayoutMap,
    ensureContractionViewSnapshots,
    findVisibleEdgeById,
    findVisibleEdgeSelectionIdByBaseEdgeId,
    findVisibleTensorById,
    getLatestAppliedStepCount,
    getSnapshotForStepCount,
    getVisibleEdges,
    getVisibleTensors,
    getViewedAppliedStepCount,
    touchContractionViewRevision,
  } = cacheSupport;
  const editingSupport = createContractionSceneEditingSupport({
    state,
    buildContractionScene,
    buildLayoutMapFromSnapshot,
    captureVisibleOperandLayoutMap,
    constants: ctx.constants,
    ensureContractionPlan: () => ctx.ensureContractionPlan(),
    ensureContractionViewSnapshots,
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    getLatestAppliedStepCount,
    getSnapshotForStepCount,
    getViewedAppliedStepCount,
    isNextOperandId,
    isPreviousOperandId,
    makeId: (prefix) => ctx.makeId(prefix),
    touchContractionViewRevision,
  });

  Object.assign(ctx, {
    applyManualContractionStep: editingSupport.applyManualContractionStep,
    applySnapshotLayoutMap: editingSupport.applySnapshotLayoutMap,
    beginPastInspection: editingSupport.beginPastInspection,
    buildContractionOperandState,
    buildContractionScene,
    canEditCurrentContractionStage: editingSupport.canEditCurrentContractionStage,
    captureVisibleOperandLayoutMap,
    clearPastInspection: editingSupport.clearPastInspection,
    ensureContractionViewSnapshots,
    findVisibleEdgeById,
    findVisibleEdgeSelectionIdByBaseEdgeId,
    findVisibleTensorById,
    getContractionPlan,
    getLatestAppliedStepCount,
    getPlanSteps,
    getSnapshotForStepCount,
    getTensorKrowchManualPlanIssue,
    getTensorKrowchManualPlanIssueMessage,
    getViewedAppliedStepCount,
    getVisibleEdges,
    getVisibleTensors,
    isContractionSceneVisible: editingSupport.isContractionSceneVisible,
    isInspectingPastStage: editingSupport.isInspectingPastStage,
    toggleFutureBadgeDisclosure: editingSupport.toggleFutureBadgeDisclosure,
    togglePastInspection: editingSupport.togglePastInspection,
    updateCurrentStageOperandLayout: editingSupport.updateCurrentStageOperandLayout,
  });
}
