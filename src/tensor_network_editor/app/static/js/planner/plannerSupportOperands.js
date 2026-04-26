import {
  buildPlannerOperandState as buildPlannerOperandStateFromSelectors,
  buildPlannerSeedOperands as buildPlannerSeedOperandsFromSelectors,
  buildPreviewOrderByVisibleTensorId as buildPreviewOrderByVisibleTensorIdFromSelectors,
  getAutomaticAnalysisByMode as getAutomaticAnalysisByModeFromSelectors,
} from "../state/plannerSelectors.js";

export function createPlannerOperandSupport({
  ctx,
  state,
  previousOperandId,
  nextOperandId,
  guards,
}) {
  const {
    clearPlannerTransientState,
    isBenchmarkBasePosition,
    isGridPeriodicMode,
    isTreePeriodicMode,
    resetPlannerBadgeDisclosureState,
  } = guards;

  function ensureContractionPlan() {
    if (!state.spec.contraction_plan) {
      state.spec.contraction_plan = {
        id: ctx.makeId("plan"),
        name: "Manual path",
        steps: [],
        metadata: {},
      };
    }
    return state.spec.contraction_plan;
  }

  function getCurrentPlanSteps() {
    return state.spec &&
      state.spec.contraction_plan &&
      Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
  }

  function hasFreshContractibleCollections() {
    const tensors = Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [];
    const edges = Array.isArray(state.spec && state.spec.edges) ? state.spec.edges : [];
    return (
      state.contractibleCacheRevision === state.specRevision &&
      state.contractibleCacheTensorRef === tensors &&
      state.contractibleCacheTensorCount === tensors.length &&
      state.contractibleCacheEdgeRef === edges &&
      state.contractibleCacheEdgeCount === edges.length
    );
  }

  function buildSeedOperandsForTensors(tensors) {
    return buildPlannerSeedOperandsFromSelectors({
      tensors,
      specTensors: state.spec && state.spec.tensors,
      isLinearPeriodicMode:
        typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode(),
      isLinearPeriodicBoundaryTensor: (tensor) => ctx.isLinearPeriodicBoundaryTensor(tensor),
      getLinearPeriodicReservedOperandIdForTensor: (tensor) =>
        ctx.getLinearPeriodicReservedOperandIdForTensor(tensor),
      isGridPeriodicMode:
        typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode(),
      isGridPeriodicBoundaryTensor: (tensor) =>
        typeof ctx.isGridPeriodicBoundaryTensor === "function" &&
        ctx.isGridPeriodicBoundaryTensor(tensor),
      getGridPeriodicReservedOperandIdForTensor: (tensor) =>
        typeof ctx.getGridPeriodicReservedOperandIdForTensor === "function"
          ? ctx.getGridPeriodicReservedOperandIdForTensor(tensor)
          : null,
      isTreePeriodicMode:
        typeof ctx.isTreePeriodicMode === "function" && ctx.isTreePeriodicMode(),
      isTreePeriodicBoundaryTensor: (tensor) =>
        typeof ctx.isTreePeriodicBoundaryTensor === "function" &&
        ctx.isTreePeriodicBoundaryTensor(tensor),
      getTreePeriodicReservedOperandIdForTensor: (tensor) =>
        typeof ctx.getTreePeriodicReservedOperandIdForTensor === "function"
          ? ctx.getTreePeriodicReservedOperandIdForTensor(tensor)
          : null,
      syntheticOperands: getSyntheticAnalysisOperands(),
    });
  }

  function getSyntheticAnalysisOperands() {
    return state.contractionAnalysis &&
      state.contractionAnalysis.status === "ready" &&
      state.contractionAnalysis.payload &&
      Array.isArray(state.contractionAnalysis.payload.synthetic_operands)
      ? state.contractionAnalysis.payload.synthetic_operands
      : [];
  }

  function buildPlannerOperandStateForSteps(steps, tensors) {
    return buildPlannerOperandStateFromSelectors({
      tensors,
      steps,
      seedOperands: buildSeedOperandsForTensors(tensors),
      previousOperandId,
      nextOperandId,
    });
  }

  function getPlannerOperandState() {
    const planSteps = getCurrentPlanSteps();
    const cacheIsFresh =
      hasFreshContractibleCollections() &&
      state.plannerOperandStateCacheRevision === state.specRevision &&
      state.plannerOperandStateCacheStepsRef === planSteps &&
      state.plannerOperandStateCacheStepCount === planSteps.length &&
      state.plannerOperandStateCacheContractibleToken === state.contractibleCacheToken &&
      state.plannerOperandStateCache;
    if (cacheIsFresh) {
      return state.plannerOperandStateCache;
    }

    const tensors = ctx.getContractibleTensors();
    const contractibleToken = state.contractibleCacheToken;
    const plannerOperandState = buildPlannerOperandStateForSteps(planSteps, tensors);
    state.plannerOperandStateCacheRevision = state.specRevision;
    state.plannerOperandStateCacheStepsRef = planSteps;
    state.plannerOperandStateCacheStepCount = planSteps.length;
    state.plannerOperandStateCacheContractibleToken = contractibleToken;
    state.plannerOperandStateCache = plannerOperandState;
    return plannerOperandState;
  }

  function getPlannerOperandStateForSteps(steps) {
    const planSteps = getCurrentPlanSteps();
    if (steps === planSteps) {
      return getPlannerOperandState();
    }
    return buildPlannerOperandStateForSteps(steps, ctx.getContractibleTensors());
  }

  function buildStepOrdersByTensorId(steps) {
    return getPlannerOperandStateForSteps(steps).stepOrdersByTensorId;
  }

  function syncPlannerOrderBadges() {
    state.plannerManualOrderByTensorId = {};
    if (
      state.plannerPreviewMode &&
      state.contractionAnalysis &&
      state.contractionAnalysis.status === "ready"
    ) {
      const previewAnalysis = getAutomaticAnalysisByModeFromSelectors(
        state.contractionAnalysis.payload,
        state.plannerPreviewMode
      );
      const visibleTensors =
        typeof ctx.getVisibleTensors === "function"
          ? ctx
              .getVisibleTensors()
              .filter((tensor) =>
                typeof ctx.isForBoundaryTensor === "function"
                  ? !ctx.isForBoundaryTensor(tensor)
                  : !ctx.isLinearPeriodicBoundaryTensor(tensor)
              )
          : ctx.getContractibleTensors();
      state.plannerPreviewOrderByTensorId = previewAnalysis
        ? buildPreviewOrderByVisibleTensorIdFromSelectors(
            visibleTensors,
            previewAnalysis.steps
          )
        : {};
      return;
    }
    state.plannerPreviewOrderByTensorId = {};
  }

  function resolvePlannerOperandId(operandId) {
    if (typeof operandId !== "string" || !operandId) {
      return null;
    }
    const plannerOperandState = getPlannerOperandState();
    return (
      plannerOperandState.representativeByOperandId[operandId] ||
      plannerOperandState.representativeByTensorId[operandId] ||
      null
    );
  }

  function repairContractionPlan() {
    if (isTreePeriodicMode()) {
      clearPlannerTransientState({ clearInspectionStepCount: true });
    }
    if (isGridPeriodicMode()) {
      clearPlannerTransientState({ clearInspectionStepCount: true });
    }
    if (isBenchmarkBasePosition()) {
      state.spec.contraction_plan = null;
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
    }
    const plan = state.spec.contraction_plan;
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      if (plan) {
        plan.view_snapshots = [];
      }
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
    }
    const plannerOperandState = getPlannerOperandState();
    if (!plannerOperandState.validSteps.length) {
      state.spec.contraction_plan = null;
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
    }
    plan.steps = plannerOperandState.validSteps;
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
    const latestAppliedStepCount =
      typeof ctx.getLatestAppliedStepCount === "function"
        ? ctx.getLatestAppliedStepCount()
        : plannerOperandState.validSteps.length;
    if (
      Number.isInteger(state.plannerInspectionStepCount) &&
      state.plannerInspectionStepCount >= latestAppliedStepCount
    ) {
      state.plannerInspectionStepCount = null;
    }
    resetPlannerBadgeDisclosureState();
  }

  function getPlannerRemainingOperandIds() {
    return getPlannerOperandState().activeOperandIds;
  }

  function isPlannerOperandAvailable(operandId) {
    return resolvePlannerOperandId(operandId) !== null;
  }

  function getPlannerOperandSourceTensorIds(operandId) {
    const representativeOperandId = resolvePlannerOperandId(operandId) || operandId;
    const plannerOperandState = getPlannerOperandState();
    return plannerOperandState.sourceTensorIdsByOperandId[representativeOperandId]
      ? [...plannerOperandState.sourceTensorIdsByOperandId[representativeOperandId]]
      : [];
  }

  function getPlannerOperandLabel(operandId) {
    const tensor = ctx.findTensorById(operandId);
    if (tensor) {
      return tensor.name;
    }
    if (operandId === previousOperandId) {
      return "Previous cell";
    }
    if (operandId === nextOperandId) {
      return "Next cell";
    }
    if (typeof ctx.getGridPeriodicReservedOperandLabel === "function") {
      const gridLabel = ctx.getGridPeriodicReservedOperandLabel(operandId);
      if (gridLabel) {
        return gridLabel;
      }
    }
    if (typeof ctx.getTreePeriodicReservedOperandLabel === "function") {
      const treeLabel = ctx.getTreePeriodicReservedOperandLabel(operandId);
      if (treeLabel) {
        return treeLabel;
      }
    }
    const syntheticOperand = getSyntheticAnalysisOperands().find(
      (operand) => operand && operand.operand_id === operandId
    );
    if (syntheticOperand && syntheticOperand.name) {
      return syntheticOperand.name;
    }
    const planSteps =
      state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
        ? state.spec.contraction_plan.steps
        : [];
    const stepIndex = planSteps.findIndex((step) => step.id === operandId);
    if (stepIndex >= 0) {
      return `Result ${stepIndex + 1}`;
    }
    if (/^auto_future_step_\d+$/.test(operandId)) {
      return `Auto future ${operandId.replace("auto_future_step_", "step ")}`;
    }
    if (/__auto_past_\d+$/.test(operandId)) {
      return `Auto past ${operandId.split("__auto_past_")[1]}`;
    }
    return operandId;
  }

  return {
    ensureContractionPlan,
    getCurrentPlanSteps,
    getPlannerOperandState,
    buildStepOrdersByTensorId,
    syncPlannerOrderBadges,
    resolvePlannerOperandId,
    repairContractionPlan,
    getPlannerRemainingOperandIds,
    isPlannerOperandAvailable,
    getPlannerOperandSourceTensorIds,
    getPlannerOperandLabel,
  };
}
