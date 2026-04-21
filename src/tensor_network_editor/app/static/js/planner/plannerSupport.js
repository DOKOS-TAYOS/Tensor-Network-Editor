import { createPlannerCommands } from "../actions/plannerCommands.js";
import { createPlannerAnalysisService } from "../services/plannerAnalysisService.js";
import { createPlannerAutomaticSupport } from "./plannerAutomaticSupport.js";
import {
  buildPlannerOperandState as buildPlannerOperandStateFromSelectors,
  buildPlannerSeedOperands as buildPlannerSeedOperandsFromSelectors,
  buildPreviewOrderByVisibleTensorId as buildPreviewOrderByVisibleTensorIdFromSelectors,
  getAutomaticAnalysisByMode as getAutomaticAnalysisByModeFromSelectors,
} from "../state/plannerSelectors.js";

export function createPlannerSupport({
  ctx,
  state,
  analysisRefreshDelayMs,
  setTimer,
  clearTimer,
  getRenderPlanner,
}) {
  const previousOperandId =
    typeof ctx.getLinearPeriodicReservedOperandId === "function"
      ? ctx.getLinearPeriodicReservedOperandId("previous")
      : "__linear_previous__";
  const nextOperandId =
    typeof ctx.getLinearPeriodicReservedOperandId === "function"
      ? ctx.getLinearPeriodicReservedOperandId("next")
      : "__linear_next__";
  const benchmarkBaseStatusMessage =
    typeof ctx.benchmarkBaseStatusHint === "string" && ctx.benchmarkBaseStatusHint
      ? ctx.benchmarkBaseStatusHint
      : "Move right to edit or create a contraction scheme.";
  const gridPeriodicStatusMessage =
    "Contractions are disabled in For bidimensional mode.";
  const treePeriodicStatusMessage =
    "Contractions are disabled in For Tree mode.";
  const hyperedgeStatusMessage =
    "Manual contraction planning is unavailable while the design contains hyperedges.";
  let pendingContractionAnalysisOptions = null;
  let plannerCommands = null;

  function getCachedContractionAnalysisPayload() {
    return state.contractionAnalysisCacheRevision === state.specRevision
      ? state.contractionAnalysisCachePayload
      : null;
  }

  function markContractionAnalysisDirty() {
    state.contractionAnalysisDirty = true;
  }

  function isBenchmarkBasePosition() {
    return (
      typeof ctx.isBenchmarkBasePosition === "function" &&
      ctx.isBenchmarkBasePosition()
    );
  }

  function isGridPeriodicMode() {
    return typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode();
  }

  function isTreePeriodicMode() {
    return typeof ctx.isTreePeriodicMode === "function" && ctx.isTreePeriodicMode();
  }

  function hasHyperedges() {
    return Boolean(Array.isArray(state.spec?.hyperedges) && state.spec.hyperedges.length);
  }

  function resetPlannerBadgeDisclosureState() {
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
  }

  function clearPlannerTransientState({ clearInspectionStepCount = false } = {}) {
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerPreviewMode = null;
    if (clearInspectionStepCount) {
      state.plannerInspectionStepCount = null;
    }
    resetPlannerBadgeDisclosureState();
  }

  function guardBenchmarkBasePlannerAction(message = benchmarkBaseStatusMessage) {
    if (!isBenchmarkBasePosition()) {
      return false;
    }
    state.plannerMode = false;
    clearPlannerTransientState({ clearInspectionStepCount: true });
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(message);
    return true;
  }

  function guardGridPeriodicPlannerAction(message = gridPeriodicStatusMessage) {
    if (!isGridPeriodicMode()) {
      return false;
    }
    state.plannerMode = false;
    clearPlannerTransientState({ clearInspectionStepCount: true });
    if (state.spec) {
      state.spec.contraction_plan = null;
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(message);
    return true;
  }

  function guardTreePeriodicPlannerAction(message = treePeriodicStatusMessage) {
    if (!isTreePeriodicMode()) {
      return false;
    }
    state.plannerMode = false;
    clearPlannerTransientState({ clearInspectionStepCount: true });
    if (state.spec) {
      state.spec.contraction_plan = null;
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(message);
    return true;
  }

  function guardHyperedgePlannerAction(message = hyperedgeStatusMessage) {
    if (!hasHyperedges()) {
      return false;
    }
    state.plannerMode = false;
    clearPlannerTransientState({ clearInspectionStepCount: true });
    if (state.spec) {
      state.spec.contraction_plan = null;
    }
    state.contractionAnalysis = {
      status: "hyperedgesDisabled",
      message,
    };
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(message);
    return true;
  }

  function renderPlanner() {
    getRenderPlanner()();
  }

  const analysisService = createPlannerAnalysisService({
    analysisRefreshDelayMs,
    analyze: (payload) => ctx.apiPost("/api/analyze-contraction", payload),
    cancel: (timerId) => clearTimer(timerId),
    onAnalysisError: (error) => {
      state.contractionAnalysisCacheRevision = -1;
      state.contractionAnalysisCachePayload = null;
      state.contractionAnalysis = {
        status: "error",
        message: error.message,
      };
    },
    onAnalysisResult: (payload) => {
      if (!payload.ok) {
        state.contractionAnalysis = {
          status: "issues",
          issues: payload.issues || [],
        };
        return;
      }
      state.contractionAnalysisCacheRevision = state.specRevision;
      state.contractionAnalysisCachePayload = payload;
      state.contractionAnalysis = {
        status: "ready",
        payload,
      };
    },
    onRequestStarted: (options) => {
      if (options.focusTab && typeof ctx.setActiveSidebarTab === "function") {
        ctx.setActiveSidebarTab("planner");
      }
      state.contractionAnalysisRequestId += 1;
      state.contractionAnalysis = { status: "loading" };
      renderPlanner();
    },
    onRenderRequested: () => {
      renderPlanner();
      ctx.renderOverlayDecorations();
    },
    schedule: (callback, delay) => setTimer(callback, delay),
    serializeCurrentSpec: (options) => ctx.serializeCurrentSpec(options),
  });

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
    return state.spec && state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
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
    });
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

  function buildStepOrdersByTensorId(steps) {
    return getPlannerOperandStateForSteps(steps).stepOrdersByTensorId;
  }

  function getPlannerOperandStateForSteps(steps) {
    const planSteps = getCurrentPlanSteps();
    if (steps === planSteps) {
      return getPlannerOperandState();
    }
    return buildPlannerOperandStateForSteps(steps, ctx.getContractibleTensors());
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
              .filter((tensor) => !ctx.isLinearPeriodicBoundaryTensor(tensor))
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
    if (hasHyperedges()) {
      state.spec.contraction_plan = null;
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
    }
    if (isTreePeriodicMode()) {
      state.spec.contraction_plan = null;
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
    }
    if (isGridPeriodicMode()) {
      state.spec.contraction_plan = null;
      clearPlannerTransientState({ clearInspectionStepCount: true });
      return;
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

  function handlePlannerOperandClick(operandId) {
    if (guardHyperedgePlannerAction()) {
      return;
    }
    if (guardTreePeriodicPlannerAction()) {
      return;
    }
    if (guardGridPeriodicPlannerAction()) {
      return;
    }
    if (guardBenchmarkBasePlannerAction()) {
      return;
    }
    return plannerCommands.handlePlannerOperandClick(operandId);
  }

  function trimContractionPlanInPlace(stepCount) {
    const plan = state.spec.contraction_plan;
    if (!plan) {
      return false;
    }
    if (stepCount <= 0) {
      state.spec.contraction_plan = null;
    } else {
      plan.steps = plan.steps.slice(0, stepCount);
    }
    state.plannerPreviewMode = null;
    resetPlannerBadgeDisclosureState();
    state.plannerInspectionStepCount =
      stepCount <= 0
        ? null
        : Number.isInteger(state.plannerInspectionStepCount)
        ? Math.min(state.plannerInspectionStepCount, stepCount - 1)
        : null;
    return true;
  }

  function trimContractionPlan(stepCount) {
    if (guardHyperedgePlannerAction()) {
      return;
    }
    if (guardTreePeriodicPlannerAction()) {
      return;
    }
    if (guardGridPeriodicPlannerAction()) {
      return;
    }
    if (guardBenchmarkBasePlannerAction()) {
      return;
    }
    const plan = state.spec.contraction_plan;
    if (!plan) {
      return;
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    ctx.applyDesignChange(
      () => {
        trimContractionPlanInPlace(stepCount);
      },
      {
        statusMessage:
          stepCount <= 0 ? "Reset the manual contraction path." : "Trimmed the manual contraction path.",
      }
    );
  }

  function togglePlannerMode() {
    if (guardHyperedgePlannerAction()) {
      return;
    }
    if (guardTreePeriodicPlannerAction()) {
      return;
    }
    if (guardGridPeriodicPlannerAction()) {
      return;
    }
    if (guardBenchmarkBasePlannerAction()) {
      return;
    }
    state.plannerMode = !state.plannerMode;
    if (!state.plannerMode) {
      state.pendingPlannerOperandId = null;
      state.pendingPlannerSelectionId = null;
    }
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    } else if (state.plannerMode && state.contractionAnalysisDirty) {
      refreshContractionAnalysis();
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(
      state.plannerMode
        ? "Manual planner mode active. Click visible tensors or result tensors to define the next contraction step."
        : "Manual planner mode disabled."
    );
  }

  function runContractionAnalysisRequest(options = {}) {
    pendingContractionAnalysisOptions = options;
    return analysisService.flushQueue();
  }

  function flushContractionAnalysisQueue() {
    const queuedOptions = pendingContractionAnalysisOptions || {};
    pendingContractionAnalysisOptions = null;
    return runContractionAnalysisRequest(queuedOptions);
  }

  function shouldRefreshContractionAnalysisImmediately(options = {}) {
    return (
      Boolean(options.immediate) ||
      state.activeSidebarTab === "planner" ||
      state.plannerMode ||
      Boolean(state.plannerPreviewMode) ||
      (typeof ctx.isInspectingPastStage === "function" &&
        ctx.isInspectingPastStage()) ||
      (typeof ctx.isContractionSceneVisible === "function" &&
        ctx.isContractionSceneVisible())
    );
  }

  function refreshContractionAnalysis(options = {}) {
    if (hasHyperedges()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "hyperedgesDisabled",
        message: hyperedgeStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isTreePeriodicMode()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "treePeriodicDisabled",
        message: treePeriodicStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isGridPeriodicMode()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "gridPeriodicDisabled",
        message: gridPeriodicStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isBenchmarkBasePosition()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      state.contractionAnalysis = { status: "benchmarkBase" };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    state.contractionAnalysisDirty = false;
    const cachedPayload = getCachedContractionAnalysisPayload();
    if (cachedPayload) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysis = {
        status: "ready",
        payload: cachedPayload,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return Promise.resolve(cachedPayload);
    }
    pendingContractionAnalysisOptions = {
      focusTab:
        Boolean(options.focusTab) ||
        Boolean(
          pendingContractionAnalysisOptions &&
            pendingContractionAnalysisOptions.focusTab
        ),
    };
    return analysisService.requestRefresh(pendingContractionAnalysisOptions, {
      immediate: shouldRefreshContractionAnalysisImmediately(options),
    });
  }

  function togglePlannerDisclosure(disclosureKey) {
    state.plannerDisclosureState[disclosureKey] = !state.plannerDisclosureState[disclosureKey];
    renderPlanner();
  }


  const automaticPlannerSupport = createPlannerAutomaticSupport({
    ctx,
    state,
    ensureContractionPlan,
    getPlannerOperandState,
    getCurrentPlanSteps,
    renderPlanner,
  });

  function startAutomaticPreview(mode) {
    if (guardHyperedgePlannerAction()) {
      return;
    }
    if (guardTreePeriodicPlannerAction()) {
      return;
    }
    if (guardGridPeriodicPlannerAction()) {
      return;
    }
    if (guardBenchmarkBasePlannerAction()) {
      return;
    }
    automaticPlannerSupport.startAutomaticPreview(mode);
  }

  function acceptAutomaticPlan(mode) {
    if (guardHyperedgePlannerAction()) {
      return;
    }
    if (guardTreePeriodicPlannerAction()) {
      return;
    }
    if (guardGridPeriodicPlannerAction()) {
      return;
    }
    if (guardBenchmarkBasePlannerAction()) {
      return;
    }
    automaticPlannerSupport.acceptAutomaticPlan(mode);
  }

  plannerCommands = createPlannerCommands({
    state,
    applyManualContractionStep: (leftOperandId, rightOperandId) => {
      const leftLabel = getPlannerOperandLabel(leftOperandId);
      const rightLabel = getPlannerOperandLabel(rightOperandId);
      ctx.applyDesignChange(
        () => {
          if (typeof ctx.applyManualContractionStep === "function") {
            ctx.applyManualContractionStep(leftOperandId, rightOperandId);
            return;
          }
          const plan = ensureContractionPlan();
          plan.steps.push({
            id: ctx.makeId("step"),
            left_operand_id: leftOperandId,
            right_operand_id: rightOperandId,
            metadata: {},
          });
        },
        {
          statusMessage: `Added manual contraction step ${leftLabel} × ${rightLabel}.`,
        }
      );
    },
    getPlannerOperandLabel,
    isInspectingPastStage: () =>
      typeof ctx.isInspectingPastStage === "function" && ctx.isInspectingPastStage(),
    renderOverlayDecorations: () => ctx.renderOverlayDecorations(),
    renderPlanner,
    resolvePlannerOperandId,
    setActiveSidebarTab: (tabId) =>
      typeof ctx.setActiveSidebarTab === "function" && ctx.setActiveSidebarTab(tabId),
    setStatus: (message, level) => ctx.setStatus(message, level),
    syncPendingInteractionClasses: () =>
      typeof ctx.syncPendingInteractionClasses === "function" &&
      ctx.syncPendingInteractionClasses(),
  });

  return {
    analysisRefreshDelayMs,
    previousOperandId,
    nextOperandId,
    ensureContractionPlan,
    getPlannerOperandState,
    buildStepOrdersByTensorId,
    syncPlannerOrderBadges,
    resolvePlannerOperandId,
    repairContractionPlan,
    getPlannerRemainingOperandIds,
    isPlannerOperandAvailable,
    getPlannerOperandSourceTensorIds,
    getPlannerOperandLabel,
    handlePlannerOperandClick,
    trimContractionPlanInPlace,
    trimContractionPlan,
    togglePlannerMode,
    markContractionAnalysisDirty,
    refreshContractionAnalysis,
    isBenchmarkBasePosition,
    getAutomaticAnalysisByMode: getAutomaticAnalysisByModeFromSelectors,
    togglePlannerDisclosure,
    clearAutomaticPreview: automaticPlannerSupport.clearAutomaticPreview,
    startAutomaticPreview,
    acceptAutomaticPlan,
  };
}
