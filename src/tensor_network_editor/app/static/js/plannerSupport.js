import { createPlannerCommands } from "./actions/plannerCommands.js";
import { createPlannerAnalysisService } from "./services/plannerAnalysisService.js";
import {
  getPlannerStepId as getPlannerStepIdFromSelectors,
  buildPlannerOperandState as buildPlannerOperandStateFromSelectors,
  buildPlannerSeedOperands as buildPlannerSeedOperandsFromSelectors,
  buildPreviewOrderByVisibleTensorId as buildPreviewOrderByVisibleTensorIdFromSelectors,
  getAutomaticAnalysisByMode as getAutomaticAnalysisByModeFromSelectors,
} from "./state/plannerSelectors.js";

export function getPlannerStepId(step) {
  return getPlannerStepIdFromSelectors(step);
}

export function buildPlannerSeedOperands({
  tensors,
  specTensors,
  isLinearPeriodicMode,
  isLinearPeriodicBoundaryTensor,
  getLinearPeriodicReservedOperandIdForTensor,
}) {
  return buildPlannerSeedOperandsFromSelectors({
    tensors,
    specTensors,
    isLinearPeriodicMode,
    isLinearPeriodicBoundaryTensor,
    getLinearPeriodicReservedOperandIdForTensor,
  });
}

export function buildPlannerOperandState({
  tensors,
  steps,
  seedOperands,
  previousOperandId,
  nextOperandId,
}) {
  return buildPlannerOperandStateFromSelectors({
    tensors,
    steps,
    seedOperands,
    previousOperandId,
    nextOperandId,
  });
}

export function buildPreviewOrderByVisibleTensorId(visibleTensors, steps) {
  return buildPreviewOrderByVisibleTensorIdFromSelectors(visibleTensors, steps);
}

export function getAutomaticAnalysisByMode(payload, mode) {
  return getAutomaticAnalysisByModeFromSelectors(payload, mode);
}

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
  let pendingContractionAnalysisOptions = null;
  let plannerCommands = null;

  function renderPlanner() {
    getRenderPlanner()();
  }

  const analysisService = createPlannerAnalysisService({
    analysisRefreshDelayMs,
    analyze: (payload) => ctx.apiPost("/api/analyze-contraction", payload),
    cancel: (timerId) => clearTimer(timerId),
    onAnalysisError: (error) => {
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
    const plan = state.spec.contraction_plan;
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      if (plan) {
        plan.view_snapshots = [];
      }
      state.plannerInspectionStepCount = null;
      state.plannerFutureBadgeDisclosure = {};
      return;
    }
    const plannerOperandState = getPlannerOperandState();
    if (!plannerOperandState.validSteps.length) {
      state.spec.contraction_plan = null;
      state.plannerInspectionStepCount = null;
      state.plannerFutureBadgeDisclosure = {};
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
    state.plannerFutureBadgeDisclosure = {};
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
    state.plannerFutureBadgeDisclosure = {};
    state.plannerInspectionStepCount =
      stepCount <= 0
        ? null
        : Number.isInteger(state.plannerInspectionStepCount)
        ? Math.min(state.plannerInspectionStepCount, stepCount - 1)
        : null;
    return true;
  }

  function trimContractionPlan(stepCount) {
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

  function refreshContractionAnalysis(options = {}) {
    pendingContractionAnalysisOptions = {
      focusTab:
        Boolean(options.focusTab) ||
        Boolean(
          pendingContractionAnalysisOptions &&
            pendingContractionAnalysisOptions.focusTab
        ),
    };
    analysisService.requestRefresh(pendingContractionAnalysisOptions);
  }

  function togglePlannerDisclosure(disclosureKey) {
    state.plannerDisclosureState[disclosureKey] = !state.plannerDisclosureState[disclosureKey];
    renderPlanner();
  }

  function buildAutomaticPastRootGroups(steps) {
    const plannerOperandState = getPlannerOperandState();
    const planSteps =
      state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
        ? state.spec.contraction_plan.steps
        : [];
    const sourceTensorIdsByOperandId = plannerOperandState.sourceTensorIdsByOperandId || {};
    const stepOrdersByTensorId = plannerOperandState.stepOrdersByTensorId || {};
    const stepOrderById = Object.fromEntries(
      planSteps.map((step, index) => [step.id, index + 1])
    );
    const groups = {};

    (Array.isArray(steps) ? steps : []).forEach((step) => {
      const rootId =
        typeof step.result_operand_id === "string" &&
        Object.prototype.hasOwnProperty.call(stepOrderById, step.result_operand_id)
          ? step.result_operand_id
          : step.result_operand_id.split("__auto_past_")[0];
      if (!groups[rootId]) {
        const rootSourceTensorIds = sourceTensorIdsByOperandId[rootId] || [];
        const earliestStepOrder = rootSourceTensorIds.reduce((minimum, tensorId) => {
          const tensorStepOrders = stepOrdersByTensorId[tensorId];
          if (!Array.isArray(tensorStepOrders) || !tensorStepOrders.length) {
            return minimum;
          }
          return Math.min(minimum, tensorStepOrders[0]);
        }, Number.POSITIVE_INFINITY);
        groups[rootId] = {
          rootId,
          steps: [],
          earliestStepCount: Number.isFinite(earliestStepOrder)
            ? earliestStepOrder - 1
            : 0,
          originalStepOrder: stepOrderById[rootId] || 0,
        };
      }
      groups[rootId].steps.push(step);
    });

    return Object.values(groups).sort(
      (left, right) => left.originalStepOrder - right.originalStepOrder
    );
  }

  function clearAutomaticPreview(options = {}) {
    const previousPreviewMode = state.plannerPreviewMode;
    state.plannerPreviewMode = null;
    state.plannerPreviewOrderByTensorId = {};
    if (
      previousPreviewMode === "automaticPast" &&
      !options.preservePastInspection &&
      typeof ctx.clearPastInspection === "function"
    ) {
      ctx.clearPastInspection();
    }
  }

  function startAutomaticPreview(mode) {
    if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {
      return;
    }
    if (state.plannerPreviewMode === mode) {
      clearAutomaticPreview();
      renderPlanner();
      ctx.render();
      ctx.setStatus("Automatic preview cleared.");
      return;
    }
    const analysis = getAutomaticAnalysisByModeFromSelectors(
      state.contractionAnalysis.payload,
      mode
    );
    if (!analysis || analysis.status === "unavailable" || !Array.isArray(analysis.steps)) {
      ctx.setStatus("That automatic preview is not available yet.", "error");
      return;
    }
    clearAutomaticPreview();
    state.plannerPreviewMode = mode;
    if (mode === "automaticPast") {
      const rootGroups = buildAutomaticPastRootGroups(analysis.steps);
      if (rootGroups.length && typeof ctx.beginPastInspection === "function") {
        ctx.beginPastInspection(rootGroups[0].earliestStepCount);
      }
      renderPlanner();
      ctx.render();
      ctx.setStatus("Showing the auto past preview from the first affected contraction step.");
      return;
    }
    if (typeof ctx.clearPastInspection === "function") {
      ctx.clearPastInspection();
    }
    renderPlanner();
    ctx.render();
    ctx.setStatus("Showing the auto future preview.");
  }

  function appendAutomaticFutureSteps(steps) {
    const plan = ensureContractionPlan();
    const stepIdMap = {};
    steps.forEach((step) => {
      const nextStepId = ctx.makeId("step");
      stepIdMap[step.result_operand_id] = nextStepId;
      plan.steps.push({
        id: nextStepId,
        left_operand_id: stepIdMap[step.left_operand_id] || step.left_operand_id,
        right_operand_id: stepIdMap[step.right_operand_id] || step.right_operand_id,
        metadata: {},
      });
    });
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
  }

  function rewriteAutomaticPastSteps(steps) {
    const plan = ensureContractionPlan();
    const previousVisibleLayoutMap =
      typeof ctx.captureVisibleOperandLayoutMap === "function"
        ? ctx.captureVisibleOperandLayoutMap(
            typeof ctx.getLatestAppliedStepCount === "function"
              ? ctx.getLatestAppliedStepCount()
              : null
          )
        : {};
    const rootGroups = buildAutomaticPastRootGroups(steps);
    const plannerOperandState = getPlannerOperandState();
    const sourceTensorIdsByOperandId = plannerOperandState.sourceTensorIdsByOperandId || {};
    const sourceTensorIdsByRootId = Object.fromEntries(
      rootGroups.map((group) => [group.rootId, sourceTensorIdsByOperandId[group.rootId] || []])
    );
    const rewrittenSteps = [];

    plan.steps.forEach((step) => {
      const rootMatch = rootGroups.find((group) => {
        const rootSourceTensorIds = sourceTensorIdsByRootId[group.rootId] || [];
        const stepSourceTensorIds = sourceTensorIdsByOperandId[step.id] || [];
        return (
          rootSourceTensorIds.length &&
          stepSourceTensorIds.length &&
          stepSourceTensorIds.every((tensorId) => rootSourceTensorIds.includes(tensorId))
        );
      });
      if (!rootMatch) {
        rewrittenSteps.push(step);
        return;
      }
      if (step.id !== rootMatch.rootId) {
        return;
      }

      const existingRootStep = plan.steps.find((candidate) => candidate.id === rootMatch.rootId);
      const autoOperandIdMap = {};
      rootMatch.steps.forEach((autoStep) => {
        const isRootResult = autoStep.result_operand_id === rootMatch.rootId;
        const nextStepId = isRootResult ? rootMatch.rootId : ctx.makeId("step");
        autoOperandIdMap[autoStep.result_operand_id] = nextStepId;
        rewrittenSteps.push({
          id: nextStepId,
          left_operand_id: autoOperandIdMap[autoStep.left_operand_id] || autoStep.left_operand_id,
          right_operand_id: autoOperandIdMap[autoStep.right_operand_id] || autoStep.right_operand_id,
          metadata:
            isRootResult && existingRootStep && existingRootStep.metadata
              ? ctx.deepClone(existingRootStep.metadata)
              : {},
        });
      });
    });

    plan.steps = rewrittenSteps;
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
    if (
      typeof ctx.getLatestAppliedStepCount === "function" &&
      typeof ctx.applySnapshotLayoutMap === "function"
    ) {
      ctx.applySnapshotLayoutMap(ctx.getLatestAppliedStepCount(), previousVisibleLayoutMap);
    }
    if (
      state.plannerPreviewMode === "automaticPast" &&
      rootGroups.length &&
      typeof ctx.beginPastInspection === "function"
    ) {
      ctx.beginPastInspection(rootGroups[0].earliestStepCount);
    }
  }

  function acceptAutomaticPlan(mode) {
    if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {
      return;
    }
    const analysis = getAutomaticAnalysisByModeFromSelectors(
      state.contractionAnalysis.payload,
      mode
    );
    if (!analysis || analysis.status === "unavailable" || !Array.isArray(analysis.steps) || !analysis.steps.length) {
      ctx.setStatus("That automatic path is not available to accept.", "error");
      return;
    }
    ctx.applyDesignChange(
      () => {
        if (mode === "automaticFuture") {
          appendAutomaticFutureSteps(analysis.steps);
          if (typeof ctx.clearPastInspection === "function") {
            ctx.clearPastInspection();
          }
        } else {
          rewriteAutomaticPastSteps(analysis.steps);
        }
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
        state.plannerFutureBadgeDisclosure = {};
        clearAutomaticPreview();
      },
      {
        statusMessage:
          mode === "automaticFuture"
            ? "Assigned the remaining contraction steps from the auto future path."
            : "Rewired the contracted history with the auto past path.",
      }
    );
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
    refreshContractionAnalysis,
    getAutomaticAnalysisByMode,
    togglePlannerDisclosure,
    clearAutomaticPreview,
    startAutomaticPreview,
    acceptAutomaticPlan,
  };
}
