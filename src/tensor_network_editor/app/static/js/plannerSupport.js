export function getPlannerStepId(step) {
  if (!step || typeof step !== "object") {
    return null;
  }
  if (typeof step.id === "string" && step.id) {
    return step.id;
  }
  if (typeof step.step_id === "string" && step.step_id) {
    return step.step_id;
  }
  return null;
}

export function buildPlannerSeedOperands({
  tensors,
  specTensors,
  isLinearPeriodicMode,
  isLinearPeriodicBoundaryTensor,
  getLinearPeriodicReservedOperandIdForTensor,
}) {
  const baseOperands = (Array.isArray(tensors) ? tensors : []).map((tensor) => ({
    id: tensor.id,
    sourceTensorIds: [tensor.id],
    selectionIds: [tensor.id],
  }));
  if (!isLinearPeriodicMode) {
    return baseOperands;
  }
  const boundaryOperands = (Array.isArray(specTensors) ? specTensors : [])
    .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
    .map((tensor) => {
      const reservedOperandId =
        getLinearPeriodicReservedOperandIdForTensor(tensor);
      return reservedOperandId
        ? {
            id: reservedOperandId,
            sourceTensorIds: [tensor.id],
            selectionIds: [tensor.id, reservedOperandId],
          }
        : null;
    })
    .filter(Boolean);
  return [...baseOperands, ...boundaryOperands];
}

export function buildPlannerOperandState({
  tensors,
  steps,
  seedOperands,
  previousOperandId,
  nextOperandId,
}) {
  const activeOperands = new Map();
  const representativeByTensorId = {};
  const representativeByOperandId = {};
  const sourceTensorIdsByOperandId = {};
  const validSteps = [];
  const reservedOperandIds = new Set();
  const stepOrdersByTensorId = {};

  (Array.isArray(seedOperands) ? seedOperands : []).forEach((operand) => {
    const sourceTensorIds =
      Array.isArray(operand.sourceTensorIds) && operand.sourceTensorIds.length
        ? [...operand.sourceTensorIds]
        : [operand.id];
    const selectionIds = [
      ...new Set(
        Array.isArray(operand.selectionIds) && operand.selectionIds.length
          ? [operand.id, ...operand.selectionIds]
          : [operand.id]
      ),
    ];
    activeOperands.set(operand.id, { sourceTensorIds, selectionIds });
    representativeByOperandId[operand.id] = operand.id;
    selectionIds.forEach((selectionId) => {
      representativeByOperandId[selectionId] = operand.id;
    });
    sourceTensorIdsByOperandId[operand.id] = sourceTensorIds;
    sourceTensorIds.forEach((tensorId) => {
      representativeByTensorId[tensorId] = operand.id;
    });
    reservedOperandIds.add(operand.id);
  });

  for (const step of Array.isArray(steps) ? steps : []) {
    const stepId = getPlannerStepId(step);
    const usesPreviousOperand =
      step.left_operand_id === previousOperandId ||
      step.right_operand_id === previousOperandId;
    const usesNextOperand =
      step.left_operand_id === nextOperandId ||
      step.right_operand_id === nextOperandId;
    if (
      !step ||
      !stepId ||
      step.left_operand_id === step.right_operand_id ||
      !activeOperands.has(step.left_operand_id) ||
      !activeOperands.has(step.right_operand_id) ||
      reservedOperandIds.has(stepId) ||
      (usesPreviousOperand && usesNextOperand)
    ) {
      break;
    }
    const leftOperand = activeOperands.get(step.left_operand_id);
    const rightOperand = activeOperands.get(step.right_operand_id);
    if (!leftOperand || !rightOperand) {
      break;
    }
    const carrySourceOperand = usesNextOperand
      ? step.left_operand_id === nextOperandId
        ? rightOperand
        : leftOperand
      : null;
    const sourceTensorIds = carrySourceOperand
      ? [...carrySourceOperand.sourceTensorIds]
      : [...new Set([...leftOperand.sourceTensorIds, ...rightOperand.sourceTensorIds])];

    activeOperands.delete(step.left_operand_id);
    activeOperands.delete(step.right_operand_id);
    activeOperands.set(stepId, { sourceTensorIds });
    reservedOperandIds.add(stepId);
    validSteps.push(step);
    sourceTensorIdsByOperandId[stepId] = sourceTensorIds;

    sourceTensorIds.forEach((tensorId) => {
      representativeByTensorId[tensorId] = stepId;
      representativeByOperandId[tensorId] = stepId;
      if (!Array.isArray(stepOrdersByTensorId[tensorId])) {
        stepOrdersByTensorId[tensorId] = [];
      }
      stepOrdersByTensorId[tensorId].push(validSteps.length);
    });
    Object.keys(sourceTensorIdsByOperandId).forEach((operandId) => {
      const operandSourceTensorIds = sourceTensorIdsByOperandId[operandId] || [];
      if (operandSourceTensorIds.some((tensorId) => sourceTensorIds.includes(tensorId))) {
        representativeByOperandId[operandId] = stepId;
      }
    });
    if (usesNextOperand) {
      break;
    }
  }

  return {
    activeOperandIds: [...activeOperands.keys()],
    representativeByTensorId,
    representativeByOperandId,
    sourceTensorIdsByOperandId,
    validSteps,
    stepOrdersByTensorId,
    tensors,
  };
}

export function buildPreviewOrderByVisibleTensorId(visibleTensors, steps) {
  const previewOrderByTensorId = Object.fromEntries(
    visibleTensors.map((tensor) => [tensor.id, []])
  );
  const sourceTensorIdsByOperandId = {};

  visibleTensors.forEach((tensor) => {
    sourceTensorIdsByOperandId[tensor.id] =
      Array.isArray(tensor.sourceTensorIds) && tensor.sourceTensorIds.length
        ? [...tensor.sourceTensorIds]
        : [tensor.id];
  });

  (Array.isArray(steps) ? steps : []).forEach((step, index) => {
    const leftSourceTensorIds = sourceTensorIdsByOperandId[step.left_operand_id] || [
      step.left_operand_id,
    ];
    const rightSourceTensorIds = sourceTensorIdsByOperandId[step.right_operand_id] || [
      step.right_operand_id,
    ];
    const resultSourceTensorIds = [...new Set([...leftSourceTensorIds, ...rightSourceTensorIds])];
    sourceTensorIdsByOperandId[step.result_operand_id] = resultSourceTensorIds;

    visibleTensors.forEach((tensor) => {
      const visibleSourceTensorIds = sourceTensorIdsByOperandId[tensor.id] || [tensor.id];
      if (
        resultSourceTensorIds.some((tensorId) => visibleSourceTensorIds.includes(tensorId))
      ) {
        previewOrderByTensorId[tensor.id].push(index + 1);
      }
    });
  });

  return previewOrderByTensorId;
}

export function getAutomaticAnalysisByMode(payload, mode) {
  if (!payload) {
    return null;
  }
  if (mode === "automaticFuture") {
    return payload.automatic_future || null;
  }
  if (mode === "automaticPast") {
    return payload.automatic_past || null;
  }
  return null;
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
  let contractionAnalysisDebounceId = null;
  let contractionAnalysisRequestPending = false;
  let pendingContractionAnalysisOptions = null;

  function renderPlanner() {
    getRenderPlanner()();
  }

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
    return buildPlannerSeedOperands({
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
    return buildPlannerOperandState({
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
      const previewAnalysis = getAutomaticAnalysisByMode(
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
        ? buildPreviewOrderByVisibleTensorId(visibleTensors, previewAnalysis.steps)
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
    if (!state.plannerMode) {
      return;
    }
    if (
      typeof ctx.isInspectingPastStage === "function" &&
      ctx.isInspectingPastStage()
    ) {
      ctx.setStatus(
        "Past contraction steps are read-only. Return to the latest step before adding a new contraction.",
        "error"
      );
      return;
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    const resolvedOperandId = resolvePlannerOperandId(operandId);
    if (!resolvedOperandId) {
      ctx.setStatus("That operand is not available for the next manual contraction step.", "error");
      return;
    }
    if (!state.pendingPlannerOperandId) {
      state.pendingPlannerOperandId = resolvedOperandId;
      state.pendingPlannerSelectionId = operandId;
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
      renderPlanner();
      ctx.renderOverlayDecorations();
      ctx.setStatus(`Selected ${getPlannerOperandLabel(resolvedOperandId)} as the first manual operand.`);
      return;
    }
    if (state.pendingPlannerOperandId === resolvedOperandId) {
      state.pendingPlannerOperandId = null;
      state.pendingPlannerSelectionId = null;
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
      renderPlanner();
      ctx.renderOverlayDecorations();
      ctx.setStatus("Selection cleared.");
      return;
    }
    const leftOperandId = state.pendingPlannerOperandId;
    const rightOperandId = resolvedOperandId;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    const leftLabel = getPlannerOperandLabel(leftOperandId);
    const rightLabel = getPlannerOperandLabel(rightOperandId);
    ctx.applyDesignChange(
      () => {
        if (typeof ctx.applyManualContractionStep === "function") {
          ctx.applyManualContractionStep(leftOperandId, rightOperandId);
        } else {
          const plan = ensureContractionPlan();
          plan.steps.push({
            id: ctx.makeId("step"),
            left_operand_id: leftOperandId,
            right_operand_id: rightOperandId,
            metadata: {},
          });
        }
      },
      {
        statusMessage: `Added manual contraction step ${leftLabel} \u00d7 ${rightLabel}.`,
      }
    );
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

  async function runContractionAnalysisRequest(options = {}) {
    if (options.focusTab && typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    const requestId = state.contractionAnalysisRequestId + 1;
    state.contractionAnalysisRequestId = requestId;
    contractionAnalysisRequestPending = true;
    state.contractionAnalysis = { status: "loading" };
    renderPlanner();
    try {
      const payload = await ctx.apiPost("/api/analyze-contraction", {
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (state.contractionAnalysisRequestId !== requestId) {
        return;
      }
      if (!payload.ok) {
        state.contractionAnalysis = {
          status: "issues",
          issues: payload.issues || [],
        };
      } else {
        state.contractionAnalysis = {
          status: "ready",
          payload,
        };
      }
    } catch (error) {
      if (state.contractionAnalysisRequestId !== requestId) {
        return;
      }
      state.contractionAnalysis = {
        status: "error",
        message: error.message,
      };
    } finally {
      contractionAnalysisRequestPending = false;
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
  }

  function flushContractionAnalysisQueue() {
    contractionAnalysisDebounceId = null;
    if (contractionAnalysisRequestPending) {
      return;
    }
    const queuedOptions = pendingContractionAnalysisOptions || {};
    pendingContractionAnalysisOptions = null;
    runContractionAnalysisRequest(queuedOptions).finally(() => {
      if (
        pendingContractionAnalysisOptions &&
        contractionAnalysisDebounceId === null
      ) {
        contractionAnalysisDebounceId = setTimer(
          flushContractionAnalysisQueue,
          analysisRefreshDelayMs
        );
      }
    });
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
    if (contractionAnalysisDebounceId !== null) {
      clearTimer(contractionAnalysisDebounceId);
    }
    contractionAnalysisDebounceId = setTimer(
      flushContractionAnalysisQueue,
      analysisRefreshDelayMs
    );
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
        const earliestStepCount = planSteps.reduce((minimum, candidate, index) => {
          const candidateSourceTensorIds = sourceTensorIdsByOperandId[candidate.id] || [];
          const belongsToRoot =
            rootSourceTensorIds.length &&
            candidateSourceTensorIds.every((tensorId) => rootSourceTensorIds.includes(tensorId));
          if (!belongsToRoot) {
            return minimum;
          }
          return Math.min(minimum, index);
        }, Number.POSITIVE_INFINITY);
        groups[rootId] = {
          rootId,
          steps: [],
          earliestStepCount: Number.isFinite(earliestStepCount) ? earliestStepCount : 0,
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
    const analysis = getAutomaticAnalysisByMode(state.contractionAnalysis.payload, mode);
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
    const analysis = getAutomaticAnalysisByMode(state.contractionAnalysis.payload, mode);
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
