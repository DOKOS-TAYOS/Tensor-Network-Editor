import { getAutomaticAnalysisByMode as getAutomaticAnalysisByModeFromSelectors } from "../state/plannerSelectors.js";

export function createPlannerAutomaticSupport({
  ctx,
  state,
  ensureContractionPlan,
  getPlannerOperandState,
  getCurrentPlanSteps,
  renderPlanner,
}) {
  function buildAutomaticPastRootGroups(steps) {
    const plannerOperandState = getPlannerOperandState();
    const planSteps = getCurrentPlanSteps();
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
          right_operand_id:
            autoOperandIdMap[autoStep.right_operand_id] || autoStep.right_operand_id,
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
    if (
      !analysis ||
      analysis.status === "unavailable" ||
      !Array.isArray(analysis.steps) ||
      !analysis.steps.length
    ) {
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
    clearAutomaticPreview,
    startAutomaticPreview,
    acceptAutomaticPlan,
  };
}
