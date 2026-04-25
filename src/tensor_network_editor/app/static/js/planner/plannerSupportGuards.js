export function createPlannerGuardSupport({
  ctx,
  state,
  renderPlanner,
  statusMessages,
}) {
  const {
    benchmarkBaseStatusMessage,
    gridPeriodicStatusMessage,
    treePeriodicStatusMessage,
    hyperedgeStatusMessage,
  } = statusMessages;

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
    return false;
  }

  function guardTreePeriodicPlannerAction(message = treePeriodicStatusMessage) {
    return false;
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

  return {
    benchmarkBaseStatusMessage,
    gridPeriodicStatusMessage,
    treePeriodicStatusMessage,
    hyperedgeStatusMessage,
    isBenchmarkBasePosition,
    isGridPeriodicMode,
    isTreePeriodicMode,
    hasHyperedges,
    resetPlannerBadgeDisclosureState,
    clearPlannerTransientState,
    guardBenchmarkBasePlannerAction,
    guardGridPeriodicPlannerAction,
    guardTreePeriodicPlannerAction,
    guardHyperedgePlannerAction,
  };
}
