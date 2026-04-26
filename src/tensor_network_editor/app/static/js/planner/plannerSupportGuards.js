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

  return {
    benchmarkBaseStatusMessage,
    gridPeriodicStatusMessage,
    treePeriodicStatusMessage,
    isBenchmarkBasePosition,
    isGridPeriodicMode,
    isTreePeriodicMode,
    hasHyperedges,
    resetPlannerBadgeDisclosureState,
    clearPlannerTransientState,
    guardBenchmarkBasePlannerAction,
    guardGridPeriodicPlannerAction,
    guardTreePeriodicPlannerAction,
  };
}
