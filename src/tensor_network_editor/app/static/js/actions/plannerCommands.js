export function createPlannerCommands({
  state,
  applyManualContractionStep = () => {},
  getPlannerOperandLabel,
  isInspectingPastStage = () => false,
  renderOverlayDecorations = () => {},
  renderPlanner = () => {},
  resolvePlannerOperandId,
  setActiveSidebarTab = () => {},
  setStatus,
  syncPendingInteractionClasses = () => {},
}) {
  function handlePlannerOperandClick(operandId) {
    if (!state.plannerMode) {
      return;
    }
    if (isInspectingPastStage()) {
      setStatus(
        "Past contraction steps are read-only. Return to the latest step before adding a new contraction.",
        "error"
      );
      return;
    }
    setActiveSidebarTab("planner");
    const resolvedOperandId = resolvePlannerOperandId(operandId);
    if (!resolvedOperandId) {
      setStatus(
        "That operand is not available for the next manual contraction step.",
        "error"
      );
      return;
    }
    if (!state.pendingPlannerOperandId) {
      state.pendingPlannerOperandId = resolvedOperandId;
      state.pendingPlannerSelectionId = operandId;
      syncPendingInteractionClasses();
      renderPlanner();
      renderOverlayDecorations();
      setStatus(
        `Selected ${getPlannerOperandLabel(
          resolvedOperandId
        )} as the first manual operand.`
      );
      return;
    }

    if (resolvedOperandId === state.pendingPlannerOperandId) {
      state.pendingPlannerOperandId = null;
      state.pendingPlannerSelectionId = null;
      syncPendingInteractionClasses();
      renderPlanner();
      renderOverlayDecorations();
      setStatus("Selection cleared.");
      return;
    }

    const firstOperandId = state.pendingPlannerOperandId;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    syncPendingInteractionClasses();
    applyManualContractionStep(firstOperandId, resolvedOperandId);
    renderPlanner();
    renderOverlayDecorations();
  }

  return {
    handlePlannerOperandClick,
  };
}
