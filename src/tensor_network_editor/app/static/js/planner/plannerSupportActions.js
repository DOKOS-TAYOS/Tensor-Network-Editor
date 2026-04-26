import { createPlannerCommands } from "../actions/plannerCommands.js";
import { createPlannerAutomaticSupport } from "./plannerAutomaticSupport.js";

export function createPlannerActionSupport({
  ctx,
  state,
  renderPlanner,
  guards,
  operandSupport,
  analysisSupport,
}) {
  const {
    guardBenchmarkBasePlannerAction,
    guardGridPeriodicPlannerAction,
    guardTreePeriodicPlannerAction,
    resetPlannerBadgeDisclosureState,
  } = guards;
  const {
    ensureContractionPlan,
    getCurrentPlanSteps,
    getPlannerOperandLabel,
    getPlannerOperandState,
    resolvePlannerOperandId,
  } = operandSupport;
  let plannerCommands = null;

  const automaticPlannerSupport = createPlannerAutomaticSupport({
    ctx,
    state,
    ensureContractionPlan,
    getPlannerOperandState,
    getCurrentPlanSteps,
    renderPlanner,
  });

  function handlePlannerOperandClick(operandId) {
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
          stepCount <= 0
            ? "Reset the manual contraction path."
            : "Trimmed the manual contraction path.",
      }
    );
  }

  function togglePlannerMode() {
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
      analysisSupport.refreshContractionAnalysis();
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(
      state.plannerMode
        ? "Manual planner mode active. Click visible tensors or result tensors to define the next contraction step."
        : "Manual planner mode disabled."
    );
  }

  function startAutomaticPreview(mode) {
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
          statusMessage: `Added manual contraction step ${leftLabel} \u00d7 ${rightLabel}.`,
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
    handlePlannerOperandClick,
    trimContractionPlanInPlace,
    trimContractionPlan,
    togglePlannerMode,
    clearAutomaticPreview: automaticPlannerSupport.clearAutomaticPreview,
    startAutomaticPreview,
    acceptAutomaticPlan,
  };
}
