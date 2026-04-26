import { getAutomaticAnalysisByMode as getAutomaticAnalysisByModeFromSelectors } from "../state/plannerSelectors.js";
import {
  LINEAR_PERIODIC_NEXT_OPERAND_ID,
  LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
} from "../utils/utilitiesLinearPeriodicState.js";
import { createPlannerActionSupport } from "./plannerSupportActions.js";
import { createPlannerAnalysisSupport } from "./plannerSupportAnalysis.js";
import { createPlannerGuardSupport } from "./plannerSupportGuards.js";
import { createPlannerOperandSupport } from "./plannerSupportOperands.js";

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
      : LINEAR_PERIODIC_PREVIOUS_OPERAND_ID;
  const nextOperandId =
    typeof ctx.getLinearPeriodicReservedOperandId === "function"
      ? ctx.getLinearPeriodicReservedOperandId("next")
      : LINEAR_PERIODIC_NEXT_OPERAND_ID;
  const renderPlanner = () => getRenderPlanner()();
  const guardSupport = createPlannerGuardSupport({
    ctx,
    state,
    renderPlanner,
    statusMessages: {
      benchmarkBaseStatusMessage:
        typeof ctx.benchmarkBaseStatusHint === "string" && ctx.benchmarkBaseStatusHint
          ? ctx.benchmarkBaseStatusHint
          : "Move right to edit or create a contraction scheme.",
      gridPeriodicStatusMessage: "Contractions are disabled in For bidimensional mode.",
      treePeriodicStatusMessage: "Contractions are disabled in For Tree mode.",
      hyperedgeStatusMessage:
        "Hyperedges are analyzed as generated copy tensors; the visual model is unchanged.",
    },
  });
  const operandSupport = createPlannerOperandSupport({
    ctx,
    state,
    previousOperandId,
    nextOperandId,
    guards: guardSupport,
  });
  const analysisSupport = createPlannerAnalysisSupport({
    ctx,
    state,
    analysisRefreshDelayMs,
    setTimer,
    clearTimer,
    renderPlanner,
    guards: guardSupport,
  });
  const actionSupport = createPlannerActionSupport({
    ctx,
    state,
    renderPlanner,
    guards: guardSupport,
    operandSupport,
    analysisSupport,
  });

  return {
    analysisRefreshDelayMs,
    previousOperandId,
    nextOperandId,
    ensureContractionPlan: operandSupport.ensureContractionPlan,
    getPlannerOperandState: operandSupport.getPlannerOperandState,
    buildStepOrdersByTensorId: operandSupport.buildStepOrdersByTensorId,
    syncPlannerOrderBadges: operandSupport.syncPlannerOrderBadges,
    resolvePlannerOperandId: operandSupport.resolvePlannerOperandId,
    repairContractionPlan: operandSupport.repairContractionPlan,
    getPlannerRemainingOperandIds: operandSupport.getPlannerRemainingOperandIds,
    isPlannerOperandAvailable: operandSupport.isPlannerOperandAvailable,
    getPlannerOperandSourceTensorIds: operandSupport.getPlannerOperandSourceTensorIds,
    getPlannerOperandLabel: operandSupport.getPlannerOperandLabel,
    handlePlannerOperandClick: actionSupport.handlePlannerOperandClick,
    trimContractionPlanInPlace: actionSupport.trimContractionPlanInPlace,
    trimContractionPlan: actionSupport.trimContractionPlan,
    togglePlannerMode: actionSupport.togglePlannerMode,
    markContractionAnalysisDirty: analysisSupport.markContractionAnalysisDirty,
    refreshContractionAnalysis: analysisSupport.refreshContractionAnalysis,
    isBenchmarkBasePosition: guardSupport.isBenchmarkBasePosition,
    getAutomaticAnalysisByMode: getAutomaticAnalysisByModeFromSelectors,
    togglePlannerDisclosure: analysisSupport.togglePlannerDisclosure,
    clearAutomaticPreview: actionSupport.clearAutomaticPreview,
    startAutomaticPreview: actionSupport.startAutomaticPreview,
    acceptAutomaticPlan: actionSupport.acceptAutomaticPlan,
  };
}
