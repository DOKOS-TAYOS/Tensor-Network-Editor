import { createPlannerRenderers } from "./plannerRenderers.js";
import { createPlannerSupport } from "./plannerSupport.js";

export function registerPlannerFeature(ctx) {
  const plannerDocument =
    ctx.document ||
    (typeof globalThis.document !== "undefined" ? globalThis.document : null);
  const ANALYSIS_REFRESH_DELAY_MS = 200;
  const setTimer =
    typeof ctx.window?.setTimeout === "function"
      ? ctx.window.setTimeout.bind(ctx.window)
      : globalThis.setTimeout.bind(globalThis);
  const clearTimer =
    typeof ctx.window?.clearTimeout === "function"
      ? ctx.window.clearTimeout.bind(ctx.window)
      : globalThis.clearTimeout.bind(globalThis);

  let renderPlanner = () => {};
  const support = createPlannerSupport({
    ctx,
    state: ctx.state,
    analysisRefreshDelayMs: ANALYSIS_REFRESH_DELAY_MS,
    setTimer,
    clearTimer,
    getRenderPlanner: () => renderPlanner,
  });
  const renderers = createPlannerRenderers({
    ctx,
    state: ctx.state,
    plannerPanel: ctx.dom.plannerPanel,
    plannerDocument,
    support,
  });
  renderPlanner = renderers.renderPlanner;

  Object.assign(ctx, {
    repairContractionPlan: support.repairContractionPlan,
    ensureContractionPlan: support.ensureContractionPlan,
    getPlannerRemainingOperandIds: support.getPlannerRemainingOperandIds,
    isPlannerOperandAvailable: support.isPlannerOperandAvailable,
    getPlannerOperandSourceTensorIds: support.getPlannerOperandSourceTensorIds,
    getPlannerOperandLabel: support.getPlannerOperandLabel,
    resolvePlannerOperandId: support.resolvePlannerOperandId,
    handlePlannerOperandClick: support.handlePlannerOperandClick,
    trimContractionPlan: support.trimContractionPlan,
    trimContractionPlanInPlace: support.trimContractionPlanInPlace,
    togglePlannerMode: support.togglePlannerMode,
    refreshContractionAnalysis: support.refreshContractionAnalysis,
    renderPlanner: renderers.renderPlanner,
    buildPlannerOperandState: support.getPlannerOperandState,
    buildStepOrdersByTensorId: support.buildStepOrdersByTensorId,
    syncPlannerOrderBadges: support.syncPlannerOrderBadges,
    startAutomaticPreview: support.startAutomaticPreview,
    acceptAutomaticPlan: support.acceptAutomaticPlan,
    clearAutomaticPreview: support.clearAutomaticPreview,
  });
}
