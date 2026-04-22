import {
  formatBytes,
  formatNumber,
  formatShape,
  formatSignedDelta,
  getAnalysisMemoryDtype,
  getShapeElementCount,
  getPeakMemoryBytes,
  renderShapeElementDetail,
} from "./plannerAnalysisFormatting.js";
import { createPlannerPanelBindings } from "./plannerPanelBindings.js";
import { createPlannerAutomaticRendererSupport } from "./plannerRenderersAutomatic.js";
import { createPlannerRendererCommonSupport } from "./plannerRenderersCommon.js";
import { createPlannerManualRendererSupport } from "./plannerRenderersManual.js";
import { createPlannerPanelRendererSupport } from "./plannerRenderersPanel.js";

export function createPlannerRenderers({
  ctx,
  state,
  plannerPanel,
  plannerDocument,
  support,
  actions,
}) {
  const {
    syncPlannerOrderBadges,
    getPlannerOperandLabel,
    isBenchmarkBasePosition,
  } = support;
  const plannerPanelBindings = createPlannerPanelBindings({
    plannerPanel,
    plannerDocument,
    actions,
  });
  const common = createPlannerRendererCommonSupport({ ctx });
  const automaticRenderer = createPlannerAutomaticRendererSupport({
    ctx,
    state,
    common,
    getPlannerOperandLabel,
    formatters: {
      formatBytes,
      formatNumber,
      formatSignedDelta,
      getPeakMemoryBytes,
    },
  });
  const manualRenderer = createPlannerManualRendererSupport({
    ctx,
    state,
    common,
    getPlannerOperandLabel,
    formatters: {
      formatBytes,
      formatNumber,
      formatShape,
      getPeakMemoryBytes,
      renderShapeElementDetail,
    },
  });
  const panelRenderer = createPlannerPanelRendererSupport({
    ctx,
    state,
    plannerPanel,
    plannerPanelBindings,
    actions,
    syncPlannerOrderBadges,
    getPlannerOperandLabel,
    isBenchmarkBasePosition,
    automaticRenderer,
    manualRenderer,
    formatters: {
      formatShape,
      getAnalysisMemoryDtype,
      renderShapeElementDetail,
    },
  });

  return {
    renderPlanner: panelRenderer.renderPlanner,
    formatShape,
    formatNumber,
    getShapeElementCount,
  };
}
