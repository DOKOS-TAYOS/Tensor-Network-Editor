export function createPlannerPanelBindings({
  plannerPanel,
  plannerDocument,
  actions,
}) {
  function bindPlannerPanelInteractions() {
    plannerDocument
      ?.getElementById("toggle-planner-mode-button")
      ?.addEventListener("click", actions.togglePlannerMode);
    plannerDocument
      ?.getElementById("planner-reset-button")
      ?.addEventListener("click", () => actions.trimContractionPlan(0));
    plannerPanel.querySelectorAll("[data-trim-step]").forEach((button) => {
      button.addEventListener("click", () => {
        actions.trimContractionPlan(Number(button.dataset.trimStep));
      });
    });
    plannerPanel.querySelectorAll("[data-inspect-step]").forEach((button) => {
      button.addEventListener("click", () => {
        actions.togglePastInspection(Number(button.dataset.inspectStep));
        actions.clearAutomaticPreview({ preservePastInspection: true });
        actions.renderPlanner();
        actions.renderEditor();
        actions.renderOverlayDecorations();
      });
    });
    plannerPanel.querySelectorAll("[data-disclosure]").forEach((button) => {
      button.addEventListener("click", () => {
        actions.togglePlannerDisclosure(button.dataset.disclosure);
      });
    });
    plannerPanel.querySelectorAll("[data-preview-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        actions.startAutomaticPreview(button.dataset.previewMode);
      });
    });
    plannerPanel.querySelectorAll("[data-accept-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        actions.acceptAutomaticPlan(button.dataset.acceptMode);
      });
    });
  }

  return {
    bindPlannerPanelInteractions,
  };
}
