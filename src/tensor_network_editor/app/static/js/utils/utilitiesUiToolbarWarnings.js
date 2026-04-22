export function createUiToolbarWarningSupport({
  state,
  dom,
  setTooltipDescription,
  getTensorKrowchManualPlanIssueMessage,
}) {
  const {
    codeGenerationWarning,
    templateCatalogWarning,
    subnetworkCatalogWarning,
  } = dom;

  function syncCodeGenerationWarning() {
    if (!codeGenerationWarning) {
      return;
    }
    const warningMessage = getTensorKrowchManualPlanIssueMessage();
    if (!codeGenerationWarning.dataset) {
      codeGenerationWarning.dataset = {};
    }
    codeGenerationWarning.dataset.tooltipEnabled = "true";
    codeGenerationWarning.dataset.shortcutLabel = "Code generation warning";
    codeGenerationWarning.textContent = warningMessage;
    setTooltipDescription(codeGenerationWarning, warningMessage);
    codeGenerationWarning.hidden = !warningMessage;
  }

  function syncTemplateCatalogWarning() {
    if (!templateCatalogWarning) {
      return;
    }
    const warningMessages = Array.isArray(state.templateCatalogWarnings)
      ? state.templateCatalogWarnings.filter(
          (warningMessage) => typeof warningMessage === "string" && warningMessage
        )
      : [];
    if (!templateCatalogWarning.dataset) {
      templateCatalogWarning.dataset = {};
    }
    templateCatalogWarning.dataset.tooltipEnabled = "true";
    templateCatalogWarning.dataset.shortcutLabel = "Template warnings";
    if (!warningMessages.length) {
      templateCatalogWarning.textContent = "";
      setTooltipDescription(templateCatalogWarning, "");
      templateCatalogWarning.hidden = true;
      return;
    }
    const extraWarningCount = warningMessages.length - 1;
    templateCatalogWarning.textContent =
      extraWarningCount > 0
        ? `${warningMessages[0]} (+${extraWarningCount} more)`
        : warningMessages[0];
    setTooltipDescription(templateCatalogWarning, warningMessages.join("\n"));
    templateCatalogWarning.hidden = false;
  }

  function syncSubnetworkCatalogWarning() {
    if (!subnetworkCatalogWarning) {
      return;
    }
    const warningMessages = Array.isArray(state.subnetworkCatalogWarnings)
      ? state.subnetworkCatalogWarnings.filter(
          (warningMessage) => typeof warningMessage === "string" && warningMessage
        )
      : [];
    if (!subnetworkCatalogWarning.dataset) {
      subnetworkCatalogWarning.dataset = {};
    }
    subnetworkCatalogWarning.dataset.tooltipEnabled = "true";
    subnetworkCatalogWarning.dataset.shortcutLabel = "Subnetwork warnings";
    if (!warningMessages.length) {
      subnetworkCatalogWarning.textContent = "";
      setTooltipDescription(subnetworkCatalogWarning, "");
      subnetworkCatalogWarning.hidden = true;
      return;
    }
    const extraWarningCount = warningMessages.length - 1;
    subnetworkCatalogWarning.textContent =
      extraWarningCount > 0
        ? `${warningMessages[0]} (+${extraWarningCount} more)`
        : warningMessages[0];
    setTooltipDescription(subnetworkCatalogWarning, warningMessages.join("\n"));
    subnetworkCatalogWarning.hidden = false;
  }

  return {
    syncCodeGenerationWarning,
    syncTemplateCatalogWarning,
    syncSubnetworkCatalogWarning,
  };
}
