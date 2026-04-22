import { createUiToolbarActionStateSupport } from "./utilitiesUiToolbarActionState.js";
import { createUiToolbarDerivedStateSupport } from "./utilitiesUiToolbarDerivedState.js";
import { createUiToolbarModeControlSupport } from "./utilitiesUiToolbarModeControls.js";
import { createUiToolbarWarningSupport } from "./utilitiesUiToolbarWarnings.js";

export function createUtilityUiToolbarSupport({
  state,
  dom,
  runtime,
  setElementHidden,
  setTooltipDescription,
  setButtonGroupDisabled,
  setMenuItemChecked,
  getSelectedTensorIds,
  findTensorById,
  getTensorKrowchManualPlanIssueMessage,
  syncToolbarTransientUi,
  syncHelpModalState,
  syncTemplateManagerModalState,
  syncSubnetworkLibraryModalState,
  syncGeneratedCodeActionState,
  syncGeneratedCodeModalState,
}) {
  const warningSupport = createUiToolbarWarningSupport({
    state,
    dom,
    setTooltipDescription,
    getTensorKrowchManualPlanIssueMessage,
  });
  const derivedStateSupport = createUiToolbarDerivedStateSupport({
    state,
    dom,
    runtime,
    getSelectedTensorIds,
    findTensorById,
  });
  const modeControlSupport = createUiToolbarModeControlSupport({
    dom,
    runtime,
    setTooltipDescription,
    setMenuItemChecked,
  });
  const actionStateSupport = createUiToolbarActionStateSupport({
    state,
    dom,
    runtime,
    setElementHidden,
    setTooltipDescription,
    setButtonGroupDisabled,
    syncSubnetworkLibraryModalState,
  });

  function updateToolbarState() {
    const derivedState = derivedStateSupport.getToolbarDerivedState();

    runtime.enforceLinearPeriodicEngineSupport();
    warningSupport.syncCodeGenerationWarning();
    warningSupport.syncTemplateCatalogWarning();
    warningSupport.syncSubnetworkCatalogWarning();
    syncHelpModalState();
    syncTemplateManagerModalState();
    syncSubnetworkLibraryModalState();
    syncGeneratedCodeActionState();
    syncGeneratedCodeModalState();
    actionStateSupport.syncToolbarActionState(derivedState);
    modeControlSupport.syncToolbarModeControls(derivedState);
    syncToolbarTransientUi();
  }

  return {
    syncCodeGenerationWarning: warningSupport.syncCodeGenerationWarning,
    syncTemplateCatalogWarning: warningSupport.syncTemplateCatalogWarning,
    syncSubnetworkCatalogWarning: warningSupport.syncSubnetworkCatalogWarning,
    updateToolbarState,
  };
}
