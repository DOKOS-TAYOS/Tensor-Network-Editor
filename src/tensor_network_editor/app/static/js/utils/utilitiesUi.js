import { createUtilityUiDomSupport } from "./utilitiesUiDom.js";
import { createUtilityUiGeneratedCodeSupport } from "./utilitiesUiGeneratedCode.js";
import { createUtilityUiPanelsSupport } from "./utilitiesUiPanels.js";
import { createUtilityUiStatusSupport } from "./utilitiesUiStatus.js";
import { createUtilityUiToolbarSupport } from "./utilitiesUiToolbar.js";

export function createUtilityUiBindings({ ctx, state, dom, runtime }) {
  const domSupport = createUtilityUiDomSupport({
    windowRef: ctx.window,
  });
  const panelsSupport = createUtilityUiPanelsSupport({
    state,
    dom,
    positionFloatingPanel: domSupport.positionFloatingPanel,
    setExpandedState: domSupport.setExpandedState,
    toggleElementClass: domSupport.toggleElementClass,
  });
  const generatedCodeSupport = createUtilityUiGeneratedCodeSupport({
    state,
    dom,
    setTooltipDescription: domSupport.setTooltipDescription,
    syncToolbarTransientUi: panelsSupport.syncToolbarTransientUi,
    highlightCodeElement: (element) =>
      typeof runtime.highlightCodeElement === "function"
        ? runtime.highlightCodeElement(element)
        : undefined,
  });
  const toolbarSupport = createUtilityUiToolbarSupport({
    state,
    dom,
    runtime,
    setElementHidden: domSupport.setElementHidden,
    setTooltipDescription: domSupport.setTooltipDescription,
    setButtonGroupDisabled: domSupport.setButtonGroupDisabled,
    setMenuItemChecked: domSupport.setMenuItemChecked,
    getSelectedTensorIds: () =>
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [],
    findTensorById: (tensorId) =>
      typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null,
    getTensorKrowchManualPlanIssueMessage: () =>
      typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "",
    syncToolbarTransientUi: panelsSupport.syncToolbarTransientUi,
    syncHelpModalState: panelsSupport.syncHelpModalState,
    syncTemplateManagerModalState: panelsSupport.syncTemplateManagerModalState,
    syncSubnetworkLibraryModalState: panelsSupport.syncSubnetworkLibraryModalState,
    syncGeneratedCodeActionState: generatedCodeSupport.syncGeneratedCodeActionState,
    syncGeneratedCodeModalState: generatedCodeSupport.syncGeneratedCodeModalState,
  });
  const statusSupport = createUtilityUiStatusSupport({
    statusMessage: dom.statusMessage,
  });

  return {
    syncToolbarTransientUi: panelsSupport.syncToolbarTransientUi,
    closeTransientToolbarUi: panelsSupport.closeTransientToolbarUi,
    openToolbarMenu: panelsSupport.openToolbarMenu,
    toggleToolbarMenu: panelsSupport.toggleToolbarMenu,
    toggleTemplateSettingsPopover: panelsSupport.toggleTemplateSettingsPopover,
    toggleReflowLayoutPopover: panelsSupport.toggleReflowLayoutPopover,
    syncGeneratedCodeModalState: generatedCodeSupport.syncGeneratedCodeModalState,
    toggleGeneratedCodeModal: generatedCodeSupport.toggleGeneratedCodeModal,
    renderGeneratedCodePreview: generatedCodeSupport.renderGeneratedCodePreview,
    updateToolbarState: toolbarSupport.updateToolbarState,
    syncCodeGenerationWarning: toolbarSupport.syncCodeGenerationWarning,
    syncTemplateCatalogWarning: toolbarSupport.syncTemplateCatalogWarning,
    syncSubnetworkCatalogWarning: toolbarSupport.syncSubnetworkCatalogWarning,
    syncHelpModalState: panelsSupport.syncHelpModalState,
    toggleHelpModal: panelsSupport.toggleHelpModal,
    openHelpSection: panelsSupport.openHelpSection,
    syncTemplateManagerModalState: panelsSupport.syncTemplateManagerModalState,
    toggleTemplateManager: panelsSupport.toggleTemplateManager,
    setTemplateManagerValidationMessage:
      panelsSupport.setTemplateManagerValidationMessage,
    syncSubnetworkLibraryModalState: panelsSupport.syncSubnetworkLibraryModalState,
    toggleSubnetworkLibrary: panelsSupport.toggleSubnetworkLibrary,
    formatIssues: statusSupport.formatIssues,
    setStatus: statusSupport.setStatus,
  };
}
