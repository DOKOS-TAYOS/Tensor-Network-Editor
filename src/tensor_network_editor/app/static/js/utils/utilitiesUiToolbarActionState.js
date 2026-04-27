export function createUiToolbarActionStateSupport({
  state,
  dom,
  runtime,
  setElementHidden,
  setTooltipDescription,
  setButtonGroupDisabled,
  syncSubnetworkLibraryModalState,
}) {
  const {
    undoButton,
    redoButton,
    exportMenuItem,
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    exportTikzMenuItem,
    exportDotMenuItem,
    exportShowTensorNamesMenuItem,
    exportShowIndexNamesMenuItem,
    exportShowBondNamesMenuItem,
    templateSelect,
    templateSettingsButton,
    insertTemplateButton,
    saveSessionTemplateMenuItem,
    saveSubnetworkLibraryMenuItem,
    loadSessionTemplateMenuItem,
    exportSessionTemplateMenuItem,
    editSessionTemplateMenuItem,
    openSubnetworkLibraryMenuItem,
    reflowImportedButton,
    reflowAlignLeftButton,
    reflowAlignRightButton,
    reflowAlignTopButton,
    reflowAlignMiddleButton,
    reflowAlignBottomButton,
    reflowIndicesLeftButton,
    reflowIndicesRightButton,
    reflowIndicesTopButton,
    reflowIndicesResetButton,
    reflowIndicesBottomButton,
    reflowArrangeChainButton,
    reflowArrangeTreeButton,
    reflowArrangeGridButton,
    reflowAutoLayoutButton,
    reflowDistributeHorizontalButton,
    reflowDistributeVerticalButton,
    reflowSnapGridButton,
    createGroupButton,
    generateButton,
    benchmarkCompareModal,
    benchmarkCompareTableBody,
  } = dom;

  function syncAcademicExportLabelMenuItem(menuItem, isChecked) {
    if (!menuItem) {
      return;
    }
    menuItem.classList.toggle("is-checked", Boolean(isChecked));
    menuItem.setAttribute("aria-checked", String(Boolean(isChecked)));
  }

  function syncToolbarActionState(derivedState) {
    const {
      forMode,
      benchmarkSchemeView,
      primaryToolbarGroup,
      primaryToolbarDivider,
      templateToolbarGroup,
      selectedTemplateValue,
      selectedTensorIds,
      graphTensorCount,
      autoLayoutTensorCount,
      hasSelectedIndices,
    } = derivedState;

    if (undoButton) {
      undoButton.disabled = state.undoStack.length === 0;
    }
    if (redoButton) {
      redoButton.disabled = state.redoStack.length === 0;
    }
    if (generateButton) {
      generateButton.disabled = !state.spec || !state.selectedEngine;
    }
    if (exportMenuItem) {
      exportMenuItem.disabled = false;
    }
    if (exportPythonMenuItem) {
      exportPythonMenuItem.disabled = !state.spec || !state.selectedEngine;
    }
    if (exportPngMenuItem) {
      exportPngMenuItem.disabled = !state.spec;
    }
    if (exportSvgMenuItem) {
      exportSvgMenuItem.disabled = !state.spec;
    }
    if (exportTikzMenuItem) {
      exportTikzMenuItem.disabled = !state.spec;
    }
    if (exportDotMenuItem) {
      exportDotMenuItem.disabled = !state.spec;
    }
    syncAcademicExportLabelMenuItem(
      exportShowTensorNamesMenuItem,
      state.academicExportLabels.tensor
    );
    syncAcademicExportLabelMenuItem(
      exportShowIndexNamesMenuItem,
      state.academicExportLabels.index
    );
    syncAcademicExportLabelMenuItem(
      exportShowBondNamesMenuItem,
      state.academicExportLabels.bond
    );
    if (saveSessionTemplateMenuItem) {
      saveSessionTemplateMenuItem.disabled = forMode || selectedTensorIds.length === 0;
    }
    if (saveSubnetworkLibraryMenuItem) {
      saveSubnetworkLibraryMenuItem.disabled =
        forMode || benchmarkSchemeView || selectedTensorIds.length === 0;
    }
    if (loadSessionTemplateMenuItem) {
      loadSessionTemplateMenuItem.disabled = false;
    }
    if (exportSessionTemplateMenuItem) {
      exportSessionTemplateMenuItem.disabled =
        forMode || selectedTensorIds.length === 0;
    }
    if (editSessionTemplateMenuItem) {
      editSessionTemplateMenuItem.disabled = state.availableTemplates.length === 0;
    }
    if (openSubnetworkLibraryMenuItem) {
      openSubnetworkLibraryMenuItem.disabled = forMode || benchmarkSchemeView;
    }
    setElementHidden(primaryToolbarGroup, benchmarkSchemeView);
    setElementHidden(primaryToolbarDivider, benchmarkSchemeView);
    setElementHidden(templateToolbarGroup, benchmarkSchemeView);
    if (insertTemplateButton) {
      insertTemplateButton.disabled = benchmarkSchemeView || !selectedTemplateValue;
      insertTemplateButton.hidden = benchmarkSchemeView;
    }
    if (reflowImportedButton) {
      reflowImportedButton.disabled =
        benchmarkSchemeView ||
        (selectedTensorIds.length === 0 && graphTensorCount < 2);
      setElementHidden(
        reflowImportedButton.parentElement || reflowImportedButton,
        benchmarkSchemeView
      );
      setTooltipDescription(
        reflowImportedButton,
        benchmarkSchemeView
          ? "Layout tools are unavailable while viewing a benchmark scheme."
          : selectedTensorIds.length === 0 && graphTensorCount < 2
            ? "Add or select at least two tensors first."
            : selectedTensorIds.length === 0
              ? "Open layout tools. Auto layout will arrange the whole graph."
              : selectedTensorIds.length === 1
                ? "Reflow indices for the selected tensor."
                : "Choose a layout for the selected tensors or reflow their indices."
      );
    }
    setButtonGroupDisabled(
      [
        reflowAlignLeftButton,
        reflowAlignRightButton,
        reflowAlignTopButton,
        reflowAlignMiddleButton,
        reflowAlignBottomButton,
        reflowArrangeChainButton,
        reflowArrangeTreeButton,
        reflowArrangeGridButton,
        reflowDistributeHorizontalButton,
        reflowDistributeVerticalButton,
        reflowSnapGridButton,
      ],
      selectedTensorIds.length < 2,
      selectedTensorIds.length < 2
        ? "Select at least two tensors first."
        : "Reflow the selected tensors."
    );
    setButtonGroupDisabled(
      [reflowAutoLayoutButton],
      autoLayoutTensorCount < 2,
      autoLayoutTensorCount < 2
        ? "Add or select at least two tensors first."
        : selectedTensorIds.length
          ? "Auto-arrange the selected tensors."
          : "Auto-arrange the whole graph."
    );
    setButtonGroupDisabled(
      [
        reflowIndicesLeftButton,
        reflowIndicesRightButton,
        reflowIndicesTopButton,
        reflowIndicesResetButton,
        reflowIndicesBottomButton,
      ],
      selectedTensorIds.length === 0 || !hasSelectedIndices,
      selectedTensorIds.length === 0
        ? "Select at least one tensor first."
        : !hasSelectedIndices
          ? "The selected tensors have no indices to reflow."
          : selectedTensorIds.length === 1
            ? "Reflow indices for the selected tensor."
            : "Reflow indices for the selected tensors."
    );
    if (templateSettingsButton) {
      templateSettingsButton.disabled =
        benchmarkSchemeView || !selectedTemplateValue || forMode;
      setElementHidden(
        templateSettingsButton.parentElement || templateSettingsButton,
        benchmarkSchemeView
      );
      setTooltipDescription(
        templateSettingsButton,
        benchmarkSchemeView
          ? "Template parameters are unavailable while viewing a benchmark scheme."
          : !selectedTemplateValue
            ? "Choose a template first."
            : forMode
              ? "Template parameters are not editable in For mode."
              : "Edit template parameters."
      );
    }
    if (
      (benchmarkSchemeView || !selectedTemplateValue || forMode) &&
      state.isTemplateSettingsOpen
    ) {
      state.isTemplateSettingsOpen = false;
    }
    if (
      (benchmarkSchemeView ||
        (selectedTensorIds.length === 0 && graphTensorCount < 2)) &&
      state.isReflowLayoutOpen
    ) {
      state.isReflowLayoutOpen = false;
    }
    if ((benchmarkSchemeView || forMode) && state.isSubnetworkLibraryOpen) {
      state.isSubnetworkLibraryOpen = false;
      syncSubnetworkLibraryModalState();
    }
    if (templateSelect) {
      templateSelect.disabled = benchmarkSchemeView;
      setElementHidden(templateSelect.parentElement || templateSelect, benchmarkSchemeView);
    }
    if (createGroupButton) {
      createGroupButton.disabled = selectedTensorIds.length < 2;
    }
    if (benchmarkCompareModal && benchmarkCompareTableBody) {
      if (typeof runtime.syncBenchmarkCompareModalState === "function") {
        runtime.syncBenchmarkCompareModalState();
      }
    }
  }

  return {
    syncToolbarActionState,
  };
}
