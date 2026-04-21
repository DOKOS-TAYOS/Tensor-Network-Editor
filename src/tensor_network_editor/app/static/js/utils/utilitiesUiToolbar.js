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
  syncGeneratedCodeActionState,
  syncGeneratedCodeModalState,
}) {
  const {
    codeGenerationWarning,
    undoButton,
    redoButton,
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    singleModeMenuItem,
    linearPeriodicModeMenuItem,
    gridPeriodicModeMenuItem,
    treeModeMenuItem,
    benchmarkModeMenuItem,
    toolbarModeControls,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    gridPeriodicUpCellButton,
    gridPeriodicDownCellButton,
    linearPeriodicNextCellButton,
    benchmarkSchemeNameInput,
    benchmarkCompareButton,
    templateSelect,
    templateSettingsButton,
    insertTemplateButton,
    saveSessionTemplateMenuItem,
    loadSessionTemplateMenuItem,
    exportSessionTemplateMenuItem,
    editSessionTemplateMenuItem,
    templateCatalogWarning,
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
    reflowDistributeHorizontalButton,
    reflowDistributeVerticalButton,
    reflowSnapGridButton,
    createGroupButton,
    connectButton,
    addNoteButton,
    generateButton,
    benchmarkCompareModal,
    benchmarkCompareTableBody,
  } = dom;

  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };
  const GRID_PERIODIC_CELL_LABELS = {
    top_left: "Top-left cell",
    top: "Top cell",
    top_right: "Top-right cell",
    left: "Left cell",
    center: "Center cell",
    right: "Right cell",
    bottom_left: "Bottom-left cell",
    bottom: "Bottom cell",
    bottom_right: "Bottom-right cell",
  };
  const TREE_PERIODIC_CELL_LABELS = {
    root: "Root cell",
    branch: "Branch cell",
    leaf: "Leaf cell",
  };

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

  function updateToolbarState() {
    const linearPeriodicMode = runtime.isLinearPeriodicMode();
    const activeLinearPeriodicCell = runtime.getActiveLinearPeriodicCellName();
    const gridPeriodicMode =
      typeof runtime.isGridPeriodicMode === "function" && runtime.isGridPeriodicMode();
    const activeGridPeriodicCell =
      gridPeriodicMode && typeof runtime.getActiveGridPeriodicCellName === "function"
        ? runtime.getActiveGridPeriodicCellName()
        : null;
    const treePeriodicMode =
      typeof runtime.isTreePeriodicMode === "function" && runtime.isTreePeriodicMode();
    const activeTreePeriodicCell =
      treePeriodicMode && typeof runtime.getActiveTreePeriodicCellName === "function"
        ? runtime.getActiveTreePeriodicCellName()
        : null;
    const forMode = linearPeriodicMode || gridPeriodicMode || treePeriodicMode;
    const canSwitchGridPeriodicCell = (direction) =>
      typeof runtime.canSwitchGridPeriodicCell === "function" &&
      runtime.canSwitchGridPeriodicCell(direction);
    const canSwitchTreePeriodicCell = (direction) =>
      typeof runtime.canSwitchTreePeriodicCell === "function" &&
      runtime.canSwitchTreePeriodicCell(direction);
    const benchmarkMode =
      typeof runtime.isBenchmarkMode === "function" && runtime.isBenchmarkMode();
    const benchmarkSession =
      benchmarkMode && typeof runtime.getBenchmarkSession === "function"
        ? runtime.getBenchmarkSession()
        : null;
    const benchmarkActivePosition = benchmarkSession ? benchmarkSession.activePosition : 0;
    const benchmarkSchemeView = benchmarkMode && benchmarkActivePosition > 0;
    const primaryToolbarGroup =
      connectButton && connectButton.parentElement
        ? connectButton.parentElement
        : createGroupButton && createGroupButton.parentElement
          ? createGroupButton.parentElement
          : addNoteButton && addNoteButton.parentElement
            ? addNoteButton.parentElement
            : null;
    const primaryToolbarDivider =
      primaryToolbarGroup && primaryToolbarGroup.nextElementSibling
        ? primaryToolbarGroup.nextElementSibling
        : null;
    const templateToolbarGroup =
      templateSelect && templateSelect.parentElement
        ? templateSelect.parentElement.parentElement || templateSelect.parentElement
        : templateSettingsButton && templateSettingsButton.parentElement
          ? templateSettingsButton.parentElement.parentElement
              || templateSettingsButton.parentElement
          : reflowImportedButton && reflowImportedButton.parentElement
            ? reflowImportedButton.parentElement.parentElement
                || reflowImportedButton.parentElement
            : insertTemplateButton
              ? insertTemplateButton.parentElement || insertTemplateButton
              : null;
    const activeBenchmarkScheme =
      benchmarkMode && typeof runtime.getActiveBenchmarkScheme === "function"
        ? runtime.getActiveBenchmarkScheme()
        : null;
    const selectedTemplateValue = templateSelect ? templateSelect.value : "";
    const rawSelectedTensorIds = getSelectedTensorIds();
    const selectedTensorIds = Array.isArray(rawSelectedTensorIds)
      ? rawSelectedTensorIds
      : [];
    const selectedTensors = selectedTensorIds
      .map((tensorId) => findTensorById(tensorId))
      .filter(Boolean);
    const hasSelectedIndices = selectedTensors.some(
      (tensor) => Array.isArray(tensor.indices) && tensor.indices.length > 0
    );
    runtime.enforceLinearPeriodicEngineSupport();
    syncCodeGenerationWarning();
    syncTemplateCatalogWarning();
    syncHelpModalState();
    syncTemplateManagerModalState();
    syncGeneratedCodeActionState();
    syncGeneratedCodeModalState();

    if (undoButton) {
      undoButton.disabled = state.undoStack.length === 0;
    }
    if (redoButton) {
      redoButton.disabled = state.redoStack.length === 0;
    }
    if (generateButton) {
      generateButton.disabled = !state.spec || !state.selectedEngine;
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
    if (saveSessionTemplateMenuItem) {
      saveSessionTemplateMenuItem.disabled = forMode || selectedTensorIds.length === 0;
    }
    if (loadSessionTemplateMenuItem) {
      loadSessionTemplateMenuItem.disabled = false;
    }
    if (exportSessionTemplateMenuItem) {
      exportSessionTemplateMenuItem.disabled = forMode || selectedTensorIds.length === 0;
    }
    if (editSessionTemplateMenuItem) {
      editSessionTemplateMenuItem.disabled = state.availableTemplates.length === 0;
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
        benchmarkSchemeView || selectedTensorIds.length === 0;
      setElementHidden(
        reflowImportedButton.parentElement || reflowImportedButton,
        benchmarkSchemeView
      );
      setTooltipDescription(
        reflowImportedButton,
        benchmarkSchemeView
          ? "Layout tools are unavailable while viewing a benchmark scheme."
          : selectedTensorIds.length === 0
          ? "Select at least one tensor first."
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
      (benchmarkSchemeView || !selectedTemplateValue || forMode)
      && state.isTemplateSettingsOpen
    ) {
      state.isTemplateSettingsOpen = false;
    }
    if ((benchmarkSchemeView || selectedTensorIds.length === 0) && state.isReflowLayoutOpen) {
      state.isReflowLayoutOpen = false;
    }
    if (templateSelect) {
      templateSelect.disabled = benchmarkSchemeView;
      setElementHidden(templateSelect.parentElement || templateSelect, benchmarkSchemeView);
    }
    if (createGroupButton) {
      createGroupButton.disabled = selectedTensorIds.length < 2;
    }
    setMenuItemChecked(singleModeMenuItem, !forMode && !benchmarkMode);
    setMenuItemChecked(linearPeriodicModeMenuItem, linearPeriodicMode);
    if (gridPeriodicModeMenuItem) {
      setMenuItemChecked(gridPeriodicModeMenuItem, gridPeriodicMode);
    }
    if (treeModeMenuItem) {
      setMenuItemChecked(treeModeMenuItem, treePeriodicMode);
    }
    if (benchmarkModeMenuItem) {
      setMenuItemChecked(benchmarkModeMenuItem, benchmarkMode);
      benchmarkModeMenuItem.disabled = forMode;
      setTooltipDescription(
        benchmarkModeMenuItem,
        forMode
          ? "Benchmark mode is unavailable while a For mode is active."
          : "Compare manual contraction schemes on the current tensor network."
      );
    }
    if (toolbarModeControls) {
      toolbarModeControls.hidden = !(forMode || benchmarkMode);
    }
    if (linearPeriodicCellLabel && !benchmarkMode) {
      linearPeriodicCellLabel.hidden = false;
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : gridPeriodicMode
          ? GRID_PERIODIC_CELL_LABELS[activeGridPeriodicCell] || "Grid cell"
          : treePeriodicMode
            ? TREE_PERIODIC_CELL_LABELS[activeTreePeriodicCell] || "Tree cell"
          : "Single";
    }
    if (benchmarkSchemeNameInput && !benchmarkMode) {
      benchmarkSchemeNameInput.hidden = true;
      benchmarkSchemeNameInput.disabled = true;
      benchmarkSchemeNameInput.value = "";
    }
    if (benchmarkCompareButton && !benchmarkMode) {
      benchmarkCompareButton.hidden = true;
      benchmarkCompareButton.disabled = true;
      setTooltipDescription(
        benchmarkCompareButton,
        "Compare the saved contraction schemes."
      );
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicPreviousCellButton.disabled = benchmarkMode
        ? benchmarkActivePosition === 0
        : gridPeriodicMode
          ? !canSwitchGridPeriodicCell("left")
          : !linearPeriodicMode || activeLinearPeriodicCell === "initial";
      setTooltipDescription(
        linearPeriodicPreviousCellButton,
        benchmarkMode
          ? benchmarkActivePosition === 0
            ? "You are already at the tensor network view. In benchmark mode, use Previous and Next to move between the base network and the saved contraction schemes."
            : "Open the previous saved benchmark scheme. Use Previous and Next to move between the tensor network view and each saved scheme."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("left")
              ? "Move to the cell on the left. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the left edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
            : !linearPeriodicMode
              ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
            : activeLinearPeriodicCell === "initial"
              ? "You are already at the initial cell of the three-cell unidimensional workflow."
              : "Move to the previous cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicNextCellButton.disabled =
        !benchmarkMode &&
        (gridPeriodicMode
          ? !canSwitchGridPeriodicCell("right")
          : !linearPeriodicMode || activeLinearPeriodicCell === "final");
      linearPeriodicNextCellButton.textContent = benchmarkMode
        ? typeof runtime.getBenchmarkNextButtonLabel === "function"
          ? runtime.getBenchmarkNextButtonLabel()
          : ">"
        : ">";
      setTooltipDescription(
        linearPeriodicNextCellButton,
        benchmarkMode
          ? linearPeriodicNextCellButton.textContent === "+"
            ? "Create a new benchmark scheme after the current one. Use Next repeatedly to add schemes and then compare them in Planner."
            : "Open the next saved benchmark scheme. Use Previous and Next to move through the benchmark chain."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("right")
              ? "Move to the cell on the right. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the right edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
            : !linearPeriodicMode
              ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
            : activeLinearPeriodicCell === "final"
              ? "You are already at the final cell of the three-cell unidimensional workflow."
              : "Move to the next cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (gridPeriodicUpCellButton) {
      gridPeriodicUpCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicUpCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("up")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("up");
      setTooltipDescription(
        gridPeriodicUpCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("up")
            ? "Move to the upper cell."
            : "You are already at the top edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("up")
              ? "Move to the parent-facing cell above."
              : "You are already at the root cell."
      );
    }
    if (gridPeriodicDownCellButton) {
      gridPeriodicDownCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicDownCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("down")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("down");
      setTooltipDescription(
        gridPeriodicDownCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("down")
            ? "Move to the lower cell."
            : "You are already at the bottom edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("down")
              ? "Move to the child-facing cell below."
              : "You are already at the leaf cell."
      );
    }
    if (benchmarkMode) {
      if (linearPeriodicCellLabel) {
        linearPeriodicCellLabel.hidden = true;
        linearPeriodicCellLabel.textContent =
          typeof runtime.getBenchmarkBaseLabel === "function"
            ? runtime.getBenchmarkBaseLabel()
            : "Tensor network";
      }
      if (benchmarkSchemeNameInput) {
        benchmarkSchemeNameInput.hidden = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.disabled = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.value =
          benchmarkActivePosition > 0
            ? activeBenchmarkScheme && typeof activeBenchmarkScheme.name === "string"
              ? activeBenchmarkScheme.name
              : typeof runtime.getBenchmarkSchemeName === "function"
                ? runtime.getBenchmarkSchemeName(benchmarkActivePosition - 1)
                : `Scheme ${benchmarkActivePosition}`
            : "";
      }
      if (benchmarkCompareButton) {
        benchmarkCompareButton.hidden = false;
        benchmarkCompareButton.disabled = !(
          typeof runtime.canOpenBenchmarkCompare === "function" &&
          runtime.canOpenBenchmarkCompare()
        );
        setTooltipDescription(
          benchmarkCompareButton,
          benchmarkCompareButton.disabled
            ? "Create at least one scheme first."
            : "Compare the saved contraction schemes."
        );
      }
    }
    if (benchmarkCompareModal && benchmarkCompareTableBody) {
      if (typeof runtime.syncBenchmarkCompareModalState === "function") {
        runtime.syncBenchmarkCompareModalState();
      }
    }
    syncToolbarTransientUi();
  }

  return {
    syncCodeGenerationWarning,
    syncTemplateCatalogWarning,
    updateToolbarState,
  };
}
