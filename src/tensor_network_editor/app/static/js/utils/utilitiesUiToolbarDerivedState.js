export function createUiToolbarDerivedStateSupport({
  state,
  dom,
  runtime,
  getSelectedTensorIds,
  findTensorById,
}) {
  const {
    connectButton,
    createGroupButton,
    addNoteButton,
    templateSelect,
    templateSettingsButton,
    reflowImportedButton,
    insertTemplateButton,
  } = dom;

  function getPrimaryToolbarGroup() {
    return connectButton && connectButton.parentElement
      ? connectButton.parentElement
      : createGroupButton && createGroupButton.parentElement
        ? createGroupButton.parentElement
        : addNoteButton && addNoteButton.parentElement
          ? addNoteButton.parentElement
          : null;
  }

  function getTemplateToolbarGroup() {
    return templateSelect && templateSelect.parentElement
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
  }

  function isStructuralBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (
          (typeof runtime.isForBoundaryTensor === "function" &&
            runtime.isForBoundaryTensor(tensor)) ||
          (typeof runtime.isLinearPeriodicBoundaryTensor === "function" &&
            runtime.isLinearPeriodicBoundaryTensor(tensor)) ||
          (typeof runtime.isTreePeriodicBoundaryTensor === "function" &&
            runtime.isTreePeriodicBoundaryTensor(tensor)) ||
          tensor.linear_periodic_role === "previous" ||
          tensor.linear_periodic_role === "next" ||
          tensor.grid_periodic_role === "up" ||
          tensor.grid_periodic_role === "right" ||
          tensor.grid_periodic_role === "down" ||
          tensor.grid_periodic_role === "left" ||
          tensor.tree_periodic_role === "parent" ||
          tensor.tree_periodic_role === "child"
        )
    );
  }

  function getToolbarDerivedState() {
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
    const benchmarkMode =
      typeof runtime.isBenchmarkMode === "function" && runtime.isBenchmarkMode();
    const benchmarkSession =
      benchmarkMode && typeof runtime.getBenchmarkSession === "function"
        ? runtime.getBenchmarkSession()
        : null;
    const benchmarkActivePosition = benchmarkSession ? benchmarkSession.activePosition : 0;
    const benchmarkSchemeView = benchmarkMode && benchmarkActivePosition > 0;
    const primaryToolbarGroup = getPrimaryToolbarGroup();
    const primaryToolbarDivider =
      primaryToolbarGroup && primaryToolbarGroup.nextElementSibling
        ? primaryToolbarGroup.nextElementSibling
        : null;
    const templateToolbarGroup = getTemplateToolbarGroup();
    const activeBenchmarkScheme =
      benchmarkMode && typeof runtime.getActiveBenchmarkScheme === "function"
        ? runtime.getActiveBenchmarkScheme()
        : null;
    const selectedTemplateValue = templateSelect ? templateSelect.value : "";
    const rawSelectedTensorIds = getSelectedTensorIds();
    const selectedTensorIds = Array.isArray(rawSelectedTensorIds)
      ? rawSelectedTensorIds
      : [];
    const graphTensorCount =
      state.spec && Array.isArray(state.spec.tensors) ? state.spec.tensors.length : 0;
    const autoLayoutTensorCount = selectedTensorIds.length || graphTensorCount;
    const selectedTensors = selectedTensorIds
      .map((tensorId) => findTensorById(tensorId))
      .filter(Boolean);
    const selectedExportableTensorIds = selectedTensors
      .filter((tensor) => !isStructuralBoundaryTensor(tensor))
      .map((tensor) => tensor.id);
    const hasSelectedIndices = selectedTensors.some(
      (tensor) => Array.isArray(tensor.indices) && tensor.indices.length > 0
    );
    const hasHyperedges = Boolean(
      Array.isArray(state.spec?.hyperedges) && state.spec.hyperedges.length
    );

    return {
      linearPeriodicMode,
      activeLinearPeriodicCell,
      gridPeriodicMode,
      activeGridPeriodicCell,
      treePeriodicMode,
      activeTreePeriodicCell,
      forMode,
      canSwitchGridPeriodicCell: (direction) =>
        typeof runtime.canSwitchGridPeriodicCell === "function" &&
        runtime.canSwitchGridPeriodicCell(direction),
      canSwitchTreePeriodicCell: (direction) =>
        typeof runtime.canSwitchTreePeriodicCell === "function" &&
        runtime.canSwitchTreePeriodicCell(direction),
      benchmarkMode,
      benchmarkActivePosition,
      benchmarkSchemeView,
      primaryToolbarGroup,
      primaryToolbarDivider,
      templateToolbarGroup,
      activeBenchmarkScheme,
      selectedTemplateValue,
      selectedTensorIds,
      selectedExportableTensorIds,
      graphTensorCount,
      autoLayoutTensorCount,
      hasSelectedIndices,
      hasHyperedges,
    };
  }

  return {
    getToolbarDerivedState,
  };
}
