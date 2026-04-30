import { createOverviewPropertiesBindings } from "./overviewPropertiesBindings.js";
import {
  buildMultiSelectionPropertiesMarkup,
  buildNetworkPropertiesMarkup,
} from "./overviewPropertiesMarkup.js";

export function createOverviewPropertiesRenderers({
  state,
  document,
  propertiesPanel,
  support,
  actions,
}) {
  const {
    buildMetadataEditorMarkup,
    renderTrashIcon,
    getSelectionTotalElementCount,
    formatTotalElementCount,
  } = support;
  const bindings = createOverviewPropertiesBindings({
    documentRef: document,
    support,
    actions,
  });

  function isStructuralBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (
          (typeof actions.isForBoundaryTensor === "function" &&
            actions.isForBoundaryTensor(tensor)) ||
          (typeof actions.isLinearPeriodicBoundaryTensor === "function" &&
            actions.isLinearPeriodicBoundaryTensor(tensor)) ||
          (typeof actions.isTreePeriodicBoundaryTensor === "function" &&
            actions.isTreePeriodicBoundaryTensor(tensor)) ||
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

  function renderNetworkProperties() {
    propertiesPanel.innerHTML = buildNetworkPropertiesMarkup({
      spec: state.spec,
      connectionCount:
        (Array.isArray(state.spec?.edges) ? state.spec.edges.length : 0) +
        (Array.isArray(state.spec?.hyperedges) ? state.spec.hyperedges.length : 0),
      escapeHtml: actions.escapeHtml,
      buildMetadataEditorMarkup,
    });
    bindings.bindNetworkProperties({ state });
  }

  function renderMultiSelectionProperties() {
    const selectedEntries = actions.getSelectedEntries();
    const baseTensorCount = selectedEntries.filter(
      (entry) => entry.kind === "tensor"
    ).length;
    const tensorCount = selectedEntries.filter(
      (entry) => entry.kind === "tensor" || entry.kind === "contraction-tensor"
    ).length;
    const indexCount = selectedEntries.filter(
      (entry) => entry.kind === "index"
    ).length;
    const connectionCount = selectedEntries.filter(
      (entry) => entry.kind === "edge" || entry.kind === "hyperedge"
    ).length;
    const groupCount = selectedEntries.filter(
      (entry) => entry.kind === "group"
    ).length;
    const noteCount = selectedEntries.filter(
      (entry) => entry.kind === "note"
    ).length;
    const selectedIndexEntries = selectedEntries.filter(
      (entry) => entry.kind === "index"
    );
    const selectedIndexDimensions = [
      ...new Set(
        selectedIndexEntries
          .map((entry) => entry?.located?.index?.dimension)
          .filter((dimension) => Number.isFinite(dimension))
      ),
    ];
    const tensorsOnly =
      baseTensorCount > 0 && baseTensorCount === selectedEntries.length;
    const hasMultipleTensors = baseTensorCount > 1;
    const editableTensorIds = selectedEntries
      .filter((entry) => entry.kind === "tensor")
      .map((entry) => entry.tensor || null)
      .filter((tensor) => tensor && !isStructuralBoundaryTensor(tensor))
      .map((tensor) => tensor.id);
    const exportableTensorCount = editableTensorIds.length;
    const disableSubnetworkActions = hasMultipleTensors && exportableTensorCount === 0;
    const subnetworkActionsMessage = disableSubnetworkActions
      ? "Virtual For-mode boundary tensors cannot be exported or promoted as templates."
      : "";
    const batchColor = actions.getBatchColorValue(selectedEntries);
    const totalElementCount = getSelectionTotalElementCount(selectedEntries);
    const hyperedgeCreationCandidate =
      typeof actions.describeSelectedHyperedgeCandidate === "function"
        ? actions.describeSelectedHyperedgeCandidate(selectedEntries)
        : null;
    const multiIndexDimensionCandidate =
      hyperedgeCreationCandidate?.selectionContainsOnlyIndicesOrOwners &&
      selectedIndexEntries.length >= 2
        ? {
            canEdit: true,
            hasMixedDimensions: selectedIndexDimensions.length > 1,
            selectedIndexIds: selectedIndexEntries.map((entry) => entry.id),
            value:
              selectedIndexDimensions.length === 1
                ? String(selectedIndexDimensions[0])
                : "",
          }
        : null;

    propertiesPanel.innerHTML = buildMultiSelectionPropertiesMarkup({
      selectedEntries,
      baseTensorCount,
      tensorCount,
      indexCount,
      connectionCount,
      groupCount,
      noteCount,
      hasMultipleTensors,
      multiIndexDimensionCandidate,
      showAddIndexAction: editableTensorIds.length > 0,
      hyperedgeCreationCandidate,
      disableSubnetworkActions,
      subnetworkActionsMessage,
      batchColor,
      totalElementCount,
      formatTotalElementCount,
      renderTrashIcon,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindMultiSelectionProperties({
      state,
      selectedEntries,
      batchColor,
      editableTensorIds,
      selectedIndexIds: multiIndexDimensionCandidate?.selectedIndexIds || [],
      hyperedgeCreationCandidate,
      hasMultipleTensors,
    });
  }

  return {
    renderNetworkProperties,
    renderMultiSelectionProperties,
  };
}
