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
    const tensorsOnly =
      baseTensorCount > 0 && baseTensorCount === selectedEntries.length;
    const hasMultipleTensors = baseTensorCount > 1;
    const linearPeriodicMode =
      (typeof actions.isForMode === "function" && actions.isForMode()) ||
      (typeof actions.isLinearPeriodicMode === "function" &&
        actions.isLinearPeriodicMode());
    const batchColor = actions.getBatchColorValue(selectedEntries);
    const totalElementCount = getSelectionTotalElementCount(selectedEntries);
    const hyperedgeCreationCandidate =
      typeof actions.describeSelectedHyperedgeCandidate === "function"
        ? actions.describeSelectedHyperedgeCandidate(selectedEntries)
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
      hyperedgeCreationCandidate,
      linearPeriodicMode,
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
      hyperedgeCreationCandidate,
      hasMultipleTensors,
    });
  }

  return {
    renderNetworkProperties,
    renderMultiSelectionProperties,
  };
}
