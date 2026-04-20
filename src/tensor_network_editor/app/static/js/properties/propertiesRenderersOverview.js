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
    const edgeCount = selectedEntries.filter(
      (entry) => entry.kind === "edge"
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

    propertiesPanel.innerHTML = buildMultiSelectionPropertiesMarkup({
      selectedEntries,
      baseTensorCount,
      tensorCount,
      indexCount,
      edgeCount,
      groupCount,
      noteCount,
      hasMultipleTensors,
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
      hasMultipleTensors,
    });
  }

  return {
    renderNetworkProperties,
    renderMultiSelectionProperties,
  };
}
