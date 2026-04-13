import { createEntityPropertiesRenderers } from "./propertiesRenderersEntities.js";
import { createOverviewPropertiesRenderers } from "./propertiesRenderersOverview.js";
import { createTensorPropertiesRenderers } from "./propertiesRenderersTensor.js";

export function createPropertiesRenderers({
  ctx,
  state,
  document,
  propertiesPanel,
  support,
}) {
  const { syncPendingTensorIndexDisclosure } = support;
  const {
    renderNetworkProperties,
    renderMultiSelectionProperties,
  } = createOverviewPropertiesRenderers({
    ctx,
    state,
    document,
    propertiesPanel,
    support,
  });
  const {
    renderContractionTensorProperties,
    renderContractionIndexProperties,
    renderLinearPeriodicBoundaryTensorProperties,
    renderTensorProperties,
  } = createTensorPropertiesRenderers({
    ctx,
    document,
    propertiesPanel,
    support,
  });
  const {
    renderGroupProperties,
    renderEdgeProperties,
    renderNoteProperties,
  } = createEntityPropertiesRenderers({
    ctx,
    state,
    document,
    propertiesPanel,
    support,
  });

  function renderProperties() {
    ctx.pruneSelectionToExisting();
    syncPendingTensorIndexDisclosure();
    if (!state.selectionIds.length) {
      renderNetworkProperties();
      return;
    }
    if (state.selectionIds.length > 1) {
      renderMultiSelectionProperties();
      return;
    }
    const singleSelection = ctx.getSelectionEntry(state.selectionIds[0]);
    if (!singleSelection) {
      renderNetworkProperties();
      return;
    }
    if (singleSelection.kind === "tensor") {
      renderTensorProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "contraction-tensor") {
      renderContractionTensorProperties(singleSelection.tensor);
      return;
    }
    if (singleSelection.kind === "index") {
      renderTensorProperties(singleSelection.located.tensor.id, {
        focusedIndexId: singleSelection.id,
      });
      return;
    }
    if (singleSelection.kind === "contraction-index") {
      renderContractionIndexProperties(singleSelection.located);
      return;
    }
    if (singleSelection.kind === "edge") {
      renderEdgeProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "group") {
      renderGroupProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "note") {
      renderNoteProperties(singleSelection.id);
      return;
    }
    renderNetworkProperties();
  }

  return {
    renderProperties,
    renderNetworkProperties,
    renderMultiSelectionProperties,
    renderTensorProperties,
    renderLinearPeriodicBoundaryTensorProperties,
    renderGroupProperties,
    renderEdgeProperties,
    renderContractionIndexProperties,
    renderNoteProperties,
  };
}
