import { GRAPH_THEME } from "../core/theme.js";
import { createEntityPropertiesBindings } from "./entityPropertiesBindings.js";
import {
  buildEdgePropertiesMarkup,
  buildGroupPropertiesMarkup,
  buildHyperedgePropertiesMarkup,
  buildNotePropertiesMarkup,
} from "./entityPropertiesMarkup.js";

export function createEntityPropertiesRenderers({
  state,
  document,
  propertiesPanel,
  support,
  actions,
}) {
  const {
    buildMetadataEditorMarkup,
    renderTrashIcon,
    getTotalElementCountForTensorIds,
    formatTotalElementCount,
  } = support;
  const bindings = createEntityPropertiesBindings({
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

  function renderGroupProperties(groupId) {
    const group = actions.findGroupById(groupId);
    if (!group) {
      actions.clearSelection();
      return;
    }
    const groupColor = actions.getMetadataColor(
      group.metadata,
      GRAPH_THEME.groupDefault
    );
    const exportableTensorCount = (Array.isArray(group.tensor_ids)
      ? group.tensor_ids
      : []
    )
      .map((tensorId) =>
        typeof actions.findTensorById === "function"
          ? actions.findTensorById(tensorId)
          : null
      )
      .filter((tensor) => tensor && !isStructuralBoundaryTensor(tensor)).length;
    const disableSubnetworkActions = exportableTensorCount === 0;
    const subnetworkActionsMessage = disableSubnetworkActions
      ? "Virtual For-mode boundary tensors cannot be exported or promoted as templates."
      : "";
    const totalElementCount = getTotalElementCountForTensorIds(
      Array.isArray(group.tensor_ids) ? group.tensor_ids : []
    );

    propertiesPanel.innerHTML = buildGroupPropertiesMarkup({
      group,
      groupColor,
      disableSubnetworkActions,
      subnetworkActionsMessage,
      totalElementCount,
      formatTotalElementCount,
      renderTrashIcon,
      buildMetadataEditorMarkup,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindGroupProperties({ group, groupColor });
  }

  function renderEdgeProperties(edgeId) {
    const edge = actions.findEdgeById(edgeId);
    if (!edge) {
      actions.clearSelection();
      return;
    }
    const edgeColor = actions.getMetadataColor(edge.metadata, GRAPH_THEME.edge);

    propertiesPanel.innerHTML = buildEdgePropertiesMarkup({
      edge,
      edgeColor,
      renderTrashIcon,
      buildMetadataEditorMarkup,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindEdgeProperties({ edge, edgeColor });
  }

  function renderHyperedgeProperties(hyperedgeId) {
    const hyperedge = actions.findHyperedgeById(hyperedgeId);
    if (!hyperedge) {
      actions.clearSelection();
      return;
    }
    const hyperedgeColor = actions.getMetadataColor(
      hyperedge.metadata,
      GRAPH_THEME.edge
    );
    const endpointCount = Array.isArray(hyperedge.endpoints)
      ? hyperedge.endpoints.length
      : 0;

    propertiesPanel.innerHTML = buildHyperedgePropertiesMarkup({
      hyperedge,
      hyperedgeColor,
      endpointCount,
      renderTrashIcon,
      buildMetadataEditorMarkup,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindHyperedgeProperties({ hyperedge, hyperedgeColor });
  }

  function renderNoteProperties(noteId) {
    const note = actions.findNoteById(noteId);
    if (!note) {
      actions.clearSelection();
      return;
    }
    const noteColor = actions.getMetadataColor(
      note.metadata,
      GRAPH_THEME.noteDefault
    );

    propertiesPanel.innerHTML = buildNotePropertiesMarkup({
      note,
      noteColor,
      renderTrashIcon,
      buildMetadataEditorMarkup,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindNoteProperties({ note, noteColor });
  }

  return {
    renderGroupProperties,
    renderEdgeProperties,
    renderHyperedgeProperties,
    renderNoteProperties,
  };
}
