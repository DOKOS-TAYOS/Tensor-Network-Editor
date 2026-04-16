import { createEntityPropertiesBindings } from "./properties/entityPropertiesBindings.js";
import {
  buildEdgePropertiesMarkup,
  buildGroupPropertiesMarkup,
  buildNotePropertiesMarkup,
} from "./properties/entityPropertiesMarkup.js";

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

  function renderGroupProperties(groupId) {
    const group = actions.findGroupById(groupId);
    if (!group) {
      actions.clearSelection();
      return;
    }
    const groupColor = actions.getMetadataColor(group.metadata, "#61a8ff");
    const linearPeriodicMode = actions.isLinearPeriodicMode();
    const totalElementCount = getTotalElementCountForTensorIds(
      Array.isArray(group.tensor_ids) ? group.tensor_ids : []
    );

    propertiesPanel.innerHTML = buildGroupPropertiesMarkup({
      group,
      groupColor,
      linearPeriodicMode,
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
    const edgeColor = actions.getMetadataColor(edge.metadata, "#8da1c3");

    propertiesPanel.innerHTML = buildEdgePropertiesMarkup({
      edge,
      edgeColor,
      renderTrashIcon,
      buildMetadataEditorMarkup,
      escapeHtml: actions.escapeHtml,
    });
    bindings.bindEdgeProperties({ edge, edgeColor });
  }

  function renderNoteProperties(noteId) {
    const note = actions.findNoteById(noteId);
    if (!note) {
      actions.clearSelection();
      return;
    }
    const noteColor = actions.getMetadataColor(note.metadata, "#5f95ff");

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
    renderNoteProperties,
  };
}
