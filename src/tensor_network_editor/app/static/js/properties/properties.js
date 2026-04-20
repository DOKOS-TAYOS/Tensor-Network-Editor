import { createPropertyCommands } from "../actions/propertyCommands.js";
import { createPropertiesRenderers } from "./propertiesRenderers.js";
import { createPropertiesSupport } from "./propertiesSupport.js";

function resolveContextAction(ctx, actionName, fallback = () => {}) {
  const candidate = ctx[actionName];
  return typeof candidate === "function" ? candidate.bind(ctx) : fallback;
}

function createPropertyActions(ctx) {
  return {
    escapeHtml: (value) => ctx.escapeHtml(value),
    clearSelection: resolveContextAction(ctx, "clearSelection"),
    getSelectedEntries: () => ctx.getSelectedEntries(),
    getBatchColorValue: (selectedEntries) => ctx.getBatchColorValue(selectedEntries),
    getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
    isLinearPeriodicMode: resolveContextAction(ctx, "isLinearPeriodicMode", () => false),
    isForMode: resolveContextAction(ctx, "isForMode", () => false),
    deleteSelection: resolveContextAction(ctx, "deleteSelection"),
    alignSelectedTensors: (alignment) => ctx.alignSelectedTensors(alignment),
    arrangeSelectedTensors: (layoutKind) => ctx.arrangeSelectedTensors(layoutKind),
    distributeSelectedTensors: (axis) => ctx.distributeSelectedTensors(axis),
    snapSelectedTensorsToGrid: () => ctx.snapSelectedTensorsToGrid(),
    exportSelectedSubnetwork: () => ctx.exportSelectedSubnetwork(),
    promoteSelectedSubnetworkToTemplate: () => ctx.promoteSelectedSubnetworkToTemplate(),
    createGroupFromSelection: () => ctx.createGroupFromSelection(),
    findGroupById: (groupId) => ctx.findGroupById(groupId),
    findEdgeById: (edgeId) => ctx.findEdgeById(edgeId),
    findNoteById: (noteId) => ctx.findNoteById(noteId),
    getMetadataColor: (metadata, fallbackColor) =>
      ctx.getMetadataColor(metadata, fallbackColor),
    toggleGroupCollapse: (groupId) => ctx.toggleGroupCollapse(groupId),
    exportGroupSubnetwork: (groupId) => ctx.exportGroupSubnetwork(groupId),
    promoteGroupToTemplate: (groupId) => ctx.promoteGroupToTemplate(groupId),
  };
}

function createCommands(ctx) {
  return createPropertyCommands({
    applyDesignChange: (mutate, options) => ctx.applyDesignChange(mutate, options),
    applyColorToSelection: (nextColor) => ctx.applyColorToSelection(nextColor),
    centerTensor: (tensorId) => ctx.centerTensor(tensorId),
    createIndex: (tensor, indexPosition) => ctx.createIndex(tensor, indexPosition),
    deleteSelection: () => ctx.deleteSelection(),
    findIndexOwner: (indexId) => ctx.findIndexOwner(indexId),
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
    moveIndex: (tensorId, indexPosition, direction) =>
      ctx.moveIndex(tensorId, indexPosition, direction),
    removeEdge: (edgeId) => ctx.removeEdge(edgeId),
    removeGroup: (groupId) => {
      ctx.state.spec.groups = ctx.state.spec.groups.filter(
        (candidate) => candidate.id !== groupId
      );
    },
    removeIndex: (tensorId, indexId) => ctx.removeIndex(tensorId, indexId),
    removeNote: (noteId) => ctx.removeNote(noteId),
    removeTensor: (tensorId) => ctx.removeTensor(tensorId),
    setStatus: (message, level) => ctx.setStatus(message, level),
    syncConnectedIndexDimension: (indexId, nextDimension) =>
      ctx.syncConnectedIndexDimension(indexId, nextDimension),
    tensorIndexNameExists: (tensor, proposedName, ignoredIndexId) =>
      ctx.tensorIndexNameExists(tensor, proposedName, ignoredIndexId),
  });
}

export function registerProperties(ctx) {
  const commands = createCommands(ctx);
  const support = createPropertiesSupport({
    ctx,
    state: ctx.state,
    window: ctx.window,
    commands,
  });
  const actions = createPropertyActions(ctx);
  const renderers = createPropertiesRenderers({
    ctx,
    state: ctx.state,
    document: ctx.document,
    propertiesPanel: ctx.dom.propertiesPanel,
    support,
    actions,
  });

  Object.assign(ctx, {
    bindDebouncedAutosave: support.bindDebouncedAutosave,
    bindImmediateAutosave: support.bindImmediateAutosave,
    buildMetadataEditorMarkup: support.buildMetadataEditorMarkup,
    bindMetadataEditors: support.bindMetadataEditors,
    propertyCommands: commands,
    propertyInvalidation: support.propertyInvalidation,
    ...renderers,
  });
}
