import { createPropertyCommands } from "../actions/propertyCommands.js";
import { createPropertiesRenderers } from "./propertiesRenderers.js";
import { createPropertiesSupport } from "./propertiesSupport.js";

function resolveContextAction(ctx, actionName, fallback = () => {}) {
  const candidate = ctx[actionName];
  return typeof candidate === "function" ? candidate.bind(ctx) : fallback;
}

function createPropertyActions(ctx) {
  function describeSelectedHyperedgeCandidate(selectedEntries = []) {
    const entries = Array.isArray(selectedEntries) ? selectedEntries : [];
    const indexEntries = entries.filter((entry) => entry.kind === "index");
    const selectedIndexIds = indexEntries.map((entry) => entry.id);
    if (!selectedIndexIds.length) {
      return {
        canCreate: false,
        message: "Select indices to create a hyperedge.",
        selectedIndexIds,
      };
    }
    if (indexEntries.length !== entries.length) {
      return {
        canCreate: false,
        message: "Select only normal-mode indices to create a hyperedge.",
        selectedIndexIds,
      };
    }
    if (typeof ctx.describeHyperedgeCandidate !== "function") {
      return {
        canCreate: false,
        message: "Hyperedge creation is unavailable in this session.",
        selectedIndexIds,
      };
    }
    return {
      ...ctx.describeHyperedgeCandidate(selectedIndexIds),
      selectedIndexIds,
    };
  }

  return {
    escapeHtml: (value) => ctx.escapeHtml(value),
    clearSelection: resolveContextAction(ctx, "clearSelection"),
    getSelectedEntries: () => ctx.getSelectedEntries(),
    getBatchColorValue: (selectedEntries) => ctx.getBatchColorValue(selectedEntries),
    getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
    isLinearPeriodicMode: resolveContextAction(ctx, "isLinearPeriodicMode", () => false),
    isForMode: resolveContextAction(ctx, "isForMode", () => false),
    deleteSelection: resolveContextAction(ctx, "deleteSelection"),
    describeSelectedHyperedgeCandidate,
    alignSelectedTensors: (alignment) => ctx.alignSelectedTensors(alignment),
    arrangeSelectedTensors: (layoutKind) => ctx.arrangeSelectedTensors(layoutKind),
    distributeSelectedTensors: (axis) => ctx.distributeSelectedTensors(axis),
    snapSelectedTensorsToGrid: () => ctx.snapSelectedTensorsToGrid(),
    exportSelectedSubnetwork: () => ctx.exportSelectedSubnetwork(),
    saveSelectionToSubnetworkLibrary: () => ctx.saveSelectionToSubnetworkLibrary(),
    promoteSelectedSubnetworkToTemplate: () => ctx.promoteSelectedSubnetworkToTemplate(),
    createGroupFromSelection: () => ctx.createGroupFromSelection(),
    findGroupById: (groupId) => ctx.findGroupById(groupId),
    findEdgeById: (edgeId) => ctx.findEdgeById(edgeId),
    findHyperedgeById: (hyperedgeId) => ctx.findHyperedgeById(hyperedgeId),
    findNoteById: (noteId) => ctx.findNoteById(noteId),
    getMetadataColor: (metadata, fallbackColor) =>
      ctx.getMetadataColor(metadata, fallbackColor),
    toggleGroupCollapse: (groupId) => ctx.toggleGroupCollapse(groupId),
    exportGroupSubnetwork: (groupId) => ctx.exportGroupSubnetwork(groupId),
    saveGroupToSubnetworkLibrary: (groupId) => ctx.saveGroupToSubnetworkLibrary(groupId),
    promoteGroupToTemplate: (groupId) => ctx.promoteGroupToTemplate(groupId),
  };
}

function createCommands(ctx) {
  return createPropertyCommands({
    applyDesignChange: (mutate, options) => ctx.applyDesignChange(mutate, options),
    applyColorToSelection: (nextColor) => ctx.applyColorToSelection(nextColor),
    centerTensor: (tensorId) => ctx.centerTensor(tensorId),
    createHyperedge: (indexIds) => ctx.createHyperedge(indexIds),
    createIndex: (tensor, indexPosition) => ctx.createIndex(tensor, indexPosition),
    describeHyperedgeCandidate: (indexIds) => ctx.describeHyperedgeCandidate(indexIds),
    deleteSelection: () => ctx.deleteSelection(),
    findIndexOwner: (indexId) => ctx.findIndexOwner(indexId),
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
    moveIndex: (tensorId, indexPosition, direction) =>
      ctx.moveIndex(tensorId, indexPosition, direction),
    removeEdge: (edgeId) => ctx.removeEdge(edgeId),
    removeHyperedge: (hyperedgeId) => ctx.removeHyperedge(hyperedgeId),
    removeGroup: (groupId) => {
      ctx.state.spec.groups = ctx.state.spec.groups.filter(
        (candidate) => candidate.id !== groupId
      );
    },
    removeIndex: (tensorId, indexId) => ctx.removeIndex(tensorId, indexId),
    removeNote: (noteId) => ctx.removeNote(noteId),
    removeTensor: (tensorId) => ctx.removeTensor(tensorId),
    resolveHyperedgeSelectionId: (hyperedgeId) => ctx.hyperedgeHubNodeId(hyperedgeId),
    setStatus: (message, level) => ctx.setStatus(message, level),
    setSelection: (selectionIds, options) => ctx.setSelection(selectionIds, options),
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
