import { createMetadataEditorSupport } from "./properties/metadataEditors.js";
import { createPropertyAutosaveBindings } from "./properties/propertyAutosave.js";
import {
  createPropertyInvalidationSupport,
  propertyInvalidationForContext,
  selectionColorInvalidationForContext,
} from "./properties/propertyInvalidation.js";
import {
  formatTotalElementCount as formatTotalElementCountFromSummaries,
  getContractionTensorTotalElementCount as getContractionTensorTotalElementCountFromSummaries,
  getSelectionEntryTensorIds as getSelectionEntryTensorIdsFromSummaries,
  getSelectionTotalElementCount as getSelectionTotalElementCountFromSummaries,
  getTensorTotalElementCount as getTensorTotalElementCountFromSummaries,
  getTotalElementCountForTensorIds as getTotalElementCountForTensorIdsFromSummaries,
  normalizeElementDimension,
} from "./properties/propertySummaries.js";

export {
  normalizeElementDimension,
  propertyInvalidationForContext,
  selectionColorInvalidationForContext,
};

export function renderTrashIcon() {
  return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
      </svg>
    `;
}

export function createPropertiesSupport({ ctx, state, window, commands }) {
  function collectTagSuggestionsByScope() {
    const scopedTags = {
      edge: [],
      group: [],
      index: [],
      network: [],
      note: [],
      tensor: [],
    };
    const seenByScope = Object.fromEntries(
      Object.keys(scopedTags).map((scope) => [scope, new Set()])
    );

    function addTags(scope, metadata) {
      const tags = Array.isArray(metadata && metadata.tags) ? metadata.tags : [];
      tags.forEach((tag) => {
        if (typeof tag !== "string" || !tag.trim()) {
          return;
        }
        const normalizedTag = tag.trim().toLowerCase();
        if (seenByScope[scope].has(normalizedTag)) {
          return;
        }
        seenByScope[scope].add(normalizedTag);
        scopedTags[scope].push(tag.trim());
      });
    }

    if (!ctx.isObject(state.spec)) {
      return scopedTags;
    }

    addTags("network", state.spec.metadata);
    (Array.isArray(state.spec.tensors) ? state.spec.tensors : []).forEach((tensor) => {
      addTags("tensor", tensor.metadata);
      (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
        addTags("index", index.metadata);
      });
    });
    (Array.isArray(state.spec.edges) ? state.spec.edges : []).forEach((edge) => {
      addTags("edge", edge.metadata);
    });
    (Array.isArray(state.spec.groups) ? state.spec.groups : []).forEach((group) => {
      addTags("group", group.metadata);
    });
    (Array.isArray(state.spec.notes) ? state.spec.notes : []).forEach((note) => {
      addTags("note", note.metadata);
    });
    return scopedTags;
  }

  const autosave = createPropertyAutosaveBindings({
    windowRef: window || globalThis,
  });
  const invalidationSupport = createPropertyInvalidationSupport({
    isLinearPeriodicMode: () =>
      typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode(),
  });
  const metadataSupport = createMetadataEditorSupport({
    annotationDefinitionsByScope: () => state.annotationDefinitions || {},
    tagSuggestionsByScope: collectTagSuggestionsByScope,
    escapeHtml: (value) => ctx.escapeHtml(value),
    isObject: (value) => ctx.isObject(value),
  });

  function propertyInvalidation(overrides = {}) {
    return invalidationSupport.propertyInvalidation(overrides);
  }

  function selectionColorInvalidation(selectedEntries) {
    return invalidationSupport.selectionColorInvalidation(selectedEntries);
  }

  function tensorDisclosureState(tensorId) {
    if (!state.tensorIndexDisclosureState[tensorId]) {
      state.tensorIndexDisclosureState[tensorId] = {};
    }
    return state.tensorIndexDisclosureState[tensorId];
  }

  function isTensorIndexDisclosureOpen(tensorId, indexId) {
    return Boolean(tensorDisclosureState(tensorId)[indexId]);
  }

  function setTensorIndexDisclosureOpen(tensorId, indexId, isOpen) {
    const disclosureState = tensorDisclosureState(tensorId);
    if (isOpen) {
      disclosureState[indexId] = true;
      return;
    }
    delete disclosureState[indexId];
  }

  function syncPendingTensorIndexDisclosure() {
    const pendingIndexId = state.pendingPropertiesIndexFocusId;
    if (!pendingIndexId) {
      return;
    }

    const located = ctx.findIndexOwner(pendingIndexId);
    state.pendingPropertiesIndexFocusId = null;
    if (!located) {
      return;
    }

    const wasOpen = isTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId);
    setTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId, true);
    state.autoExpandedTensorIndex = {
      tensorId: located.tensor.id,
      indexId: pendingIndexId,
      wasOpen,
    };
  }

  function toggleTensorIndexDisclosure(tensorId, indexId) {
    const nextOpen = !isTensorIndexDisclosureOpen(tensorId, indexId);
    setTensorIndexDisclosureOpen(tensorId, indexId, nextOpen);
    if (
      state.autoExpandedTensorIndex &&
      state.autoExpandedTensorIndex.tensorId === tensorId &&
      state.autoExpandedTensorIndex.indexId === indexId
    ) {
      state.autoExpandedTensorIndex = null;
    }
    ctx.renderProperties();
  }

  return {
    bindDebouncedAutosave: autosave.bindDebouncedAutosave,
    bindImmediateAutosave: autosave.bindImmediateAutosave,
    buildMetadataEditorMarkup: (...args) =>
      metadataSupport.buildMetadataEditorMarkup(...args),
    bindMetadataEditors: (options) =>
      metadataSupport.bindMetadataEditors({
        ...options,
        applyDesignChange: (mutate, changeOptions) =>
          ctx.applyDesignChange(mutate, changeOptions),
        bindDebouncedAutosave: autosave.bindDebouncedAutosave,
        setStatus: (message, level) => ctx.setStatus(message, level),
      }),
    commands,
    propertyInvalidation: (overrides = {}) => propertyInvalidation(overrides),
    selectionColorInvalidation: (selectedEntries) =>
      selectionColorInvalidation(selectedEntries),
    renderTrashIcon,
    getTensorTotalElementCount: (tensor) =>
      getTensorTotalElementCountFromSummaries(tensor, ctx.asFiniteNumber),
    getTotalElementCountForTensorIds: (tensorIds) =>
      getTotalElementCountForTensorIdsFromSummaries(
        tensorIds,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    getSelectionEntryTensorIds: getSelectionEntryTensorIdsFromSummaries,
    getSelectionTotalElementCount: (selectedEntries) =>
      getSelectionTotalElementCountFromSummaries(
        selectedEntries,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    getContractionTensorTotalElementCount: (tensor) =>
      getContractionTensorTotalElementCountFromSummaries(
        tensor,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    formatTotalElementCount: formatTotalElementCountFromSummaries,
    tensorDisclosureState,
    isTensorIndexDisclosureOpen,
    setTensorIndexDisclosureOpen,
    syncPendingTensorIndexDisclosure,
    toggleTensorIndexDisclosure,
  };
}
