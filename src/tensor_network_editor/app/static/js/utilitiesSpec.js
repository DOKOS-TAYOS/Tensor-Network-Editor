import { createSpecLookupBindings } from "./spec/specLookups.js";
import { createSpecMutationBindings } from "./spec/specMutations.js";
import { createSpecNormalizationBindings } from "./spec/specNormalization.js";

export function createUtilitySpecBindings({ ctx, state, constants, runtime }) {
  const { document } = ctx;

  const normalizationBindings = createSpecNormalizationBindings({
    state,
    constants,
    runtime,
  });
  const lookupBindings = createSpecLookupBindings({
    ctx,
    state,
  });
  const mutationBindings = createSpecMutationBindings({
    ctx,
    state,
    constants,
    runtime,
    findTensorById: lookupBindings.findTensorById,
    findEdgeById: lookupBindings.findEdgeById,
    findIndexOwner: lookupBindings.findIndexOwner,
    findEdgeByIndexId: lookupBindings.findEdgeByIndexId,
    resolveBaseEdgeId: lookupBindings.resolveBaseEdgeId,
  });

  function serializeCurrentSpec(options = {}) {
    const { persistViewSnapshots = false } = options;
    const cacheKey = persistViewSnapshots
      ? "serializedSpecCacheWithSnapshots"
      : "serializedSpecCacheWithoutSnapshots";
    if (state.serializedSpecCacheRevision !== state.specRevision) {
      state.serializedSpecCacheRevision = state.specRevision;
      state.serializedSpecCacheWithoutSnapshots = null;
      state.serializedSpecCacheWithSnapshots = null;
    }
    if (state[cacheKey]) {
      return state[cacheKey];
    }
    if (
      persistViewSnapshots &&
      state.spec &&
      state.spec.contraction_plan &&
      typeof ctx.ensureContractionViewSnapshots === "function"
    ) {
      ctx.ensureContractionViewSnapshots();
    }
    const serializedSpec = {
      schema_version: state.schemaVersion,
      network: normalizationBindings.buildSerializedSpec(),
    };
    state[cacheKey] = serializedSpec;
    return serializedSpec;
  }

  function captureEditableFocus() {
    const activeElement = document.activeElement;
    if (!activeElement || !(activeElement instanceof HTMLElement)) {
      return null;
    }
    const focusKey = activeElement.dataset ? activeElement.dataset.focusKey : "";
    if (!focusKey) {
      return null;
    }
    const focusState = {
      key: focusKey,
      selectionStart: null,
      selectionEnd: null,
    };
    if (
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement
    ) {
      focusState.selectionStart = activeElement.selectionStart;
      focusState.selectionEnd = activeElement.selectionEnd;
    }
    return focusState;
  }

  function restoreEditableFocus(focusState) {
    if (!focusState) {
      return;
    }
    const target = Array.from(document.querySelectorAll("[data-focus-key]")).find(
      (element) => element.dataset.focusKey === focusState.key
    );
    if (!(target instanceof HTMLElement)) {
      return;
    }
    target.focus({ preventScroll: true });
    if (
      typeof focusState.selectionStart === "number" &&
      typeof focusState.selectionEnd === "number" &&
      (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)
    ) {
      target.setSelectionRange(
        focusState.selectionStart,
        focusState.selectionEnd
      );
    }
  }

  function stripImportLines(code) {
    const keptLines = code
      .split(/\r?\n/)
      .filter((line) => !/^\s*(import|from)\s+/.test(line));
    while (keptLines.length && keptLines[0].trim() === "") {
      keptLines.shift();
    }
    while (keptLines.length && keptLines[keptLines.length - 1].trim() === "") {
      keptLines.pop();
    }
    return keptLines.join("\n");
  }

  return {
    ...normalizationBindings,
    ...lookupBindings,
    serializeCurrentSpec,
    captureEditableFocus,
    restoreEditableFocus,
    stripImportLines,
    ...mutationBindings,
  };
}
