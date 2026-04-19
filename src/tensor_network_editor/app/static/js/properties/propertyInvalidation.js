export function createPropertyInvalidationSupport({
  isForMode = null,
  isLinearPeriodicMode = () => false,
}) {
  const resolveForMode = () =>
    (typeof isForMode === "function" && isForMode()) ||
    (typeof isLinearPeriodicMode === "function" && isLinearPeriodicMode());

  function propertyInvalidation(overrides = {}) {
    return {
      graph: false,
      lookups: false,
      analysis: false,
      properties: Boolean(resolveForMode()),
      overlays: false,
      planner: false,
      minimap: false,
      ...overrides,
    };
  }

  function selectionColorInvalidation(selectedEntries) {
    const entryKinds = new Set(
      (Array.isArray(selectedEntries) ? selectedEntries : []).map(
        (entry) => entry.kind
      )
    );
    const affectsGraph =
      entryKinds.has("tensor") ||
      entryKinds.has("index") ||
      entryKinds.has("edge");
    const affectsOverlays = entryKinds.has("group") || entryKinds.has("note");
    return propertyInvalidation({
      graph: affectsGraph,
      overlays: affectsOverlays,
      minimap: affectsGraph,
    });
  }

  return {
    propertyInvalidation,
    selectionColorInvalidation,
  };
}

export function propertyInvalidationForContext(ctx, overrides = {}) {
  return createPropertyInvalidationSupport({
    isForMode: () =>
      (typeof ctx.isForMode === "function" && ctx.isForMode()) ||
      (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) ||
      (typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode()),
  }).propertyInvalidation(overrides);
}

export function selectionColorInvalidationForContext(ctx, selectedEntries) {
  return createPropertyInvalidationSupport({
    isForMode: () =>
      (typeof ctx.isForMode === "function" && ctx.isForMode()) ||
      (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) ||
      (typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode()),
  }).selectionColorInvalidation(selectedEntries);
}
