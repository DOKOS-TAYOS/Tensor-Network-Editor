export function createPropertyInvalidationSupport({
  isLinearPeriodicMode = () => false,
}) {
  function propertyInvalidation(overrides = {}) {
    return {
      graph: false,
      lookups: false,
      analysis: false,
      properties: Boolean(isLinearPeriodicMode()),
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
    isLinearPeriodicMode: () =>
      typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode(),
  }).propertyInvalidation(overrides);
}

export function selectionColorInvalidationForContext(ctx, selectedEntries) {
  return createPropertyInvalidationSupport({
    isLinearPeriodicMode: () =>
      typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode(),
  }).selectionColorInvalidation(selectedEntries);
}
