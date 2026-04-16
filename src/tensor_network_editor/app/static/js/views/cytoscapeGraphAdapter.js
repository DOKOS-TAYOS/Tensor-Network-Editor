import { cloneGraphElementDescriptor } from "./graphDescriptors.js";
import { buildGraphElementUpdatePlan } from "./graphModelDiff.js";

export function createCytoscapeGraphAdapter({ state, getCy }) {
  function updateCache(model) {
    state.graphRenderCyRef = getCy();
    state.graphRenderDescriptorById = Object.fromEntries(
      model.orderedIds.map((elementId) => [
        elementId,
        cloneGraphElementDescriptor(model.descriptorsById[elementId]),
      ])
    );
    state.graphRenderDescriptorOrder = [...model.orderedIds];
    state.graphRenderVisibleSignature = model.visibleSignature;
    state.graphRenderEphemeralSignature = model.ephemeralSignature;
    state.graphRenderDescriptorRevision = state.specRevision;
  }

  function resetForCurrentCy() {
    state.graphRenderCyRef = getCy();
    state.graphRenderDescriptorById = {};
    state.graphRenderDescriptorOrder = [];
    state.graphRenderVisibleSignature = null;
    state.graphRenderEphemeralSignature = null;
    state.graphRenderDescriptorRevision = -1;
    state.cySelectionSyncedIds = [];
    state.pendingInteractionRenderedPlannerSelectionId = null;
    state.pendingInteractionRenderedIndexId = null;
  }

  function ensureForCurrentCy() {
    if (state.graphRenderCyRef !== getCy()) {
      resetForCurrentCy();
    }
  }

  function applyModel(model) {
    const cy = getCy();
    if (!cy) {
      return;
    }
    if (!state.graphRenderDescriptorOrder.length) {
      if (model.elements.length) {
        cy.add(model.elements);
      }
      updateCache(model);
      return;
    }
    const updatePlan = buildGraphElementUpdatePlan({
      previousDescriptorsById: state.graphRenderDescriptorById || {},
      nextModel: model,
    });
    updatePlan.removedIds.forEach((elementId) => {
      const element = cy.getElementById(elementId);
      if (element && element.length) {
        element.remove();
      }
    });
    updatePlan.updatedDescriptors.forEach((update) => {
      const element = cy.getElementById(update.id);
      if (!element || !element.length) {
        updatePlan.addedDescriptors.push(update.nextDescriptor);
        return;
      }
      if (update.shouldUpdateData) {
        element.data(update.nextDescriptor.data);
      }
      if (update.shouldUpdatePosition) {
        element.position(update.nextDescriptor.position);
      }
      if (update.shouldUpdateClasses) {
        element.classes(update.nextDescriptor.classes || "");
      }
      if (update.shouldUpdateSelectable) {
        element.selectable(update.nextDescriptor.selectable);
      }
      if (update.shouldUpdateGrabbable) {
        element.grabbable(update.nextDescriptor.grabbable);
      }
    });
    if (updatePlan.addedDescriptors.length) {
      cy.add(updatePlan.addedDescriptors);
    }
    updateCache(model);
  }

  return {
    applyModel,
    ensureForCurrentCy,
    resetForCurrentCy,
    updateCache,
  };
}
