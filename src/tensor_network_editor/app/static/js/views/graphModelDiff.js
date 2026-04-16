import {
  graphElementDataEqual,
  graphElementDescriptorsEqual,
  graphElementPositionEqual,
} from "./graphDescriptors.js";

export function buildGraphElementUpdatePlan({
  previousDescriptorsById = {},
  nextModel,
}) {
  const nextDescriptorsById = nextModel && nextModel.descriptorsById ? nextModel.descriptorsById : {};
  const orderedIds = Array.isArray(nextModel && nextModel.orderedIds) ? nextModel.orderedIds : [];
  const nextIdSet = new Set(orderedIds);
  const removedIds = [];
  const addedDescriptors = [];
  const updatedDescriptors = [];

  Object.keys(previousDescriptorsById).forEach((elementId) => {
    if (!nextIdSet.has(elementId)) {
      removedIds.push(elementId);
    }
  });

  orderedIds.forEach((elementId) => {
    const nextDescriptor = nextDescriptorsById[elementId];
    const previousDescriptor = previousDescriptorsById[elementId];
    if (!previousDescriptor) {
      addedDescriptors.push(nextDescriptor);
      return;
    }
    if (graphElementDescriptorsEqual(previousDescriptor, nextDescriptor)) {
      return;
    }
    updatedDescriptors.push({
      id: elementId,
      nextDescriptor,
      previousDescriptor,
      shouldUpdateClasses: previousDescriptor.classes !== nextDescriptor.classes,
      shouldUpdateData: !graphElementDataEqual(previousDescriptor.data, nextDescriptor.data),
      shouldUpdateGrabbable:
        previousDescriptor.grabbable !== nextDescriptor.grabbable,
      shouldUpdatePosition: !graphElementPositionEqual(
        previousDescriptor.position,
        nextDescriptor.position
      ),
      shouldUpdateSelectable:
        previousDescriptor.selectable !== nextDescriptor.selectable,
    });
  });

  return {
    addedDescriptors,
    removedIds,
    updatedDescriptors,
  };
}
