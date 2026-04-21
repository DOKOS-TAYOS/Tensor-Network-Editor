export function createUtilityLayoutSelectionSupport({ ctx, state }) {
  function getSelectedLayoutTensorIds() {
    return typeof ctx.getSelectedIdsByKind === "function"
      ? ctx.getSelectedIdsByKind("tensor")
      : [];
  }

  function getLayoutTensorsById(tensorIds) {
    return tensorIds
      .map((tensorId) => ctx.findTensorById(tensorId))
      .filter(Boolean);
  }

  function getSelectedLayoutTensors() {
    return getLayoutTensorsById(getSelectedLayoutTensorIds());
  }

  function applyTensorLayoutChangeForIds(
    tensorIds,
    mutator,
    statusMessage,
    primaryId = state.primarySelectionId
  ) {
    if (
      !Array.isArray(tensorIds) ||
      tensorIds.length < 2 ||
      typeof ctx.applyDesignChange !== "function"
    ) {
      return false;
    }
    ctx.applyDesignChange(mutator, {
      invalidate: {
        lookups: false,
        analysis: false,
      },
      selectionIds: [...tensorIds],
      primaryId: tensorIds.includes(primaryId)
        ? primaryId
        : tensorIds[tensorIds.length - 1],
      statusMessage,
    });
    return true;
  }

  function applyTensorLayoutChange(mutator, statusMessage) {
    return applyTensorLayoutChangeForIds(
      getSelectedLayoutTensorIds(),
      mutator,
      statusMessage
    );
  }

  function applyIndexLayoutChangeForIds(
    tensorIds,
    mutator,
    statusMessage,
    primaryId = state.primarySelectionId
  ) {
    if (
      !Array.isArray(tensorIds) ||
      tensorIds.length < 1 ||
      typeof ctx.applyDesignChange !== "function"
    ) {
      return false;
    }
    ctx.applyDesignChange(mutator, {
      invalidate: {
        lookups: false,
        analysis: false,
      },
      selectionIds: [...tensorIds],
      primaryId: tensorIds.includes(primaryId)
        ? primaryId
        : tensorIds[tensorIds.length - 1],
      statusMessage,
    });
    return true;
  }

  function applyTensorPositions(tensorIds, targetPositions, statusMessage) {
    if (!targetPositions) {
      return false;
    }
    return applyTensorLayoutChangeForIds(
      tensorIds,
      () => {
        tensorIds.forEach((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const targetPosition = targetPositions[tensorId];
          if (!tensor || !targetPosition) {
            return;
          }
          tensor.position.x = targetPosition.x;
          tensor.position.y = targetPosition.y;
        });
      },
      statusMessage
    );
  }

  return {
    applyIndexLayoutChangeForIds,
    applyTensorLayoutChange,
    applyTensorLayoutChangeForIds,
    applyTensorPositions,
    getLayoutTensorsById,
    getSelectedLayoutTensorIds,
    getSelectedLayoutTensors,
  };
}
