export function createUtilityLayoutBindings({ ctx, state, constants }) {
  const GRID_SNAP_SIZE =
    Number.isFinite(constants && constants.GRID_SNAP_SIZE) &&
    constants.GRID_SNAP_SIZE > 0
      ? constants.GRID_SNAP_SIZE
      : 20;

  function getSelectedLayoutTensorIds() {
    return typeof ctx.getSelectedIdsByKind === "function"
      ? ctx.getSelectedIdsByKind("tensor")
      : [];
  }

  function getSelectedLayoutTensors() {
    return getSelectedLayoutTensorIds()
      .map((tensorId) => ctx.findTensorById(tensorId))
      .filter(Boolean);
  }

  function applyTensorLayoutChange(mutator, statusMessage) {
    const selectionIds = getSelectedLayoutTensorIds();
    if (selectionIds.length < 2 || typeof ctx.applyDesignChange !== "function") {
      return false;
    }
    ctx.applyDesignChange(mutator, {
      invalidate: {
        lookups: false,
        analysis: false,
      },
      selectionIds,
      primaryId: selectionIds.includes(state.primarySelectionId)
        ? state.primarySelectionId
        : selectionIds[selectionIds.length - 1],
      statusMessage,
    });
    return true;
  }

  function alignSelectedTensors(mode) {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to align.");
      return false;
    }

    const left = Math.min(
      ...tensors.map((tensor) => tensor.position.x - ctx.tensorWidth(tensor) / 2)
    );
    const right = Math.max(
      ...tensors.map((tensor) => tensor.position.x + ctx.tensorWidth(tensor) / 2)
    );
    const top = Math.min(
      ...tensors.map((tensor) => tensor.position.y - ctx.tensorHeight(tensor) / 2)
    );
    const bottom = Math.max(
      ...tensors.map((tensor) => tensor.position.y + ctx.tensorHeight(tensor) / 2)
    );
    const centerX = (left + right) / 2;
    const centerY = (top + bottom) / 2;
    const statusLabels = {
      left: "Aligned tensors to the left.",
      center: "Aligned tensor centers horizontally.",
      right: "Aligned tensors to the right.",
      top: "Aligned tensors to the top.",
      middle: "Aligned tensor centers vertically.",
      bottom: "Aligned tensors to the bottom.",
    };

    return applyTensorLayoutChange(() => {
      tensors.forEach((tensor) => {
        if (mode === "left") {
          tensor.position.x = left + ctx.tensorWidth(tensor) / 2;
        } else if (mode === "center") {
          tensor.position.x = centerX;
        } else if (mode === "right") {
          tensor.position.x = right - ctx.tensorWidth(tensor) / 2;
        } else if (mode === "top") {
          tensor.position.y = top + ctx.tensorHeight(tensor) / 2;
        } else if (mode === "middle") {
          tensor.position.y = centerY;
        } else if (mode === "bottom") {
          tensor.position.y = bottom - ctx.tensorHeight(tensor) / 2;
        }
      });
    }, statusLabels[mode] || "Aligned tensors.");
  }

  function distributeSelectedTensors(axis) {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 3) {
      ctx.setStatus("Select at least three tensors to distribute.");
      return false;
    }

    const sortedTensors = [...tensors].sort((leftTensor, rightTensor) =>
      axis === "vertical"
        ? leftTensor.position.y - rightTensor.position.y
        : leftTensor.position.x - rightTensor.position.x
    );
    const firstTensor = sortedTensors[0];
    const lastTensor = sortedTensors[sortedTensors.length - 1];
    const start = axis === "vertical" ? firstTensor.position.y : firstTensor.position.x;
    const end = axis === "vertical" ? lastTensor.position.y : lastTensor.position.x;
    const step = (end - start) / (sortedTensors.length - 1);

    return applyTensorLayoutChange(() => {
      sortedTensors.forEach((tensor, index) => {
        if (axis === "vertical") {
          tensor.position.y = start + step * index;
        } else {
          tensor.position.x = start + step * index;
        }
      });
    }, axis === "vertical"
      ? "Distributed tensors vertically."
      : "Distributed tensors horizontally.");
  }

  function snapSelectedTensorsToGrid() {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to snap.");
      return false;
    }

    return applyTensorLayoutChange(() => {
      tensors.forEach((tensor) => {
        tensor.position.x =
          Math.round(tensor.position.x / GRID_SNAP_SIZE) * GRID_SNAP_SIZE;
        tensor.position.y =
          Math.round(tensor.position.y / GRID_SNAP_SIZE) * GRID_SNAP_SIZE;
      });
    }, "Snapped tensors to the grid.");
  }

  return {
    GRID_SNAP_SIZE,
    alignSelectedTensors,
    distributeSelectedTensors,
    snapSelectedTensorsToGrid,
  };
}
