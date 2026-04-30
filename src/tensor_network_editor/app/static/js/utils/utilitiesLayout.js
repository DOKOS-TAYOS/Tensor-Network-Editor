import { createUtilityLayoutAlgorithmSupport } from "./utilitiesLayoutAlgorithms.js";
import { createUtilityLayoutIndexSupport } from "./utilitiesLayoutIndices.js";
import { createUtilityLayoutSelectionSupport } from "./utilitiesLayoutSelection.js";

export function createUtilityLayoutBindings({ ctx, state, constants }) {
  const GRID_SNAP_SIZE =
    Number.isFinite(constants && constants.GRID_SNAP_SIZE) &&
    constants.GRID_SNAP_SIZE > 0
      ? constants.GRID_SNAP_SIZE
      : 20;

  const selection = createUtilityLayoutSelectionSupport({ ctx, state });
  const algorithms = createUtilityLayoutAlgorithmSupport({
    ctx,
    state,
    constants,
    selection,
  });
  const indices = createUtilityLayoutIndexSupport({ ctx, constants });
  const {
    applyIndexLayoutChangeForIds,
    applyTensorLayoutChange,
    applyTensorPositions,
    getLayoutTensorsById,
    getSelectedLayoutTensorIds,
    getSelectedLayoutTensors,
  } = selection;
  const {
    buildAlignedTensorPositions,
    buildArrangedSelectionPositions,
    buildAutoLayoutPositions,
    buildImportedReflowPositions,
    computeTensorBounds,
  } = algorithms;
  const { buildReflowIndexOffsets } = indices;

  function alignSelectedTensors(mode) {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to align.");
      return false;
    }

    const bounds = computeTensorBounds(tensors);
    const centerX = (bounds.left + bounds.right) / 2;
    const centerY = (bounds.top + bounds.bottom) / 2;
    const statusLabels = {
      left: "Aligned tensors to the left.",
      center: "Aligned tensor centers horizontally.",
      right: "Aligned tensors to the right.",
      top: "Aligned tensors to the top.",
      middle: "Aligned tensor centers vertically.",
      bottom: "Aligned tensors to the bottom.",
    };

    const targetPositions = buildAlignedTensorPositions(tensors, mode, {
      bounds,
      centerX,
      centerY,
    });
    if (!targetPositions) {
      ctx.setStatus(`Unknown alignment mode '${mode}'.`, "error");
      return false;
    }
    return applyTensorPositions(
      tensors.map((tensor) => tensor.id),
      targetPositions,
      statusLabels[mode] || "Aligned tensors."
    );
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
    const start =
      axis === "vertical" ? firstTensor.position.y : firstTensor.position.x;
    const end = axis === "vertical" ? lastTensor.position.y : lastTensor.position.x;
    const step = (end - start) / (sortedTensors.length - 1);

    return applyTensorLayoutChange(
      () => {
        sortedTensors.forEach((tensor, index) => {
          if (axis === "vertical") {
            tensor.position.y = start + step * index;
          } else {
            tensor.position.x = start + step * index;
          }
        });
      },
      axis === "vertical"
        ? "Distributed tensors vertically."
        : "Distributed tensors horizontally."
    );
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

  function rotateSelectedTensorsClockwise() {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to rotate.");
      return false;
    }

    const bounds = computeTensorBounds(tensors);
    const centerX = (bounds.left + bounds.right) / 2;
    const centerY = (bounds.top + bounds.bottom) / 2;

    return applyTensorLayoutChange(() => {
      tensors.forEach((tensor) => {
        const deltaX = tensor.position.x - centerX;
        const deltaY = tensor.position.y - centerY;
        tensor.position.x = centerX - deltaY;
        tensor.position.y = centerY + deltaX;
        tensor.indices.forEach((index) => {
          const rotatedOffset = {
            x: -index.offset.y,
            y: index.offset.x,
          };
          index.offset =
            typeof ctx.clampIndexOffset === "function"
              ? ctx.clampIndexOffset(rotatedOffset, tensor)
              : rotatedOffset;
        });
      });
    }, "Rotated the selected tensors and ports 90° clockwise.");
  }

  function arrangeSelectedTensors(mode) {
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 2) {
      ctx.setStatus("Select at least two tensors to arrange.");
      return false;
    }
    const targetPositions = buildArrangedSelectionPositions(tensorIds, mode);
    if (!targetPositions) {
      ctx.setStatus(`Unknown layout mode '${mode}'.`, "error");
      return false;
    }
    return applyTensorPositions(
      tensorIds,
      targetPositions,
      {
        chain: "Arranged selection as a chain.",
        tree: "Arranged selection as a tree.",
        grid: "Arranged selection as a grid.",
      }[mode] || "Arranged the selected tensors."
    );
  }

  function applyAutoLayout() {
    const selectedTensorIds = getSelectedLayoutTensorIds();
    const graphTensorIds = Array.isArray(state.spec && state.spec.tensors)
      ? state.spec.tensors.map((tensor) => tensor.id)
      : [];
    const tensorIds = selectedTensorIds.length ? selectedTensorIds : graphTensorIds;
    if (tensorIds.length < 2) {
      ctx.setStatus("Add or select at least two tensors first.");
      return false;
    }
    const targetPositions = buildAutoLayoutPositions(
      tensorIds,
      tensorIds.includes(state.primarySelectionId) ? state.primarySelectionId : null
    );
    return applyTensorPositions(
      tensorIds,
      targetPositions,
      selectedTensorIds.length
        ? "Auto-arranged the selected tensors."
        : "Auto-arranged the whole graph.",
      {
        selectionIds: Array.isArray(state.selectionIds) ? [...state.selectionIds] : [],
        primaryId: state.primarySelectionId,
      }
    );
  }

  function applyReflowLayoutAction(layoutAction) {
    const action = typeof layoutAction === "string" ? layoutAction : "";
    if (action === "align-horizontal") {
      return alignSelectedTensors("middle");
    }
    if (action === "align-vertical") {
      return alignSelectedTensors("center");
    }
    if (action === "rotate-90") {
      return rotateSelectedTensorsClockwise();
    }
    if (
      action === "left" ||
      action === "center" ||
      action === "right" ||
      action === "top" ||
      action === "middle" ||
      action === "bottom"
    ) {
      return alignSelectedTensors(action);
    }
    if (action === "chain" || action === "tree" || action === "grid") {
      return arrangeSelectedTensors(action);
    }
    if (action === "horizontal" || action === "vertical") {
      return distributeSelectedTensors(action);
    }
    if (action === "snap") {
      return snapSelectedTensorsToGrid();
    }
    if (action === "auto") {
      return applyAutoLayout();
    }
    if (action === "smart") {
      return reflowLastImportedTensors();
    }
    ctx.setStatus(`Unknown reflow action '${action}'.`, "error");
    return false;
  }

  function applyReflowIndicesAction(layoutAction) {
    const action = typeof layoutAction === "string" ? layoutAction : "";
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 1) {
      ctx.setStatus("Select at least one tensor to reflow indices.");
      return false;
    }
    const tensors = getLayoutTensorsById(tensorIds).filter(
      (tensor) => Array.isArray(tensor.indices) && tensor.indices.length > 0
    );
    if (!tensors.length) {
      ctx.setStatus("The selected tensors have no indices to reflow.");
      return false;
    }
    const targetOffsets = buildReflowIndexOffsets(tensors, action);
    if (!targetOffsets) {
      ctx.setStatus(`Unknown index reflow action '${action}'.`, "error");
      return false;
    }
    return applyIndexLayoutChangeForIds(
      tensorIds,
      () => {
        tensors.forEach((tensor) => {
          const tensorOffsets = targetOffsets[tensor.id];
          if (!Array.isArray(tensorOffsets)) {
            return;
          }
          tensor.indices.forEach((index, indexPosition) => {
            const targetOffset = tensorOffsets[indexPosition];
            if (!targetOffset) {
              return;
            }
            index.offset = targetOffset;
          });
        });
      },
      {
        left: "Moved selected tensor indices to the left.",
        right: "Moved selected tensor indices to the right.",
        top: "Moved selected tensor indices to the top.",
        bottom: "Moved selected tensor indices to the bottom.",
        reset: "Reset selected tensor indices.",
      }[action] || "Reflowed the selected tensor indices."
    );
  }

  function reflowLastImportedTensors() {
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 2) {
      ctx.setStatus("Select at least two tensors to reflow.");
      return false;
    }
    const targetPositions = buildImportedReflowPositions(tensorIds);
    return applyTensorPositions(
      tensorIds,
      targetPositions,
      "Reflowed the selected tensors."
    );
  }

  return {
    GRID_SNAP_SIZE,
    alignSelectedTensors,
    applyReflowIndicesAction,
    applyReflowLayoutAction,
    applyAutoLayout,
    arrangeSelectedTensors,
    distributeSelectedTensors,
    reflowLastImportedTensors,
    rotateSelectedTensorsClockwise,
    snapSelectedTensorsToGrid,
  };
}
