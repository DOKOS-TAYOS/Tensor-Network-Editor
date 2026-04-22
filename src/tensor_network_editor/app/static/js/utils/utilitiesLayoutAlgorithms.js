import { createLayoutAlgorithmGraphSupport } from "./utilitiesLayoutAlgorithmsGraph.js";
import { createLayoutAlgorithmPositionSupport } from "./utilitiesLayoutAlgorithmsPositions.js";

export function createUtilityLayoutAlgorithmSupport({
  ctx,
  state,
  constants,
  selection,
}) {
  const layoutMetrics = {
    LAYOUT_HORIZONTAL_GAP:
      Number.isFinite(constants && constants.LAYOUT_HORIZONTAL_GAP) &&
      constants.LAYOUT_HORIZONTAL_GAP >= 0
        ? constants.LAYOUT_HORIZONTAL_GAP
        : 80,
    LAYOUT_VERTICAL_GAP:
      Number.isFinite(constants && constants.LAYOUT_VERTICAL_GAP) &&
      constants.LAYOUT_VERTICAL_GAP >= 0
        ? constants.LAYOUT_VERTICAL_GAP
        : 100,
    LAYOUT_COMPONENT_GAP:
      Number.isFinite(constants && constants.LAYOUT_COMPONENT_GAP) &&
      constants.LAYOUT_COMPONENT_GAP >= 0
        ? constants.LAYOUT_COMPONENT_GAP
        : 140,
    LAYOUT_NON_OVERLAP_GAP:
      Number.isFinite(constants && constants.LAYOUT_NON_OVERLAP_GAP) &&
      constants.LAYOUT_NON_OVERLAP_GAP >= 0
        ? constants.LAYOUT_NON_OVERLAP_GAP
        : 36,
  };

  const positionSupport = createLayoutAlgorithmPositionSupport({
    ctx,
    layoutMetrics,
  });
  const graphSupport = createLayoutAlgorithmGraphSupport({
    ctx,
    state,
    selection,
    positionSupport,
  });

  return {
    buildAlignedTensorPositions: positionSupport.buildAlignedTensorPositions,
    buildArrangedSelectionPositions: graphSupport.buildArrangedSelectionPositions,
    buildAutoLayoutPositions: graphSupport.buildAutoLayoutPositions,
    buildImportedReflowPositions: graphSupport.buildImportedReflowPositions,
    computeTensorBounds: positionSupport.computeTensorBounds,
  };
}
