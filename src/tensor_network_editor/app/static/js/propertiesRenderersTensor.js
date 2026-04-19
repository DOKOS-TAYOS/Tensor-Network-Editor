import { createBoundaryTensorPropertiesRenderer } from "./properties/tensorPropertiesBoundary.js";
import { createContractionTensorPropertiesRenderer } from "./properties/tensorPropertiesContraction.js";
import { createStandardTensorPropertiesRenderer } from "./properties/tensorPropertiesStandard.js";

export function createTensorPropertiesRenderers({
  ctx,
  document,
  propertiesPanel,
  support,
}) {
  const {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    buildMetadataEditorMarkup,
    commands,
    propertyInvalidation,
    renderTrashIcon,
    getTensorTotalElementCount,
    getContractionTensorTotalElementCount,
    formatTotalElementCount,
    isTensorIndexDisclosureOpen,
    toggleTensorIndexDisclosure,
  } = support;

  const rendererDependencies = {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    buildMetadataEditorMarkup,
    commands,
    ctx,
    document,
    formatTotalElementCount,
    getContractionTensorTotalElementCount,
    getTensorTotalElementCount,
    isTensorIndexDisclosureOpen,
    propertiesPanel,
    propertyInvalidation,
    renderTrashIcon,
    toggleTensorIndexDisclosure,
  };

  const contractionRenderers = createContractionTensorPropertiesRenderer(
    rendererDependencies
  );
  const boundaryRenderers = createBoundaryTensorPropertiesRenderer(
    rendererDependencies
  );
  const standardRenderers = createStandardTensorPropertiesRenderer(
    rendererDependencies
  );

  const renderTensorProperties = (tensorId, options = {}) => {
    const tensor = ctx.findTensorById(tensorId);
    if (!tensor) {
      ctx.clearSelection();
      return;
    }
    if (
      (typeof ctx.isForBoundaryTensor === "function" && ctx.isForBoundaryTensor(tensor)) ||
      ctx.isLinearPeriodicBoundaryTensor(tensor)
    ) {
      boundaryRenderers.renderLinearPeriodicBoundaryTensorProperties(tensor);
      return;
    }
    standardRenderers.renderTensorProperties(tensor, options);
  };

  return {
    renderContractionIndexProperties:
      contractionRenderers.renderContractionIndexProperties,
    renderContractionTensorProperties:
      contractionRenderers.renderContractionTensorProperties,
    renderLinearPeriodicBoundaryTensorProperties:
      boundaryRenderers.renderLinearPeriodicBoundaryTensorProperties,
    renderTensorProperties,
  };
}
