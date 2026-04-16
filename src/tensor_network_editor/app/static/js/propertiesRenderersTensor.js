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
    bindSuggestedAnnotationEditors,
    buildMetadataEditorMarkup,
    buildSuggestedAnnotationsMarkup,
    commands,
    propertyInvalidation,
    renderTrashIcon,
    getTensorTotalElementCount,
    getContractionTensorTotalElementCount,
    formatTotalElementCount,
    isTensorIndexDisclosureOpen,
    toggleTensorIndexDisclosure,
  } = support;

  const tensorAnnotationInputId = (key) => `tensor-annotation-${key}-input`;
  const tensorAnnotationFocusKey = (tensorId, key) =>
    `tensor:${tensorId}:annotation:${key}`;
  const tensorAnnotationSuggestionButtonId = (key, suggestion) =>
    `tensor-annotation-${key}-suggestion-${ctx.sanitizeFilename(suggestion)}`;
  const indexAnnotationInputId = (indexId, key) =>
    `index-annotation-${key}-input-${indexId}`;
  const indexAnnotationFocusKey = (indexId, key) =>
    `index:${indexId}:annotation:${key}`;
  const indexAnnotationSuggestionButtonId = (indexId, key, suggestion) =>
    `index-annotation-${key}-suggestion-${ctx.sanitizeFilename(suggestion)}-${indexId}`;

  const rendererDependencies = {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    bindSuggestedAnnotationEditors,
    buildMetadataEditorMarkup,
    buildSuggestedAnnotationsMarkup,
    commands,
    ctx,
    document,
    formatTotalElementCount,
    getContractionTensorTotalElementCount,
    getTensorTotalElementCount,
    indexAnnotationFocusKey,
    indexAnnotationInputId,
    indexAnnotationSuggestionButtonId,
    isTensorIndexDisclosureOpen,
    propertiesPanel,
    propertyInvalidation,
    renderTrashIcon,
    tensorAnnotationFocusKey,
    tensorAnnotationInputId,
    tensorAnnotationSuggestionButtonId,
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
    if (ctx.isLinearPeriodicBoundaryTensor(tensor)) {
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
