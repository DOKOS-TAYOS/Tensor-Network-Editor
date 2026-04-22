import { createStandardTensorPropertiesBindingSupport } from "./tensorPropertiesStandardBindings.js";
import { createStandardTensorDataSupport } from "./tensorPropertiesStandardData.js";
import { createStandardTensorPropertiesMarkupSupport } from "./tensorPropertiesStandardMarkup.js";

export function createStandardTensorPropertiesRenderer({
  bindDebouncedAutosave,
  bindImmediateAutosave,
  bindMetadataEditors,
  buildMetadataEditorMarkup,
  commands,
  ctx,
  document,
  formatTotalElementCount,
  getTensorTotalElementCount,
  isTensorIndexDisclosureOpen,
  propertiesPanel,
  propertyInvalidation,
  renderTrashIcon,
  toggleTensorIndexDisclosure,
}) {
  const dataSupport = createStandardTensorDataSupport();
  const markupSupport = createStandardTensorPropertiesMarkupSupport({
    ctx,
    buildMetadataEditorMarkup,
    formatTotalElementCount,
    getTensorTotalElementCount,
    isTensorIndexDisclosureOpen,
    renderTrashIcon,
    dataSupport,
  });
  const bindingSupport = createStandardTensorPropertiesBindingSupport({
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    commands,
    ctx,
    document,
    propertiesPanel,
    propertyInvalidation,
    toggleTensorIndexDisclosure,
    dataSupport,
  });

  function renderTensorProperties(tensor, options = {}) {
    propertiesPanel.innerHTML = markupSupport.renderTensorPropertiesMarkup(
      tensor,
      options
    );
    bindingSupport.bindStandardTensorProperties(tensor);
  }

  return {
    renderTensorProperties,
  };
}
