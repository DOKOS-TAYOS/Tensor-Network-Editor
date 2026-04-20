import { getTemplateDefinition } from "../utils/utilitiesTemplates.js";

export function createEditorSelectors({ store }) {
  function getState() {
    return store.getState();
  }

  function getTemplateDefinitionByName(templateName) {
    return getTemplateDefinition(getState().templateDefinitions, templateName);
  }

  function getTemplateSource(templateName) {
    const definition = getTemplateDefinitionByName(templateName);
    return definition && typeof definition.source === "string"
      ? definition.source
      : "global";
  }

  function isProjectTemplate(templateName) {
    return getTemplateSource(templateName) === "project";
  }

  function isSessionTemplate(templateName) {
    return getTemplateSource(templateName) === "session";
  }

  function hasTemplateCatalogWarnings() {
    return Array.isArray(getState().templateCatalogWarnings)
      && getState().templateCatalogWarnings.length > 0;
  }

  function getSelectedEngine() {
    return getState().selectedEngine;
  }

  function getSelectedCollectionFormat() {
    return getState().selectedCollectionFormat;
  }

  return {
    getTemplateDefinition: getTemplateDefinitionByName,
    getTemplateSource,
    isProjectTemplate,
    isSessionTemplate,
    hasTemplateCatalogWarnings,
    getSelectedEngine,
    getSelectedCollectionFormat,
  };
}
