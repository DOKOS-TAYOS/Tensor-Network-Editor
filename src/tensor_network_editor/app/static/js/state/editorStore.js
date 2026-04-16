export function createEditorStore(state) {
  function getState() {
    return state;
  }

  function setSpec(spec) {
    state.spec = spec;
    return state.spec;
  }

  function setSchemaVersion(schemaVersion) {
    state.schemaVersion = schemaVersion;
    return state.schemaVersion;
  }

  function setAvailableCollectionFormats(collectionFormats) {
    state.availableCollectionFormats = Array.isArray(collectionFormats)
      ? [...collectionFormats]
      : [];
    return state.availableCollectionFormats;
  }

  function setAnnotationDefinitions(annotationDefinitions) {
    state.annotationDefinitions =
      annotationDefinitions && typeof annotationDefinitions === "object"
        ? { ...annotationDefinitions }
        : {};
    return state.annotationDefinitions;
  }

  function setSelectedEngine(engine) {
    state.selectedEngine = engine;
    return state.selectedEngine;
  }

  function setSelectedCollectionFormat(collectionFormat) {
    state.selectedCollectionFormat = collectionFormat;
    return state.selectedCollectionFormat;
  }

  function setGeneratedCode(code) {
    state.generatedCode = typeof code === "string" ? code : "";
    return state.generatedCode;
  }

  function setEditorFinished(editorFinished) {
    state.editorFinished = Boolean(editorFinished);
    return state.editorFinished;
  }

  function setLastImportedTensorIds(tensorIds) {
    state.lastImportedTensorIds = Array.isArray(tensorIds) ? [...tensorIds] : [];
    return state.lastImportedTensorIds;
  }

  function setTemplateCatalogData({
    templateNames,
    templateDefinitions,
    templateCatalogWarnings,
  }) {
    state.availableTemplates = Array.isArray(templateNames) ? [...templateNames] : [];
    state.templateDefinitions =
      templateDefinitions && typeof templateDefinitions === "object"
        ? { ...templateDefinitions }
        : {};
    state.templateCatalogWarnings = Array.isArray(templateCatalogWarnings)
      ? [...templateCatalogWarnings]
      : [];
    return {
      availableTemplates: state.availableTemplates,
      templateDefinitions: state.templateDefinitions,
      templateCatalogWarnings: state.templateCatalogWarnings,
    };
  }

  return {
    getState,
    setSpec,
    setSchemaVersion,
    setAvailableCollectionFormats,
    setAnnotationDefinitions,
    setSelectedEngine,
    setSelectedCollectionFormat,
    setGeneratedCode,
    setEditorFinished,
    setLastImportedTensorIds,
    setTemplateCatalogData,
  };
}
