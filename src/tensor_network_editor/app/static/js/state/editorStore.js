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

  function setAppMetadata(appMetadata) {
    state.appMetadata =
      appMetadata && typeof appMetadata === "object" ? { ...appMetadata } : {};
    return state.appMetadata;
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

  function setIncludeRoundtripMetadata(includeRoundtripMetadata) {
    state.includeRoundtripMetadata = Boolean(includeRoundtripMetadata);
    return state.includeRoundtripMetadata;
  }

  function setSelectedTheme(themeName) {
    state.selectedTheme = typeof themeName === "string" ? themeName : state.selectedTheme;
    return state.selectedTheme;
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
    state.catalogTemplateNames = Array.isArray(templateNames) ? [...templateNames] : [];
    state.catalogTemplateDefinitions =
      templateDefinitions && typeof templateDefinitions === "object"
        ? { ...templateDefinitions }
        : {};
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

  function setSubnetworkCatalogData({
    subnetworkNames,
    subnetworkDefinitions,
    subnetworkCatalogWarnings,
    selectedSubnetworkName = null,
  }) {
    state.availableSubnetworks = Array.isArray(subnetworkNames)
      ? [...subnetworkNames]
      : [];
    state.subnetworkDefinitions =
      subnetworkDefinitions && typeof subnetworkDefinitions === "object"
        ? { ...subnetworkDefinitions }
        : {};
    state.subnetworkCatalogWarnings = Array.isArray(subnetworkCatalogWarnings)
      ? [...subnetworkCatalogWarnings]
      : [];
    if (
      typeof selectedSubnetworkName === "string"
      && state.availableSubnetworks.includes(selectedSubnetworkName)
    ) {
      state.selectedSubnetworkName = selectedSubnetworkName;
    } else if (
      state.selectedSubnetworkName &&
      state.availableSubnetworks.includes(state.selectedSubnetworkName)
    ) {
      state.selectedSubnetworkName = state.selectedSubnetworkName;
    } else {
      state.selectedSubnetworkName = state.availableSubnetworks[0] || "";
    }
    if (
      state.subnetworkLibraryTagFilter
      && !Object.values(state.subnetworkDefinitions).some(
        (definition) =>
          definition &&
          Array.isArray(definition.tags) &&
          definition.tags.includes(state.subnetworkLibraryTagFilter)
      )
    ) {
      state.subnetworkLibraryTagFilter = "";
    }
    state.selectedSubnetworkLibraryNames = Array.isArray(
      state.selectedSubnetworkLibraryNames
    )
      ? state.selectedSubnetworkLibraryNames.filter((subnetworkName) =>
          state.availableSubnetworks.includes(subnetworkName)
        )
      : [];
    return {
      availableSubnetworks: state.availableSubnetworks,
      subnetworkDefinitions: state.subnetworkDefinitions,
      subnetworkCatalogWarnings: state.subnetworkCatalogWarnings,
      selectedSubnetworkName: state.selectedSubnetworkName,
    };
  }

  return {
    getState,
    setSpec,
    setSchemaVersion,
    setAppMetadata,
    setAvailableCollectionFormats,
    setAnnotationDefinitions,
    setSelectedEngine,
    setSelectedCollectionFormat,
    setIncludeRoundtripMetadata,
    setSelectedTheme,
    setGeneratedCode,
    setEditorFinished,
    setLastImportedTensorIds,
    setTemplateCatalogData,
    setSubnetworkCatalogData,
  };
}
