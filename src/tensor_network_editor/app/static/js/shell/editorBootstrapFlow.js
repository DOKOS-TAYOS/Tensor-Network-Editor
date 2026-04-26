const READY_STATUS_MESSAGE =
  "Editor ready. Drag the canvas to move, use Ctrl+wheel to zoom, use the wheel to pan, and right drag to box-select.";

export function createEditorBootstrapFlow({
  state,
  store,
  sessionService,
  actions,
  confirmAction = null,
}) {
  async function bootstrap() {
    const payload = await sessionService.loadBootstrap();
    const draftPayload = await loadRecoverableDraft();
    const restoredDraft = await chooseRecoverableDraft(draftPayload);
    const activeSpecPayload = restoredDraft?.spec || payload.spec;
    const activeEngine = restoredDraft?.engine || payload.default_engine;
    const activeCollectionFormat =
      restoredDraft?.collection_format || payload.default_collection_format || "list";
    store.setSpec(actions.normalizeSpec(activeSpecPayload.network));
    store.setSchemaVersion(activeSpecPayload.schema_version || payload.schema_version);
    if (typeof store.setAppMetadata === "function") {
      store.setAppMetadata(payload.app_metadata);
    } else {
      state.appMetadata =
        payload.app_metadata && typeof payload.app_metadata === "object"
          ? { ...payload.app_metadata }
          : {};
    }
    store.setAvailableCollectionFormats(
      Array.isArray(payload.collection_formats) ? payload.collection_formats : ["list"]
    );
    actions.applyTemplateCatalogPayload({
      templateNames: payload.templates,
      templateDefinitions: payload.template_definitions,
      templateCatalogWarnings: payload.template_catalog_warnings,
    });
    if (typeof store.setSubnetworkCatalogData === "function") {
      store.setSubnetworkCatalogData({
        subnetworkNames: payload.subnetworks,
        subnetworkDefinitions: payload.subnetwork_definitions,
        subnetworkCatalogWarnings: payload.subnetwork_catalog_warnings,
        selectedSubnetworkName: payload.selected_subnetwork,
      });
    } else {
      state.availableSubnetworks = Array.isArray(payload.subnetworks)
        ? [...payload.subnetworks]
        : [];
      state.subnetworkDefinitions =
        payload.subnetwork_definitions &&
        typeof payload.subnetwork_definitions === "object"
          ? { ...payload.subnetwork_definitions }
          : {};
      state.subnetworkCatalogWarnings = Array.isArray(
        payload.subnetwork_catalog_warnings
      )
        ? [...payload.subnetwork_catalog_warnings]
        : [];
      state.selectedSubnetworkName =
        typeof payload.selected_subnetwork === "string"
          ? payload.selected_subnetwork
          : state.availableSubnetworks[0] || "";
      state.selectedSubnetworkLibraryNames = Array.isArray(
        state.selectedSubnetworkLibraryNames
      )
        ? state.selectedSubnetworkLibraryNames.filter((subnetworkName) =>
            state.availableSubnetworks.includes(subnetworkName)
          )
        : [];
    }
    store.setAnnotationDefinitions(payload.annotation_definitions);
    store.setSelectedEngine(activeEngine);
    store.setSelectedCollectionFormat(activeCollectionFormat);
    actions.reconcileTensorOrder();
    actions.populateEngineOptions(payload.engines);
    actions.enforceLinearPeriodicEngineSupport();
    actions.populateCollectionFormatOptions(state.availableCollectionFormats);
    actions.initGraph();
    actions.clearHistory();
    actions.render();
    state.draftAutosaveReady = true;
    if (typeof actions.markContractionAnalysisDirty === "function") {
      actions.markContractionAnalysisDirty();
    } else {
      state.contractionAnalysisDirty = true;
    }
    if (state.templateCatalogWarnings.length) {
      actions.setStatus(state.templateCatalogWarnings[0], "error");
    } else if (state.subnetworkCatalogWarnings.length) {
      actions.setStatus(state.subnetworkCatalogWarnings[0], "error");
    } else {
      actions.setStatus(READY_STATUS_MESSAGE, "success");
    }
    return payload;
  }

  async function loadRecoverableDraft() {
    if (typeof sessionService.loadDraft !== "function") {
      return null;
    }
    const draftResponse = await sessionService.loadDraft();
    return draftResponse && draftResponse.draft ? draftResponse.draft : null;
  }

  async function chooseRecoverableDraft(draftPayload) {
    if (
      !draftPayload
      || !draftPayload.spec
      || !draftPayload.spec.network
      || typeof draftPayload.engine !== "string"
      || typeof draftPayload.collection_format !== "string"
    ) {
      return null;
    }
    const savedAtText =
      typeof draftPayload.saved_at === "string" && draftPayload.saved_at.trim()
        ? `\nSaved at: ${draftPayload.saved_at}`
        : "";
    const confirmRestore =
      typeof confirmAction === "function"
        ? confirmAction(`Previous editor session found. Restore it?${savedAtText}`)
        : false;
    if (confirmRestore) {
      return draftPayload;
    }
    if (typeof sessionService.clearDraft === "function") {
      await sessionService.clearDraft();
    }
    return null;
  }

  return {
    bootstrap,
  };
}
