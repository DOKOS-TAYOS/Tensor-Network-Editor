const READY_STATUS_MESSAGE =
  "Editor ready. Drag the canvas to move, use Ctrl+wheel to zoom, use the wheel to pan, and right drag to box-select.";

export function createEditorBootstrapFlow({
  state,
  store,
  sessionService,
  actions,
}) {
  async function bootstrap() {
    const payload = await sessionService.loadBootstrap();
    store.setSpec(actions.normalizeSpec(payload.spec.network));
    store.setSchemaVersion(payload.schema_version);
    store.setAvailableCollectionFormats(
      Array.isArray(payload.collection_formats) ? payload.collection_formats : ["list"]
    );
    actions.applyTemplateCatalogPayload({
      templateNames: payload.templates,
      templateDefinitions: payload.template_definitions,
      templateCatalogWarnings: payload.template_catalog_warnings,
    });
    store.setAnnotationDefinitions(payload.annotation_definitions);
    store.setSelectedEngine(payload.default_engine);
    store.setSelectedCollectionFormat(payload.default_collection_format || "list");
    actions.reconcileTensorOrder();
    actions.populateEngineOptions(payload.engines);
    actions.enforceLinearPeriodicEngineSupport();
    actions.populateCollectionFormatOptions(state.availableCollectionFormats);
    actions.initGraph();
    actions.clearHistory();
    actions.render();
    actions.refreshContractionAnalysis();
    if (state.templateCatalogWarnings.length) {
      actions.setStatus(state.templateCatalogWarnings[0], "error");
    } else {
      actions.setStatus(READY_STATUS_MESSAGE, "success");
    }
    return payload;
  }

  return {
    bootstrap,
  };
}
