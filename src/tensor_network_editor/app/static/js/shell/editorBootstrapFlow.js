import {
  applyEditorTheme,
  resolvePreferredEditorThemeName,
} from "../core/theme.js";

const READY_STATUS_MESSAGE =
  "Editor ready. Drag the canvas to move, use Ctrl+wheel to zoom, use the wheel to pan, and right drag to box-select.";

function warningCountFromPayload(payload) {
  const templateWarningCount = Array.isArray(payload?.template_catalog_warnings)
    ? payload.template_catalog_warnings.length
    : 0;
  const subnetworkWarningCount = Array.isArray(payload?.subnetwork_catalog_warnings)
    ? payload.subnetwork_catalog_warnings.length
    : 0;
  return templateWarningCount + subnetworkWarningCount;
}

export function createEditorBootstrapFlow({
  state,
  store,
  sessionService,
  actions,
  logger = null,
  documentRef = null,
  windowRef = null,
  confirmAction = null,
}) {
  async function bootstrap() {
    const bootstrapOperation = logger?.startOperation?.("Bootstrap", {
      operation: "bootstrap",
    });
    try {
      const payload = await sessionService.loadBootstrap();
      if (logger && typeof logger.refreshRuntimeConfig === "function") {
        logger.refreshRuntimeConfig({
          sessionId: payload?.session_id || null,
          enabled: payload?.frontend_logging?.enabled,
          level: payload?.frontend_logging?.level,
          persist: payload?.frontend_logging?.persist,
          transport_endpoint: payload?.frontend_logging?.transport_endpoint,
        });
      }
      bootstrapOperation?.branch("Loaded bootstrap payload", {
        warning_count: warningCountFromPayload(payload),
      });
      const draftPayloadPromise = loadRecoverableDraft(bootstrapOperation);
      const selectedThemeName = resolvePreferredEditorThemeName({
        bootstrapThemeName: payload.theme,
        storageRef: windowRef?.localStorage || null,
      });
      applyEditorTheme(selectedThemeName, { documentRef });
      bootstrapOperation?.branch("Applied editor theme", {
        theme: selectedThemeName,
      });
      if (typeof store.setSelectedTheme === "function") {
        store.setSelectedTheme(selectedThemeName);
      } else {
        state.selectedTheme = selectedThemeName;
      }
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
      const draftPayload = await draftPayloadPromise;
      const restoredDraft = await chooseRecoverableDraft(
        draftPayload,
        bootstrapOperation
      );
      const activeSpecPayload = restoredDraft?.spec || payload.spec;
      const activeEngine = restoredDraft?.engine || payload.default_engine;
      const activeCollectionFormat =
        restoredDraft?.collection_format
        || payload.default_collection_format
        || "list";
      store.setSpec(actions.normalizeSpec(activeSpecPayload.network));
      store.setSchemaVersion(activeSpecPayload.schema_version || payload.schema_version);
      store.setSelectedEngine(activeEngine);
      store.setSelectedCollectionFormat(activeCollectionFormat);
      actions.reconcileTensorOrder();
      actions.populateEngineOptions(payload.engines);
      actions.enforceLinearPeriodicEngineSupport();
      actions.populateCollectionFormatOptions(state.availableCollectionFormats);
      actions.initGraph();
      actions.clearHistory();
      actions.render();
      if (typeof actions.updateToolbarState === "function") {
        actions.updateToolbarState();
      }
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
      bootstrapOperation?.finish({
        engine: activeEngine,
        collection_format: activeCollectionFormat,
        restored_draft: Boolean(restoredDraft),
        warning_count:
          state.templateCatalogWarnings.length + state.subnetworkCatalogWarnings.length,
      });
      return payload;
    } catch (error) {
      bootstrapOperation?.fail(error);
      throw error;
    }
  }

  async function loadRecoverableDraft(bootstrapOperation = null) {
    if (typeof sessionService.loadDraft !== "function") {
      bootstrapOperation?.branch?.("Draft recovery unavailable");
      return null;
    }
    const draftResponse = await sessionService.loadDraft();
    const draftPayload = draftResponse && draftResponse.draft ? draftResponse.draft : null;
    bootstrapOperation?.branch?.(
      draftPayload ? "Loaded recoverable draft metadata" : "No recoverable draft found",
      {
        restored_draft: Boolean(draftPayload),
      }
    );
    return draftPayload;
  }

  async function chooseRecoverableDraft(draftPayload, bootstrapOperation = null) {
    if (
      !draftPayload
      || !draftPayload.spec
      || !draftPayload.spec.network
      || typeof draftPayload.engine !== "string"
      || typeof draftPayload.collection_format !== "string"
    ) {
      bootstrapOperation?.branch?.("No recoverable draft to restore", {
        restored_draft: false,
      });
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
      bootstrapOperation?.branch?.("Recovered saved draft", {
        restored_draft: true,
      });
      return draftPayload;
    }
    if (typeof sessionService.clearDraft === "function") {
      await sessionService.clearDraft();
    }
    bootstrapOperation?.branch?.("Discarded recoverable draft", {
      restored_draft: false,
    });
    return null;
  }

  return {
    bootstrap,
  };
}
