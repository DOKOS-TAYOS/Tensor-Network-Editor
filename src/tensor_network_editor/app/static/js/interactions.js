import { createInteractionCanvasBindings } from "./interactionsCanvas.js";
import { createInteractionEditorBindings } from "./interactionsEditor.js";
import { createInteractionSessionBindings } from "./interactionsSession.js";
import { createInteractionShortcutBindings } from "./interactionsShortcuts.js";
import { createSessionUiAdapters } from "./session/sessionUiAdapters.js";

function createSessionActions(ctx) {
  return {
    ensureCodePanelVisible() {
      if (typeof ctx.toggleSidebarCollapsed === "function") {
        ctx.toggleSidebarCollapsed(false);
      }
      if (typeof ctx.setActiveSidebarTab === "function") {
        ctx.setActiveSidebarTab("code");
      }
    },
    syncCodeGenerationWarning() {
      if (typeof ctx.syncCodeGenerationWarning === "function") {
        ctx.syncCodeGenerationWarning();
      }
    },
    getTensorKrowchManualPlanIssueMessage() {
      return typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "";
    },
    getSelectedTensorIds() {
      return typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [];
    },
    findGroupById(groupId) {
      return typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
    },
    isLinearPeriodicMode() {
      return typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode();
    },
    syncGeneratedCodePreview:
      typeof ctx.renderGeneratedCodePreview === "function"
        ? (code) => ctx.renderGeneratedCodePreview(code)
        : null,
    setStatus: (message, level) => ctx.setStatus(message, level),
    serializeCurrentSpec: (options) => ctx.serializeCurrentSpec(options),
    formatIssues: (issues) => ctx.formatIssues(issues),
    stripImportLines: (code) => ctx.stripImportLines(code),
    sanitizeFilename: (value) => ctx.sanitizeFilename(value),
    resetDesignState: (spec, message, schemaVersion) =>
      ctx.resetDesignState(spec, message, schemaVersion),
    downloadPngExport: () => ctx.downloadPngExport(),
    downloadSvgExport: () => ctx.downloadSvgExport(),
    applyTemplateCatalogPayload: (payload) => ctx.applyTemplateCatalogPayload(payload),
    normalizeSpec: (spec) => ctx.normalizeSpec(spec),
    applyDesignChange: (mutate, options) => ctx.applyDesignChange(mutate, options),
    bringTensorToFront: (tensorId) => ctx.bringTensorToFront(tensorId),
    formatTemplateLabel: (templateName) => ctx.formatTemplateLabel(templateName),
    persistTemplateParametersFromControls: () =>
      ctx.persistTemplateParametersFromControls(),
    uniquifyImportedSpec: (spec, prefix) => ctx.uniquifyImportedSpec(spec, prefix),
    makeId: (prefix) => ctx.makeId(prefix),
    translateImportedSpec: (spec, targetCenter) =>
      ctx.translateImportedSpec(spec, targetCenter),
    suggestTensorPosition: (position) => ctx.suggestTensorPosition(position),
    viewportCenterPosition: () => ctx.viewportCenterPosition(),
  };
}

export function registerInteractions(ctx) {
  const runtime = {};
  const sessionUi = createSessionUiAdapters({
    windowRef: ctx.window,
    documentRef: ctx.document,
  });
  const env = {
    ctx,
    state: ctx.state,
    constants: ctx.constants,
    dom: ctx.dom,
    runtime,
  };

  Object.assign(runtime, createInteractionCanvasBindings(env));
  Object.assign(runtime, createInteractionShortcutBindings(env));
  Object.assign(runtime, createInteractionEditorBindings(env));
  Object.assign(
    runtime,
    createInteractionSessionBindings({
      ...env,
      store: ctx.store,
      selectors: ctx.selectors,
      services: ctx.services,
      sessionUi,
      sessionActions: createSessionActions(ctx),
    })
  );

  Object.assign(ctx, runtime);
}
