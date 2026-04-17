function noop() {}

function resolveOptionalAction(ctx, name, fallback = noop) {
  return typeof ctx[name] === "function" ? ctx[name].bind(ctx) : fallback;
}

function resolveOptionalValue(ctx, name, fallback) {
  return typeof ctx[name] === "function" ? ctx[name].bind(ctx) : fallback;
}

export function createEditorActionGroups(ctx) {
  return {
    session: {
      ensureCodePanelVisible() {
        resolveOptionalAction(ctx, "toggleSidebarCollapsed")(false);
        resolveOptionalAction(ctx, "setActiveSidebarTab")("code");
      },
      syncCodeGenerationWarning: resolveOptionalAction(
        ctx,
        "syncCodeGenerationWarning"
      ),
      getTensorKrowchManualPlanIssueMessage:
        resolveOptionalValue(ctx, "getTensorKrowchManualPlanIssueMessage", () => ""),
      getSelectedTensorIds: () =>
        resolveOptionalValue(ctx, "getSelectedIdsByKind", () => [])("tensor"),
      findGroupById: resolveOptionalValue(ctx, "findGroupById", () => null),
      isLinearPeriodicMode:
        resolveOptionalValue(ctx, "isLinearPeriodicMode", () => false),
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
    },
    editor: {
      bumpSpecRevision: resolveOptionalAction(ctx, "bumpSpecRevision"),
      enforceLinearPeriodicEngineSupport: resolveOptionalAction(
        ctx,
        "enforceLinearPeriodicEngineSupport"
      ),
      refreshContractionAnalysis: resolveOptionalAction(
        ctx,
        "refreshContractionAnalysis"
      ),
      findVisibleTensorById:
        resolveOptionalValue(ctx, "findVisibleTensorById", (tensorId) =>
          ctx.findTensorById(tensorId)
        ),
      canEditCurrentContractionStage:
        resolveOptionalValue(ctx, "canEditCurrentContractionStage", () => false),
      updateCurrentStageOperandLayout: resolveOptionalAction(
        ctx,
        "updateCurrentStageOperandLayout"
      ),
      isInspectingPastStage:
        resolveOptionalValue(ctx, "isInspectingPastStage", () => false),
      resolveConnectableIndexOwner:
        resolveOptionalValue(ctx, "resolveConnectableIndexOwner", (indexId) =>
          ctx.findIndexOwner(indexId)
        ),
      toggleSidebarCollapsed: resolveOptionalAction(ctx, "toggleSidebarCollapsed"),
      setActiveSidebarTab: resolveOptionalAction(ctx, "setActiveSidebarTab"),
      syncPendingInteractionClasses: resolveOptionalAction(
        ctx,
        "syncPendingInteractionClasses"
      ),
      findVisibleEdgeSelectionIdByBaseEdgeId:
        resolveOptionalValue(
          ctx,
          "findVisibleEdgeSelectionIdByBaseEdgeId",
          (edgeId) => edgeId
        ),
      removeNote: resolveOptionalAction(ctx, "removeNote"),
    },
    shortcuts: {
      toggleSidebarCollapsed: resolveOptionalAction(ctx, "toggleSidebarCollapsed"),
      setActiveSidebarTab: resolveOptionalAction(ctx, "setActiveSidebarTab"),
      enforceLinearPeriodicEngineSupport: resolveOptionalAction(
        ctx,
        "enforceLinearPeriodicEngineSupport"
      ),
      renderPlanner: resolveOptionalAction(ctx, "renderPlanner"),
      startAutomaticPreview: resolveOptionalAction(ctx, "startAutomaticPreview"),
      acceptAutomaticPlan: resolveOptionalAction(ctx, "acceptAutomaticPlan"),
      toggleMinimapVisibility: resolveOptionalAction(ctx, "toggleMinimapVisibility"),
      syncPendingInteractionClasses: resolveOptionalAction(
        ctx,
        "syncPendingInteractionClasses"
      ),
      clearAutomaticPreview: resolveOptionalAction(ctx, "clearAutomaticPreview"),
      clearPastInspection: resolveOptionalAction(ctx, "clearPastInspection"),
      closeTransientToolbarUi: resolveOptionalAction(ctx, "closeTransientToolbarUi"),
      copySelectedSubgraphToClipboard: resolveOptionalAction(
        ctx,
        "copySelectedSubgraphToClipboard"
      ),
      pasteClipboardToCanvas: resolveOptionalAction(ctx, "pasteClipboardToCanvas"),
      trimContractionPlan: resolveOptionalAction(ctx, "trimContractionPlan"),
      togglePlannerMode: resolveOptionalAction(ctx, "togglePlannerMode"),
      createGroupFromSelection: resolveOptionalAction(
        ctx,
        "createGroupFromSelection"
      ),
      addNoteAtCenter: resolveOptionalAction(ctx, "addNoteAtCenter"),
      toggleLinearPeriodicMode: resolveOptionalAction(
        ctx,
        "toggleLinearPeriodicMode"
      ),
    },
  };
}
