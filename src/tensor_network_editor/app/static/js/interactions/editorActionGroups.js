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
      isForMode: resolveOptionalValue(ctx, "isForMode", () => false),
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
      getTemplateSource: (templateName) => ctx.getTemplateSource(templateName),
      getTemplateSpec: (templateName) => ctx.getTemplateSpec(templateName),
      listTemplateEntries: () => ctx.listTemplateEntries(),
      hasTemplateDisplayName: (displayName, excludedTemplateName) =>
        ctx.hasTemplateDisplayName(displayName, excludedTemplateName),
      getNextSessionTemplateDisplayName: (baseDisplayName) =>
        ctx.getNextSessionTemplateDisplayName(baseDisplayName),
      addSessionTemplate: (payload) => ctx.addSessionTemplate(payload),
      updateSessionTemplateDisplayNames: (updates) =>
        ctx.updateSessionTemplateDisplayNames(updates),
      removeSessionTemplate: (templateName) => ctx.removeSessionTemplate(templateName),
      toggleTemplateManager: (forceOpen) => ctx.toggleTemplateManager(forceOpen),
      syncTemplateManagerModalState: () => ctx.syncTemplateManagerModalState(),
      toggleSubnetworkLibrary: (forceOpen) => ctx.toggleSubnetworkLibrary(forceOpen),
      syncSubnetworkLibraryModalState: () => ctx.syncSubnetworkLibraryModalState(),
      setTemplateManagerValidationMessage: (message) =>
        ctx.setTemplateManagerValidationMessage(message),
      updateToolbarState: () => ctx.updateToolbarState(),
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
      createHyperedgeFromSelection: resolveOptionalAction(
        ctx,
        "createHyperedgeFromSelection"
      ),
      addNoteAtCenter: resolveOptionalAction(ctx, "addNoteAtCenter"),
      toggleTemplateManager: resolveOptionalAction(ctx, "toggleTemplateManager"),
      toggleLinearPeriodicMode: resolveOptionalAction(
        ctx,
        "toggleLinearPeriodicMode"
      ),
      switchLinearPeriodicCell: resolveOptionalAction(
        ctx,
        "switchLinearPeriodicCell"
      ),
      switchGridPeriodicCell: resolveOptionalAction(
        ctx,
        "switchGridPeriodicCell"
      ),
      switchTreePeriodicCell: resolveOptionalAction(
        ctx,
        "switchTreePeriodicCell"
      ),
      switchBenchmarkPosition: resolveOptionalAction(
        ctx,
        "switchBenchmarkPosition"
      ),
      nudgeSelectedElements: (...args) => {
        if (typeof ctx.nudgeSelectedElements === "function") {
          return ctx.nudgeSelectedElements(...args);
        }
        return false;
      },
      closeBenchmarkCompareModal: resolveOptionalAction(
        ctx,
        "closeBenchmarkCompareModal"
      ),
    },
  };
}
