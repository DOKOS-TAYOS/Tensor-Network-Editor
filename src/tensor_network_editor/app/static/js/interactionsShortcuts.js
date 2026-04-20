function resolveContextAction(ctx, name) {
  return (...args) => {
    if (typeof ctx[name] === "function") {
      return ctx[name](...args);
    }
    return undefined;
  };
}

export function createInteractionShortcutBindings({
  ctx,
  state,
  dom,
  runtime,
  shortcutActions = {},
}) {
  const { engineSelect, loadInput } = dom;
  const resolvedShortcutActions = {
    toggleSidebarCollapsed:
      shortcutActions.toggleSidebarCollapsed ||
      resolveContextAction(ctx, "toggleSidebarCollapsed"),
    setActiveSidebarTab:
      shortcutActions.setActiveSidebarTab ||
      resolveContextAction(ctx, "setActiveSidebarTab"),
    enforceLinearPeriodicEngineSupport:
      shortcutActions.enforceLinearPeriodicEngineSupport ||
      resolveContextAction(ctx, "enforceLinearPeriodicEngineSupport"),
    renderPlanner:
      shortcutActions.renderPlanner || resolveContextAction(ctx, "renderPlanner"),
    startAutomaticPreview:
      shortcutActions.startAutomaticPreview ||
      resolveContextAction(ctx, "startAutomaticPreview"),
    acceptAutomaticPlan:
      shortcutActions.acceptAutomaticPlan ||
      resolveContextAction(ctx, "acceptAutomaticPlan"),
    toggleMinimapVisibility:
      shortcutActions.toggleMinimapVisibility ||
      resolveContextAction(ctx, "toggleMinimapVisibility"),
    syncPendingInteractionClasses:
      shortcutActions.syncPendingInteractionClasses ||
      resolveContextAction(ctx, "syncPendingInteractionClasses"),
    clearAutomaticPreview:
      shortcutActions.clearAutomaticPreview ||
      resolveContextAction(ctx, "clearAutomaticPreview"),
    clearPastInspection:
      shortcutActions.clearPastInspection ||
      resolveContextAction(ctx, "clearPastInspection"),
    closeTransientToolbarUi:
      shortcutActions.closeTransientToolbarUi ||
      resolveContextAction(ctx, "closeTransientToolbarUi"),
    copySelectedSubgraphToClipboard:
      shortcutActions.copySelectedSubgraphToClipboard ||
      resolveContextAction(ctx, "copySelectedSubgraphToClipboard"),
    pasteClipboardToCanvas:
      shortcutActions.pasteClipboardToCanvas ||
      resolveContextAction(ctx, "pasteClipboardToCanvas"),
    trimContractionPlan:
      shortcutActions.trimContractionPlan ||
      resolveContextAction(ctx, "trimContractionPlan"),
    togglePlannerMode:
      shortcutActions.togglePlannerMode ||
      resolveContextAction(ctx, "togglePlannerMode"),
    createGroupFromSelection:
      shortcutActions.createGroupFromSelection ||
      resolveContextAction(ctx, "createGroupFromSelection"),
    selectAllTensors:
      shortcutActions.selectAllTensors ||
      resolveContextAction(ctx, "selectAllTensors"),
    addNoteAtCenter:
      shortcutActions.addNoteAtCenter || resolveContextAction(ctx, "addNoteAtCenter"),
    toggleTemplateManager:
      shortcutActions.toggleTemplateManager
      || resolveContextAction(ctx, "toggleTemplateManager"),
    toggleLinearPeriodicMode:
      shortcutActions.toggleLinearPeriodicMode ||
      resolveContextAction(ctx, "toggleLinearPeriodicMode"),
    setLinearPeriodicMode:
      shortcutActions.setLinearPeriodicMode ||
      resolveContextAction(ctx, "setLinearPeriodicMode"),
    setGridPeriodicMode:
      shortcutActions.setGridPeriodicMode ||
      resolveContextAction(ctx, "setGridPeriodicMode"),
    setBenchmarkMode:
      shortcutActions.setBenchmarkMode ||
      resolveContextAction(ctx, "setBenchmarkMode"),
    openSessionTemplatePicker:
      shortcutActions.openSessionTemplatePicker
      || resolveContextAction(ctx, "openSessionTemplatePicker"),
    exportSelectedTemplateSpec:
      shortcutActions.exportSelectedTemplateSpec
      || resolveContextAction(ctx, "exportSelectedTemplateSpec"),
    closeBenchmarkCompareModal:
      shortcutActions.closeBenchmarkCompareModal
      || resolveContextAction(ctx, "closeBenchmarkCompareModal"),
  };
  const {
    toggleSidebarCollapsed,
    setActiveSidebarTab,
    enforceLinearPeriodicEngineSupport,
    renderPlanner,
    startAutomaticPreview,
    acceptAutomaticPlan,
    toggleMinimapVisibility,
    syncPendingInteractionClasses,
    clearAutomaticPreview,
    clearPastInspection,
    closeTransientToolbarUi,
    copySelectedSubgraphToClipboard,
    pasteClipboardToCanvas,
    trimContractionPlan,
    togglePlannerMode,
    createGroupFromSelection,
    selectAllTensors,
    addNoteAtCenter,
    toggleTemplateManager,
    toggleLinearPeriodicMode,
    setLinearPeriodicMode,
    setGridPeriodicMode,
    setBenchmarkMode,
    openSessionTemplatePicker,
    exportSelectedTemplateSpec,
    closeBenchmarkCompareModal,
  } = resolvedShortcutActions;

  function openSidebarTab(tabName) {
    toggleSidebarCollapsed(false);
    setActiveSidebarTab(tabName);
  }

  function setSelectedEngine(engineName) {
    if (!engineSelect) {
      return;
    }
    const hasEngine = Array.from(engineSelect.options).some(
      (option) => option.value === engineName
    );
    if (!hasEngine) {
      ctx.setStatus(`The ${engineName} engine is not available in this session.`, "error");
      return;
    }
    state.selectedEngine = engineName;
    engineSelect.value = engineName;
    enforceLinearPeriodicEngineSupport();
    renderPlanner();
    ctx.updateToolbarState();
    ctx.setStatus(`Engine set to ${ctx.formatEngineLabel(engineName)}.`, "success");
  }

  function toggleAutomaticPreview(mode) {
    openSidebarTab("planner");
    startAutomaticPreview(mode);
  }

  function acceptAutomaticShortcut(mode) {
    openSidebarTab("planner");
    acceptAutomaticPlan(mode);
  }

  function toggleSidebarVisibility() {
    toggleSidebarCollapsed();
    ctx.setStatus(state.sidebarCollapsed ? "Sidebar collapsed." : "Sidebar expanded.");
  }

  function toggleMinimapShortcut() {
    toggleMinimapVisibility();
    ctx.setStatus(state.minimapHidden ? "Minimap hidden." : "Minimap shown.");
  }

  function activateSingleMode() {
    setBenchmarkMode(false);
    setLinearPeriodicMode(false);
    setGridPeriodicMode(false);
  }

  function activateGridPeriodicMode() {
    setBenchmarkMode(false);
    setGridPeriodicMode(true);
  }

  function activateBenchmarkMode() {
    setLinearPeriodicMode(false);
    setGridPeriodicMode(false);
    setBenchmarkMode(true);
  }

  function announceTreeModeUnavailable() {
    if (typeof ctx.setStatus === "function") {
      ctx.setStatus("For Tree mode is not available yet.", "error");
    }
  }

  function handleKeydown(event) {
    const activeElement = ctx.document.activeElement;
    const inTextInput = ctx.isTextInput(event.target) || ctx.isTextInput(activeElement);
    const lowerKey = event.key.toLowerCase();
    const hasSystemModifier = event.ctrlKey || event.metaKey;
    const hasAnyModifier = hasSystemModifier || event.altKey || event.shiftKey;

    if (event.key === "Escape") {
      event.preventDefault();
      if (state.benchmarkSession?.compareModal?.open) {
        closeBenchmarkCompareModal();
        return;
      }
      if (closeTransientToolbarUi()) {
        return;
      }
      if (state.isTemplateManagerOpen) {
        toggleTemplateManager(false);
        return;
      }
      if (state.isHelpOpen) {
        ctx.toggleHelpModal(false);
        return;
      }
      if (state.boxSelection) {
        ctx.finishBoxSelection(true);
        return;
      }
      if (state.connectMode) {
        state.pendingIndexId = null;
        state.connectMode = false;
        syncPendingInteractionClasses();
        ctx.render();
        ctx.setStatus("Connect mode cancelled.");
        return;
      }
      if (state.pendingPlannerOperandId) {
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
        syncPendingInteractionClasses();
        renderPlanner();
        ctx.renderOverlayDecorations();
        ctx.setStatus("Manual planner operand selection cleared.");
        return;
      }
      if (state.plannerPreviewMode) {
        clearAutomaticPreview();
        state.plannerPreviewMode = null;
        ctx.render();
        ctx.setStatus("Automatic preview cleared.");
        return;
      }
      if (Number.isInteger(state.plannerInspectionStepCount)) {
        clearPastInspection();
        ctx.render();
        ctx.setStatus("Returned to the latest contracted view.");
        return;
      }
      ctx.clearSelection();
      return;
    }

    if (inTextInput) {
      return;
    }

    if (hasSystemModifier && lowerKey === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        ctx.performRedo();
      } else {
        ctx.performUndo();
      }
      return;
    }
    if (hasSystemModifier && event.altKey && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticFuture");
      return;
    }
    if (hasSystemModifier && event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticPast");
      return;
    }
    if (hasSystemModifier && lowerKey === "a") {
      event.preventDefault();
      selectAllTensors();
      return;
    }
    if (hasSystemModifier && lowerKey === "s") {
      event.preventDefault();
      ctx.saveDesign();
      return;
    }
    if (hasSystemModifier && lowerKey === "l") {
      event.preventDefault();
      loadInput.click();
      return;
    }
    if (hasSystemModifier && lowerKey === "y") {
      event.preventDefault();
      setSelectedEngine("einsum_numpy");
      return;
    }
    if (hasSystemModifier && lowerKey === "p") {
      event.preventDefault();
      setSelectedEngine("einsum_torch");
      return;
    }
    if (hasSystemModifier && lowerKey === "k") {
      event.preventDefault();
      setSelectedEngine("tensorkrowch");
      return;
    }
    if (hasSystemModifier && lowerKey === "q") {
      event.preventDefault();
      setSelectedEngine("quimb");
      return;
    }
    if (hasSystemModifier && lowerKey === "t") {
      event.preventDefault();
      setSelectedEngine("tensornetwork");
      return;
    }
    if (hasSystemModifier && lowerKey === "c") {
      event.preventDefault();
      copySelectedSubgraphToClipboard();
      return;
    }
    if (hasSystemModifier && lowerKey === "v") {
      event.preventDefault();
      pasteClipboardToCanvas();
      return;
    }
    if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "r") {
      event.preventDefault();
      if (state.spec.contraction_plan) {
        trimContractionPlan(0);
      } else {
        ctx.setStatus("There is no contraction path to reset.");
      }
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      ctx.deleteSelection();
      return;
    }
    if (!hasSystemModifier && event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticPast");
      return;
    }
    if (!hasSystemModifier && event.altKey && lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticFuture");
      return;
    }
    if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "m") {
      event.preventDefault();
      toggleMinimapShortcut();
      return;
    }
    if (!hasAnyModifier && lowerKey === "m") {
      event.preventDefault();
      openSidebarTab("planner");
      togglePlannerMode();
      return;
    }
    if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "g") {
      event.preventDefault();
      ctx.generateCode();
      return;
    }
    if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "s") {
      event.preventDefault();
      activateSingleMode();
      return;
    }
    if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "e") {
      event.preventDefault();
      exportSelectedTemplateSpec();
      return;
    }
    if (!hasAnyModifier && lowerKey === "s") {
      event.preventDefault();
      toggleSidebarVisibility();
      return;
    }
    if (!hasAnyModifier && lowerKey === "g") {
      event.preventDefault();
      createGroupFromSelection();
      return;
    }
    if (!hasAnyModifier && lowerKey === "t") {
      event.preventDefault();
      ctx.insertTemplate();
      return;
    }
    if (!hasAnyModifier && lowerKey === "p") {
      event.preventDefault();
      addNoteAtCenter();
      return;
    }
    if (!hasAnyModifier && lowerKey === "f") {
      event.preventDefault();
      toggleLinearPeriodicMode();
      return;
    }
    if (!hasAnyModifier && lowerKey === "d") {
      event.preventDefault();
      activateGridPeriodicMode();
      return;
    }
    if (!hasAnyModifier && lowerKey === "b") {
      event.preventDefault();
      activateBenchmarkMode();
      return;
    }
    if (!hasAnyModifier && lowerKey === "l") {
      event.preventDefault();
      openSessionTemplatePicker();
      return;
    }
    if (!hasAnyModifier && lowerKey === "e") {
      event.preventDefault();
      announceTreeModeUnavailable();
      return;
    }
    if (!hasAnyModifier && lowerKey === "n") {
      event.preventDefault();
      ctx.addTensorAtCenter();
      return;
    }
    if (!hasAnyModifier && lowerKey === "c") {
      event.preventDefault();
      ctx.toggleConnectMode();
      return;
    }
    if (event.key === "?") {
      event.preventDefault();
      if (typeof ctx.openHelpSection === "function") {
        ctx.openHelpSection("info");
        return;
      }
      ctx.toggleHelpModal(true, "info");
    }
  }

  function sendCancelBeacon() {
    if (state.editorFinished || !navigator.sendBeacon) {
      return;
    }
    const payload = new Blob([JSON.stringify({})], { type: "application/json" });
    navigator.sendBeacon("/api/cancel", payload);
  }

  function handleWindowResize() {
    if (state.cy) {
      state.cy.resize();
    }
    ctx.renderOverlayDecorations();
    ctx.renderMinimap();
    if (typeof ctx.syncToolbarTransientUi === "function") {
      ctx.syncToolbarTransientUi();
    }
  }

  return {
    openSidebarTab,
    setSelectedEngine,
    toggleAutomaticPreview,
    acceptAutomaticShortcut,
    toggleSidebarVisibility,
    handleKeydown,
    sendCancelBeacon,
    handleWindowResize,
  };
}
