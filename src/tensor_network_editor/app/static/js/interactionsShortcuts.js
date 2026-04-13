export function createInteractionShortcutBindings({
  ctx,
  state,
  dom,
  runtime,
}) {
  const {
    engineSelect,
    helpCloseButton,
    helpModal,
    loadInput,
  } = dom;

  function openSidebarTab(tabName) {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab(tabName);
    }
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
    if (typeof ctx.enforceLinearPeriodicEngineSupport === "function") {
      ctx.enforceLinearPeriodicEngineSupport();
    }
    if (typeof ctx.renderPlanner === "function") {
      ctx.renderPlanner();
    }
    ctx.updateToolbarState();
    ctx.setStatus(`Engine set to ${ctx.formatEngineLabel(engineName)}.`, "success");
  }

  function toggleAutomaticPreview(mode) {
    openSidebarTab("planner");
    if (typeof ctx.startAutomaticPreview === "function") {
      ctx.startAutomaticPreview(mode);
    }
  }

  function acceptAutomaticShortcut(mode) {
    openSidebarTab("planner");
    if (typeof ctx.acceptAutomaticPlan === "function") {
      ctx.acceptAutomaticPlan(mode);
    }
  }

  function toggleSidebarVisibility() {
    if (typeof ctx.toggleSidebarCollapsed !== "function") {
      return;
    }
    ctx.toggleSidebarCollapsed();
    ctx.setStatus(state.sidebarCollapsed ? "Sidebar collapsed." : "Sidebar expanded.");
  }

  function toggleMinimapVisibility() {
    if (typeof ctx.toggleMinimapVisibility !== "function") {
      return;
    }
    ctx.toggleMinimapVisibility();
    ctx.setStatus(state.minimapHidden ? "Minimap hidden." : "Minimap shown.");
  }

  function toggleHelpModal(forceOpen) {
    state.isHelpOpen = typeof forceOpen === "boolean" ? forceOpen : !state.isHelpOpen;
    helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    if (state.isHelpOpen) {
      helpCloseButton.focus();
    }
  }

  function handleKeydown(event) {
    const activeElement = ctx.document.activeElement;
    const inTextInput = ctx.isTextInput(event.target) || ctx.isTextInput(activeElement);
    const lowerKey = event.key.toLowerCase();

    if (event.key === "Escape") {
      event.preventDefault();
      if (state.isHelpOpen) {
        toggleHelpModal(false);
        return;
      }
      if (state.boxSelection) {
        ctx.finishBoxSelection(true);
        return;
      }
      if (state.connectMode) {
        state.pendingIndexId = null;
        state.connectMode = false;
        if (typeof ctx.syncPendingInteractionClasses === "function") {
          ctx.syncPendingInteractionClasses();
        }
        ctx.render();
        ctx.setStatus("Connect mode cancelled.");
        return;
      }
      if (state.pendingPlannerOperandId) {
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
        if (typeof ctx.syncPendingInteractionClasses === "function") {
          ctx.syncPendingInteractionClasses();
        }
        if (typeof ctx.renderPlanner === "function") {
          ctx.renderPlanner();
        }
        ctx.renderOverlayDecorations();
        ctx.setStatus("Manual planner operand selection cleared.");
        return;
      }
      if (state.plannerPreviewMode) {
        if (typeof ctx.clearAutomaticPreview === "function") {
          ctx.clearAutomaticPreview();
        } else {
          state.plannerPreviewMode = null;
        }
        ctx.render();
        ctx.setStatus("Automatic preview cleared.");
        return;
      }
      if (
        Number.isInteger(state.plannerInspectionStepCount) &&
        typeof ctx.clearPastInspection === "function"
      ) {
        ctx.clearPastInspection();
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

    const hasModifier = event.ctrlKey || event.metaKey;
    if (hasModifier && lowerKey === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        ctx.performRedo();
      } else {
        ctx.performUndo();
      }
      return;
    }
    if (hasModifier && event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticPast");
      return;
    }
    if (hasModifier && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticFuture");
      return;
    }
    if (hasModifier && lowerKey === "s") {
      event.preventDefault();
      ctx.saveDesign();
      return;
    }
    if (hasModifier && lowerKey === "l") {
      event.preventDefault();
      loadInput.click();
      return;
    }
    if (hasModifier && lowerKey === "y") {
      event.preventDefault();
      setSelectedEngine("einsum_numpy");
      return;
    }
    if (hasModifier && lowerKey === "p") {
      event.preventDefault();
      setSelectedEngine("einsum_torch");
      return;
    }
    if (hasModifier && lowerKey === "k") {
      event.preventDefault();
      setSelectedEngine("tensorkrowch");
      return;
    }
    if (hasModifier && lowerKey === "q") {
      event.preventDefault();
      setSelectedEngine("quimb");
      return;
    }
    if (hasModifier && lowerKey === "t") {
      event.preventDefault();
      setSelectedEngine("tensornetwork");
      return;
    }
    if (hasModifier && lowerKey === "c") {
      event.preventDefault();
      if (typeof ctx.copySelectedSubgraphToClipboard === "function") {
        ctx.copySelectedSubgraphToClipboard();
      }
      return;
    }
    if (hasModifier && lowerKey === "v") {
      event.preventDefault();
      if (typeof ctx.pasteClipboardToCanvas === "function") {
        ctx.pasteClipboardToCanvas();
      }
      return;
    }
    if (event.shiftKey && lowerKey === "r") {
      event.preventDefault();
      if (typeof ctx.trimContractionPlan === "function" && state.spec.contraction_plan) {
        ctx.trimContractionPlan(0);
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
    if (event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticPast");
      return;
    }
    if (event.shiftKey && lowerKey === "m") {
      event.preventDefault();
      toggleMinimapVisibility();
      return;
    }
    if (lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticFuture");
      return;
    }
    if (lowerKey === "m") {
      event.preventDefault();
      openSidebarTab("planner");
      if (typeof ctx.togglePlannerMode === "function") {
        ctx.togglePlannerMode();
      }
      return;
    }
    if (event.shiftKey && lowerKey === "g") {
      event.preventDefault();
      ctx.generateCode();
      return;
    }
    if (lowerKey === "s") {
      event.preventDefault();
      toggleSidebarVisibility();
      return;
    }
    if (lowerKey === "g") {
      event.preventDefault();
      if (typeof ctx.createGroupFromSelection === "function") {
        ctx.createGroupFromSelection();
      }
      return;
    }
    if (lowerKey === "t") {
      event.preventDefault();
      ctx.insertTemplate();
      return;
    }
    if (lowerKey === "p") {
      event.preventDefault();
      if (typeof ctx.addNoteAtCenter === "function") {
        ctx.addNoteAtCenter();
      }
      return;
    }
    if (lowerKey === "f") {
      event.preventDefault();
      if (typeof ctx.toggleLinearPeriodicMode === "function") {
        ctx.toggleLinearPeriodicMode();
      }
      return;
    }
    if (lowerKey === "n") {
      event.preventDefault();
      ctx.addTensorAtCenter();
      return;
    }
    if (lowerKey === "c") {
      event.preventDefault();
      ctx.toggleConnectMode();
      return;
    }
    if (event.key === "?") {
      event.preventDefault();
      toggleHelpModal(true);
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
  }

  return {
    openSidebarTab,
    setSelectedEngine,
    toggleAutomaticPreview,
    acceptAutomaticShortcut,
    toggleSidebarVisibility,
    toggleMinimapVisibility,
    handleKeydown,
    toggleHelpModal,
    sendCancelBeacon,
    handleWindowResize,
  };
}
