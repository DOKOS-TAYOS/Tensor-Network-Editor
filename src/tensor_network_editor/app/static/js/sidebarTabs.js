export function registerSidebarTabs(ctx) {
  const state = ctx.state;
  const windowRef =
    ctx.window || (typeof window !== "undefined" ? window : null);
  const SIDEBAR_DEFAULT_WIDTH = 360;
  const SIDEBAR_MIN_WIDTH = 280;
  const SIDEBAR_MAX_WIDTH = 640;
  const SIDEBAR_KEYBOARD_STEP = 24;
  const {
    workspace,
    sidebar,
    sidebarPanel,
    sidebarResizeHandle,
    sidebarToggleButton,
    sidebarTabSelection,
    sidebarTabPlanner,
    sidebarTabCode,
    sidebarPaneSelection,
    sidebarPanePlanner,
    sidebarPaneCode,
  } = ctx.dom;

  const tabConfig = {
    selection: {
      button: sidebarTabSelection,
      pane: sidebarPaneSelection,
    },
    planner: {
      button: sidebarTabPlanner,
      pane: sidebarPanePlanner,
    },
    code: {
      button: sidebarTabCode,
      pane: sidebarPaneCode,
    },
  };

  function normalizeSidebarTab(tabName) {
    return Object.prototype.hasOwnProperty.call(tabConfig, tabName) ? tabName : "selection";
  }

  function clampSidebarWidth(width) {
    const numericWidth = Number(width);
    if (!Number.isFinite(numericWidth)) {
      return SIDEBAR_DEFAULT_WIDTH;
    }
    return Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, Math.round(numericWidth)));
  }

  function refreshSidebarLayout() {
    const cy = state.cy || ctx.cy;
    if (cy && typeof cy.resize === "function") {
      cy.resize();
    }
    if (typeof ctx.renderOverlayDecorations === "function") {
      ctx.renderOverlayDecorations();
    }
    if (typeof ctx.renderMinimap === "function") {
      ctx.renderMinimap();
    }
  }

  function syncSidebarWidth() {
    state.sidebarWidth = clampSidebarWidth(state.sidebarWidth);
    if (workspace && workspace.style) {
      workspace.style.setProperty("--sidebar-width", `${state.sidebarWidth}px`);
    }
    if (sidebarResizeHandle) {
      sidebarResizeHandle.setAttribute("aria-valuemin", String(SIDEBAR_MIN_WIDTH));
      sidebarResizeHandle.setAttribute("aria-valuemax", String(SIDEBAR_MAX_WIDTH));
      sidebarResizeHandle.setAttribute("aria-valuenow", String(state.sidebarWidth));
    }
  }

  function setSidebarWidth(nextWidth, options = {}) {
    const previousWidth = state.sidebarWidth;
    state.sidebarWidth = clampSidebarWidth(nextWidth);
    syncSidebarWidth();
    if (state.sidebarWidth !== previousWidth && options.refresh !== false) {
      refreshSidebarLayout();
    }
  }

  function renderSidebarTabs() {
    const activeTab = normalizeSidebarTab(state.activeSidebarTab);
    state.activeSidebarTab = activeTab;
    const isCollapsed = Boolean(state.sidebarCollapsed);
    syncSidebarWidth();

    if (workspace) {
      workspace.classList.toggle("sidebar-is-collapsed", isCollapsed);
    }
    if (sidebar) {
      sidebar.classList.toggle("is-collapsed", isCollapsed);
    }
    if (sidebarPanel) {
      sidebarPanel.classList.toggle("is-collapsed", isCollapsed);
    }
    if (sidebarResizeHandle) {
      sidebarResizeHandle.hidden = isCollapsed;
    }
    if (sidebarToggleButton) {
      sidebarToggleButton.innerHTML = isCollapsed ? "&lt;&lt;" : "&gt;&gt;";
      sidebarToggleButton.setAttribute("aria-expanded", String(!isCollapsed));
      sidebarToggleButton.dataset.shortcut = "S";
      sidebarToggleButton.dataset.shortcutLabel = isCollapsed
        ? "Expand sidebar"
        : "Collapse sidebar";
      sidebarToggleButton.setAttribute(
        "aria-label",
        `${sidebarToggleButton.dataset.shortcutLabel} (S)`
      );
      sidebarToggleButton.removeAttribute("title");
    }

    Object.entries(tabConfig).forEach(([tabName, config]) => {
      const isActive = tabName === activeTab;
      if (config.button) {
        config.button.classList.toggle("is-active", isActive);
        config.button.setAttribute("aria-selected", String(isActive));
        config.button.setAttribute("tabindex", isActive ? "0" : "-1");
        config.button.hidden = isCollapsed;
      }
      if (config.pane) {
        config.pane.classList.toggle("is-active", isActive && !isCollapsed);
        config.pane.hidden = isCollapsed || !isActive;
      }
    });
  }

  function setActiveSidebarTab(tabName) {
    state.activeSidebarTab = normalizeSidebarTab(tabName);
    renderSidebarTabs();
  }

  function toggleSidebarCollapsed(forceCollapsed) {
    state.sidebarCollapsed =
      typeof forceCollapsed === "boolean"
        ? forceCollapsed
        : !state.sidebarCollapsed;
    if (state.sidebarCollapsed) {
      state.activeSidebarResize = null;
      if (sidebar) {
        sidebar.classList.remove("is-resizing");
      }
    }
    renderSidebarTabs();
    refreshSidebarLayout();
  }

  function startSidebarResize(event) {
    if (!sidebarResizeHandle || state.sidebarCollapsed) {
      return;
    }
    if (typeof event.button === "number" && event.button !== 0) {
      return;
    }
    event.preventDefault();
    state.activeSidebarResize = {
      startClientX: event.clientX,
      startWidth: clampSidebarWidth(state.sidebarWidth),
    };
    if (sidebar) {
      sidebar.classList.add("is-resizing");
    }
  }

  function handleSidebarResizeMove(event) {
    if (!state.activeSidebarResize) {
      return;
    }
    event.preventDefault();
    const nextWidth =
      state.activeSidebarResize.startWidth +
      state.activeSidebarResize.startClientX -
      event.clientX;
    setSidebarWidth(nextWidth);
  }

  function stopSidebarResize() {
    if (!state.activeSidebarResize) {
      return;
    }
    state.activeSidebarResize = null;
    if (sidebar) {
      sidebar.classList.remove("is-resizing");
    }
  }

  function handleSidebarResizeKeydown(event) {
    if (state.sidebarCollapsed) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setSidebarWidth(state.sidebarWidth + SIDEBAR_KEYBOARD_STEP);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setSidebarWidth(state.sidebarWidth - SIDEBAR_KEYBOARD_STEP);
    }
  }

  function attachSidebarTabHandlers() {
    Object.entries(tabConfig).forEach(([tabName, config]) => {
      if (!config.button) {
        return;
      }
      config.button.addEventListener("click", () => {
        setActiveSidebarTab(tabName);
      });
    });
    if (sidebarToggleButton) {
      sidebarToggleButton.addEventListener("click", () => {
        toggleSidebarCollapsed();
      });
    }
    if (sidebarResizeHandle) {
      sidebarResizeHandle.addEventListener("mousedown", startSidebarResize);
      sidebarResizeHandle.addEventListener("keydown", handleSidebarResizeKeydown);
    }
    if (windowRef && typeof windowRef.addEventListener === "function") {
      windowRef.addEventListener("mousemove", handleSidebarResizeMove);
      windowRef.addEventListener("mouseup", stopSidebarResize);
    }
  }

  attachSidebarTabHandlers();
  renderSidebarTabs();

  Object.assign(ctx, {
    normalizeSidebarTab,
    renderSidebarTabs,
    setSidebarWidth,
    setActiveSidebarTab,
    toggleSidebarCollapsed,
  });
}
