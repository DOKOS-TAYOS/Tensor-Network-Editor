export function registerSidebarTabs(ctx) {
  const state = ctx.state;
  const {
    workspace,
    sidebar,
    sidebarPanel,
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

  function renderSidebarTabs() {
    const activeTab = normalizeSidebarTab(state.activeSidebarTab);
    state.activeSidebarTab = activeTab;
    const isCollapsed = Boolean(state.sidebarCollapsed);

    if (workspace) {
      workspace.classList.toggle("sidebar-is-collapsed", isCollapsed);
    }
    if (sidebar) {
      sidebar.classList.toggle("is-collapsed", isCollapsed);
    }
    if (sidebarPanel) {
      sidebarPanel.classList.toggle("is-collapsed", isCollapsed);
    }
    if (sidebarToggleButton) {
      sidebarToggleButton.innerHTML = isCollapsed ? "&lt;&lt;" : "&gt;&gt;";
      sidebarToggleButton.setAttribute("aria-expanded", String(!isCollapsed));
      sidebarToggleButton.setAttribute(
        "aria-label",
        isCollapsed ? "Expand sidebar" : "Collapse sidebar"
      );
      sidebarToggleButton.title = isCollapsed
        ? "Expand sidebar"
        : "Collapse sidebar";
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
    renderSidebarTabs();
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
  }

  attachSidebarTabHandlers();
  renderSidebarTabs();

  Object.assign(ctx, {
    normalizeSidebarTab,
    renderSidebarTabs,
    setActiveSidebarTab,
    toggleSidebarCollapsed,
  });
}
