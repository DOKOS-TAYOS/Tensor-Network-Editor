export function registerSidebarTabs(ctx) {
  const state = ctx.state;
  const {
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

    Object.entries(tabConfig).forEach(([tabName, config]) => {
      const isActive = tabName === activeTab;
      if (config.button) {
        config.button.classList.toggle("is-active", isActive);
        config.button.setAttribute("aria-selected", String(isActive));
        config.button.setAttribute("tabindex", isActive ? "0" : "-1");
      }
      if (config.pane) {
        config.pane.classList.toggle("is-active", isActive);
        config.pane.hidden = !isActive;
      }
    });
  }

  function setActiveSidebarTab(tabName) {
    state.activeSidebarTab = normalizeSidebarTab(tabName);
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
  }

  attachSidebarTabHandlers();
  renderSidebarTabs();

  Object.assign(ctx, {
    normalizeSidebarTab,
    renderSidebarTabs,
    setActiveSidebarTab,
  });
}
