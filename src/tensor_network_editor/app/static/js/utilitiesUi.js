export function createUtilityUiBindings({ ctx, state, dom, runtime }) {
  const {
    statusMessage,
    fileMenuButton,
    fileMenuPanel,
    modesMenuButton,
    modesMenuPanel,
    templatesMenuButton,
    templatesMenuPanel,
    helpMenuButton,
    helpMenuPanel,
    exportFormatSelect,
    generatedCode,
    generatedCodeView,
    codeGenerationWarning,
    undoButton,
    redoButton,
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    singleModeMenuItem,
    linearPeriodicModeMenuItem,
    gridPeriodicModeMenuItem,
    toolbarModeControls,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    linearPeriodicNextCellButton,
    templateSelect,
    templateSettingsButton,
    templateSettingsPopover,
    reflowLayoutPopover,
    insertTemplateButton,
    saveSessionTemplateMenuItem,
    loadSessionTemplateMenuItem,
    exportSessionTemplateMenuItem,
    editSessionTemplateMenuItem,
    templateCatalogWarning,
    reflowImportedButton,
    reflowAlignLeftButton,
    reflowAlignRightButton,
    reflowAlignTopButton,
    reflowAlignMiddleButton,
    reflowAlignBottomButton,
    reflowIndicesLeftButton,
    reflowIndicesRightButton,
    reflowIndicesTopButton,
    reflowIndicesResetButton,
    reflowIndicesBottomButton,
    reflowArrangeChainButton,
    reflowArrangeTreeButton,
    reflowArrangeGridButton,
    reflowDistributeHorizontalButton,
    reflowDistributeVerticalButton,
    reflowSnapGridButton,
    createGroupButton,
    generateButton,
    helpModal,
    helpCloseButton,
    helpSharedHeader,
    helpTitle,
    helpNote,
    helpInfoSection,
    helpShortcutsSection,
    helpAboutSection,
    aboutRepositoryLink,
    aboutVersion,
    aboutLicense,
    aboutAuthor,
    templateManagerModal,
    templateManagerCloseButton,
    templateManagerError,
  } = dom;

  const TOOLBAR_MENUS = {
    file: {
      button: fileMenuButton,
      panel: fileMenuPanel,
    },
    modes: {
      button: modesMenuButton,
      panel: modesMenuPanel,
    },
    templates: {
      button: templatesMenuButton,
      panel: templatesMenuPanel,
    },
    help: {
      button: helpMenuButton,
      panel: helpMenuPanel,
    },
  };
  const HELP_SECTION_CONTENT = {
    info: {
      title: "Info",
      note: "How to use the editor.",
    },
    shortcuts: {
      title: "Shortcuts",
      note: "",
    },
    about: {
      title: "About",
      note: "",
    },
  };
  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };
  const FLOATING_PANEL_MARGIN = 8;
  const FLOATING_PANEL_GAP = 4;

  function toggleElementClass(element, className, isActive) {
    if (
      !element
      || !element.classList
      || typeof element.classList.toggle !== "function"
    ) {
      return;
    }
    element.classList.toggle(className, isActive);
  }

  function setExpandedState(button, isExpanded) {
    if (!button || typeof button.setAttribute !== "function") {
      return;
    }
    button.setAttribute("aria-expanded", String(isExpanded));
  }

  function setButtonGroupDisabled(buttons, isDisabled, title) {
    buttons.forEach((button) => {
      if (!button) {
        return;
      }
      button.disabled = isDisabled;
      if (typeof title === "string") {
        button.title = title;
      }
    });
  }

  function setMenuItemChecked(menuItem, checked) {
    if (!menuItem) {
      return;
    }
    toggleElementClass(menuItem, "is-checked", checked);
    if (typeof menuItem.setAttribute === "function") {
      menuItem.setAttribute("aria-checked", String(checked));
    }
  }

  function setStyleVariable(element, propertyName, value) {
    if (!element || !element.style) {
      return;
    }
    if (typeof element.style.setProperty === "function") {
      element.style.setProperty(propertyName, value);
      return;
    }
    element.style[propertyName] = value;
  }

  function normalizeRect(rect, fallbackWidth = 0, fallbackHeight = 0) {
    const left = Number.isFinite(rect?.left) ? rect.left : 0;
    const top = Number.isFinite(rect?.top) ? rect.top : 0;
    const widthFromEdges =
      Number.isFinite(rect?.right) && Number.isFinite(rect?.left)
        ? Math.max(rect.right - rect.left, 0)
        : 0;
    const heightFromEdges =
      Number.isFinite(rect?.bottom) && Number.isFinite(rect?.top)
        ? Math.max(rect.bottom - rect.top, 0)
        : 0;
    const width =
      Number.isFinite(rect?.width) && rect.width > 0
        ? rect.width
        : Math.max(widthFromEdges, fallbackWidth);
    const height =
      Number.isFinite(rect?.height) && rect.height > 0
        ? rect.height
        : Math.max(heightFromEdges, fallbackHeight);

    return {
      left,
      top,
      width,
      height,
      right: Number.isFinite(rect?.right) ? rect.right : left + width,
      bottom: Number.isFinite(rect?.bottom) ? rect.bottom : top + height,
    };
  }

  function getElementRect(element, fallbackWidth = 0, fallbackHeight = 0) {
    const rawRect =
      element && typeof element.getBoundingClientRect === "function"
        ? element.getBoundingClientRect()
        : null;
    return normalizeRect(rawRect, fallbackWidth, fallbackHeight);
  }

  function clampFloatingOffset(offset, panelSize, viewportSize) {
    const minOffset = FLOATING_PANEL_MARGIN;
    const maxOffset = Math.max(
      FLOATING_PANEL_MARGIN,
      viewportSize - panelSize - FLOATING_PANEL_MARGIN
    );
    return Math.min(Math.max(offset, minOffset), maxOffset);
  }

  function positionFloatingPanel(
    panel,
    anchor,
    {
      align = "left",
      leftVariable,
      topVariable,
      fallbackWidth = 0,
      fallbackHeight = 0,
    }
  ) {
    if (!panel || !anchor) {
      return;
    }

    const windowRef = ctx.window && typeof ctx.window === "object" ? ctx.window : globalThis;
    const anchorRect = getElementRect(anchor);
    const panelRect = getElementRect(panel, fallbackWidth, fallbackHeight);
    const viewportWidth = Number.isFinite(windowRef.innerWidth)
      ? windowRef.innerWidth
      : anchorRect.right + panelRect.width + FLOATING_PANEL_MARGIN;
    const viewportHeight = Number.isFinite(windowRef.innerHeight)
      ? windowRef.innerHeight
      : anchorRect.bottom + panelRect.height + FLOATING_PANEL_MARGIN;
    const rawLeft =
      align === "right" ? anchorRect.right - panelRect.width : anchorRect.left;
    const left = clampFloatingOffset(rawLeft, panelRect.width, viewportWidth);
    const top = clampFloatingOffset(
      anchorRect.bottom + FLOATING_PANEL_GAP,
      panelRect.height,
      viewportHeight
    );

    setStyleVariable(panel, leftVariable, `${Math.round(left)}px`);
    setStyleVariable(panel, topVariable, `${Math.round(top)}px`);
  }

  function syncToolbarTransientUi() {
    const openToolbarMenu =
      typeof state.openToolbarMenu === "string" ? state.openToolbarMenu : null;
    Object.entries(TOOLBAR_MENUS).forEach(([menuName, elements]) => {
      const isOpen =
        openToolbarMenu === menuName && elements.button && !elements.button.disabled;
      if (elements.panel) {
        elements.panel.hidden = !isOpen;
        if (isOpen) {
          positionFloatingPanel(elements.panel, elements.button, {
            leftVariable: "--toolbar-menu-left",
            topVariable: "--toolbar-menu-top",
            fallbackWidth: 240,
            fallbackHeight: 240,
          });
        }
      }
      setExpandedState(elements.button, isOpen);
      toggleElementClass(elements.button, "is-active", isOpen);
    });
    const isTemplateSettingsOpen =
      Boolean(state.isTemplateSettingsOpen)
      && templateSettingsButton
      && !templateSettingsButton.disabled;
    if (templateSettingsPopover) {
      templateSettingsPopover.hidden = !isTemplateSettingsOpen;
      if (isTemplateSettingsOpen) {
        positionFloatingPanel(templateSettingsPopover, templateSettingsButton, {
          align: "right",
          leftVariable: "--template-settings-popover-left",
          topVariable: "--template-settings-popover-top",
          fallbackWidth: 280,
          fallbackHeight: 220,
        });
      }
    }
    setExpandedState(templateSettingsButton, isTemplateSettingsOpen);
    toggleElementClass(templateSettingsButton, "is-active", isTemplateSettingsOpen);
    const isReflowLayoutOpen =
      Boolean(state.isReflowLayoutOpen)
      && reflowImportedButton
      && !reflowImportedButton.disabled;
    if (reflowLayoutPopover) {
      reflowLayoutPopover.hidden = !isReflowLayoutOpen;
      if (isReflowLayoutOpen) {
        positionFloatingPanel(reflowLayoutPopover, reflowImportedButton, {
          align: "right",
          leftVariable: "--reflow-layout-popover-left",
          topVariable: "--reflow-layout-popover-top",
          fallbackWidth: 360,
          fallbackHeight: 340,
        });
      }
    }
    setExpandedState(reflowImportedButton, isReflowLayoutOpen);
    toggleElementClass(reflowImportedButton, "is-active", isReflowLayoutOpen);
  }

  function closeTransientToolbarUi() {
    const hadOpenUi = Boolean(
      state.openToolbarMenu || state.isTemplateSettingsOpen || state.isReflowLayoutOpen
    );
    if (!hadOpenUi) {
      return false;
    }
    state.openToolbarMenu = null;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    return true;
  }

  function openToolbarMenu(menuName) {
    if (!Object.prototype.hasOwnProperty.call(TOOLBAR_MENUS, menuName)) {
      return state.openToolbarMenu;
    }
    state.openToolbarMenu = menuName;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    return state.openToolbarMenu;
  }

  function toggleToolbarMenu(menuName) {
    if (!Object.prototype.hasOwnProperty.call(TOOLBAR_MENUS, menuName)) {
      return state.openToolbarMenu;
    }
    state.openToolbarMenu = state.openToolbarMenu === menuName ? null : menuName;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    return state.openToolbarMenu;
  }

  function toggleTemplateSettingsPopover() {
    if (!templateSettingsButton || templateSettingsButton.disabled) {
      return state.isTemplateSettingsOpen;
    }
    state.isTemplateSettingsOpen = !state.isTemplateSettingsOpen;
    state.openToolbarMenu = null;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    return state.isTemplateSettingsOpen;
  }

  function toggleReflowLayoutPopover() {
    if (!reflowImportedButton || reflowImportedButton.disabled) {
      return state.isReflowLayoutOpen;
    }
    state.isReflowLayoutOpen = !state.isReflowLayoutOpen;
    state.openToolbarMenu = null;
    state.isTemplateSettingsOpen = false;
    syncToolbarTransientUi();
    return state.isReflowLayoutOpen;
  }

  function renderGeneratedCodePreview(code = state.generatedCode) {
    const renderedCode = typeof code === "string" ? code : "";
    if (generatedCode) {
      generatedCode.value = renderedCode;
    }
    if (!generatedCodeView) {
      return;
    }
    generatedCodeView.textContent = renderedCode;
    const prism =
      ctx.window && typeof ctx.window === "object" ? ctx.window.Prism : null;
    if (prism && typeof prism.highlightElement === "function") {
      prism.highlightElement(generatedCodeView);
    }
  }

  function syncCodeGenerationWarning() {
    if (!codeGenerationWarning) {
      return;
    }
    const warningMessage =
      typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "";
    codeGenerationWarning.textContent = warningMessage;
    codeGenerationWarning.title = warningMessage;
    codeGenerationWarning.hidden = !warningMessage;
  }

  function syncTemplateCatalogWarning() {
    if (!templateCatalogWarning) {
      return;
    }
    const warningMessages = Array.isArray(state.templateCatalogWarnings)
      ? state.templateCatalogWarnings.filter(
          (warningMessage) => typeof warningMessage === "string" && warningMessage
        )
      : [];
    if (!warningMessages.length) {
      templateCatalogWarning.textContent = "";
      templateCatalogWarning.title = "";
      templateCatalogWarning.hidden = true;
      return;
    }
    const extraWarningCount = warningMessages.length - 1;
    templateCatalogWarning.textContent =
      extraWarningCount > 0
        ? `${warningMessages[0]} (+${extraWarningCount} more)`
        : warningMessages[0];
    templateCatalogWarning.title = warningMessages.join("\n");
    templateCatalogWarning.hidden = false;
  }

  function syncHelpModalState() {
    const helpSection = HELP_SECTION_CONTENT[state.activeHelpSection]
      ? state.activeHelpSection
      : "info";
    const sectionContent = HELP_SECTION_CONTENT[helpSection];
    const showSharedHelpHeader = true;
    const showSharedHelpNote = Boolean(sectionContent.note);
    if (helpSharedHeader) {
      helpSharedHeader.hidden = !showSharedHelpHeader;
    }
    if (helpTitle) {
      helpTitle.textContent = sectionContent.title;
      helpTitle.hidden = !showSharedHelpHeader;
    }
    if (helpNote) {
      helpNote.textContent = sectionContent.note;
      helpNote.hidden = !showSharedHelpNote;
    }
    if (helpInfoSection) {
      helpInfoSection.hidden = helpSection !== "info";
    }
    if (helpShortcutsSection) {
      helpShortcutsSection.hidden = helpSection !== "shortcuts";
    }
    if (helpAboutSection) {
      helpAboutSection.hidden = helpSection !== "about";
    }
    const appMetadata =
      state.appMetadata && typeof state.appMetadata === "object"
        ? state.appMetadata
        : {};
    if (aboutRepositoryLink) {
      const repositoryUrl =
        typeof appMetadata.repository_url === "string" && appMetadata.repository_url
          ? appMetadata.repository_url
          : "#";
      aboutRepositoryLink.href = repositoryUrl;
      aboutRepositoryLink.textContent = repositoryUrl === "#" ? "-" : repositoryUrl;
    }
    if (aboutVersion) {
      aboutVersion.textContent =
        typeof appMetadata.version === "string" && appMetadata.version
          ? appMetadata.version
          : "-";
    }
    if (aboutLicense) {
      aboutLicense.textContent =
        typeof appMetadata.license_name === "string" && appMetadata.license_name
          ? appMetadata.license_name
          : "-";
    }
    if (aboutAuthor) {
      aboutAuthor.textContent =
        typeof appMetadata.author_name === "string" && appMetadata.author_name
          ? appMetadata.author_name
          : "-";
    }
    if (helpModal) {
      helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    }
    if (state.isHelpOpen && helpCloseButton && typeof helpCloseButton.focus === "function") {
      helpCloseButton.focus();
    }
  }

  function toggleHelpModal(forceOpen, section = null) {
    if (typeof section === "string") {
      state.activeHelpSection = section;
    }
    state.isHelpOpen = typeof forceOpen === "boolean" ? forceOpen : !state.isHelpOpen;
    syncHelpModalState();
    return state.isHelpOpen;
  }

  function openHelpSection(section) {
    state.activeHelpSection = HELP_SECTION_CONTENT[section] ? section : "info";
    state.isHelpOpen = true;
    state.openToolbarMenu = null;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    syncHelpModalState();
  }

  function syncTemplateManagerModalState() {
    if (templateManagerModal) {
      templateManagerModal.classList.toggle(
        "is-hidden",
        !state.isTemplateManagerOpen
      );
    }
    if (state.isTemplateManagerOpen) {
      if (
        templateManagerCloseButton
        && typeof templateManagerCloseButton.focus === "function"
      ) {
        templateManagerCloseButton.focus();
      }
    } else if (templateManagerError) {
      templateManagerError.hidden = true;
      templateManagerError.textContent = "";
    }
  }

  function toggleTemplateManager(forceOpen) {
    state.isTemplateManagerOpen =
      typeof forceOpen === "boolean" ? forceOpen : !state.isTemplateManagerOpen;
    syncTemplateManagerModalState();
    return state.isTemplateManagerOpen;
  }

  function setTemplateManagerValidationMessage(message = "") {
    if (!templateManagerError) {
      return;
    }
    templateManagerError.textContent = message;
    templateManagerError.hidden = !message;
  }

  function updateToolbarState() {
    const linearPeriodicMode = runtime.isLinearPeriodicMode();
    const activeLinearPeriodicCell = runtime.getActiveLinearPeriodicCellName();
    const selectedTensorIds =
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [];
    const selectedTensors = selectedTensorIds
      .map((tensorId) =>
        typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null
      )
      .filter(Boolean);
    const hasSelectedIndices = selectedTensors.some(
      (tensor) => Array.isArray(tensor.indices) && tensor.indices.length > 0
    );
    runtime.enforceLinearPeriodicEngineSupport();
    syncCodeGenerationWarning();
    syncTemplateCatalogWarning();
    syncHelpModalState();
    syncTemplateManagerModalState();

    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
    if (generateButton) {
      generateButton.disabled = !state.spec || !state.selectedEngine;
    }
    if (exportPythonMenuItem) {
      exportPythonMenuItem.disabled = !state.spec || !state.selectedEngine;
    }
    if (exportPngMenuItem) {
      exportPngMenuItem.disabled = !state.spec;
    }
    if (exportSvgMenuItem) {
      exportSvgMenuItem.disabled = !state.spec;
    }
    if (saveSessionTemplateMenuItem) {
      saveSessionTemplateMenuItem.disabled =
        linearPeriodicMode || selectedTensorIds.length === 0;
    }
    if (loadSessionTemplateMenuItem) {
      loadSessionTemplateMenuItem.disabled = false;
    }
    if (exportSessionTemplateMenuItem) {
      exportSessionTemplateMenuItem.disabled =
        linearPeriodicMode || selectedTensorIds.length === 0;
    }
    if (editSessionTemplateMenuItem) {
      editSessionTemplateMenuItem.disabled = state.availableTemplates.length === 0;
    }
    if (insertTemplateButton) {
      insertTemplateButton.disabled = !templateSelect.value;
    }
    if (reflowImportedButton) {
      reflowImportedButton.disabled = selectedTensorIds.length === 0;
      reflowImportedButton.title =
        selectedTensorIds.length === 0
          ? "Select at least one tensor first."
          : selectedTensorIds.length === 1
            ? "Reflow indices for the selected tensor."
            : "Choose a layout for the selected tensors or reflow their indices.";
    }
    setButtonGroupDisabled(
      [
        reflowAlignLeftButton,
        reflowAlignRightButton,
        reflowAlignTopButton,
        reflowAlignMiddleButton,
        reflowAlignBottomButton,
        reflowArrangeChainButton,
        reflowArrangeTreeButton,
        reflowArrangeGridButton,
        reflowDistributeHorizontalButton,
        reflowDistributeVerticalButton,
        reflowSnapGridButton,
      ],
      selectedTensorIds.length < 2,
      selectedTensorIds.length < 2
        ? "Select at least two tensors first."
        : "Reflow the selected tensors."
    );
    setButtonGroupDisabled(
      [
        reflowIndicesLeftButton,
        reflowIndicesRightButton,
        reflowIndicesTopButton,
        reflowIndicesResetButton,
        reflowIndicesBottomButton,
      ],
      selectedTensorIds.length === 0 || !hasSelectedIndices,
      selectedTensorIds.length === 0
        ? "Select at least one tensor first."
        : !hasSelectedIndices
          ? "The selected tensors have no indices to reflow."
          : selectedTensorIds.length === 1
            ? "Reflow indices for the selected tensor."
            : "Reflow indices for the selected tensors."
    );
    if (templateSettingsButton) {
      templateSettingsButton.disabled = !templateSelect.value || linearPeriodicMode;
      templateSettingsButton.title = !templateSelect.value
        ? "Choose a template first."
        : linearPeriodicMode
          ? "Template parameters are not editable in For mode."
          : "Edit template parameters.";
    }
    if ((!templateSelect.value || linearPeriodicMode) && state.isTemplateSettingsOpen) {
      state.isTemplateSettingsOpen = false;
    }
    if (selectedTensorIds.length === 0 && state.isReflowLayoutOpen) {
      state.isReflowLayoutOpen = false;
    }
    createGroupButton.disabled = selectedTensorIds.length < 2;
    setMenuItemChecked(singleModeMenuItem, !linearPeriodicMode);
    setMenuItemChecked(linearPeriodicModeMenuItem, linearPeriodicMode);
    if (gridPeriodicModeMenuItem) {
      setMenuItemChecked(gridPeriodicModeMenuItem, false);
    }
    if (toolbarModeControls) {
      toolbarModeControls.hidden = !linearPeriodicMode;
    }
    if (linearPeriodicCellLabel) {
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : "Single";
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "initial";
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "final";
    }
    syncToolbarTransientUi();
  }

  function formatIssues(issues) {
    if (!issues || !issues.length) {
      return "The design is not valid yet.";
    }
    return issues
      .slice(0, 3)
      .map((issue) => issue.message)
      .join(" ");
  }

  function setStatus(message, kind = "info") {
    if (!statusMessage) {
      return;
    }
    statusMessage.textContent = message;
    statusMessage.classList.remove("status-error", "status-success");
    if (kind === "error") {
      statusMessage.classList.add("status-error");
    }
    if (kind === "success") {
      statusMessage.classList.add("status-success");
    }
  }

  return {
    syncToolbarTransientUi,
    closeTransientToolbarUi,
    openToolbarMenu,
    toggleToolbarMenu,
    toggleTemplateSettingsPopover,
    toggleReflowLayoutPopover,
    renderGeneratedCodePreview,
    updateToolbarState,
    syncCodeGenerationWarning,
    syncTemplateCatalogWarning,
    syncHelpModalState,
    toggleHelpModal,
    openHelpSection,
    syncTemplateManagerModalState,
    toggleTemplateManager,
    setTemplateManagerValidationMessage,
    formatIssues,
    setStatus,
  };
}
