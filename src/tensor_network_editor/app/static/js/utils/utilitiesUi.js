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
    treeModeMenuItem,
    benchmarkModeMenuItem,
    toolbarModeControls,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    gridPeriodicUpCellButton,
    gridPeriodicDownCellButton,
    linearPeriodicNextCellButton,
    benchmarkSchemeNameInput,
    benchmarkCompareButton,
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
    connectButton,
    addNoteButton,
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
    aboutSchemaVersion,
    aboutLicense,
    aboutAuthor,
    templateManagerModal,
    templateManagerSaveButton,
    templateManagerDiscardButton,
    templateManagerError,
    benchmarkCompareModal,
    benchmarkCompareTableBody,
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
  const GRID_PERIODIC_CELL_LABELS = {
    top_left: "Top-left cell",
    top: "Top cell",
    top_right: "Top-right cell",
    left: "Left cell",
    center: "Center cell",
    right: "Right cell",
    bottom_left: "Bottom-left cell",
    bottom: "Bottom cell",
    bottom_right: "Bottom-right cell",
  };
  const TREE_PERIODIC_CELL_LABELS = {
    root: "Root cell",
    branch: "Branch cell",
    leaf: "Leaf cell",
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

  function setElementHidden(element, isHidden) {
    if (!element) {
      return;
    }
    element.hidden = Boolean(isHidden);
  }

  function setTooltipDescription(button, description) {
    if (!button) {
      return;
    }
    if (!button.dataset) {
      button.dataset = {};
    }
    if (typeof description === "string" && description) {
      button.dataset.shortcutDescription = description;
    } else {
      delete button.dataset.shortcutDescription;
    }
    const label =
      typeof button.dataset.shortcutLabel === "string"
        ? button.dataset.shortcutLabel.trim()
        : "";
    const shortcut =
      typeof button.dataset.shortcut === "string"
        ? button.dataset.shortcut.trim()
        : "";
    const header = shortcut ? `${label} (${shortcut})` : label;
    if (header && typeof button.setAttribute === "function") {
      button.setAttribute(
        "aria-label",
        description ? `${header}. ${description}` : header
      );
    }
    if (typeof button.removeAttribute === "function") {
      button.removeAttribute("title");
    }
  }

  function setButtonGroupDisabled(buttons, isDisabled, title) {
    buttons.forEach((button) => {
      if (!button) {
        return;
      }
      button.disabled = isDisabled;
      if (typeof title === "string") {
        setTooltipDescription(button, title);
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
    if (typeof runtime.highlightCodeElement === "function") {
      void runtime.highlightCodeElement(generatedCodeView);
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
    if (!codeGenerationWarning.dataset) {
      codeGenerationWarning.dataset = {};
    }
    codeGenerationWarning.dataset.tooltipEnabled = "true";
    codeGenerationWarning.dataset.shortcutLabel = "Code generation warning";
    codeGenerationWarning.textContent = warningMessage;
    setTooltipDescription(codeGenerationWarning, warningMessage);
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
    if (!templateCatalogWarning.dataset) {
      templateCatalogWarning.dataset = {};
    }
    templateCatalogWarning.dataset.tooltipEnabled = "true";
    templateCatalogWarning.dataset.shortcutLabel = "Template warnings";
    if (!warningMessages.length) {
      templateCatalogWarning.textContent = "";
      setTooltipDescription(templateCatalogWarning, "");
      templateCatalogWarning.hidden = true;
      return;
    }
    const extraWarningCount = warningMessages.length - 1;
    templateCatalogWarning.textContent =
      extraWarningCount > 0
        ? `${warningMessages[0]} (+${extraWarningCount} more)`
        : warningMessages[0];
    setTooltipDescription(templateCatalogWarning, warningMessages.join("\n"));
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
    if (aboutSchemaVersion) {
      aboutSchemaVersion.textContent =
        Number.isInteger(state.schemaVersion) || typeof state.schemaVersion === "string"
          ? String(state.schemaVersion)
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
        templateManagerSaveButton
        && typeof templateManagerSaveButton.focus === "function"
      ) {
        templateManagerSaveButton.focus();
      } else if (
        templateManagerDiscardButton
        && typeof templateManagerDiscardButton.focus === "function"
      ) {
        templateManagerDiscardButton.focus();
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
    const gridPeriodicMode =
      typeof runtime.isGridPeriodicMode === "function" && runtime.isGridPeriodicMode();
    const activeGridPeriodicCell =
      gridPeriodicMode && typeof runtime.getActiveGridPeriodicCellName === "function"
        ? runtime.getActiveGridPeriodicCellName()
        : null;
    const treePeriodicMode =
      typeof runtime.isTreePeriodicMode === "function" && runtime.isTreePeriodicMode();
    const activeTreePeriodicCell =
      treePeriodicMode && typeof runtime.getActiveTreePeriodicCellName === "function"
        ? runtime.getActiveTreePeriodicCellName()
        : null;
    const forMode = linearPeriodicMode || gridPeriodicMode || treePeriodicMode;
    const canSwitchGridPeriodicCell = (direction) =>
      typeof runtime.canSwitchGridPeriodicCell === "function" &&
      runtime.canSwitchGridPeriodicCell(direction);
    const canSwitchTreePeriodicCell = (direction) =>
      typeof runtime.canSwitchTreePeriodicCell === "function" &&
      runtime.canSwitchTreePeriodicCell(direction);
    const benchmarkMode =
      typeof runtime.isBenchmarkMode === "function" && runtime.isBenchmarkMode();
    const benchmarkSession =
      benchmarkMode && typeof runtime.getBenchmarkSession === "function"
        ? runtime.getBenchmarkSession()
        : null;
    const benchmarkActivePosition = benchmarkSession ? benchmarkSession.activePosition : 0;
    const benchmarkSchemeView = benchmarkMode && benchmarkActivePosition > 0;
    const primaryToolbarGroup =
      connectButton && connectButton.parentElement
        ? connectButton.parentElement
        : createGroupButton && createGroupButton.parentElement
          ? createGroupButton.parentElement
          : addNoteButton && addNoteButton.parentElement
            ? addNoteButton.parentElement
            : null;
    const primaryToolbarDivider =
      primaryToolbarGroup && primaryToolbarGroup.nextElementSibling
        ? primaryToolbarGroup.nextElementSibling
        : null;
    const templateToolbarGroup =
      templateSelect && templateSelect.parentElement
        ? templateSelect.parentElement.parentElement || templateSelect.parentElement
        : templateSettingsButton && templateSettingsButton.parentElement
          ? templateSettingsButton.parentElement.parentElement
              || templateSettingsButton.parentElement
          : reflowImportedButton && reflowImportedButton.parentElement
            ? reflowImportedButton.parentElement.parentElement
                || reflowImportedButton.parentElement
            : insertTemplateButton
              ? insertTemplateButton.parentElement || insertTemplateButton
              : null;
    const activeBenchmarkScheme =
      benchmarkMode && typeof runtime.getActiveBenchmarkScheme === "function"
        ? runtime.getActiveBenchmarkScheme()
        : null;
    const selectedTemplateValue = templateSelect ? templateSelect.value : "";
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

    if (undoButton) {
      undoButton.disabled = state.undoStack.length === 0;
    }
    if (redoButton) {
      redoButton.disabled = state.redoStack.length === 0;
    }
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
      saveSessionTemplateMenuItem.disabled = forMode || selectedTensorIds.length === 0;
    }
    if (loadSessionTemplateMenuItem) {
      loadSessionTemplateMenuItem.disabled = false;
    }
    if (exportSessionTemplateMenuItem) {
      exportSessionTemplateMenuItem.disabled = forMode || selectedTensorIds.length === 0;
    }
    if (editSessionTemplateMenuItem) {
      editSessionTemplateMenuItem.disabled = state.availableTemplates.length === 0;
    }
    setElementHidden(primaryToolbarGroup, benchmarkSchemeView);
    setElementHidden(primaryToolbarDivider, benchmarkSchemeView);
    setElementHidden(templateToolbarGroup, benchmarkSchemeView);
    if (insertTemplateButton) {
      insertTemplateButton.disabled = benchmarkSchemeView || !selectedTemplateValue;
      insertTemplateButton.hidden = benchmarkSchemeView;
    }
    if (reflowImportedButton) {
      reflowImportedButton.disabled =
        benchmarkSchemeView || selectedTensorIds.length === 0;
      setElementHidden(
        reflowImportedButton.parentElement || reflowImportedButton,
        benchmarkSchemeView
      );
      setTooltipDescription(
        reflowImportedButton,
        benchmarkSchemeView
          ? "Layout tools are unavailable while viewing a benchmark scheme."
          : selectedTensorIds.length === 0
          ? "Select at least one tensor first."
          : selectedTensorIds.length === 1
            ? "Reflow indices for the selected tensor."
            : "Choose a layout for the selected tensors or reflow their indices."
      );
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
      templateSettingsButton.disabled =
        benchmarkSchemeView || !selectedTemplateValue || forMode;
      setElementHidden(
        templateSettingsButton.parentElement || templateSettingsButton,
        benchmarkSchemeView
      );
      setTooltipDescription(
        templateSettingsButton,
        benchmarkSchemeView
          ? "Template parameters are unavailable while viewing a benchmark scheme."
          : !selectedTemplateValue
          ? "Choose a template first."
          : forMode
            ? "Template parameters are not editable in For mode."
            : "Edit template parameters."
      );
    }
    if (
      (benchmarkSchemeView || !selectedTemplateValue || forMode)
      && state.isTemplateSettingsOpen
    ) {
      state.isTemplateSettingsOpen = false;
    }
    if ((benchmarkSchemeView || selectedTensorIds.length === 0) && state.isReflowLayoutOpen) {
      state.isReflowLayoutOpen = false;
    }
    if (templateSelect) {
      templateSelect.disabled = benchmarkSchemeView;
      setElementHidden(templateSelect.parentElement || templateSelect, benchmarkSchemeView);
    }
    if (createGroupButton) {
      createGroupButton.disabled = selectedTensorIds.length < 2;
    }
    setMenuItemChecked(singleModeMenuItem, !forMode && !benchmarkMode);
    setMenuItemChecked(linearPeriodicModeMenuItem, linearPeriodicMode);
    if (gridPeriodicModeMenuItem) {
      setMenuItemChecked(gridPeriodicModeMenuItem, gridPeriodicMode);
    }
    if (treeModeMenuItem) {
      setMenuItemChecked(treeModeMenuItem, treePeriodicMode);
    }
    if (benchmarkModeMenuItem) {
      setMenuItemChecked(benchmarkModeMenuItem, benchmarkMode);
      benchmarkModeMenuItem.disabled = forMode;
      setTooltipDescription(
        benchmarkModeMenuItem,
        forMode
          ? "Benchmark mode is unavailable while a For mode is active."
          : "Compare manual contraction schemes on the current tensor network."
      );
    }
    if (toolbarModeControls) {
      toolbarModeControls.hidden = !(forMode || benchmarkMode);
    }
    if (linearPeriodicCellLabel && !benchmarkMode) {
      linearPeriodicCellLabel.hidden = false;
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : gridPeriodicMode
          ? GRID_PERIODIC_CELL_LABELS[activeGridPeriodicCell] || "Grid cell"
          : treePeriodicMode
            ? TREE_PERIODIC_CELL_LABELS[activeTreePeriodicCell] || "Tree cell"
          : "Single";
    }
    if (benchmarkSchemeNameInput && !benchmarkMode) {
      benchmarkSchemeNameInput.hidden = true;
      benchmarkSchemeNameInput.disabled = true;
      benchmarkSchemeNameInput.value = "";
    }
    if (benchmarkCompareButton && !benchmarkMode) {
      benchmarkCompareButton.hidden = true;
      benchmarkCompareButton.disabled = true;
      setTooltipDescription(
        benchmarkCompareButton,
        "Compare the saved contraction schemes."
      );
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicPreviousCellButton.disabled = benchmarkMode
        ? benchmarkActivePosition === 0
        : gridPeriodicMode
          ? !canSwitchGridPeriodicCell("left")
          : !linearPeriodicMode || activeLinearPeriodicCell === "initial";
      setTooltipDescription(
        linearPeriodicPreviousCellButton,
        benchmarkMode
          ? benchmarkActivePosition === 0
            ? "You are already at the tensor network view. In benchmark mode, use Previous and Next to move between the base network and the saved contraction schemes."
            : "Open the previous saved benchmark scheme. Use Previous and Next to move between the tensor network view and each saved scheme."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("left")
              ? "Move to the cell on the left. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the left edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
            : !linearPeriodicMode
              ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
            : activeLinearPeriodicCell === "initial"
              ? "You are already at the initial cell of the three-cell unidimensional workflow."
              : "Move to the previous cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicNextCellButton.disabled =
        !benchmarkMode &&
        (gridPeriodicMode
          ? !canSwitchGridPeriodicCell("right")
          : !linearPeriodicMode || activeLinearPeriodicCell === "final");
      linearPeriodicNextCellButton.textContent = benchmarkMode
        ? typeof runtime.getBenchmarkNextButtonLabel === "function"
          ? runtime.getBenchmarkNextButtonLabel()
          : ">"
        : ">";
      setTooltipDescription(
        linearPeriodicNextCellButton,
        benchmarkMode
          ? linearPeriodicNextCellButton.textContent === "+"
            ? "Create a new benchmark scheme after the current one. Use Next repeatedly to add schemes and then compare them in Planner."
            : "Open the next saved benchmark scheme. Use Previous and Next to move through the benchmark chain."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("right")
              ? "Move to the cell on the right. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the right edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
            : !linearPeriodicMode
              ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
            : activeLinearPeriodicCell === "final"
              ? "You are already at the final cell of the three-cell unidimensional workflow."
              : "Move to the next cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (gridPeriodicUpCellButton) {
      gridPeriodicUpCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicUpCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("up")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("up");
      setTooltipDescription(
        gridPeriodicUpCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("up")
            ? "Move to the upper cell."
            : "You are already at the top edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("up")
              ? "Move to the parent-facing cell above."
              : "You are already at the root cell."
      );
    }
    if (gridPeriodicDownCellButton) {
      gridPeriodicDownCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicDownCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("down")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("down");
      setTooltipDescription(
        gridPeriodicDownCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("down")
            ? "Move to the lower cell."
            : "You are already at the bottom edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("down")
              ? "Move to the child-facing cell below."
              : "You are already at the leaf cell."
      );
    }
    if (benchmarkMode) {
      if (linearPeriodicCellLabel) {
        linearPeriodicCellLabel.hidden = benchmarkActivePosition > 0;
        linearPeriodicCellLabel.textContent =
          typeof runtime.getBenchmarkBaseLabel === "function"
            ? runtime.getBenchmarkBaseLabel()
            : "Tensor network";
      }
      if (benchmarkSchemeNameInput) {
        benchmarkSchemeNameInput.hidden = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.disabled = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.value =
          benchmarkActivePosition > 0
            ? activeBenchmarkScheme && typeof activeBenchmarkScheme.name === "string"
              ? activeBenchmarkScheme.name
              : typeof runtime.getBenchmarkSchemeName === "function"
                ? runtime.getBenchmarkSchemeName(benchmarkActivePosition - 1)
                : `Scheme ${benchmarkActivePosition}`
            : "";
      }
      if (benchmarkCompareButton) {
        benchmarkCompareButton.hidden = false;
        benchmarkCompareButton.disabled = !(
          typeof runtime.canOpenBenchmarkCompare === "function" &&
          runtime.canOpenBenchmarkCompare()
        );
        setTooltipDescription(
          benchmarkCompareButton,
          benchmarkCompareButton.disabled
            ? "Create at least one scheme first."
            : "Compare the saved contraction schemes."
        );
      }
    }
    if (benchmarkCompareModal && benchmarkCompareTableBody) {
      if (typeof runtime.syncBenchmarkCompareModalState === "function") {
        runtime.syncBenchmarkCompareModalState();
      }
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
