export function createUtilityUiPanelsSupport({
  state,
  dom,
  positionFloatingPanel,
  setExpandedState,
  toggleElementClass,
}) {
  const {
    fileMenuButton,
    fileMenuPanel,
    themeMenuButton,
    themeMenuPanel,
    modesMenuButton,
    modesMenuPanel,
    templatesMenuButton,
    templatesMenuPanel,
    helpMenuButton,
    helpMenuPanel,
    exportMenuItem,
    exportSubmenuPanel,
    templateSettingsButton,
    templateSettingsPopover,
    reflowLayoutPopover,
    reflowImportedButton,
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
    templateManagerCloseButton,
    templateManagerSaveButton,
    templateManagerDiscardButton,
    templateManagerError,
    subnetworkLibraryModal,
    subnetworkLibraryCloseButton,
    subnetworkLibraryWarning,
  } = dom;

  const TOOLBAR_MENUS = {
    file: {
      button: fileMenuButton,
      panel: fileMenuPanel,
    },
    theme: {
      button: themeMenuButton,
      panel: themeMenuPanel,
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
      note: "Quick guide to the main editor workflows and current limits.",
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
    const isExportSubmenuOpen =
      openToolbarMenu === "file" &&
      state.openToolbarSubmenu === "export" &&
      exportMenuItem &&
      !exportMenuItem.disabled;
    if (exportSubmenuPanel) {
      exportSubmenuPanel.hidden = !isExportSubmenuOpen;
    }
    setExpandedState(exportMenuItem, isExportSubmenuOpen);
    toggleElementClass(exportMenuItem, "is-active", isExportSubmenuOpen);
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
    state.openToolbarSubmenu = null;
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
    state.openToolbarSubmenu = null;
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
    state.openToolbarSubmenu = null;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    return state.openToolbarMenu;
  }

  function openToolbarSubmenu(submenuName) {
    if (!state.openToolbarMenu || !submenuName) {
      return state.openToolbarSubmenu;
    }
    state.openToolbarSubmenu = submenuName;
    syncToolbarTransientUi();
    return state.openToolbarSubmenu;
  }

  function closeToolbarSubmenu(submenuName = null) {
    if (submenuName && state.openToolbarSubmenu !== submenuName) {
      return state.openToolbarSubmenu;
    }
    state.openToolbarSubmenu = null;
    syncToolbarTransientUi();
    return state.openToolbarSubmenu;
  }

  function toggleToolbarSubmenu(submenuName) {
    if (!state.openToolbarMenu || !submenuName) {
      return state.openToolbarSubmenu;
    }
    state.openToolbarSubmenu =
      state.openToolbarSubmenu === submenuName ? null : submenuName;
    syncToolbarTransientUi();
    return state.openToolbarSubmenu;
  }

  function toggleTemplateSettingsPopover() {
    if (!templateSettingsButton || templateSettingsButton.disabled) {
      return state.isTemplateSettingsOpen;
    }
    state.isTemplateSettingsOpen = !state.isTemplateSettingsOpen;
    state.openToolbarMenu = null;
    state.openToolbarSubmenu = null;
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
    state.openToolbarSubmenu = null;
    state.isTemplateSettingsOpen = false;
    syncToolbarTransientUi();
    return state.isReflowLayoutOpen;
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
    state.openToolbarSubmenu = null;
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
      } else if (
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

  function syncSubnetworkLibraryModalState() {
    if (subnetworkLibraryModal) {
      subnetworkLibraryModal.classList.toggle(
        "is-hidden",
        !state.isSubnetworkLibraryOpen
      );
    }
    if (state.isSubnetworkLibraryOpen) {
      if (
        subnetworkLibraryCloseButton &&
        typeof subnetworkLibraryCloseButton.focus === "function"
      ) {
        subnetworkLibraryCloseButton.focus();
      }
    } else if (subnetworkLibraryWarning) {
      subnetworkLibraryWarning.hidden = true;
      subnetworkLibraryWarning.textContent = "";
    }
  }

  function toggleSubnetworkLibrary(forceOpen) {
    state.isSubnetworkLibraryOpen =
      typeof forceOpen === "boolean"
        ? forceOpen
        : !state.isSubnetworkLibraryOpen;
    syncSubnetworkLibraryModalState();
    return state.isSubnetworkLibraryOpen;
  }

  function setTemplateManagerValidationMessage(message = "") {
    if (!templateManagerError) {
      return;
    }
    templateManagerError.textContent = message;
    templateManagerError.hidden = !message;
  }

  return {
    syncToolbarTransientUi,
    closeTransientToolbarUi,
    openToolbarMenu,
    openToolbarSubmenu,
    closeToolbarSubmenu,
    toggleToolbarSubmenu,
    toggleToolbarMenu,
    toggleTemplateSettingsPopover,
    toggleReflowLayoutPopover,
    syncHelpModalState,
    toggleHelpModal,
    openHelpSection,
    syncTemplateManagerModalState,
    toggleTemplateManager,
    setTemplateManagerValidationMessage,
    syncSubnetworkLibraryModalState,
    toggleSubnetworkLibrary,
  };
}
