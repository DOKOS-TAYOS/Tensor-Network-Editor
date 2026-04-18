function extractFileStem(filename) {
  if (typeof filename !== "string" || !filename.trim()) {
    return "template";
  }
  const trimmedFilename = filename.trim();
  const lastDotIndex = trimmedFilename.lastIndexOf(".");
  return lastDotIndex > 0
    ? trimmedFilename.slice(0, lastDotIndex)
    : trimmedFilename;
}

function buildExportTemplatePayload(displayName, serializedSpec, sanitizeFilename) {
  return {
    schema_version: 1,
    templates: [
      {
        name: sanitizeFilename(displayName).replaceAll("-", "_") || "template",
        display_name: displayName,
        spec: serializedSpec,
      },
    ],
  };
}

function renderTrashIcon() {
  return `
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
    </svg>
  `;
}

export function createSessionTemplateFlows({
  dom,
  state,
  store,
  selectors,
  services,
  commands,
  sessionUi,
  actions,
}) {
  const {
    subnetworkLoadInput,
    templateLoadInput,
    templateManagerList,
  } = dom;
  const sessionService = services.session;
  const subnetworkService = services.subnetwork;
  let templateManagerDraft = null;
  const documentRef =
    templateManagerList && templateManagerList.ownerDocument
      ? templateManagerList.ownerDocument
      : globalThis.document;

  function getGroupById(groupId) {
    return actions.findGroupById(groupId);
  }

  function getSelectedTensorIds() {
    return actions.getSelectedTensorIds();
  }

  function getSubnetworkTargetCenter() {
    return actions.suggestTensorPosition(actions.viewportCenterPosition());
  }

  function getCurrentTemplateName() {
    return typeof dom.templateSelect.value === "string"
      ? dom.templateSelect.value.trim()
      : "";
  }

  function openSubnetworkPicker() {
    if (actions.isLinearPeriodicMode()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    sessionUi.openFilePicker(subnetworkLoadInput);
  }

  async function extractTemplateSpecByTensorIds(tensorIds, emptySelectionMessage) {
    if (actions.isLinearPeriodicMode()) {
      actions.setStatus(
        "Templates are only available in normal graph mode.",
        "error"
      );
      return null;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus(emptySelectionMessage);
      return null;
    }
    try {
      const payload = await subnetworkService.extractSubnetwork({
        serializedSpec: actions.serializeCurrentSpec({
          persistViewSnapshots: false,
        }),
        tensorIds,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return null;
      }
      return payload.spec;
    } catch (error) {
      actions.setStatus(`Could not prepare the template: ${error.message}`, "error");
      return null;
    }
  }

  async function exportSubnetworkByTensorIds(
    tensorIds,
    label = "subnetwork"
  ) {
    if (actions.isLinearPeriodicMode()) {
      actions.setStatus(
        "Subnetwork export is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus("Select one or more tensors to extract a subnetwork.");
      return;
    }
    try {
      const payload = await subnetworkService.extractSubnetwork({
        serializedSpec: actions.serializeCurrentSpec({
          persistViewSnapshots: false,
        }),
        tensorIds,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      sessionUi.downloadText(
        `${actions.sanitizeFilename(label || payload.spec.network.name || "subnetwork")}.json`,
        JSON.stringify(payload.spec, null, 2),
        "application/json;charset=utf-8"
      );
      actions.setStatus(
        `Saved ${payload.spec.network.name || "subnetwork"} as JSON.`,
        "success"
      );
    } catch (error) {
      actions.setStatus(`Could not export the subnetwork: ${error.message}`, "error");
    }
  }

  async function exportSelectedSubnetwork() {
    await exportSubnetworkByTensorIds(getSelectedTensorIds(), "subnetwork");
  }

  async function exportGroupSubnetwork(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to extract.", "error");
      return;
    }
    await exportSubnetworkByTensorIds(group.tensor_ids, group.name || "group");
  }

  async function saveTemplateByTensorIds(tensorIds, baseDisplayName) {
    const serializedSpec = await extractTemplateSpecByTensorIds(
      tensorIds,
      "Select one or more tensors to save as a template."
    );
    if (!serializedSpec) {
      return;
    }
    const resolvedDisplayName = promptForTemplateDisplayName(
      actions.getNextSessionTemplateDisplayName(baseDisplayName),
      "Template save cancelled."
    );
    if (!resolvedDisplayName) {
      return;
    }
    if (actions.hasTemplateDisplayName(resolvedDisplayName)) {
      actions.setStatus(
        `Template name '${resolvedDisplayName}' is already in use.`,
        "error"
      );
      return;
    }
    const addResult = actions.addSessionTemplate({
      displayName: resolvedDisplayName,
      spec: serializedSpec,
    });
    if (!addResult.ok) {
      actions.setStatus("Could not save the selected template.", "error");
      return;
    }
    actions.setStatus(`Saved ${resolvedDisplayName} for this session.`, "success");
  }

  async function saveSelectionAsSessionTemplate() {
    await saveTemplateByTensorIds(getSelectedTensorIds(), "Selection Template");
  }

  async function promoteSelectedSubnetworkToTemplate() {
    await saveSelectionAsSessionTemplate();
  }

  async function promoteGroupToTemplate(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to promote.", "error");
      return;
    }
    const baseDisplayName =
      typeof group.name === "string" && group.name.trim()
        ? group.name.trim()
        : "Group Template";
    await saveTemplateByTensorIds(group.tensor_ids, baseDisplayName);
  }

  function openSessionTemplatePicker() {
    sessionUi.openFilePicker(templateLoadInput);
  }

  function normalizeSerializedSpec(parsedValue) {
    if (
      parsedValue
      && typeof parsedValue === "object"
      && parsedValue.network
      && typeof parsedValue.network === "object"
    ) {
      return parsedValue;
    }
    if (
      parsedValue
      && typeof parsedValue === "object"
      && parsedValue.spec
      && typeof parsedValue.spec === "object"
      && parsedValue.spec.network
      && typeof parsedValue.spec.network === "object"
    ) {
      return parsedValue.spec;
    }
    return null;
  }

  function buildTemplateImportsFromFile(parsedValue, filename) {
    const templateEntries = [];
    if (
      parsedValue
      && typeof parsedValue === "object"
      && Array.isArray(parsedValue.templates)
    ) {
      parsedValue.templates.forEach((entry, index) => {
        if (!entry || typeof entry !== "object") {
          return;
        }
        const serializedSpec = normalizeSerializedSpec(entry.spec || entry);
        if (!serializedSpec) {
          return;
        }
        const displayName =
          (typeof entry.display_name === "string" && entry.display_name.trim())
          || (typeof entry.displayName === "string" && entry.displayName.trim())
          || (typeof entry.name === "string" && entry.name.trim())
          || (serializedSpec.network
            && typeof serializedSpec.network.name === "string"
            && serializedSpec.network.name.trim())
          || `${extractFileStem(filename)} ${index + 1}`;
        templateEntries.push({
          displayName,
          serializedSpec,
        });
      });
      return templateEntries;
    }
    const serializedSpec = normalizeSerializedSpec(parsedValue);
    if (!serializedSpec) {
      return templateEntries;
    }
    const displayName =
      (parsedValue
        && typeof parsedValue === "object"
        && typeof parsedValue.display_name === "string"
        && parsedValue.display_name.trim())
      || (serializedSpec.network
        && typeof serializedSpec.network.name === "string"
        && serializedSpec.network.name.trim())
      || extractFileStem(filename);
    templateEntries.push({
      displayName,
      serializedSpec,
    });
    return templateEntries;
  }

  async function loadSessionTemplatesFromFile(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    let loadedCount = 0;
    let duplicateCount = 0;
    let invalidCount = 0;
    const reservedDisplayNames = new Set(
      actions.listTemplateEntries().map((entry) => entry.displayName)
    );

    try {
      for (const file of files) {
        const fileText = await sessionUi.requestFileText(file, "utf-8");
        const parsedValue = JSON.parse(fileText);
        const templateImports = buildTemplateImportsFromFile(parsedValue, file.name);
        for (const templateImport of templateImports) {
          const displayName = templateImport.displayName.trim();
          if (!displayName || reservedDisplayNames.has(displayName)) {
            duplicateCount += 1;
            continue;
          }
          const validationResponse = await sessionService.validateSerializedSpec(
            templateImport.serializedSpec
          );
          if (!validationResponse.ok) {
            invalidCount += 1;
            continue;
          }
          const addResult = actions.addSessionTemplate({
            displayName,
            spec: validationResponse.spec,
            selected: false,
          });
          if (!addResult.ok) {
            duplicateCount += 1;
            continue;
          }
          reservedDisplayNames.add(displayName);
          loadedCount += 1;
        }
      }
      if (!loadedCount && !duplicateCount && !invalidCount) {
        actions.setStatus("No reusable templates were found in the selected files.", "error");
        return;
      }
      const summaryParts = [];
      if (loadedCount) {
        summaryParts.push(`Loaded ${loadedCount} template(s)`);
      }
      if (duplicateCount) {
        summaryParts.push(`skipped ${duplicateCount} duplicate name(s)`);
      }
      if (invalidCount) {
        summaryParts.push(`skipped ${invalidCount} invalid template(s)`);
      }
      actions.setStatus(`${summaryParts.join(", ")}.`, loadedCount ? "success" : "error");
    } catch (error) {
      actions.setStatus(`Could not load templates: ${error.message}`, "error");
    } finally {
      if (templateLoadInput) {
        templateLoadInput.value = "";
      }
    }
  }

  async function exportSelectedTemplateSpec() {
    const serializedSpec = await extractTemplateSpecByTensorIds(
      getSelectedTensorIds(),
      "Select one or more tensors to export as a template."
    );
    if (!serializedSpec) {
      return;
    }
    const displayName = promptForTemplateDisplayName(
      (
        serializedSpec.network
        && typeof serializedSpec.network.name === "string"
        && serializedSpec.network.name.trim()
      ) || "Selection Template",
      "Template export cancelled."
    );
    if (!displayName) {
      return;
    }
    const payload = buildExportTemplatePayload(
      displayName,
      serializedSpec,
      actions.sanitizeFilename
    );
    sessionUi.downloadText(
      `${actions.sanitizeFilename(displayName || "template")}.json`,
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
    actions.setStatus(`Exported ${displayName} as a reusable template.`, "success");
  }

  function promptForTemplateDisplayName(defaultDisplayName, cancelledStatus) {
    const promptedDisplayName = sessionUi.promptText(
      "Choose a name for this template.",
      defaultDisplayName
    );
    if (typeof promptedDisplayName !== "string") {
      actions.setStatus(cancelledStatus);
      return null;
    }
    const trimmedDisplayName = promptedDisplayName.trim();
    if (!trimmedDisplayName) {
      actions.setStatus("Template names cannot be empty.", "error");
      return null;
    }
    return trimmedDisplayName;
  }

  function buildTemplateManagerRow(entry) {
    const row = documentRef.createElement("div");
    const nameField = documentRef.createElement("label");
    const nameInput = documentRef.createElement("input");
    row.className = "template-manager-row";
    nameInput.value =
      templateManagerDraft.nameByTemplateName.get(entry.templateName) || entry.displayName;
    nameInput.dataset.templateName = entry.templateName;
    nameInput.setAttribute("aria-label", `Template name for ${entry.displayName}`);
    nameInput.disabled = false;
    nameField.append(nameInput);
    row.append(nameField);
    const deleteButton = documentRef.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "icon-button index-action-button danger";
    deleteButton.setAttribute("aria-label", `Delete ${entry.displayName}`);
    deleteButton.title = `Delete ${entry.displayName}`;
    deleteButton.innerHTML = renderTrashIcon();
    deleteButton.addEventListener("click", () => {
      templateManagerDraft.deletedTemplateNames.add(entry.templateName);
      renderTemplateManager();
    });
    row.appendChild(deleteButton);
    return row;
  }

  function renderTemplateManager() {
    if (!templateManagerList) {
      return;
    }
    templateManagerList.innerHTML = "";
    actions.listTemplateEntries().forEach((entry) => {
      if (entry.source !== "session") {
        return;
      }
      if (
        templateManagerDraft.deletedTemplateNames.has(entry.templateName)
      ) {
        return;
      }
      templateManagerList.appendChild(buildTemplateManagerRow(entry));
    });
  }

  function openTemplateManager() {
    const sessionEntries = actions
      .listTemplateEntries()
      .filter((entry) => entry.source === "session");
    templateManagerDraft = {
      nameByTemplateName: new Map(
        sessionEntries.map((entry) => [entry.templateName, entry.displayName])
      ),
      deletedTemplateNames: new Set(),
    };
    state.isTemplateManagerOpen = true;
    actions.syncTemplateManagerModalState();
    actions.setTemplateManagerValidationMessage("");
    renderTemplateManager();
  }

  function collectTemplateManagerUpdates() {
    const sessionEntries = actions
      .listTemplateEntries()
      .filter((entry) => entry.source === "session");
    return sessionEntries
      .filter((entry) => !templateManagerDraft.deletedTemplateNames.has(entry.templateName))
      .map((entry) => {
        const input = templateManagerList.querySelector(
          `input[data-template-name="${entry.templateName}"]`
        );
        return {
          templateName: entry.templateName,
          displayName:
            input && typeof input.value === "string" ? input.value.trim() : entry.displayName,
        };
      });
  }

  function validateTemplateManagerUpdates(updates) {
    const lockedDisplayNames = new Set(
      actions
        .listTemplateEntries()
        .filter((entry) => entry.source !== "session")
        .map((entry) => entry.displayName)
    );
    const seenSessionNames = new Set();
    for (const update of updates) {
      if (!update.displayName) {
        return "Template names cannot be empty.";
      }
      if (lockedDisplayNames.has(update.displayName) || seenSessionNames.has(update.displayName)) {
        return `Template name '${update.displayName}' is already in use.`;
      }
      seenSessionNames.add(update.displayName);
    }
    return "";
  }

  function closeTemplateManager() {
    if (!state.isTemplateManagerOpen) {
      return false;
    }
    const updates = collectTemplateManagerUpdates();
    const validationMessage = validateTemplateManagerUpdates(updates);
    if (validationMessage) {
      actions.setTemplateManagerValidationMessage(validationMessage);
      return true;
    }
    templateManagerDraft.deletedTemplateNames.forEach((templateName) => {
      actions.removeSessionTemplate(templateName);
    });
    actions.updateSessionTemplateDisplayNames(updates);
    state.isTemplateManagerOpen = false;
    templateManagerDraft = null;
    actions.setTemplateManagerValidationMessage("");
    actions.syncTemplateManagerModalState();
    actions.updateToolbarState();
    actions.setStatus("Updated session templates.", "success");
    return false;
  }

  function toggleTemplateManager(forceOpen) {
    const shouldOpen =
      typeof forceOpen === "boolean" ? forceOpen : !state.isTemplateManagerOpen;
    if (shouldOpen) {
      openTemplateManager();
      return true;
    }
    return closeTemplateManager();
  }

  async function loadSubnetworkFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    if (actions.isLinearPeriodicMode()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      if (subnetworkLoadInput) {
        subnetworkLoadInput.value = "";
      }
      return;
    }

    try {
      const fileText = await sessionUi.requestFileText(file, "utf-8");
      const parsed = JSON.parse(fileText);
      const serializedSpec =
        parsed && typeof parsed === "object" && parsed.network
          ? parsed
          : {
              schema_version: state.schemaVersion,
              network: parsed,
            };
      const response = await subnetworkService.prepareSubnetworkForInsert({
        serializedSpec,
        targetCenter: getSubnetworkTargetCenter(),
      });
      if (!response.ok) {
        actions.setStatus(
          response.message || actions.formatIssues(response.issues),
          "error"
        );
        return;
      }
      commands.insertPreparedSubnetwork(
        response.spec.network,
        response.spec.network.name || file.name
      );
    } catch (error) {
      actions.setStatus(`Could not insert ${file.name}: ${error.message}`, "error");
    } finally {
      if (subnetworkLoadInput) {
        subnetworkLoadInput.value = "";
      }
    }
  }

  async function insertTemplate() {
    const templateName = dom.templateSelect.value;
    if (!templateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    const templateSource = actions.getTemplateSource(templateName);
    try {
      let importedNetwork = null;
      if (templateSource === "session") {
        const serializedSpec = actions.getTemplateSpec(templateName);
        if (!serializedSpec || !serializedSpec.network) {
          actions.setStatus("Could not read the selected session template.", "error");
          return;
        }
        importedNetwork = serializedSpec.network;
      } else {
        const parameters = actions.persistTemplateParametersFromControls();
        const payload = await sessionService.buildTemplate({
          templateName,
          parameters,
        });
        importedNetwork = payload.spec.network;
      }
      const importedSpec = actions.uniquifyImportedSpec(
        importedNetwork,
        actions.makeId("template")
      );
      const translatedSpec = actions.translateImportedSpec(
        importedSpec,
        getSubnetworkTargetCenter()
      );
      actions.applyDesignChange(
        () => {
          state.spec.tensors.push(...translatedSpec.tensors);
          state.spec.edges.push(...translatedSpec.edges);
          state.spec.groups.push(...translatedSpec.groups);
          store.setLastImportedTensorIds(
            translatedSpec.tensors.map((tensor) => tensor.id)
          );
        },
        {
          invalidate: { lookups: true },
          selectionIds: translatedSpec.tensors.map((tensor) => tensor.id),
          primaryId: translatedSpec.tensors.length
            ? translatedSpec.tensors[translatedSpec.tensors.length - 1].id
            : null,
          statusMessage: `Inserted ${translatedSpec.name}.`,
        }
      );
    } catch (error) {
      actions.setStatus(`Could not insert the template: ${error.message}`, "error");
    }
  }

  async function renameSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isSessionTemplate(currentTemplateName)) {
      actions.setStatus(
        "Built-in and project templates are read-only in this editor.",
        "error"
      );
      return;
    }
    const currentEntry = actions
      .listTemplateEntries()
      .find((entry) => entry.templateName === currentTemplateName);
    const nextDisplayName = sessionUi.promptText(
      "Choose a new name for this session template.",
      currentEntry ? currentEntry.displayName : ""
    );
    if (typeof nextDisplayName !== "string") {
      actions.setStatus("Template rename cancelled.");
      return;
    }
    const trimmedDisplayName = nextDisplayName.trim();
    if (!trimmedDisplayName) {
      actions.setStatus("Template names cannot be empty.", "error");
      return;
    }
    if (actions.hasTemplateDisplayName(trimmedDisplayName, currentTemplateName)) {
      actions.setStatus(
        `Template name '${trimmedDisplayName}' is already in use.`,
        "error"
      );
      return;
    }
    actions.updateSessionTemplateDisplayNames([
      {
        templateName: currentTemplateName,
        displayName: trimmedDisplayName,
      },
    ]);
    actions.setStatus(`Renamed the template to ${trimmedDisplayName}.`, "success");
  }

  async function deleteSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isSessionTemplate(currentTemplateName)) {
      actions.setStatus(
        "Built-in and project templates are read-only in this editor.",
        "error"
      );
      return;
    }
    const currentEntry = actions
      .listTemplateEntries()
      .find((entry) => entry.templateName === currentTemplateName);
    if (
      !sessionUi.confirmAction(
        `Delete '${currentEntry ? currentEntry.displayName : "this template"}' from this session?`
      )
    ) {
      actions.setStatus("Template deletion cancelled.");
      return;
    }
    actions.removeSessionTemplate(currentTemplateName);
    actions.setStatus("Deleted the session template.", "success");
  }

  return {
    openSubnetworkPicker,
    exportSelectedSubnetwork,
    exportGroupSubnetwork,
    saveSelectionAsSessionTemplate,
    openSessionTemplatePicker,
    loadSessionTemplatesFromFile,
    exportSelectedTemplateSpec,
    promoteSelectedSubnetworkToTemplate,
    promoteGroupToTemplate,
    toggleTemplateManager,
    renameSelectedTemplate,
    deleteSelectedTemplate,
    loadSubnetworkFromFile,
    insertTemplate,
  };
}
