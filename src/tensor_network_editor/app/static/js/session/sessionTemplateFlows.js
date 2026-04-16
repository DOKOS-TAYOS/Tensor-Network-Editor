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
  const { subnetworkLoadInput, templateSelect } = dom;
  const sessionService = services.session;
  const templateCatalogService = services.templateCatalog;
  const subnetworkService = services.subnetwork;

  function applyTemplateCatalogUpdate(
    payload,
    successMessage = "Updated the template catalog."
  ) {
    commands.applyTemplateCatalogUpdate(payload, successMessage);
  }

  function requestTemplateName(defaultTemplateName = "fragment_template") {
    const proposedName = sessionUi.promptText(
      "Choose a template name using lowercase letters, digits, and underscores.",
      defaultTemplateName
    );
    if (typeof proposedName !== "string") {
      return null;
    }
    const normalizedName = proposedName.trim();
    return normalizedName || null;
  }

  function confirmTemplateOverwrite(templateName, operationLabel) {
    return sessionUi.confirmAction(
      `${operationLabel} '${actions.formatTemplateLabel(templateName)}'? This replaces the existing project template.`
    );
  }

  function resolveTemplateOverwriteDecision(templateName, operationLabel) {
    const existingDefinition = selectors.getTemplateDefinition(templateName);
    if (!existingDefinition) {
      return { overwrite: false };
    }
    if (!selectors.isProjectTemplate(templateName)) {
      actions.setStatus(
        `Template '${templateName}' is registered globally and cannot be replaced.`,
        "error"
      );
      return null;
    }
    if (!confirmTemplateOverwrite(templateName, operationLabel)) {
      actions.setStatus(`${operationLabel} cancelled.`);
      return null;
    }
    return { overwrite: true };
  }

  function getGroupById(groupId) {
    return actions.findGroupById(groupId);
  }

  function getSelectedTensorIds() {
    return actions.getSelectedTensorIds();
  }

  function getSubnetworkTargetCenter() {
    return actions.suggestTensorPosition(actions.viewportCenterPosition());
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

  async function promoteSubnetworkByTensorIds(
    tensorIds,
    defaultTemplateName = "fragment_template"
  ) {
    if (actions.isLinearPeriodicMode()) {
      actions.setStatus(
        "Template promotion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus("Select one or more tensors to promote as a template.");
      return;
    }
    const templateName = requestTemplateName(defaultTemplateName);
    if (!templateName) {
      actions.setStatus("Template promotion cancelled.");
      return;
    }
    const overwriteDecision = resolveTemplateOverwriteDecision(
      templateName,
      "Replace template"
    );
    if (overwriteDecision === null) {
      return;
    }
    try {
      const payload = await templateCatalogService.promoteTemplate({
        serializedSpec: actions.serializeCurrentSpec({
          persistViewSnapshots: false,
        }),
        tensorIds,
        templateName,
        overwrite: overwriteDecision.overwrite,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Saved ${actions.formatTemplateLabel(templateName)} to the template catalog.`
      );
    } catch (error) {
      actions.setStatus(`Could not save the template: ${error.message}`, "error");
    }
  }

  async function promoteSelectedSubnetworkToTemplate() {
    await promoteSubnetworkByTensorIds(getSelectedTensorIds(), "selection_template");
  }

  async function promoteGroupToTemplate(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to promote.", "error");
      return;
    }
    await promoteSubnetworkByTensorIds(
      group.tensor_ids,
      typeof group.name === "string" && group.name
        ? actions.sanitizeFilename(group.name).replaceAll("-", "_")
        : "group_template"
    );
  }

  function getCurrentTemplateName() {
    return typeof templateSelect.value === "string" ? templateSelect.value.trim() : "";
  }

  async function renameSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isProjectTemplate(currentTemplateName)) {
      actions.setStatus(
        "Only project-local templates can be renamed from the editor.",
        "error"
      );
      return;
    }
    const newTemplateName = requestTemplateName(currentTemplateName);
    if (!newTemplateName) {
      actions.setStatus("Template rename cancelled.");
      return;
    }
    if (newTemplateName === currentTemplateName) {
      actions.setStatus("Template name unchanged.");
      return;
    }
    const overwriteDecision = resolveTemplateOverwriteDecision(
      newTemplateName,
      "Replace template"
    );
    if (overwriteDecision === null) {
      return;
    }
    try {
      const payload = await templateCatalogService.renameTemplate({
        templateName: currentTemplateName,
        newTemplateName,
        overwrite: overwriteDecision.overwrite,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Renamed ${actions.formatTemplateLabel(newTemplateName)}.`
      );
    } catch (error) {
      actions.setStatus(`Could not rename the template: ${error.message}`, "error");
    }
  }

  async function deleteSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isProjectTemplate(currentTemplateName)) {
      actions.setStatus(
        "Only project-local templates can be deleted from the editor.",
        "error"
      );
      return;
    }
    const currentTemplateLabel = actions.formatTemplateLabel(currentTemplateName);
    if (
      !sessionUi.confirmAction(
        `Delete template '${currentTemplateLabel}' from this project?`
      )
    ) {
      actions.setStatus("Template deletion cancelled.");
      return;
    }
    try {
      const payload = await templateCatalogService.deleteTemplate({
        templateName: currentTemplateName,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Deleted ${currentTemplateLabel} from the template catalog.`
      );
    } catch (error) {
      actions.setStatus(`Could not delete the template: ${error.message}`, "error");
    }
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
    const templateName = templateSelect.value;
    if (!templateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    const parameters = actions.persistTemplateParametersFromControls();
    try {
      const payload = await sessionService.buildTemplate({
        templateName,
        parameters,
      });
      const importedSpec = actions.uniquifyImportedSpec(
        payload.spec.network,
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

  return {
    openSubnetworkPicker,
    exportSelectedSubnetwork,
    exportGroupSubnetwork,
    promoteSelectedSubnetworkToTemplate,
    promoteGroupToTemplate,
    renameSelectedTemplate,
    deleteSelectedTemplate,
    loadSubnetworkFromFile,
    insertTemplate,
  };
}
