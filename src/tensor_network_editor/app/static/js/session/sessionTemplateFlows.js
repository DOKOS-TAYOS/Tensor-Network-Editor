import {
  buildExportTemplatePayload,
  createSessionTemplateImportSupport,
} from "./sessionTemplateImports.js";
import { createSessionTemplateDialogs } from "./sessionTemplateDialogs.js";
import { createSessionTemplateManager } from "./sessionTemplateManager.js";

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
  const { subnetworkLoadInput, templateLoadInput, templateManagerList } = dom;
  const sessionService = services.session;
  const subnetworkService = services.subnetwork;
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

  function isForModeActive() {
    return (
      (typeof actions.isForMode === "function" && actions.isForMode()) ||
      (typeof actions.isLinearPeriodicMode === "function" &&
        actions.isLinearPeriodicMode())
    );
  }

  function openSubnetworkPicker() {
    if (isForModeActive()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    sessionUi.openFilePicker(subnetworkLoadInput);
  }

  const dialogs = createSessionTemplateDialogs({
    actions,
    sessionUi,
  });
  const {
    promptForTemplateDisplayName,
    promptForSubnetworkName,
  } = dialogs;
  const importSupport = createSessionTemplateImportSupport({
    templateLoadInput,
    subnetworkLoadInput,
    state,
    sessionService,
    subnetworkService,
    commands,
    sessionUi,
    actions,
    getSubnetworkTargetCenter,
    isForModeActive,
  });
  const {
    loadSessionTemplatesFromFile,
    loadSubnetworkFromFile,
  } = importSupport;
  const templateManager = createSessionTemplateManager({
    templateManagerList,
    documentRef,
    state,
    actions,
  });
  const {
    discardTemplateManagerChanges,
    saveTemplateManagerChanges,
    toggleTemplateManager,
  } = templateManager;

  async function extractTemplateSpecByTensorIds(tensorIds, emptySelectionMessage) {
    if (isForModeActive()) {
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

  async function exportSubnetworkByTensorIds(tensorIds, label = "subnetwork") {
    if (isForModeActive()) {
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
      const resolvedDisplayName = promptForSubnetworkName(
        (payload.spec &&
          payload.spec.network &&
          typeof payload.spec.network.name === "string" &&
          payload.spec.network.name.trim()) ||
          label ||
          "subnetwork",
        "Subnetwork export cancelled."
      );
      if (!resolvedDisplayName) {
        return;
      }
      if (payload.spec && payload.spec.network) {
        payload.spec.network.name = resolvedDisplayName;
      }
      sessionUi.downloadText(
        `${actions.sanitizeFilename(resolvedDisplayName || "subnetwork")}.json`,
        JSON.stringify(payload.spec, null, 2),
        "application/json;charset=utf-8"
      );
      actions.setStatus(`Saved ${resolvedDisplayName} as JSON.`, "success");
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

  async function exportSelectedTemplateSpec() {
    const serializedSpec = await extractTemplateSpecByTensorIds(
      getSelectedTensorIds(),
      "Select one or more tensors to export as a template."
    );
    if (!serializedSpec) {
      return;
    }
    const displayName = promptForTemplateDisplayName(
      (serializedSpec.network &&
        typeof serializedSpec.network.name === "string" &&
        serializedSpec.network.name.trim()) ||
        "Selection Template",
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
    saveTemplateManagerChanges,
    discardTemplateManagerChanges,
    renameSelectedTemplate,
    deleteSelectedTemplate,
    loadSubnetworkFromFile,
    insertTemplate,
  };
}
