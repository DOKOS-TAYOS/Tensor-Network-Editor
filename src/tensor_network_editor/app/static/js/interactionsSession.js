import { createSessionCommands } from "./actions/sessionCommands.js";
import { createEditorSessionService } from "./services/editorSessionService.js";
import { createSubnetworkService } from "./services/subnetworkService.js";
import { createTemplateCatalogService } from "./services/templateCatalogService.js";
import { createEditorSelectors } from "./state/editorSelectors.js";
import { createEditorStore } from "./state/editorStore.js";

export function createInteractionSessionBindings({ ctx, state, dom }) {
  const {
    exportFormatSelect,
    generatedCode,
    loadInput,
    subnetworkLoadInput,
    templateSelect,
  } = dom;
  const { apiGet, apiPost, document, window } = ctx;
  const store = ctx.store || createEditorStore(state);
  const selectors = ctx.selectors || createEditorSelectors({ store });
  const sessionService =
    ctx.services && ctx.services.session
      ? ctx.services.session
      : createEditorSessionService({ apiGet, apiPost });
  const templateCatalogService =
    ctx.services && ctx.services.templateCatalog
      ? ctx.services.templateCatalog
      : createTemplateCatalogService({ apiPost });
  const subnetworkService =
    ctx.services && ctx.services.subnetwork
      ? ctx.services.subnetwork
      : createSubnetworkService({ apiPost });
  const commands = createSessionCommands({
    dom,
    state,
    store,
    window,
    setStatus: ctx.setStatus,
    applyTemplateCatalogPayload: ctx.applyTemplateCatalogPayload,
    normalizeSpec: ctx.normalizeSpec,
    applyDesignChange: ctx.applyDesignChange,
    bringTensorToFront: ctx.bringTensorToFront,
  });
  const clipboard =
    window &&
    window.navigator &&
    window.navigator.clipboard &&
    typeof window.navigator.clipboard.writeText === "function"
      ? window.navigator.clipboard
      : null;

  function isLinearPeriodicMode() {
    return typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode();
  }

  function ensureCodePanelVisible() {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
  }

  function syncCodeGenerationWarning() {
    if (typeof ctx.syncCodeGenerationWarning === "function") {
      ctx.syncCodeGenerationWarning();
    }
  }

  function getTensorKrowchManualPlanIssueMessage() {
    return typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
      ? ctx.getTensorKrowchManualPlanIssueMessage()
      : "";
  }

  function getSelectedTensorIds() {
    return typeof ctx.getSelectedIdsByKind === "function"
      ? ctx.getSelectedIdsByKind("tensor")
      : [];
  }

  function getGroupById(groupId) {
    return typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
  }

  function syncGeneratedCodePreview(code) {
    if (typeof ctx.renderGeneratedCodePreview === "function") {
      ctx.renderGeneratedCodePreview(code);
    } else {
      commands.syncGeneratedCodePreview(code);
    }
  }

  function showCodeGenerationError(message) {
    const safeMessage = message || "Code generation failed.";
    store.setGeneratedCode(`Code generation failed:\n${safeMessage}`);
    syncGeneratedCodePreview(state.generatedCode);
    ctx.setStatus(safeMessage, "error");
  }

  async function generateCode() {
    ensureCodePanelVisible();
    syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      ctx.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await sessionService.generateCode({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        showCodeGenerationError(payload.message || ctx.formatIssues(payload.issues));
        return;
      }
      store.setGeneratedCode(ctx.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      ctx.setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      showCodeGenerationError(`Code generation failed: ${error.message}`);
    }
  }

  async function completeEditor() {
    try {
      const payload = await sessionService.completeSession({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      store.setEditorFinished(true);
      ctx.setStatus("Returning the design to Python. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      ctx.setStatus(`Could not finish the editor session: ${error.message}`, "error");
    }
  }

  async function cancelEditor() {
    try {
      store.setEditorFinished(true);
      await sessionService.cancelSession();
      ctx.setStatus("Editor cancelled. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      ctx.setStatus(`Could not cancel the editor session: ${error.message}`, "error");
    }
  }

  function saveDesign() {
    const blob = new Blob(
      [JSON.stringify(ctx.serializeCurrentSpec({ persistViewSnapshots: true }), null, 2)],
      {
        type: "application/json",
      }
    );
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${ctx.sanitizeFilename(state.spec.name || "tensor-network")}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    ctx.setStatus("Design downloaded as JSON.");
  }

  function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const fileText = typeof reader.result === "string" ? reader.result : "";
        const isPythonSource = file.name.toLowerCase().endsWith(".py");
        const response = isPythonSource
          ? await sessionService.validatePythonCode(fileText)
          : await sessionService.validateSerializedSpec(JSON.parse(fileText));
        if (!response.ok) {
          ctx.setStatus(ctx.formatIssues(response.issues), "error");
          return;
        }
        ctx.resetDesignState(
          response.spec.network,
          `Loaded design from ${file.name}. History cleared.`,
          response.spec.schema_version
        );
      } catch (error) {
        ctx.setStatus(`Could not load ${file.name}: ${error.message}`, "error");
      } finally {
        loadInput.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  }

  function openSubnetworkPicker() {
    if (isLinearPeriodicMode()) {
      ctx.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (subnetworkLoadInput) {
      subnetworkLoadInput.click();
    }
  }

  function applyTemplateCatalogUpdate(
    payload,
    successMessage = "Updated the template catalog."
  ) {
    commands.applyTemplateCatalogUpdate(payload, successMessage);
  }

  function requestTemplateName(defaultTemplateName = "fragment_template") {
    const proposedName =
      window && typeof window.prompt === "function"
        ? window.prompt(
            "Choose a template name using lowercase letters, digits, and underscores.",
            defaultTemplateName
          )
        : defaultTemplateName;
    if (typeof proposedName !== "string") {
      return null;
    }
    const normalizedName = proposedName.trim();
    return normalizedName || null;
  }

  function confirmTemplateOverwrite(templateName, operationLabel) {
    if (!window || typeof window.confirm !== "function") {
      return true;
    }
    return window.confirm(
      `${operationLabel} '${ctx.formatTemplateLabel(templateName)}'? This replaces the existing project template.`
    );
  }

  function resolveTemplateOverwriteDecision(templateName, operationLabel) {
    const existingDefinition = selectors.getTemplateDefinition(templateName);
    if (!existingDefinition) {
      return { overwrite: false };
    }
    if (!selectors.isProjectTemplate(templateName)) {
      ctx.setStatus(
        `Template '${templateName}' is registered globally and cannot be replaced.`,
        "error"
      );
      return null;
    }
    if (!confirmTemplateOverwrite(templateName, operationLabel)) {
      ctx.setStatus(`${operationLabel} cancelled.`);
      return null;
    }
    return { overwrite: true };
  }

  function insertPreparedSubnetwork(preparedSpec, label = null) {
    commands.insertPreparedSubnetwork(preparedSpec, label);
  }

  async function exportSubnetworkByTensorIds(tensorIds, label = "subnetwork") {
    if (isLinearPeriodicMode()) {
      ctx.setStatus(
        "Subnetwork export is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      ctx.setStatus("Select one or more tensors to extract a subnetwork.");
      return;
    }
    try {
      const payload = await subnetworkService.extractSubnetwork({
        serializedSpec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
        tensorIds,
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      ctx.downloadBlob(
        `${ctx.sanitizeFilename(label || payload.spec.network.name || "subnetwork")}.json`,
        new Blob([JSON.stringify(payload.spec, null, 2)], {
          type: "application/json;charset=utf-8",
        })
      );
      ctx.setStatus(
        `Saved ${payload.spec.network.name || "subnetwork"} as JSON.`,
        "success"
      );
    } catch (error) {
      ctx.setStatus(`Could not export the subnetwork: ${error.message}`, "error");
    }
  }

  async function exportSelectedSubnetwork() {
    await exportSubnetworkByTensorIds(getSelectedTensorIds(), "subnetwork");
  }

  async function exportGroupSubnetwork(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      ctx.setStatus("This group does not contain any tensors to extract.", "error");
      return;
    }
    await exportSubnetworkByTensorIds(group.tensor_ids, group.name || "group");
  }

  async function promoteSubnetworkByTensorIds(
    tensorIds,
    defaultTemplateName = "fragment_template"
  ) {
    if (isLinearPeriodicMode()) {
      ctx.setStatus(
        "Template promotion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      ctx.setStatus("Select one or more tensors to promote as a template.");
      return;
    }
    const templateName = requestTemplateName(defaultTemplateName);
    if (!templateName) {
      ctx.setStatus("Template promotion cancelled.");
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
        serializedSpec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
        tensorIds,
        templateName,
        overwrite: overwriteDecision.overwrite,
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Saved ${ctx.formatTemplateLabel(templateName)} to the template catalog.`
      );
    } catch (error) {
      ctx.setStatus(`Could not save the template: ${error.message}`, "error");
    }
  }

  async function promoteSelectedSubnetworkToTemplate() {
    await promoteSubnetworkByTensorIds(getSelectedTensorIds(), "selection_template");
  }

  async function promoteGroupToTemplate(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      ctx.setStatus("This group does not contain any tensors to promote.", "error");
      return;
    }
    await promoteSubnetworkByTensorIds(
      group.tensor_ids,
      typeof group.name === "string" && group.name
        ? ctx.sanitizeFilename(group.name).replaceAll("-", "_")
        : "group_template"
    );
  }

  async function renameSelectedTemplate() {
    const currentTemplateName =
      typeof templateSelect.value === "string" ? templateSelect.value.trim() : "";
    if (!currentTemplateName) {
      ctx.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isProjectTemplate(currentTemplateName)) {
      ctx.setStatus(
        "Only project-local templates can be renamed from the editor.",
        "error"
      );
      return;
    }
    const newTemplateName = requestTemplateName(currentTemplateName);
    if (!newTemplateName) {
      ctx.setStatus("Template rename cancelled.");
      return;
    }
    if (newTemplateName === currentTemplateName) {
      ctx.setStatus("Template name unchanged.");
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
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Renamed ${ctx.formatTemplateLabel(newTemplateName)}.`
      );
    } catch (error) {
      ctx.setStatus(`Could not rename the template: ${error.message}`, "error");
    }
  }

  async function deleteSelectedTemplate() {
    const currentTemplateName =
      typeof templateSelect.value === "string" ? templateSelect.value.trim() : "";
    if (!currentTemplateName) {
      ctx.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isProjectTemplate(currentTemplateName)) {
      ctx.setStatus(
        "Only project-local templates can be deleted from the editor.",
        "error"
      );
      return;
    }
    const currentTemplateLabel = ctx.formatTemplateLabel(currentTemplateName);
    if (
      window &&
      typeof window.confirm === "function" &&
      !window.confirm(`Delete template '${currentTemplateLabel}' from this project?`)
    ) {
      ctx.setStatus("Template deletion cancelled.");
      return;
    }
    try {
      const payload = await templateCatalogService.deleteTemplate({
        templateName: currentTemplateName,
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      applyTemplateCatalogUpdate(
        payload,
        `Deleted ${currentTemplateLabel} from the template catalog.`
      );
    } catch (error) {
      ctx.setStatus(`Could not delete the template: ${error.message}`, "error");
    }
  }

  function loadSubnetworkFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    if (isLinearPeriodicMode()) {
      ctx.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      if (subnetworkLoadInput) {
        subnetworkLoadInput.value = "";
      }
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const fileText = typeof reader.result === "string" ? reader.result : "";
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
          targetCenter: ctx.suggestTensorPosition(ctx.viewportCenterPosition()),
        });
        if (!response.ok) {
          ctx.setStatus(
            response.message || ctx.formatIssues(response.issues),
            "error"
          );
          return;
        }
        insertPreparedSubnetwork(
          response.spec.network,
          response.spec.network.name || file.name
        );
      } catch (error) {
        ctx.setStatus(`Could not insert ${file.name}: ${error.message}`, "error");
      } finally {
        if (subnetworkLoadInput) {
          subnetworkLoadInput.value = "";
        }
      }
    };
    reader.readAsText(file, "utf-8");
  }

  async function copyGeneratedCode() {
    const codeToCopy = ctx.stripImportLines(generatedCode.value);
    if (!codeToCopy.trim()) {
      ctx.setStatus("There is no generated code to copy yet.");
      return;
    }
    if (!clipboard) {
      ctx.setStatus("Clipboard access is not available in this browser.", "error");
      return;
    }
    try {
      await clipboard.writeText(codeToCopy);
      ctx.setStatus("Generated code copied to the clipboard without import lines.", "success");
    } catch (error) {
      ctx.setStatus(`Could not copy the generated code: ${error.message}`, "error");
    }
  }

  async function downloadPythonExport() {
    ensureCodePanelVisible();
    syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      ctx.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await sessionService.generateCode({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        showCodeGenerationError(payload.message || ctx.formatIssues(payload.issues));
        return;
      }
      store.setGeneratedCode(ctx.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      ctx.downloadBlob(
        `${ctx.sanitizeFilename(state.spec.name || "tensor-network")}-${ctx.sanitizeFilename(selectors.getSelectedEngine() || "engine")}.py`,
        new Blob([payload.code], { type: "text/x-python;charset=utf-8" })
      );
      ctx.setStatus(`Exported ${payload.engine} Python code.`, "success");
    } catch (error) {
      showCodeGenerationError(`Could not export Python code: ${error.message}`);
    }
  }

  async function downloadSelectedExport() {
    switch (exportFormatSelect.value) {
      case "png":
        ctx.downloadPngExport();
        break;
      case "svg":
        ctx.downloadSvgExport();
        break;
      case "py":
        await ctx.downloadPythonExport();
        break;
      default:
        ctx.setStatus(`Unsupported export format: ${exportFormatSelect.value}`, "error");
        break;
    }
  }

  async function insertTemplate() {
    const templateName = templateSelect.value;
    if (!templateName) {
      ctx.setStatus("Choose a template first.");
      return;
    }
    const parameters = ctx.persistTemplateParametersFromControls();
    try {
      const payload = await sessionService.buildTemplate({
        templateName,
        parameters,
      });
      const importedSpec = ctx.uniquifyImportedSpec(
        payload.spec.network,
        ctx.makeId("template")
      );
      const translatedSpec = ctx.translateImportedSpec(
        importedSpec,
        ctx.suggestTensorPosition(ctx.viewportCenterPosition())
      );
      ctx.applyDesignChange(
        () => {
          state.spec.tensors.push(...translatedSpec.tensors);
          state.spec.edges.push(...translatedSpec.edges);
          state.spec.groups.push(...translatedSpec.groups);
          store.setLastImportedTensorIds(translatedSpec.tensors.map(
            (tensor) => tensor.id
          ));
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
      ctx.setStatus(`Could not insert the template: ${error.message}`, "error");
    }
  }

  return {
    showCodeGenerationError,
    generateCode,
    completeEditor,
    cancelEditor,
    saveDesign,
    loadDesignFromFile,
    copyGeneratedCode,
    downloadSelectedExport,
    downloadPythonExport,
    openSubnetworkPicker,
    exportSelectedSubnetwork,
    exportGroupSubnetwork,
    promoteSelectedSubnetworkToTemplate,
    promoteGroupToTemplate,
    renameSelectedTemplate,
    deleteSelectedTemplate,
    insertPreparedSubnetwork,
    loadSubnetworkFromFile,
    insertTemplate,
  };
}
