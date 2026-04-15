export function createInteractionSessionBindings({ ctx, state, dom }) {
  const {
    exportFormatSelect,
    generatedCode,
    generatedCodeView,
    loadInput,
    subnetworkLoadInput,
    templateSelect,
  } = dom;
  const { apiPost, document, window } = ctx;

  function syncGeneratedCodePreview(code) {
    if (typeof ctx.renderGeneratedCodePreview === "function") {
      ctx.renderGeneratedCodePreview(code);
      return;
    }
    const renderedCode = typeof code === "string" ? code : "";
    if (generatedCode) {
      generatedCode.value = renderedCode;
    }
    if (!generatedCodeView) {
      return;
    }
    generatedCodeView.textContent = renderedCode;
    const prism =
      window && typeof window === "object" ? window.Prism : null;
    if (prism && typeof prism.highlightElement === "function") {
      prism.highlightElement(generatedCodeView);
    }
  }

  function showCodeGenerationError(message) {
    const safeMessage = message || "Code generation failed.";
    state.generatedCode = `Code generation failed:\n${safeMessage}`;
    syncGeneratedCodePreview(state.generatedCode);
    ctx.setStatus(safeMessage, "error");
  }

  async function generateCode() {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    if (typeof ctx.syncCodeGenerationWarning === "function") {
      ctx.syncCodeGenerationWarning();
    }
    const tensorKrowchPlanIssue =
      typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "";
    if (tensorKrowchPlanIssue) {
      ctx.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        showCodeGenerationError(payload.message || ctx.formatIssues(payload.issues));
        return;
      }
      state.generatedCode = ctx.stripImportLines(payload.code);
      syncGeneratedCodePreview(state.generatedCode);
      ctx.setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      showCodeGenerationError(`Code generation failed: ${error.message}`);
    }
  }

  async function completeEditor() {
    try {
      const payload = await apiPost("/api/complete", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      state.editorFinished = true;
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
      state.editorFinished = true;
      await apiPost("/api/cancel", {});
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
          ? await apiPost("/api/validate", { python_code: fileText })
          : await apiPost("/api/validate", { spec: JSON.parse(fileText) });
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
    if (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) {
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
    ctx.applyTemplateCatalogPayload({
      templateNames: payload.templates,
      templateDefinitions: payload.template_definitions,
      selectedTemplate:
        typeof payload.selected_template === "string"
          ? payload.selected_template
          : null,
      templateCatalogWarnings: payload.template_catalog_warnings,
    });
    if (
      Array.isArray(payload.template_catalog_warnings) &&
      payload.template_catalog_warnings.length
    ) {
      ctx.setStatus(payload.template_catalog_warnings[0], "error");
      return;
    }
    ctx.setStatus(successMessage, "success");
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

  function getTemplateDefinition(templateName) {
    return typeof ctx.getTemplateDefinition === "function"
      ? ctx.getTemplateDefinition(templateName)
      : null;
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
    const existingDefinition = getTemplateDefinition(templateName);
    if (!existingDefinition) {
      return { overwrite: false };
    }
    if (existingDefinition.source === "global") {
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
    const normalizedSpec = ctx.normalizeSpec(preparedSpec);
    ctx.applyDesignChange(
      () => {
        state.spec.tensors.push(...normalizedSpec.tensors);
        state.spec.edges.push(...normalizedSpec.edges);
        state.spec.groups.push(...normalizedSpec.groups);
        normalizedSpec.tensors.forEach((tensor) => {
          ctx.bringTensorToFront(tensor.id);
        });
        state.lastImportedTensorIds = normalizedSpec.tensors.map(
          (tensor) => tensor.id
        );
      },
      {
        invalidate: { lookups: true },
        selectionIds: normalizedSpec.tensors.map((tensor) => tensor.id),
        primaryId: normalizedSpec.tensors.length
          ? normalizedSpec.tensors[normalizedSpec.tensors.length - 1].id
          : null,
        statusMessage: `Inserted ${label || normalizedSpec.name || "subnetwork"}.`,
      }
    );
  }

  async function exportSubnetworkByTensorIds(tensorIds, label = "subnetwork") {
    if (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) {
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
      const payload = await apiPost("/api/subnetwork/extract", {
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
        tensor_ids: tensorIds,
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
    await exportSubnetworkByTensorIds(
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [],
      "subnetwork"
    );
  }

  async function exportGroupSubnetwork(groupId) {
    const group = typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
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
    if (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) {
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
      const payload = await apiPost("/api/template/promote", {
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
        tensor_ids: tensorIds,
        template_name: templateName,
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
    await promoteSubnetworkByTensorIds(
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [],
      "selection_template"
    );
  }

  async function promoteGroupToTemplate(groupId) {
    const group =
      typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
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
    const currentDefinition = getTemplateDefinition(currentTemplateName);
    if (!currentDefinition || currentDefinition.source !== "project") {
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
      const payload = await apiPost("/api/template/rename", {
        template_name: currentTemplateName,
        new_template_name: newTemplateName,
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
    const currentDefinition = getTemplateDefinition(currentTemplateName);
    if (!currentDefinition || currentDefinition.source !== "project") {
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
      const payload = await apiPost("/api/template/delete", {
        template_name: currentTemplateName,
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
    if (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) {
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
        const response = await apiPost("/api/subnetwork/prepare-insert", {
          spec: serializedSpec,
          target_center: ctx.suggestTensorPosition(ctx.viewportCenterPosition()),
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
    try {
      await navigator.clipboard.writeText(codeToCopy);
      ctx.setStatus("Generated code copied to the clipboard without import lines.", "success");
    } catch (error) {
      ctx.setStatus(`Could not copy the generated code: ${error.message}`, "error");
    }
  }

  async function downloadPythonExport() {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    if (typeof ctx.syncCodeGenerationWarning === "function") {
      ctx.syncCodeGenerationWarning();
    }
    const tensorKrowchPlanIssue =
      typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "";
    if (tensorKrowchPlanIssue) {
      ctx.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        showCodeGenerationError(payload.message || ctx.formatIssues(payload.issues));
        return;
      }
      state.generatedCode = ctx.stripImportLines(payload.code);
      syncGeneratedCodePreview(state.generatedCode);
      ctx.downloadBlob(
        `${ctx.sanitizeFilename(state.spec.name || "tensor-network")}-${ctx.sanitizeFilename(state.selectedEngine || "engine")}.py`,
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
      const payload = await apiPost("/api/template", {
        template: templateName,
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
          state.lastImportedTensorIds = translatedSpec.tensors.map(
            (tensor) => tensor.id
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
