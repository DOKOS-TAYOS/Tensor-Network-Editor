export function createInteractionSessionBindings({ ctx, state, dom }) {
  const {
    exportFormatSelect,
    generatedCode,
    loadInput,
    templateSelect,
  } = dom;
  const { apiPost, document, window } = ctx;

  function showCodeGenerationError(message) {
    const safeMessage = message || "Code generation failed.";
    state.generatedCode = `Code generation failed:\n${safeMessage}`;
    if (generatedCode) {
      generatedCode.value = state.generatedCode;
    }
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
      generatedCode.value = state.generatedCode;
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
      generatedCode.value = state.generatedCode;
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
    insertTemplate,
  };
}
