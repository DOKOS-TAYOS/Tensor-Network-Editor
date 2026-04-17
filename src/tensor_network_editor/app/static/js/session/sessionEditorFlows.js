export function createSessionEditorFlows({
  dom,
  state,
  store,
  selectors,
  services,
  commands,
  sessionUi,
  actions,
}) {
  const { exportFormatSelect, generatedCode, loadInput } = dom;
  const sessionService = services.session;

  function syncGeneratedCodePreview(code) {
    if (typeof actions.syncGeneratedCodePreview === "function") {
      actions.syncGeneratedCodePreview(code);
      return;
    }
    commands.syncGeneratedCodePreview(code);
  }

  function showCodeGenerationError(message) {
    const safeMessage = message || "Code generation failed.";
    store.setGeneratedCode(`Code generation failed:\n${safeMessage}`);
    syncGeneratedCodePreview(state.generatedCode);
    actions.setStatus(safeMessage, "error");
  }

  async function requestGeneratedCode() {
    return sessionService.generateCode({
      engine: selectors.getSelectedEngine(),
      collectionFormat: selectors.getSelectedCollectionFormat(),
      spec: actions.serializeCurrentSpec({ persistViewSnapshots: false }),
    });
  }

  async function generateCode() {
    actions.ensureCodePanelVisible();
    actions.syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = actions.getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      actions.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await requestGeneratedCode();
      if (!payload.ok) {
        showCodeGenerationError(
          payload.message || actions.formatIssues(payload.issues)
        );
        return;
      }
      store.setGeneratedCode(actions.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      actions.setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      showCodeGenerationError(`Code generation failed: ${error.message}`);
    }
  }

  async function completeEditor() {
    try {
      const payload = await sessionService.completeSession({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        spec: actions.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      store.setEditorFinished(true);
      actions.setStatus(
        "Returning the design to Python. You can close this tab.",
        "success"
      );
      sessionUi.schedule(() => {
        sessionUi.closeWindow();
      }, 150);
    } catch (error) {
      actions.setStatus(
        `Could not finish the editor session: ${error.message}`,
        "error"
      );
    }
  }

  async function cancelEditor() {
    try {
      store.setEditorFinished(true);
      await sessionService.cancelSession();
      actions.setStatus("Editor cancelled. You can close this tab.", "success");
      sessionUi.schedule(() => {
        sessionUi.closeWindow();
      }, 150);
    } catch (error) {
      actions.setStatus(
        `Could not cancel the editor session: ${error.message}`,
        "error"
      );
    }
  }

  function saveDesign() {
    sessionUi.downloadText(
      `${actions.sanitizeFilename(state.spec.name || "tensor-network")}.json`,
      JSON.stringify(
        actions.serializeCurrentSpec({ persistViewSnapshots: true }),
        null,
        2
      ),
      "application/json;charset=utf-8"
    );
    actions.setStatus("Design downloaded as JSON.");
  }

  async function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    try {
      const fileText = await sessionUi.requestFileText(file, "utf-8");
      const isPythonSource = file.name.toLowerCase().endsWith(".py");
      const response = isPythonSource
        ? await sessionService.validatePythonCode(fileText)
        : await sessionService.validateSerializedSpec(JSON.parse(fileText));
      if (!response.ok) {
        actions.setStatus(actions.formatIssues(response.issues), "error");
        return;
      }
      actions.resetDesignState(
        response.spec.network,
        `Loaded design from ${file.name}. History cleared.`,
        response.spec.schema_version
      );
    } catch (error) {
      actions.setStatus(`Could not load ${file.name}: ${error.message}`, "error");
    } finally {
      if (loadInput) {
        loadInput.value = "";
      }
    }
  }

  async function copyGeneratedCode() {
    const codeToCopy = actions.stripImportLines(generatedCode.value);
    if (!codeToCopy.trim()) {
      actions.setStatus("There is no generated code to copy yet.");
      return;
    }
    try {
      await sessionUi.copyText(codeToCopy);
      actions.setStatus(
        "Generated code copied to the clipboard without import lines.",
        "success"
      );
    } catch (error) {
      actions.setStatus(
        `Could not copy the generated code: ${error.message}`,
        "error"
      );
    }
  }

  async function downloadPythonExport() {
    actions.ensureCodePanelVisible();
    actions.syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = actions.getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      actions.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await requestGeneratedCode();
      if (!payload.ok) {
        showCodeGenerationError(
          payload.message || actions.formatIssues(payload.issues)
        );
        return;
      }
      store.setGeneratedCode(actions.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      sessionUi.downloadText(
        `${actions.sanitizeFilename(state.spec.name || "tensor-network")}-${actions.sanitizeFilename(selectors.getSelectedEngine() || "engine")}.py`,
        payload.code,
        "text/x-python;charset=utf-8"
      );
      actions.setStatus(`Exported ${payload.engine} Python code.`, "success");
    } catch (error) {
      showCodeGenerationError(`Could not export Python code: ${error.message}`);
    }
  }

  async function downloadSelectedExport() {
    switch (exportFormatSelect.value) {
      case "png":
        actions.downloadPngExport();
        break;
      case "svg":
        actions.downloadSvgExport();
        break;
      case "py":
        await downloadPythonExport();
        break;
      default:
        actions.setStatus(
          `Unsupported export format: ${exportFormatSelect.value}`,
          "error"
        );
        break;
    }
  }

  async function downloadExportAs(format) {
    if (!exportFormatSelect) {
      actions.setStatus("Export controls are not available in this browser.", "error");
      return;
    }
    const previousFormat = exportFormatSelect.value;
    exportFormatSelect.value = format;
    try {
      await downloadSelectedExport();
    } finally {
      exportFormatSelect.value = previousFormat;
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
    downloadExportAs,
    downloadPythonExport,
  };
}
