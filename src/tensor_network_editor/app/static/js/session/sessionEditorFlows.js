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
  const DRAFT_AUTOSAVE_DELAY_MS = 500;

  function syncGeneratedCodePreview(code) {
    if (typeof actions.syncGeneratedCodePreview === "function") {
      actions.syncGeneratedCodePreview(code);
      return;
    }
    commands.syncGeneratedCodePreview(code);
  }

  function ensureAcademicExportLabels() {
    const labels = state.academicExportLabels || {};
    state.academicExportLabels = {
      tensor: labels.tensor !== false,
      index: labels.index !== false,
      bond: labels.bond !== false,
    };
  }

  function decodeBase64ToUint8Array(value) {
    const base64Decoder =
      typeof globalThis.atob === "function"
        ? globalThis.atob.bind(globalThis)
        : typeof Buffer !== "undefined"
          ? (encoded) => Buffer.from(encoded, "base64").toString("binary")
          : null;
    if (!base64Decoder || typeof Uint8Array !== "function") {
      throw new Error("Binary downloads are not available in this browser.");
    }
    const decoded = base64Decoder(value);
    return Uint8Array.from(decoded, (character) => character.charCodeAt(0));
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

  function canAutosaveDraft() {
    return (
      Boolean(state.spec)
      && !state.editorFinished
      && state.draftAutosaveReady
      && typeof sessionService.saveDraft === "function"
    );
  }

  function scheduleDraftAutosave() {
    if (!canAutosaveDraft()) {
      return;
    }
    state.draftAutosaveDirty = true;
    if (state.draftAutosaveTimer !== null) {
      return;
    }
    state.draftAutosaveTimer = sessionUi.schedule(async () => {
      state.draftAutosaveTimer = null;
      if (!state.draftAutosaveDirty || !canAutosaveDraft()) {
        return;
      }
      state.draftAutosaveDirty = false;
      await saveCurrentDraft({ silent: true });
      if (state.draftAutosaveDirty) {
        scheduleDraftAutosave();
      }
    }, DRAFT_AUTOSAVE_DELAY_MS);
  }

  async function saveCurrentDraft({ silent = false } = {}) {
    if (!canAutosaveDraft() || state.draftAutosaveSaving) {
      return false;
    }
    state.draftAutosaveSaving = true;
    try {
      const payload = await sessionService.saveDraft({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        spec: actions.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok && !silent) {
        actions.setStatus(payload.message || "Could not save the local draft.", "error");
      }
      return Boolean(payload.ok);
    } catch (error) {
      if (!silent) {
        actions.setStatus(`Could not save the local draft: ${error.message}`, "error");
      }
      return false;
    } finally {
      state.draftAutosaveSaving = false;
    }
  }

  async function clearSavedDraft({ silent = false, resumeAutosave = true } = {}) {
    const wasReady = state.draftAutosaveReady;
    state.draftAutosaveReady = false;
    state.draftAutosaveDirty = false;
    state.draftAutosaveTimer = null;
    if (typeof sessionService.clearDraft !== "function") {
      state.draftAutosaveReady = resumeAutosave && wasReady && !state.editorFinished;
      return true;
    }
    try {
      const payload = await sessionService.clearDraft();
      if (!payload.ok && !silent) {
        actions.setStatus(payload.message || "Could not clear the local draft.", "error");
      }
      return Boolean(payload.ok);
    } catch (error) {
      if (!silent) {
        actions.setStatus(`Could not clear the local draft: ${error.message}`, "error");
      }
      return false;
    } finally {
      state.draftAutosaveReady = resumeAutosave && wasReady && !state.editorFinished;
    }
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
      await clearSavedDraft({ silent: true, resumeAutosave: false });
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
      await clearSavedDraft({ silent: true, resumeAutosave: false });
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
    void clearSavedDraft({ silent: true, resumeAutosave: true });
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
      let pythonImportMode = "static";
      let pythonObjectName = null;
      if (isPythonSource && sessionUi.confirmAction(
        "Run this Python file in a subprocess to import a live tensor network object? Choose Cancel to keep the static parser."
      )) {
        pythonImportMode = "live";
        const promptedObjectName = sessionUi.promptText(
          "Optional global object name for the live import. Leave blank to auto-detect it.",
          ""
        );
        pythonObjectName =
          typeof promptedObjectName === "string" && promptedObjectName.trim()
            ? promptedObjectName.trim()
            : null;
      }
      const response = isPythonSource
        ? await sessionService.validatePythonCode({
            pythonCode: fileText,
            pythonImportMode,
            pythonReconstructionLevel: "auto",
            pythonObjectName,
            sourceProfile: "auto",
          })
        : await sessionService.validateSerializedSpec(JSON.parse(fileText));
      if (!response.ok) {
        actions.setStatus(
          response.message || actions.formatIssues(response.issues),
          "error"
        );
        return;
      }
      if (Array.isArray(response.warnings) && response.warnings.length) {
        actions.setStatus(response.warnings[0]);
      }
      actions.resetDesignState(
        response.spec.network,
        `Loaded design from ${file.name}. History cleared.`,
        response.spec.schema_version
      );
      scheduleDraftAutosave();
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

  function buildAcademicRenderRequest(format) {
    ensureAcademicExportLabels();
    return {
      format,
      spec: actions.serializeCurrentSpec({ persistViewSnapshots: true }),
      showTensorNames: state.academicExportLabels.tensor,
      showIndexNames: state.academicExportLabels.index,
      showBondNames: state.academicExportLabels.bond,
    };
  }

  async function tryDownloadPngViaSvgFallback(filename) {
    if (typeof sessionUi.rasterizeSvgToPng !== "function") {
      return false;
    }
    const svgPayload = await sessionService.renderSpec(buildAcademicRenderRequest("svg"));
    if (!svgPayload.ok) {
      throw new Error(svgPayload.message || actions.formatIssues(svgPayload.issues));
    }
    const pngBlob = await sessionUi.rasterizeSvgToPng({
      filename,
      svgText: svgPayload.text || "",
      sourceContentType:
        svgPayload.content_type || "image/svg+xml;charset=utf-8",
    });
    sessionUi.downloadBlob(filename, pngBlob);
    actions.setStatus("Exported a PNG file.", "success");
    return true;
  }

  async function downloadAcademicExport(format) {
    const exportDetails = {
      svg: {
        extension: "svg",
        label: "SVG",
        contentType: "image/svg+xml;charset=utf-8",
        responseKind: "text",
      },
      png: {
        extension: "png",
        label: "PNG",
        contentType: "image/png",
        responseKind: "binary",
      },
      pdf: {
        extension: "pdf",
        label: "PDF",
        contentType: "application/pdf",
        responseKind: "binary",
      },
      tikz: {
        extension: "tex",
        label: "TikZ/LaTeX",
        contentType: "text/x-tex;charset=utf-8",
        responseKind: "text",
      },
      dot: {
        extension: "dot",
        label: "Graphviz/DOT",
        contentType: "text/vnd.graphviz;charset=utf-8",
        responseKind: "text",
      },
    }[format];
    if (!exportDetails) {
      actions.setStatus(`Unsupported export format: ${format}`, "error");
      return;
    }
    const filename = `${actions.sanitizeFilename(state.spec.name || "tensor-network")}.${exportDetails.extension}`;
    try {
      const payload = await sessionService.renderSpec(
        buildAcademicRenderRequest(format)
      );
      if (!payload.ok) {
        if (
          format === "png" &&
          await tryDownloadPngViaSvgFallback(filename)
        ) {
          return;
        }
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      if (exportDetails.responseKind === "binary") {
        sessionUi.downloadBlob(
          filename,
          new Blob([decodeBase64ToUint8Array(payload.base64 || "")], {
            type: payload.content_type || exportDetails.contentType,
          })
        );
      } else {
        sessionUi.downloadText(
          filename,
          payload.text || "",
          payload.content_type || exportDetails.contentType
        );
      }
      actions.setStatus(`Exported a ${exportDetails.label} file.`, "success");
    } catch (error) {
      if (
        format === "png" &&
        await tryDownloadPngViaSvgFallback(filename).catch(() => false)
      ) {
        return;
      }
      actions.setStatus(
        `Could not export ${exportDetails.label}: ${error.message}`,
        "error"
      );
    }
  }

  async function downloadSelectedExport() {
    switch (exportFormatSelect.value) {
      case "png":
        await downloadAcademicExport("png");
        break;
      case "svg":
        await downloadAcademicExport("svg");
        break;
      case "pdf":
        await downloadAcademicExport("pdf");
        break;
      case "tikz":
        await downloadAcademicExport("tikz");
        break;
      case "dot":
        await downloadAcademicExport("dot");
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
    scheduleDraftAutosave,
    saveCurrentDraft,
    clearSavedDraft,
    loadDesignFromFile,
    copyGeneratedCode,
    downloadSelectedExport,
    downloadExportAs,
    downloadPythonExport,
    downloadAcademicExport,
  };
}
