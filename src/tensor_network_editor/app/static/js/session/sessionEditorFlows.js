export function createSessionEditorFlows({
  dom,
  state,
  logger = null,
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

  function startFlowOperation(name, context = {}) {
    return logger && typeof logger.startOperation === "function"
      ? logger.startOperation(name, context)
      : null;
  }

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

  function wasFileSaved(saveResult) {
    return saveResult !== false;
  }

  async function downloadTextFile(filename, text, contentType) {
    return wasFileSaved(
      await sessionUi.downloadText(filename, text, contentType)
    );
  }

  async function downloadBlobFile(filename, blobLike) {
    return wasFileSaved(await sessionUi.downloadBlob(filename, blobLike));
  }

  async function requestGeneratedCode() {
    return sessionService.generateCode({
      engine: selectors.getSelectedEngine(),
      collectionFormat: selectors.getSelectedCollectionFormat(),
      includeRoundtripMetadata: state.includeRoundtripMetadata === true,
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
      logger?.debug?.("Skipped draft autosave scheduling", {
        operation: "draft.autosave",
      });
      return;
    }
    state.draftAutosaveDirty = true;
    if (state.draftAutosaveTimer !== null) {
      logger?.debug?.("Draft autosave already scheduled", {
        operation: "draft.autosave",
      });
      return;
    }
    logger?.debug?.("Scheduled draft autosave", {
      operation: "draft.autosave",
    });
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
    const saveOperation = startFlowOperation("Save current draft", {
      operation: "draft.save",
      engine: selectors.getSelectedEngine(),
      collection_format: selectors.getSelectedCollectionFormat(),
    });
    if (!canAutosaveDraft() || state.draftAutosaveSaving) {
      saveOperation?.finish({
        outcome: "skipped",
      });
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
      saveOperation?.finish({
        outcome: payload.ok ? "saved" : "rejected",
      });
      return Boolean(payload.ok);
    } catch (error) {
      saveOperation?.fail(error, {
        outcome: "error",
      });
      if (!silent) {
        actions.setStatus(`Could not save the local draft: ${error.message}`, "error");
      }
      return false;
    } finally {
      state.draftAutosaveSaving = false;
    }
  }

  async function clearSavedDraft({ silent = false, resumeAutosave = true } = {}) {
    const clearOperation = startFlowOperation("Clear saved draft", {
      operation: "draft.clear",
    });
    const wasReady = state.draftAutosaveReady;
    state.draftAutosaveReady = false;
    state.draftAutosaveDirty = false;
    state.draftAutosaveTimer = null;
    if (typeof sessionService.clearDraft !== "function") {
      state.draftAutosaveReady = resumeAutosave && wasReady && !state.editorFinished;
      clearOperation?.finish({
        outcome: "unavailable",
      });
      return true;
    }
    try {
      const payload = await sessionService.clearDraft();
      if (!payload.ok && !silent) {
        actions.setStatus(payload.message || "Could not clear the local draft.", "error");
      }
      clearOperation?.finish({
        outcome: payload.ok ? "cleared" : "rejected",
      });
      return Boolean(payload.ok);
    } catch (error) {
      clearOperation?.fail(error, {
        outcome: "error",
      });
      if (!silent) {
        actions.setStatus(`Could not clear the local draft: ${error.message}`, "error");
      }
      return false;
    } finally {
      state.draftAutosaveReady = resumeAutosave && wasReady && !state.editorFinished;
    }
  }

  async function generateCode() {
    const generateOperation = startFlowOperation("Generate code", {
      operation: "generate",
      engine: selectors.getSelectedEngine(),
      collection_format: selectors.getSelectedCollectionFormat(),
    });
    actions.ensureCodePanelVisible();
    actions.syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = actions.getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      generateOperation?.finish({
        outcome: "blocked",
      });
      actions.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await requestGeneratedCode();
      if (!payload.ok) {
        generateOperation?.finish({
          outcome: "rejected",
        });
        showCodeGenerationError(
          payload.message || actions.formatIssues(payload.issues)
        );
        return;
      }
      store.setGeneratedCode(actions.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      actions.setStatus(`Generated ${payload.engine} code.`, "success");
      generateOperation?.finish({
        outcome: "generated",
        engine: payload.engine,
      });
    } catch (error) {
      generateOperation?.fail(error, {
        outcome: "error",
      });
      showCodeGenerationError(`Code generation failed: ${error.message}`);
    }
  }

  async function completeEditor() {
    const completeOperation = startFlowOperation("Complete editor", {
      operation: "complete",
      engine: selectors.getSelectedEngine(),
      collection_format: selectors.getSelectedCollectionFormat(),
    });
    try {
      const payload = await sessionService.completeSession({
        engine: selectors.getSelectedEngine(),
        collectionFormat: selectors.getSelectedCollectionFormat(),
        includeRoundtripMetadata: state.includeRoundtripMetadata === true,
        spec: actions.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok) {
        completeOperation?.finish({
          outcome: "rejected",
        });
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
      completeOperation?.finish({
        outcome: "completed",
      });
    } catch (error) {
      completeOperation?.fail(error, {
        outcome: "error",
      });
      actions.setStatus(
        `Could not finish the editor session: ${error.message}`,
        "error"
      );
    }
  }

  async function cancelEditor() {
    const cancelOperation = startFlowOperation("Cancel editor", {
      operation: "cancel",
    });
    try {
      await clearSavedDraft({ silent: true, resumeAutosave: false });
      store.setEditorFinished(true);
      await sessionService.cancelSession();
      actions.setStatus("Editor cancelled. You can close this tab.", "success");
      sessionUi.schedule(() => {
        sessionUi.closeWindow();
      }, 150);
      cancelOperation?.finish({
        outcome: "cancelled",
      });
    } catch (error) {
      cancelOperation?.fail(error, {
        outcome: "error",
      });
      actions.setStatus(
        `Could not cancel the editor session: ${error.message}`,
        "error"
      );
    }
  }

  async function saveDesign() {
    const saveDesignOperation = startFlowOperation("Save design", {
      operation: "design.save",
    });
    const didSave = await downloadTextFile(
      `${actions.sanitizeFilename(state.spec.name || "tensor-network")}.json`,
      JSON.stringify(
        actions.serializeCurrentSpec({ persistViewSnapshots: true }),
        null,
        2
      ),
      "application/json;charset=utf-8"
    );
    if (!didSave) {
      actions.setStatus("Design save cancelled.");
      saveDesignOperation?.finish({
        outcome: "cancelled",
      });
      return;
    }
    void clearSavedDraft({ silent: true, resumeAutosave: true });
    actions.setStatus("Design downloaded as JSON.");
    saveDesignOperation?.finish({
      outcome: "downloaded",
    });
  }

  async function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    const loadOperation = startFlowOperation("Load design from file", {
      operation: "design.load",
      path: file.name,
    });

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
        loadOperation?.finish({
          outcome: "rejected",
          warning_count: Array.isArray(response.warnings) ? response.warnings.length : 0,
        });
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
      loadOperation?.finish({
        outcome: "loaded",
        warning_count: Array.isArray(response.warnings) ? response.warnings.length : 0,
      });
    } catch (error) {
      loadOperation?.fail(error, {
        outcome: "error",
      });
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
    const pythonExportOperation = startFlowOperation("Download Python export", {
      operation: "export.python",
      engine: selectors.getSelectedEngine(),
    });
    actions.ensureCodePanelVisible();
    actions.syncCodeGenerationWarning();
    const tensorKrowchPlanIssue = actions.getTensorKrowchManualPlanIssueMessage();
    if (tensorKrowchPlanIssue) {
      pythonExportOperation?.finish({
        outcome: "blocked",
      });
      actions.setStatus(tensorKrowchPlanIssue, "error");
      return;
    }
    try {
      const payload = await requestGeneratedCode();
      if (!payload.ok) {
        pythonExportOperation?.finish({
          outcome: "rejected",
        });
        showCodeGenerationError(
          payload.message || actions.formatIssues(payload.issues)
        );
        return;
      }
      store.setGeneratedCode(actions.stripImportLines(payload.code));
      syncGeneratedCodePreview(state.generatedCode);
      const didSave = await downloadTextFile(
        `${actions.sanitizeFilename(state.spec.name || "tensor-network")}-${actions.sanitizeFilename(selectors.getSelectedEngine() || "engine")}.py`,
        payload.code,
        "text/x-python;charset=utf-8"
      );
      if (!didSave) {
        actions.setStatus("Python export cancelled.");
        pythonExportOperation?.finish({
          outcome: "cancelled",
          engine: payload.engine,
        });
        return;
      }
      actions.setStatus(`Exported ${payload.engine} Python code.`, "success");
      pythonExportOperation?.finish({
        outcome: "downloaded",
        engine: payload.engine,
      });
    } catch (error) {
      pythonExportOperation?.fail(error, {
        outcome: "error",
      });
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
      theme: state.selectedTheme,
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
    const didSave = await downloadBlobFile(filename, pngBlob);
    if (!didSave) {
      actions.setStatus("PNG export cancelled.");
      return false;
    }
    actions.setStatus("Exported a PNG file.", "success");
    return true;
  }

  async function downloadAcademicExport(format) {
    const exportOperation = startFlowOperation("Download academic export", {
      operation: "export.academic",
      format,
    });
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
      mermaid: {
        extension: "mmd",
        label: "Mermaid",
        contentType: "text/plain;charset=utf-8",
        responseKind: "text",
      },
    }[format];
    if (!exportDetails) {
      exportOperation?.finish({
        outcome: "unsupported",
      });
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
          exportOperation?.finish({
            outcome: "downloaded",
            format,
          });
          return;
        }
        exportOperation?.finish({
          outcome: "rejected",
          format,
        });
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      if (exportDetails.responseKind === "binary") {
        const didSave = await downloadBlobFile(
          filename,
          new Blob([decodeBase64ToUint8Array(payload.base64 || "")], {
            type: payload.content_type || exportDetails.contentType,
          })
        );
        if (!didSave) {
          actions.setStatus(`${exportDetails.label} export cancelled.`);
          exportOperation?.finish({
            outcome: "cancelled",
            format,
          });
          return;
        }
      } else {
        const didSave = await downloadTextFile(
          filename,
          payload.text || "",
          payload.content_type || exportDetails.contentType
        );
        if (!didSave) {
          actions.setStatus(`${exportDetails.label} export cancelled.`);
          exportOperation?.finish({
            outcome: "cancelled",
            format,
          });
          return;
        }
      }
      actions.setStatus(`Exported a ${exportDetails.label} file.`, "success");
      exportOperation?.finish({
        outcome: "downloaded",
        format,
      });
    } catch (error) {
      if (
        format === "png" &&
        await tryDownloadPngViaSvgFallback(filename).catch(() => false)
      ) {
        exportOperation?.finish({
          outcome: "downloaded",
          format,
        });
        return;
      }
      exportOperation?.fail(error, {
        outcome: "error",
        format,
      });
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
      case "mermaid":
        await downloadAcademicExport("mermaid");
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
