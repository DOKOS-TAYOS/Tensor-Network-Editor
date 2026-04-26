function buildCodegenPayload({ engine, collectionFormat, spec }) {
  return {
    engine,
    collection_format: collectionFormat,
    spec,
  };
}

export function createEditorSessionService({ apiGet, apiPost }) {
  return {
    loadBootstrap() {
      return apiGet("/api/bootstrap");
    },
    loadDraft() {
      return apiGet("/api/draft");
    },
    saveDraft(request) {
      return apiPost("/api/draft", buildCodegenPayload(request));
    },
    clearDraft() {
      return apiPost("/api/draft/clear", {});
    },
    generateCode(request) {
      return apiPost("/api/generate", buildCodegenPayload(request));
    },
    renderSpec({ format, spec }) {
      return apiPost("/api/render", {
        format,
        spec,
      });
    },
    completeSession(request) {
      return apiPost("/api/complete", buildCodegenPayload(request));
    },
    cancelSession() {
      return apiPost("/api/cancel", {});
    },
    validateSerializedSpec(serializedSpec) {
      return apiPost("/api/validate", { spec: serializedSpec });
    },
    validatePythonCode(request) {
      const normalizedRequest =
        typeof request === "string" ? { pythonCode: request } : request || {};
      return apiPost("/api/validate", {
        python_code: normalizedRequest.pythonCode || "",
        source_profile: normalizedRequest.sourceProfile || "auto",
        python_import_mode: normalizedRequest.pythonImportMode || "static",
        python_reconstruction_level:
          normalizedRequest.pythonReconstructionLevel || "auto",
        python_object_name:
          typeof normalizedRequest.pythonObjectName === "string"
          && normalizedRequest.pythonObjectName.trim()
            ? normalizedRequest.pythonObjectName.trim()
            : null,
      });
    },
    buildTemplate({ templateName, parameters }) {
      return apiPost("/api/template", {
        template: templateName,
        parameters,
      });
    },
  };
}
