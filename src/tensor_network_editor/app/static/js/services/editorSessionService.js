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
    generateCode(request) {
      return apiPost("/api/generate", buildCodegenPayload(request));
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
    validatePythonCode(pythonCode) {
      return apiPost("/api/validate", { python_code: pythonCode });
    },
    buildTemplate({ templateName, parameters }) {
      return apiPost("/api/template", {
        template: templateName,
        parameters,
      });
    },
  };
}
