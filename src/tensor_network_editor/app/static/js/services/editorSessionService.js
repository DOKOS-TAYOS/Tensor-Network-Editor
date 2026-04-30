function buildCodegenPayload({
  engine,
  collectionFormat,
  includeRoundtripMetadata,
  spec,
}) {
  const payload = {
    engine,
    collection_format: collectionFormat,
    spec,
  };
  if (typeof includeRoundtripMetadata === "boolean") {
    payload.include_roundtrip_metadata = includeRoundtripMetadata;
  }
  return payload;
}

function summarizeSerializedSpec(serializedSpec) {
  const network =
    serializedSpec && typeof serializedSpec === "object" && serializedSpec.network
      ? serializedSpec.network
      : null;
  if (!network || typeof network !== "object") {
    return {};
  }
  return {
    tensor_count: Array.isArray(network.tensors) ? network.tensors.length : 0,
    edge_count: Array.isArray(network.edges) ? network.edges.length : 0,
    group_count: Array.isArray(network.groups) ? network.groups.length : 0,
    note_count: Array.isArray(network.notes) ? network.notes.length : 0,
    mode:
      network.linear_periodic_chain
        ? "linear_periodic"
        : network.grid_periodic_grid
          ? "grid_periodic"
          : network.tree_periodic_tree
            ? "tree_periodic"
            : "normal",
  };
}

function summarizeCodegenRequest(request) {
  return {
    engine: request.engine,
    collection_format: request.collectionFormat,
    include_roundtrip_metadata: request.includeRoundtripMetadata,
    ...summarizeSerializedSpec(request.spec),
  };
}

export function createEditorSessionService({ apiGet, apiPost }) {
  return {
    loadBootstrap() {
      return apiGet("/api/bootstrap", {
        operation: "bootstrap",
      });
    },
    loadDraft() {
      return apiGet("/api/draft", {
        operation: "draft.load",
      });
    },
    saveDraft(request) {
      return apiPost("/api/draft", buildCodegenPayload(request), {
        operation: "draft.save",
        context: summarizeCodegenRequest(request),
      });
    },
    clearDraft() {
      return apiPost("/api/draft/clear", {}, {
        operation: "draft.clear",
      });
    },
    generateCode(request) {
      return apiPost("/api/generate", buildCodegenPayload(request), {
        operation: "generate",
        context: summarizeCodegenRequest(request),
      });
    },
    renderSpec({
      format,
      spec,
      showTensorNames = true,
      showIndexNames = true,
      showBondNames = true,
    }) {
      return apiPost(
        "/api/render",
        {
          format,
          spec,
          show_tensor_names: showTensorNames,
          show_index_names: showIndexNames,
          show_bond_names: showBondNames,
        },
        {
          operation: "render",
          context: {
            format,
            ...summarizeSerializedSpec(spec),
          },
        }
      );
    },
    completeSession(request) {
      return apiPost("/api/complete", buildCodegenPayload(request), {
        operation: "complete",
        context: summarizeCodegenRequest(request),
      });
    },
    cancelSession() {
      return apiPost("/api/cancel", {}, {
        operation: "cancel",
      });
    },
    validateSerializedSpec(serializedSpec) {
      return apiPost(
        "/api/validate",
        { spec: serializedSpec },
        {
          operation: "validate.serialized",
          context: summarizeSerializedSpec(serializedSpec),
        }
      );
    },
    validatePythonCode(request) {
      const normalizedRequest =
        typeof request === "string" ? { pythonCode: request } : request || {};
      return apiPost(
        "/api/validate",
        {
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
        },
        {
          operation: "validate.python",
          context: {
            python_import_mode: normalizedRequest.pythonImportMode || "static",
            source_profile: normalizedRequest.sourceProfile || "auto",
            python_reconstruction_level:
              normalizedRequest.pythonReconstructionLevel || "auto",
            python_object_name:
              typeof normalizedRequest.pythonObjectName === "string"
              && normalizedRequest.pythonObjectName.trim()
                ? "provided"
                : "auto",
          },
        }
      );
    },
    buildTemplate({ templateName, parameters }) {
      return apiPost(
        "/api/template",
        {
          template: templateName,
          parameters,
        },
        {
          operation: "template.build",
          context: {
            template_name: templateName,
            parameter_count:
              parameters && typeof parameters === "object"
                ? Object.keys(parameters).length
                : 0,
          },
        }
      );
    },
  };
}
