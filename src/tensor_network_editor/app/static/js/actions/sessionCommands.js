import { createCodeHighlightingSupport } from "../core/codeHighlighting.js";

export function createSessionCommands({
  dom,
  state,
  store,
  document,
  window,
  setStatus,
  applyTemplateCatalogPayload,
  normalizeSpec,
  applyDesignChange,
  bringTensorToFront,
}) {
  const { generatedCode, generatedCodeView } = dom;
  const codeHighlightingSupport = createCodeHighlightingSupport({
    windowRef: window,
    documentRef: document,
  });

  function syncGeneratedCodePreview(code = state.generatedCode) {
    const renderedCode = typeof code === "string" ? code : "";
    if (generatedCode) {
      generatedCode.value = renderedCode;
    }
    if (!generatedCodeView) {
      return;
    }
    generatedCodeView.textContent = renderedCode;
    void codeHighlightingSupport.highlightElement(generatedCodeView);
  }

  function applyTemplateCatalogUpdate(
    payload,
    successMessage = "Updated the template catalog."
  ) {
    applyTemplateCatalogPayload({
      templateNames: payload.templates,
      templateDefinitions: payload.template_definitions,
      selectedTemplate:
        typeof payload.selected_template === "string"
          ? payload.selected_template
          : null,
      templateCatalogWarnings: payload.template_catalog_warnings,
    });
    if (
      Array.isArray(payload.template_catalog_warnings)
      && payload.template_catalog_warnings.length
    ) {
      setStatus(payload.template_catalog_warnings[0], "error");
      return;
    }
    setStatus(successMessage, "success");
  }

  function insertPreparedSubnetwork(preparedSpec, label = null) {
    const normalizedSpec = normalizeSpec(preparedSpec);
    applyDesignChange(
      () => {
        state.spec.tensors.push(...normalizedSpec.tensors);
        state.spec.edges.push(...normalizedSpec.edges);
        state.spec.groups.push(...normalizedSpec.groups);
        normalizedSpec.tensors.forEach((tensor) => {
          bringTensorToFront(tensor.id);
        });
        store.setLastImportedTensorIds(
          normalizedSpec.tensors.map((tensor) => tensor.id)
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

  return {
    syncGeneratedCodePreview,
    applyTemplateCatalogUpdate,
    insertPreparedSubnetwork,
  };
}
