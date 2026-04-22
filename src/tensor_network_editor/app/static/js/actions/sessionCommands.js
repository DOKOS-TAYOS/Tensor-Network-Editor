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

  function applySubnetworkCatalogUpdate(
    payload,
    successMessage = "Updated the subnetwork library."
  ) {
    if (typeof store.setSubnetworkCatalogData === "function") {
      store.setSubnetworkCatalogData({
        subnetworkNames: payload.subnetworks,
        subnetworkDefinitions: payload.subnetwork_definitions,
        subnetworkCatalogWarnings: payload.subnetwork_catalog_warnings,
        selectedSubnetworkName:
          typeof payload.selected_subnetwork === "string"
            ? payload.selected_subnetwork
            : null,
      });
    } else {
      state.availableSubnetworks = Array.isArray(payload.subnetworks)
        ? [...payload.subnetworks]
        : [];
      state.subnetworkDefinitions =
        payload.subnetwork_definitions &&
        typeof payload.subnetwork_definitions === "object"
          ? { ...payload.subnetwork_definitions }
          : {};
      state.subnetworkCatalogWarnings = Array.isArray(
        payload.subnetwork_catalog_warnings
      )
        ? [...payload.subnetwork_catalog_warnings]
        : [];
      state.selectedSubnetworkName =
        typeof payload.selected_subnetwork === "string"
          ? payload.selected_subnetwork
          : state.availableSubnetworks[0] || "";
      state.selectedSubnetworkLibraryNames = Array.isArray(
        state.selectedSubnetworkLibraryNames
      )
        ? state.selectedSubnetworkLibraryNames.filter((subnetworkName) =>
            state.availableSubnetworks.includes(subnetworkName)
          )
        : [];
    }
    if (
      Array.isArray(payload.subnetwork_catalog_warnings)
      && payload.subnetwork_catalog_warnings.length
    ) {
      setStatus(payload.subnetwork_catalog_warnings[0], "error");
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
    applySubnetworkCatalogUpdate,
    insertPreparedSubnetwork,
  };
}
