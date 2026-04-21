import { createMetadataFilterBindingsSupport } from "./metadataFiltersBindings.js";
import { createMetadataFilterRendererSupport } from "./metadataFiltersRenderers.js";
import { createMetadataFilterStateSupport } from "./metadataFiltersState.js";

export function registerMetadataFilters(ctx) {
  const state = ctx.state;
  const { document } = ctx;
  const { canvasTools } = ctx.dom;
  let shouldFocusSearchInput = false;

  function getVisibleTensors() {
    if (!state.spec) {
      return [];
    }
    return typeof ctx.getVisibleTensors === "function"
      ? ctx.getVisibleTensors()
      : state.spec.tensors;
  }

  function getVisibleEdges() {
    if (!state.spec) {
      return [];
    }
    return typeof ctx.getVisibleEdges === "function"
      ? ctx.getVisibleEdges()
      : state.spec.edges;
  }

  const stateSupport = createMetadataFilterStateSupport({
    state,
    getVisibleEdges,
    getVisibleTensors,
    findIndexOwner: (indexId) => ctx.findIndexOwner(indexId),
    getHyperedgeSelectionId: (hyperedgeId) => ctx.hyperedgeHubNodeId(hyperedgeId),
  });
  const rendererSupport = createMetadataFilterRendererSupport({
    escapeHtml:
      typeof ctx.escapeHtml === "function"
        ? (value) => ctx.escapeHtml(value)
        : (value) => String(value || ""),
    sanitizeFilename:
      typeof ctx.sanitizeFilename === "function"
        ? (value) => ctx.sanitizeFilename(value)
        : null,
    collectTagsForScope: stateSupport.collectTagsForScope,
  });
  const bindingsSupport = createMetadataFilterBindingsSupport({ document });
  const {
    collectTagsForScope,
    getMetadataFilterEntityState,
    getMetadataFilterHighlight,
    metadataFiltersEqual,
    nameSearchEqual,
    normalizeMetadataFilters,
    normalizeNameSearch,
  } = stateSupport;
  const { bindMetadataFilterControls } = bindingsSupport;
  const { buildMetadataFiltersMarkup, getCheckboxIdForTag } = rendererSupport;

  function requestHighlightRender() {
    if (typeof ctx.render !== "function") {
      return;
    }
    ctx.render({
      code: false,
      graph: true,
      minimap: true,
      overlays: false,
      planner: false,
      properties: false,
      sidebarTabs: false,
      toolbar: false,
    });
  }

  function renderMetadataFilters() {
    if (!canvasTools) {
      return;
    }

    const filters = normalizeMetadataFilters();
    const search = normalizeNameSearch();
    state.metadataFilters = filters;
    state.nameSearch = search;
    canvasTools.innerHTML = buildMetadataFiltersMarkup({
      filters,
      openCanvasToolPopover: state.openCanvasToolPopover,
      search,
    });
    bindMetadataFilterControls({
      collectTagsForScope,
      getCheckboxIdForTag,
      normalizeMetadataFilters,
      normalizeNameSearch,
      renderMetadataFilters,
      requestHighlightRender,
      setShouldFocusSearchInput: (nextValue) => {
        shouldFocusSearchInput = nextValue;
      },
      shouldFocusSearchInput,
      state,
      updateMetadataFilters,
      updateNameSearch,
    });
  }

  function updateMetadataFilters(updates, options = {}) {
    const currentFilters = normalizeMetadataFilters();
    const currentSearch = normalizeNameSearch();
    const nextFilters = normalizeMetadataFilters({
      ...currentFilters,
      ...updates,
      enabled:
        updates && Object.prototype.hasOwnProperty.call(updates, "enabled")
          ? updates.enabled
          : true,
    });
    state.metadataFilters = nextFilters;
    state.nameSearch = {
      ...currentSearch,
      enabled: false,
    };
    state.openCanvasToolPopover = options.openPopover || "filter";
    if (options.renderPanel !== false) {
      renderMetadataFilters();
    }
    if (
      !metadataFiltersEqual(currentFilters, nextFilters) ||
      currentSearch.enabled
    ) {
      requestHighlightRender();
    }
  }

  function updateNameSearch(updates, options = {}) {
    const currentFilters = normalizeMetadataFilters();
    const currentSearch = normalizeNameSearch();
    const nextSearch = normalizeNameSearch({
      ...currentSearch,
      ...updates,
    });
    state.nameSearch = nextSearch;
    state.metadataFilters = {
      ...currentFilters,
      enabled: false,
    };
    state.openCanvasToolPopover = options.openPopover || "search";
    if (options.renderPanel !== false) {
      renderMetadataFilters();
    }
    if (
      !nameSearchEqual(currentSearch, nextSearch) ||
      currentFilters.enabled
    ) {
      requestHighlightRender();
    }
  }

  Object.assign(ctx, {
    getMetadataFilterEntityState,
    getMetadataFilterHighlight,
    renderMetadataFilters,
    updateMetadataFilters,
    updateNameSearch,
  });
}
