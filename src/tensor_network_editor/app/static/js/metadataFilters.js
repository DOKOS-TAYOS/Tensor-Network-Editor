function normalizeText(value) {
  return String(value || "").trim();
}

function normalizeTextLower(value) {
  return normalizeText(value).toLowerCase();
}

const NOT_SPECIFIED_FILTER_LABEL = "Not specified";
const NOT_SPECIFIED_FILTER_VALUE = "__not_specified__";

function collectFilterOptionsForScope(collectTagsForScope, scope) {
  return [...collectTagsForScope(scope), NOT_SPECIFIED_FILTER_VALUE];
}

function scopeToEntityKind(scope) {
  if (scope === "bond") {
    return "edge";
  }
  return scope === "index" ? "index" : "tensor";
}

function createEmptyHighlight(mode = null, scope = null) {
  return {
    mode,
    scope,
    matchedTensorIds: new Set(),
    contextTensorIds: new Set(),
    matchedIndexIds: new Set(),
    contextIndexIds: new Set(),
    matchedEdgeIds: new Set(),
    contextEdgeIds: new Set(),
  };
}

export function registerMetadataFilters(ctx) {
  const state = ctx.state;
  const { document } = ctx;
  const { canvasTools } = ctx.dom;
  let shouldFocusSearchInput = false;

  function bindListener(target, eventName, handler) {
    if (!target || typeof target.addEventListener !== "function") {
      return;
    }
    target.addEventListener(eventName, handler);
  }

  function readSelectChevronExpanded(fieldElement) {
    if (!fieldElement) {
      return false;
    }
    if (typeof fieldElement.getAttribute === "function") {
      return fieldElement.getAttribute("data-expanded") === "true";
    }
    return fieldElement.attributes?.["data-expanded"] === "true";
  }

  function setSelectChevronExpanded(
    fieldElement,
    isExpanded,
    selectElement = null
  ) {
    if (!fieldElement || typeof fieldElement.setAttribute !== "function") {
      if (selectElement && selectElement.dataset) {
        selectElement.dataset.expanded = String(Boolean(isExpanded));
      }
      return;
    }
    fieldElement.setAttribute("data-expanded", String(Boolean(isExpanded)));
    if (selectElement && selectElement.dataset) {
      selectElement.dataset.expanded = String(Boolean(isExpanded));
    }
  }

  function bindSelectChevronDisclosure(selectElement, fieldElement) {
    if (!selectElement) {
      return;
    }
    setSelectChevronExpanded(fieldElement, false, selectElement);
    bindListener(selectElement, "mousedown", () => {
      setSelectChevronExpanded(
        fieldElement,
        !readSelectChevronExpanded(fieldElement),
        selectElement
      );
    });
    bindListener(selectElement, "keydown", (event) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, true, selectElement);
      }
      if (["Escape", "Tab"].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, false, selectElement);
      }
    });
    bindListener(selectElement, "change", () => {
      setSelectChevronExpanded(fieldElement, false, selectElement);
    });
    bindListener(selectElement, "blur", () => {
      setSelectChevronExpanded(fieldElement, false, selectElement);
    });
  }

  function normalizeMetadataFilters(filters = state.metadataFilters) {
    const scope =
      filters && (filters.scope === "index" || filters.scope === "bond")
        ? filters.scope
        : "tensor";
    const selectedTags = Array.isArray(filters && filters.selectedTags)
      ? filters.selectedTags
          .filter((tag) => typeof tag === "string" && tag.trim())
          .map((tag) => tag.trim())
          .filter(
            (tag, index, values) =>
              values.findIndex(
                (candidate) => candidate.toLowerCase() === tag.toLowerCase()
              ) === index
          )
      : [];
    return {
      scope,
      selectedTags,
      enabled: Boolean(filters && filters.enabled),
    };
  }

  function normalizeNameSearch(search = state.nameSearch) {
    const scope =
      search && (search.scope === "index" || search.scope === "bond")
        ? search.scope
        : "tensor";
    const query = normalizeText(search && search.query);
    return {
      scope,
      query,
      enabled: Boolean(search && search.enabled && query),
    };
  }

  function metadataFiltersEqual(leftFilters, rightFilters) {
    return (
      leftFilters.scope === rightFilters.scope &&
      leftFilters.enabled === rightFilters.enabled &&
      JSON.stringify(leftFilters.selectedTags) ===
        JSON.stringify(rightFilters.selectedTags)
    );
  }

  function nameSearchEqual(leftSearch, rightSearch) {
    return (
      leftSearch.scope === rightSearch.scope &&
      leftSearch.query === rightSearch.query &&
      leftSearch.enabled === rightSearch.enabled
    );
  }

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

  function resolveEdgeIndexIds(edge) {
    return [
      edge.leftIndexId || (edge.left && edge.left.index_id) || "",
      edge.rightIndexId || (edge.right && edge.right.index_id) || "",
    ];
  }

  function resolveEdgeTensorIds(edge) {
    const [leftIndexId, rightIndexId] = resolveEdgeIndexIds(edge);
    const leftOwner = leftIndexId ? ctx.findIndexOwner(leftIndexId) : null;
    const rightOwner = rightIndexId ? ctx.findIndexOwner(rightIndexId) : null;
    return [
      (edge.left && edge.left.tensor_id) ||
        (leftOwner && leftOwner.tensor ? leftOwner.tensor.id : ""),
      (edge.right && edge.right.tensor_id) ||
        (rightOwner && rightOwner.tensor ? rightOwner.tensor.id : ""),
    ];
  }

  function collectTagsForScope(scope) {
    const normalizedScope = scopeToEntityKind(scope);
    const seenTags = new Set();
    const tags = [];

    function addTags(metadata) {
      const metadataTags = Array.isArray(metadata && metadata.tags)
        ? metadata.tags
        : [];
      metadataTags.forEach((tag) => {
        if (typeof tag !== "string" || !tag.trim()) {
          return;
        }
        const normalizedTag = tag.trim().toLowerCase();
        if (seenTags.has(normalizedTag)) {
          return;
        }
        seenTags.add(normalizedTag);
        tags.push(tag.trim());
      });
    }

    if (!state.spec) {
      return tags;
    }

    if (normalizedScope === "tensor") {
      getVisibleTensors().forEach((tensor) => addTags(tensor.metadata));
    } else if (normalizedScope === "index") {
      getVisibleTensors().forEach((tensor) => {
        (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
          addTags(index.metadata);
        });
      });
    } else {
      getVisibleEdges().forEach((edge) => addTags(edge.metadata));
    }

    return tags;
  }

  function metadataMatchesSelectedTags(metadata, selectedTags) {
    if (!selectedTags.length) {
      return false;
    }
    const includesNotSpecified = selectedTags.includes(NOT_SPECIFIED_FILTER_VALUE);
    const normalizedSelectedTags = selectedTags.map((tag) => tag.toLowerCase());
    const metadataTags = Array.isArray(metadata && metadata.tags)
      ? metadata.tags
          .filter((tag) => typeof tag === "string" && tag.trim())
          .map((tag) => tag.trim().toLowerCase())
      : [];
    if (!metadataTags.length && includesNotSpecified) {
      return true;
    }
    return metadataTags.some((tag) => normalizedSelectedTags.includes(tag));
  }

  function nameMatchesQuery(name, query) {
    return normalizeTextLower(name) === normalizeTextLower(query);
  }

  function buildTensorHighlight(matchTensor) {
    const highlight = createEmptyHighlight("tensor", "tensor");
    const visibleTensors = getVisibleTensors();
    const visibleEdges = getVisibleEdges();

    visibleTensors.forEach((tensor) => {
      if (!matchTensor(tensor)) {
        return;
      }
      highlight.matchedTensorIds.add(tensor.id);
      (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
        highlight.matchedIndexIds.add(index.id);
      });
    });

    visibleEdges.forEach((edge) => {
      const [leftTensorId, rightTensorId] = resolveEdgeTensorIds(edge);
      if (
        leftTensorId &&
        rightTensorId &&
        highlight.matchedTensorIds.has(leftTensorId) &&
        highlight.matchedTensorIds.has(rightTensorId)
      ) {
        highlight.matchedEdgeIds.add(edge.id);
      }
    });

    return highlight;
  }

  function buildIndexHighlight(matchIndex) {
    const highlight = createEmptyHighlight("index", "index");
    const visibleTensors = getVisibleTensors();
    const visibleEdges = getVisibleEdges();

    visibleTensors.forEach((tensor) => {
      (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
        if (!matchIndex(index)) {
          return;
        }
        highlight.matchedIndexIds.add(index.id);
        highlight.contextTensorIds.add(tensor.id);
      });
    });

    visibleEdges.forEach((edge) => {
      const [leftIndexId, rightIndexId] = resolveEdgeIndexIds(edge);
      if (
        highlight.matchedIndexIds.has(leftIndexId) ||
        highlight.matchedIndexIds.has(rightIndexId)
      ) {
        highlight.matchedEdgeIds.add(edge.id);
      }
    });

    return highlight;
  }

  function buildBondHighlight(matchBond) {
    const highlight = createEmptyHighlight("bond", "bond");
    const visibleEdges = getVisibleEdges();

    visibleEdges.forEach((edge) => {
      if (!matchBond(edge)) {
        return;
      }
      highlight.matchedEdgeIds.add(edge.id);
      const [leftIndexId, rightIndexId] = resolveEdgeIndexIds(edge);
      const [leftTensorId, rightTensorId] = resolveEdgeTensorIds(edge);
      if (leftIndexId) {
        highlight.contextIndexIds.add(leftIndexId);
      }
      if (rightIndexId) {
        highlight.contextIndexIds.add(rightIndexId);
      }
      if (leftTensorId) {
        highlight.contextTensorIds.add(leftTensorId);
      }
      if (rightTensorId) {
        highlight.contextTensorIds.add(rightTensorId);
      }
    });

    return highlight;
  }

  function buildTagFilterHighlight(filters) {
    if (!filters.enabled) {
      return null;
    }
    const scopeKind = scopeToEntityKind(filters.scope);
    if (scopeKind === "tensor") {
      return buildTensorHighlight((tensor) =>
        metadataMatchesSelectedTags(tensor && tensor.metadata, filters.selectedTags)
      );
    }
    if (scopeKind === "index") {
      return buildIndexHighlight((index) =>
        metadataMatchesSelectedTags(index && index.metadata, filters.selectedTags)
      );
    }
    return buildBondHighlight((edge) =>
      metadataMatchesSelectedTags(edge && edge.metadata, filters.selectedTags)
    );
  }

  function buildNameSearchHighlight(search) {
    if (!search.enabled || !search.query) {
      return null;
    }
    const scopeKind = scopeToEntityKind(search.scope);
    if (scopeKind === "tensor") {
      return buildTensorHighlight((tensor) =>
        nameMatchesQuery(tensor && tensor.name, search.query)
      );
    }
    if (scopeKind === "index") {
      return buildIndexHighlight((index) =>
        nameMatchesQuery(index && index.name, search.query)
      );
    }
    return buildBondHighlight((edge) =>
      nameMatchesQuery(edge && (edge.name || edge.label), search.query)
    );
  }

  function requestHighlightRender() {
    if (typeof ctx.render !== "function") {
      return;
    }
    ctx.render({
      graph: true,
      properties: false,
      code: false,
      toolbar: false,
      overlays: false,
      planner: false,
      sidebarTabs: false,
      minimap: true,
    });
  }

  function filterButtonIcon() {
    return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M2 3.5a1 1 0 0 1 1-1h10a1 1 0 0 1 .78 1.62L10 8.9v3.6a1 1 0 0 1-1.45.9l-2-1A1 1 0 0 1 6 11.5V8.9L2.22 4.12A1 1 0 0 1 2 3.5Z"></path>
      </svg>
    `;
  }

  function searchButtonIcon() {
    return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M6.8 2.2a4.6 4.6 0 1 0 2.86 8.2l2.62 2.62a.75.75 0 1 0 1.06-1.06l-2.62-2.62A4.6 4.6 0 0 0 6.8 2.2Zm0 1.5a3.1 3.1 0 1 1 0 6.2 3.1 3.1 0 0 1 0-6.2Z"></path>
      </svg>
    `;
  }

  function getCheckboxIdForTag(tag) {
    if (tag === NOT_SPECIFIED_FILTER_VALUE) {
      return "canvas-metadata-filter-tag-not-specified";
    }
    const fallbackSanitize = (value) =>
      String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
    const sanitized = typeof ctx.sanitizeFilename === "function"
      ? ctx.sanitizeFilename(tag).toLowerCase()
      : fallbackSanitize(tag);
    return `canvas-metadata-filter-tag-${sanitized || "tag"}`;
  }

  function renderFilterPopover(filters) {
    const availableTags = collectTagsForScope(filters.scope);
    const filterOptions = collectFilterOptionsForScope(
      collectTagsForScope,
      filters.scope
    );
    return `
        <div class="canvas-tool-popover" data-canvas-tool-popover="filter">
        <div class="canvas-tool-popover-header">
          <div
            id="canvas-metadata-filter-scope-field"
            class="canvas-tool-scope-field select-chevron-field"
            data-expanded="false"
          >
            <select id="canvas-metadata-filter-scope-select" aria-label="Filter scope">
              <option value="tensor"${
                filters.scope === "tensor" ? " selected" : ""
              }>Tensor</option>
              <option value="index"${
                filters.scope === "index" ? " selected" : ""
              }>Index</option>
              <option value="bond"${
                filters.scope === "bond" ? " selected" : ""
              }>Bond</option>
            </select>
          </div>
          <div class="canvas-tool-actions">
            <button
              id="canvas-metadata-filter-clear-button"
              type="button"
              class="button-quiet"
            >
              Clear
            </button>
            <button
              id="canvas-metadata-filter-select-all-button"
              type="button"
              class="button-quiet"
            >
              All
            </button>
            <button
              id="canvas-metadata-filter-select-none-button"
              type="button"
              class="button-quiet"
            >
              None
            </button>
          </div>
        </div>
        <div class="canvas-tool-checkbox-list">
          ${
            filterOptions.length
              ? filterOptions
                  .map((tag) => {
                    const checkboxId = getCheckboxIdForTag(tag);
                    const isChecked = filters.selectedTags.some(
                      (selectedTag) =>
                        selectedTag.toLowerCase() === tag.toLowerCase()
                    );
                    const label =
                      tag === NOT_SPECIFIED_FILTER_VALUE
                        ? NOT_SPECIFIED_FILTER_LABEL
                        : tag;
                    return `
                      <label class="canvas-tool-checkbox" for="${checkboxId}">
                        <input
                          id="${checkboxId}"
                          type="checkbox"
                          data-filter-tag="${ctx.escapeHtml(tag)}"
                          ${isChecked ? "checked" : ""}
                        />
                        <span>${ctx.escapeHtml(label)}</span>
                      </label>
                    `;
                  })
                  .join("")
              : ""
          }
          ${
            availableTags.length
              ? ""
              : '<p class="property-meta">No tags yet for this scope.</p>'
          }
        </div>
      </div>
    `;
  }

  function renderSearchPopover(search) {
    return `
      <div class="canvas-tool-popover" data-canvas-tool-popover="search">
        <div class="canvas-tool-popover-header">
          <div
            id="canvas-name-search-scope-field"
            class="canvas-tool-scope-field select-chevron-field"
            data-expanded="false"
          >
            <select id="canvas-name-search-scope-select" aria-label="Search scope">
              <option value="tensor"${
                search.scope === "tensor" ? " selected" : ""
              }>Tensor</option>
              <option value="index"${
                search.scope === "index" ? " selected" : ""
              }>Index</option>
              <option value="bond"${
                search.scope === "bond" ? " selected" : ""
              }>Bond</option>
            </select>
          </div>
        </div>
        <div class="field-group">
          <input
            id="canvas-name-search-input"
            value="${ctx.escapeHtml(search.query)}"
            placeholder="Exact name"
            aria-label="Search by exact name"
          />
        </div>
      </div>
    `;
  }

  function renderMetadataFilters() {
    if (!canvasTools) {
      return;
    }

    const filters = normalizeMetadataFilters();
    const search = normalizeNameSearch();
    state.metadataFilters = filters;
    state.nameSearch = search;

    canvasTools.innerHTML = `
      <div class="canvas-tool-tray">
        <div class="canvas-tool">
          <button
            id="canvas-metadata-filter-button"
            type="button"
            class="canvas-tool-button${
              state.openCanvasToolPopover === "filter" ? " is-open" : ""
            }${filters.enabled ? " is-active" : ""}"
            aria-label="Filter by metadata tags"
            aria-pressed="${state.openCanvasToolPopover === "filter"}"
          >
            ${filterButtonIcon()}
          </button>
          ${
            state.openCanvasToolPopover === "filter"
              ? renderFilterPopover(filters)
              : ""
          }
        </div>
        <div class="canvas-tool">
          <button
            id="canvas-name-search-button"
            type="button"
            class="canvas-tool-button${
              state.openCanvasToolPopover === "search" ? " is-open" : ""
            }${search.enabled ? " is-active" : ""}"
            aria-label="Search by exact name"
            aria-pressed="${state.openCanvasToolPopover === "search"}"
          >
            ${searchButtonIcon()}
          </button>
          ${
            state.openCanvasToolPopover === "search"
              ? renderSearchPopover(search)
              : ""
          }
        </div>
      </div>
    `;

    const filterButton = document.getElementById("canvas-metadata-filter-button");
    const searchButton = document.getElementById("canvas-name-search-button");
    const filterScopeSelect = document.getElementById(
      "canvas-metadata-filter-scope-select"
    );
    const filterSelectAllButton = document.getElementById(
      "canvas-metadata-filter-select-all-button"
    );
    const filterScopeField = document.getElementById(
      "canvas-metadata-filter-scope-field"
    );
    const filterClearButton = document.getElementById(
      "canvas-metadata-filter-clear-button"
    );
    const filterSelectNoneButton = document.getElementById(
      "canvas-metadata-filter-select-none-button"
    );
    const searchScopeSelect = document.getElementById(
      "canvas-name-search-scope-select"
    );
    const searchScopeField = document.getElementById(
      "canvas-name-search-scope-field"
    );
    const searchInput = document.getElementById("canvas-name-search-input");

    bindSelectChevronDisclosure(filterScopeSelect, filterScopeField);
    bindSelectChevronDisclosure(searchScopeSelect, searchScopeField);

    if (filterButton) {
      filterButton.addEventListener("click", () => {
        const nextPopover =
          state.openCanvasToolPopover === "filter" ? null : "filter";
        shouldFocusSearchInput = false;
        state.openCanvasToolPopover = nextPopover;
        state.nameSearch = {
          ...normalizeNameSearch(),
          enabled: false,
        };
        renderMetadataFilters();
        requestHighlightRender();
      });
    }

    if (searchButton) {
      searchButton.addEventListener("click", () => {
        const nextPopover =
          state.openCanvasToolPopover === "search" ? null : "search";
        shouldFocusSearchInput = nextPopover === "search";
        state.openCanvasToolPopover = nextPopover;
        state.metadataFilters = {
          ...normalizeMetadataFilters(),
          enabled: false,
        };
        renderMetadataFilters();
        requestHighlightRender();
      });
    }

    if (filterScopeSelect) {
      filterScopeSelect.addEventListener("change", () => {
        const nextScope = filterScopeSelect.value;
        const nextSelectedTags = normalizeMetadataFilters().selectedTags.filter((tag) =>
          tag === NOT_SPECIFIED_FILTER_VALUE ||
          collectTagsForScope(nextScope).some(
            (candidate) => candidate.toLowerCase() === tag.toLowerCase()
          )
        );
        updateMetadataFilters({
          scope: nextScope,
          selectedTags: nextSelectedTags,
          enabled: true,
        });
      });
    }

    if (filterSelectAllButton) {
      filterSelectAllButton.addEventListener("click", () => {
        updateMetadataFilters({
          selectedTags: collectFilterOptionsForScope(
            collectTagsForScope,
            filters.scope
          ),
          enabled: true,
        });
      });
    }

    if (filterClearButton) {
      filterClearButton.addEventListener("click", () => {
        updateMetadataFilters(
          {
            selectedTags: [],
            enabled: false,
          },
          { openPopover: "filter" }
        );
      });
    }

    if (filterSelectNoneButton) {
      filterSelectNoneButton.addEventListener("click", () => {
        updateMetadataFilters({
          selectedTags: [],
          enabled: true,
        });
      });
    }

    collectFilterOptionsForScope(collectTagsForScope, filters.scope).forEach((tag) => {
      const checkbox = document.getElementById(getCheckboxIdForTag(tag));
      if (!checkbox) {
        return;
      }
      checkbox.addEventListener("change", () => {
        const nextSelectedTags = collectFilterOptionsForScope(
          collectTagsForScope,
          filters.scope
        ).filter((candidate) => {
          const checkboxElement = document.getElementById(
            getCheckboxIdForTag(candidate)
          );
          return checkboxElement && checkboxElement.checked;
        });
        updateMetadataFilters({
          selectedTags: nextSelectedTags,
          enabled: true,
        });
      });
    });

    if (searchScopeSelect) {
      searchScopeSelect.addEventListener("change", () => {
        updateNameSearch({
          scope: searchScopeSelect.value,
          enabled: Boolean(normalizeText(searchInput ? searchInput.value : search.query)),
        });
      });
    }

    if (searchInput) {
      searchInput.addEventListener("input", () => {
        updateNameSearch(
          {
            query: searchInput.value,
            enabled: Boolean(normalizeText(searchInput.value)),
          },
          { renderPanel: false }
        );
      });
      if (
        shouldFocusSearchInput &&
        state.openCanvasToolPopover === "search" &&
        typeof searchInput.focus === "function"
      ) {
        searchInput.focus();
      }
    }
    shouldFocusSearchInput = false;
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

  function getMetadataFilterHighlight() {
    const filters = normalizeMetadataFilters();
    const search = normalizeNameSearch();
    state.metadataFilters = filters;
    state.nameSearch = search;
    if (!state.spec) {
      return null;
    }
    if (search.enabled && search.query) {
      return buildNameSearchHighlight(search);
    }
    if (filters.enabled) {
      return buildTagFilterHighlight(filters);
    }
    return null;
  }

  function getMetadataFilterEntityState(entityKind, entityId, highlight = null) {
    const resolvedHighlight = highlight || getMetadataFilterHighlight();
    if (!resolvedHighlight || !entityId) {
      return "none";
    }
    if (entityKind === "tensor") {
      if (resolvedHighlight.matchedTensorIds.has(entityId)) {
        return "match";
      }
      if (resolvedHighlight.contextTensorIds.has(entityId)) {
        return "context";
      }
      return "dim";
    }
    if (entityKind === "index") {
      if (resolvedHighlight.matchedIndexIds.has(entityId)) {
        return "match";
      }
      if (resolvedHighlight.contextIndexIds.has(entityId)) {
        return "context";
      }
      return "dim";
    }
    if (entityKind === "edge") {
      if (resolvedHighlight.matchedEdgeIds.has(entityId)) {
        return "match";
      }
      if (resolvedHighlight.contextEdgeIds.has(entityId)) {
        return "context";
      }
      return "dim";
    }
    return "none";
  }

  Object.assign(ctx, {
    renderMetadataFilters,
    getMetadataFilterHighlight,
    getMetadataFilterEntityState,
    updateMetadataFilters,
    updateNameSearch,
  });
}
