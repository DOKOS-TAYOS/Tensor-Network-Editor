export function registerMetadataFilters(ctx) {
  const state = ctx.state;
  const { metadataFiltersPanel } = ctx.dom;
  const { document } = ctx;

  function getAnnotationDefinitions(scope) {
    const scopedDefinitions =
      scope &&
      state.annotationDefinitions &&
      Array.isArray(state.annotationDefinitions[scope])
        ? state.annotationDefinitions[scope]
        : [];
    return scopedDefinitions
      .filter(
        (definition) =>
          ctx.isObject(definition) &&
          typeof definition.key === "string" &&
          definition.key.trim() &&
          typeof definition.label === "string"
      )
      .map((definition) => ({
        key: definition.key.trim(),
        label: definition.label,
        placeholder:
          typeof definition.placeholder === "string"
            ? definition.placeholder
            : "",
      }));
  }

  function normalizeFilterText(value) {
    return String(value || "").trim();
  }

  function normalizeMetadataFilters(filters = state.metadataFilters) {
    const scope =
      filters && filters.scope === "index" ? "index" : "tensor";
    const tag = normalizeFilterText(filters && filters.tag);
    const annotationKey = normalizeFilterText(filters && filters.annotationKey);
    const annotationValue = normalizeFilterText(
      filters && filters.annotationValue
    );
    const validAnnotationKeys = new Set(
      getAnnotationDefinitions(scope).map((definition) => definition.key)
    );
    return {
      scope,
      tag,
      annotationKey: validAnnotationKeys.has(annotationKey) ? annotationKey : "",
      annotationValue: validAnnotationKeys.has(annotationKey)
        ? annotationValue
        : "",
    };
  }

  function metadataFiltersEqual(leftFilters, rightFilters) {
    return (
      leftFilters.scope === rightFilters.scope &&
      leftFilters.tag === rightFilters.tag &&
      leftFilters.annotationKey === rightFilters.annotationKey &&
      leftFilters.annotationValue === rightFilters.annotationValue
    );
  }

  function isMetadataFilterActive(filters = normalizeMetadataFilters()) {
    return Boolean(filters.tag || filters.annotationKey || filters.annotationValue);
  }

  function normalizeComparisonValue(value) {
    return normalizeFilterText(value).toLowerCase();
  }

  function metadataMatchesTag(metadata, tagQuery) {
    if (!tagQuery) {
      return true;
    }
    const tags = Array.isArray(metadata && metadata.tags)
      ? metadata.tags.filter((tag) => typeof tag === "string" && tag.trim())
      : [];
    const normalizedQuery = normalizeComparisonValue(tagQuery);
    return tags.some(
      (tag) => normalizeComparisonValue(tag) === normalizedQuery
    );
  }

  function metadataMatchesAnnotation(metadata, annotationKey, annotationValue) {
    if (!annotationKey) {
      return true;
    }
    if (!metadata || !Object.prototype.hasOwnProperty.call(metadata, annotationKey)) {
      return false;
    }
    if (!annotationValue) {
      return true;
    }
    return (
      normalizeComparisonValue(metadata[annotationKey]) ===
      normalizeComparisonValue(annotationValue)
    );
  }

  function tensorMatchesMetadataFilter(tensor, filters) {
    return (
      metadataMatchesTag(tensor && tensor.metadata, filters.tag) &&
      metadataMatchesAnnotation(
        tensor && tensor.metadata,
        filters.annotationKey,
        filters.annotationValue
      )
    );
  }

  function indexMatchesMetadataFilter(index, filters) {
    return (
      metadataMatchesTag(index && index.metadata, filters.tag) &&
      metadataMatchesAnnotation(
        index && index.metadata,
        filters.annotationKey,
        filters.annotationValue
      )
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
    const leftIndexId = edge.leftIndexId || (edge.left && edge.left.index_id);
    const rightIndexId = edge.rightIndexId || (edge.right && edge.right.index_id);
    const leftOwner = leftIndexId ? ctx.findIndexOwner(leftIndexId) : null;
    const rightOwner = rightIndexId ? ctx.findIndexOwner(rightIndexId) : null;
    return [
      (edge.left && edge.left.tensor_id) ||
        (leftOwner && leftOwner.tensor ? leftOwner.tensor.id : ""),
      (edge.right && edge.right.tensor_id) ||
        (rightOwner && rightOwner.tensor ? rightOwner.tensor.id : ""),
    ];
  }

  function getMetadataFilterHighlight() {
    const filters = normalizeMetadataFilters();
    state.metadataFilters = filters;
    if (!state.spec || !isMetadataFilterActive(filters)) {
      return null;
    }

    const visibleTensors = getVisibleTensors();
    const visibleEdges = getVisibleEdges();
    const matchedTensorIds = new Set();
    const contextTensorIds = new Set();
    const matchedIndexIds = new Set();
    const matchedEdgeIds = new Set();

    if (filters.scope === "tensor") {
      visibleTensors.forEach((tensor) => {
        if (!tensorMatchesMetadataFilter(tensor, filters)) {
          return;
        }
        matchedTensorIds.add(tensor.id);
        (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
          matchedIndexIds.add(index.id);
        });
      });
      visibleEdges.forEach((edge) => {
        const [leftTensorId, rightTensorId] = resolveEdgeTensorIds(edge);
        if (
          leftTensorId &&
          rightTensorId &&
          matchedTensorIds.has(leftTensorId) &&
          matchedTensorIds.has(rightTensorId)
        ) {
          matchedEdgeIds.add(edge.id);
        }
      });
    } else {
      visibleTensors.forEach((tensor) => {
        (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
          if (!indexMatchesMetadataFilter(index, filters)) {
            return;
          }
          matchedIndexIds.add(index.id);
          contextTensorIds.add(tensor.id);
        });
      });
      visibleEdges.forEach((edge) => {
        const [leftIndexId, rightIndexId] = resolveEdgeIndexIds(edge);
        if (
          matchedIndexIds.has(leftIndexId) ||
          matchedIndexIds.has(rightIndexId)
        ) {
          matchedEdgeIds.add(edge.id);
        }
      });
    }

    return {
      scope: filters.scope,
      matchedTensorIds,
      contextTensorIds,
      matchedIndexIds,
      matchedEdgeIds,
    };
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
      return resolvedHighlight.matchedIndexIds.has(entityId) ? "match" : "dim";
    }
    if (entityKind === "edge") {
      return resolvedHighlight.matchedEdgeIds.has(entityId) ? "match" : "dim";
    }
    return "none";
  }

  function renderMetadataFilters() {
    if (!metadataFiltersPanel) {
      return;
    }

    const filters = normalizeMetadataFilters();
    state.metadataFilters = filters;
    const annotationDefinitions = getAnnotationDefinitions(filters.scope);
    const annotationValuePlaceholder = filters.annotationKey
      ? (
          annotationDefinitions.find(
            (definition) => definition.key === filters.annotationKey
          ) || { placeholder: "" }
        ).placeholder
      : "";
    const keepDisclosureOpen =
      /<details[^>]*\sopen\b/i.test(metadataFiltersPanel.innerHTML) ||
      isMetadataFilterActive(filters);

    metadataFiltersPanel.innerHTML = `
      <details class="metadata-filters-card metadata-filters-disclosure"${
        keepDisclosureOpen ? " open" : ""
      }>
        <summary class="properties-disclosure-summary">Metadata filters</summary>
        <div class="properties-disclosure-body metadata-filters-body">
          <div class="metadata-filters-header">
            <button
              id="clear-metadata-filters-button"
              type="button"
              class="button-quiet"
            >
              Clear
            </button>
          </div>
          <div class="field-group">
            <label for="metadata-filter-scope-select">Scope</label>
            <select id="metadata-filter-scope-select">
              <option value="tensor"${
                filters.scope === "tensor" ? " selected" : ""
              }>Tensor</option>
              <option value="index"${
                filters.scope === "index" ? " selected" : ""
              }>Index</option>
            </select>
          </div>
          <div class="field-group">
            <label for="metadata-filter-tag-input">Tag</label>
            <input
              id="metadata-filter-tag-input"
              value="${ctx.escapeHtml(filters.tag)}"
              placeholder="physical"
            />
          </div>
          <div class="field-group">
            <label for="metadata-filter-key-select">Suggested annotation</label>
            <select id="metadata-filter-key-select">
              <option value="">Any guided field</option>
              ${annotationDefinitions
                .map(
                  (definition) => `
                    <option value="${ctx.escapeHtml(definition.key)}"${
                      filters.annotationKey === definition.key ? " selected" : ""
                    }>
                      ${ctx.escapeHtml(definition.label)}
                    </option>
                  `
                )
                .join("")}
            </select>
          </div>
          <div class="field-group">
            <label for="metadata-filter-value-input">Annotation value</label>
            <input
              id="metadata-filter-value-input"
              value="${ctx.escapeHtml(filters.annotationValue)}"
              placeholder="${ctx.escapeHtml(annotationValuePlaceholder)}"
              ${filters.annotationKey ? "" : "disabled"}
            />
          </div>
          <p class="property-meta">
            Matching elements stay bright and the rest fade.
          </p>
        </div>
      </details>
    `;

    const scopeSelect = document.getElementById("metadata-filter-scope-select");
    const tagInput = document.getElementById("metadata-filter-tag-input");
    const keySelect = document.getElementById("metadata-filter-key-select");
    const valueInput = document.getElementById("metadata-filter-value-input");
    const clearButton = document.getElementById("clear-metadata-filters-button");

    if (scopeSelect) {
      scopeSelect.value = filters.scope;
      scopeSelect.addEventListener("change", () => {
        updateMetadataFilters({
          scope: scopeSelect.value,
          annotationKey: "",
          annotationValue: "",
        });
      });
    }
    if (tagInput) {
      tagInput.value = filters.tag;
      tagInput.addEventListener("input", () => {
        updateMetadataFilters({ tag: tagInput.value }, { renderPanel: false });
      });
      tagInput.addEventListener("blur", () => {
        updateMetadataFilters({ tag: tagInput.value });
      });
    }
    if (keySelect) {
      keySelect.value = filters.annotationKey;
      keySelect.addEventListener("change", () => {
        updateMetadataFilters({
          annotationKey: keySelect.value,
          annotationValue: "",
        });
      });
    }
    if (valueInput) {
      valueInput.value = filters.annotationValue;
      valueInput.addEventListener("input", () => {
        updateMetadataFilters(
          { annotationValue: valueInput.value },
          { renderPanel: false }
        );
      });
      valueInput.addEventListener("blur", () => {
        updateMetadataFilters({ annotationValue: valueInput.value });
      });
    }
    if (clearButton) {
      clearButton.addEventListener("click", () => {
        updateMetadataFilters({
          tag: "",
          annotationKey: "",
          annotationValue: "",
        });
      });
    }
  }

  function updateMetadataFilters(updates, options = {}) {
    const currentFilters = normalizeMetadataFilters();
    const nextFilters = normalizeMetadataFilters({
      ...currentFilters,
      ...updates,
    });
    state.metadataFilters = nextFilters;
    if (options.renderPanel !== false) {
      renderMetadataFilters();
    }
    if (!metadataFiltersEqual(currentFilters, nextFilters) && typeof ctx.render === "function") {
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
  }

  Object.assign(ctx, {
    renderMetadataFilters,
    getMetadataFilterHighlight,
    getMetadataFilterEntityState,
  });
}
