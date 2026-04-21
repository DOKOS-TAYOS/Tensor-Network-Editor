export const NOT_SPECIFIED_FILTER_LABEL = "Not specified";
export const NOT_SPECIFIED_FILTER_VALUE = "__not_specified__";

export function normalizeText(value) {
  return String(value || "").trim();
}

function normalizeTextLower(value) {
  return normalizeText(value).toLowerCase();
}

export function collectFilterOptionsForScope(collectTagsForScope, scope) {
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

export function createMetadataFilterStateSupport({
  state,
  getVisibleTensors,
  getVisibleEdges,
  findIndexOwner,
}) {
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
      enabled: Boolean(filters && filters.enabled),
      scope,
      selectedTags,
    };
  }

  function normalizeNameSearch(search = state.nameSearch) {
    const scope =
      search && (search.scope === "index" || search.scope === "bond")
        ? search.scope
        : "tensor";
    const query = normalizeText(search && search.query);
    return {
      enabled: Boolean(search && search.enabled && query),
      query,
      scope,
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

  function resolveEdgeIndexIds(edge) {
    return [
      edge.leftIndexId || (edge.left && edge.left.index_id) || "",
      edge.rightIndexId || (edge.right && edge.right.index_id) || "",
    ];
  }

  function resolveEdgeTensorIds(edge) {
    const [leftIndexId, rightIndexId] = resolveEdgeIndexIds(edge);
    const leftOwner = leftIndexId ? findIndexOwner(leftIndexId) : null;
    const rightOwner = rightIndexId ? findIndexOwner(rightIndexId) : null;
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

  return {
    collectTagsForScope,
    getMetadataFilterEntityState,
    getMetadataFilterHighlight,
    metadataFiltersEqual,
    nameSearchEqual,
    normalizeMetadataFilters,
    normalizeNameSearch,
  };
}
