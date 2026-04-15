const AUTOSAVE_DELAY_MS = 300;
const RESERVED_METADATA_KEYS = new Set(["color", "collapsed", "tags"]);

export function propertyInvalidation(ctx, overrides = {}) {
  const isLinearPeriodicMode =
    typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode();
  return {
    graph: false,
    lookups: false,
    analysis: false,
    properties: isLinearPeriodicMode,
    overlays: false,
    planner: false,
    minimap: false,
    ...overrides,
  };
}

export function selectionColorInvalidation(ctx, selectedEntries) {
  const entryKinds = new Set(
    (Array.isArray(selectedEntries) ? selectedEntries : []).map(
      (entry) => entry.kind
    )
  );
  const affectsGraph =
    entryKinds.has("tensor") ||
    entryKinds.has("index") ||
    entryKinds.has("edge");
  const affectsOverlays = entryKinds.has("group") || entryKinds.has("note");
  return propertyInvalidation(ctx, {
    graph: affectsGraph,
    overlays: affectsOverlays,
    minimap: affectsGraph,
  });
}

export function renderTrashIcon() {
  return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
      </svg>
    `;
}

export function normalizeElementDimension(value, asFiniteNumber) {
  return BigInt(Math.max(1, Math.round(asFiniteNumber(value, 1))));
}

export function getTensorTotalElementCount(tensor, asFiniteNumber) {
  const indices = Array.isArray(tensor && tensor.indices) ? tensor.indices : [];
  return indices.reduce(
    (product, index) =>
      product * normalizeElementDimension(index.dimension, asFiniteNumber),
    1n
  );
}

export function getTotalElementCountForTensorIds(
  tensorIds,
  findTensorById,
  asFiniteNumber
) {
  const uniqueTensorIds = [...new Set(Array.isArray(tensorIds) ? tensorIds : [])];
  let resolvedTensorCount = 0;
  const totalElementCount = uniqueTensorIds.reduce((sum, tensorId) => {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return sum;
    }
    resolvedTensorCount += 1;
    return sum + getTensorTotalElementCount(tensor, asFiniteNumber);
  }, 0n);
  return resolvedTensorCount ? totalElementCount : null;
}

export function getSelectionEntryTensorIds(entry) {
  if (!entry) {
    return [];
  }
  if (entry.kind === "group") {
    return Array.isArray(entry.group && entry.group.tensor_ids)
      ? [...entry.group.tensor_ids]
      : [];
  }
  if (entry.kind === "contraction-tensor") {
    return Array.isArray(entry.tensor && entry.tensor.sourceTensorIds)
      ? [...entry.tensor.sourceTensorIds]
      : [];
  }
  if (entry.kind === "tensor" && entry.id) {
    return [entry.id];
  }
  return [];
}

export function getSelectionTotalElementCount(
  selectedEntries,
  findTensorById,
  asFiniteNumber
) {
  const tensorIds = selectedEntries.flatMap((entry) =>
    getSelectionEntryTensorIds(entry)
  );
  return getTotalElementCountForTensorIds(
    tensorIds,
    findTensorById,
    asFiniteNumber
  );
}

export function getContractionTensorTotalElementCount(
  tensor,
  findTensorById,
  asFiniteNumber
) {
  const sourceTensorIds = Array.isArray(tensor && tensor.sourceTensorIds)
    ? tensor.sourceTensorIds
    : [];
  const totalElementCount = getTotalElementCountForTensorIds(
    sourceTensorIds,
    findTensorById,
    asFiniteNumber
  );
  return totalElementCount === null
    ? getTensorTotalElementCount(tensor, asFiniteNumber)
    : totalElementCount;
}

export function formatTotalElementCount(totalElementCount) {
  return totalElementCount === null ? "" : totalElementCount.toString();
}

export function createPropertiesSupport({ ctx, state, window }) {
  const autosaveTimers = new Map();

  function clearAutosaveTimer(fieldKey) {
    const timerId = autosaveTimers.get(fieldKey);
    if (typeof timerId === "number") {
      window.clearTimeout(timerId);
    }
    autosaveTimers.delete(fieldKey);
  }

  function commitAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    commit();
  }

  function scheduleAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    autosaveTimers.set(
      fieldKey,
      window.setTimeout(() => {
        autosaveTimers.delete(fieldKey);
        commit();
      }, AUTOSAVE_DELAY_MS)
    );
  }

  function bindDebouncedAutosave(element, fieldKey, commit, options = {}) {
    if (!element) {
      return;
    }
    element.dataset.focusKey = fieldKey;
    element.addEventListener("input", () => {
      scheduleAutosave(fieldKey, commit);
    });
    element.addEventListener("blur", () => {
      commitAutosave(fieldKey, commit);
    });
    if (options.commitOnEnter !== false) {
      element.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.shiftKey) {
          return;
        }
        event.preventDefault();
        commitAutosave(fieldKey, commit);
      });
    }
  }

  function bindImmediateAutosave(
    element,
    fieldKey,
    commit,
    eventName = "change"
  ) {
    if (!element) {
      return;
    }
    if (fieldKey) {
      element.dataset.focusKey = fieldKey;
    }
    element.addEventListener(eventName, () => {
      commit();
    });
  }

  function normalizeMetadataTarget(target) {
    if (!target || !ctx.isObject(target)) {
      return null;
    }
    target.metadata = ctx.isObject(target.metadata) ? target.metadata : {};
    return target.metadata;
  }

  function getReservedMetadata(target) {
    return getProtectedMetadata(target);
  }

  function getAnnotationDefinitions(annotationScope) {
    const scopedDefinitions =
      annotationScope &&
      state.annotationDefinitions &&
      Array.isArray(state.annotationDefinitions[annotationScope])
        ? state.annotationDefinitions[annotationScope]
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
        suggestions: Array.isArray(definition.suggestions)
          ? definition.suggestions.filter(
              (suggestion) =>
                typeof suggestion === "string" && suggestion.trim()
            )
          : [],
      }));
  }

  function getGuidedAnnotationKeys(annotationScope) {
    return new Set(
      getAnnotationDefinitions(annotationScope).map((definition) => definition.key)
    );
  }

  function getProtectedMetadata(target, annotationScope = null) {
    const metadata = normalizeMetadataTarget(target);
    if (!metadata) {
      return {};
    }
    const reserved = {};
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    RESERVED_METADATA_KEYS.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(metadata, key)) {
        reserved[key] = ctx.deepClone(metadata[key]);
      }
    });
    guidedAnnotationKeys.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(metadata, key)) {
        reserved[key] = ctx.deepClone(metadata[key]);
      }
    });
    return reserved;
  }

  function getCustomMetadata(target, annotationScope = null) {
    const metadata = normalizeMetadataTarget(target);
    if (!metadata) {
      return {};
    }
    const customMetadata = {};
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    Object.keys(metadata).forEach((key) => {
      if (
        !RESERVED_METADATA_KEYS.has(key) &&
        !guidedAnnotationKeys.has(key)
      ) {
        customMetadata[key] = ctx.deepClone(metadata[key]);
      }
    });
    return customMetadata;
  }

  function formatTagsValue(target) {
    const metadata = normalizeMetadataTarget(target);
    const tags = Array.isArray(metadata && metadata.tags)
      ? metadata.tags.filter((tag) => typeof tag === "string" && tag.trim())
      : [];
    return tags.join(", ");
  }

  function normalizeTagsValue(rawValue) {
    const tags = [];
    const seenTags = new Set();
    String(rawValue || "")
      .split(",")
      .map((candidate) => candidate.trim())
      .filter(Boolean)
      .forEach((tag) => {
        if (seenTags.has(tag)) {
          return;
        }
        seenTags.add(tag);
        tags.push(tag);
      });
    return tags;
  }

  function formatAnnotationValue(target, key) {
    const metadata = normalizeMetadataTarget(target);
    if (!metadata || !Object.prototype.hasOwnProperty.call(metadata, key)) {
      return "";
    }
    const value = metadata[key];
    if (typeof value === "string") {
      return value;
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    return "";
  }

  function normalizeAnnotationValue(rawValue) {
    return String(rawValue || "").trim();
  }

  function formatCustomMetadataValue(target, annotationScope = null) {
    const customMetadata = getCustomMetadata(target, annotationScope);
    return Object.keys(customMetadata).length
      ? JSON.stringify(customMetadata, null, 2)
      : "";
  }

  function parseCustomMetadataValue(rawValue, annotationScope = null) {
    if (!String(rawValue || "").trim()) {
      return {};
    }
    let parsedValue;
    try {
      parsedValue = JSON.parse(rawValue);
    } catch (error) {
      throw new Error("Custom metadata must be valid JSON.");
    }
    if (!ctx.isObject(parsedValue)) {
      throw new Error("Custom metadata must be a JSON object.");
    }
    const sanitizedMetadata = {};
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    Object.keys(parsedValue).forEach((key) => {
      if (
        !RESERVED_METADATA_KEYS.has(key) &&
        !guidedAnnotationKeys.has(key)
      ) {
        sanitizedMetadata[key] = parsedValue[key];
      }
    });
    return sanitizedMetadata;
  }

  function metadataValuesEqual(leftValue, rightValue) {
    return JSON.stringify(leftValue) === JSON.stringify(rightValue);
  }

  function buildMetadataEditorMarkup({
    tagsInputId,
    tagsFocusKey,
    customMetadataInputId,
    customMetadataFocusKey,
    target,
    annotationScope = null,
    suggestedAnnotationsMarkup = "",
    collapsible = false,
    summaryLabel = "Metadata",
  }) {
    const metadataEditorMarkup = `
      <div class="field-group">
        <label for="${tagsInputId}">Tags</label>
        <input
          id="${tagsInputId}"
          data-focus-key="${tagsFocusKey}"
          value="${ctx.escapeHtml(formatTagsValue(target))}"
          placeholder="physical, observable, left-leg"
        />
      </div>
      ${suggestedAnnotationsMarkup}
      <div class="field-group">
        <label for="${customMetadataInputId}">Custom metadata (JSON)</label>
        <textarea
          id="${customMetadataInputId}"
          data-focus-key="${customMetadataFocusKey}"
          rows="5"
          placeholder='{"role": "physical"}'
        >${ctx.escapeHtml(
          formatCustomMetadataValue(target, annotationScope)
        )}</textarea>
      </div>
    `;
    if (!collapsible) {
      return metadataEditorMarkup;
    }
    return `
      <details class="metadata-editor-disclosure properties-disclosure">
        <summary class="properties-disclosure-summary">${ctx.escapeHtml(
          summaryLabel
        )}</summary>
        <div class="properties-disclosure-body metadata-editor-disclosure-body">
          ${metadataEditorMarkup}
        </div>
      </details>
    `;
  }

  function buildSuggestedAnnotationsMarkup({
    annotationScope,
    target,
    inputIdForKey,
    focusKeyForKey,
    suggestionButtonIdForValue,
  }) {
    const definitions = getAnnotationDefinitions(annotationScope);
    if (!definitions.length) {
      return "";
    }
    return `
      <section class="suggested-annotations">
        <div class="properties-section-heading">Suggested annotations</div>
        ${definitions
          .map(
            (definition) => `
              <div class="field-group">
                <label for="${inputIdForKey(definition.key)}">${ctx.escapeHtml(
                  definition.label
                )}</label>
                <input
                  id="${inputIdForKey(definition.key)}"
                  data-focus-key="${focusKeyForKey(definition.key)}"
                  value="${ctx.escapeHtml(
                    formatAnnotationValue(target, definition.key)
                  )}"
                  placeholder="${ctx.escapeHtml(definition.placeholder)}"
                />
                ${
                  definition.suggestions.length
                    ? `
                      <div class="suggested-annotation-chips">
                        ${definition.suggestions
                          .map(
                            (suggestion) => `
                              <button
                                id="${suggestionButtonIdForValue(
                                  definition.key,
                                  suggestion
                                )}"
                                type="button"
                                class="annotation-chip"
                              >
                                ${ctx.escapeHtml(suggestion)}
                              </button>
                            `
                          )
                          .join("")}
                      </div>
                    `
                    : ""
                }
              </div>
            `
          )
          .join("")}
      </section>
    `;
  }

  function bindMetadataEditors({
    target,
    tagsInput,
    tagsFieldKey,
    customMetadataInput,
    customMetadataFieldKey,
    statusMessage,
    invalidate = null,
    annotationScope = null,
  }) {
    const metadataInvalidation =
      invalidate || propertyInvalidation(ctx);

    bindDebouncedAutosave(tagsInput, tagsFieldKey, () => {
      const nextTags = normalizeTagsValue(tagsInput.value);
      const currentTags = normalizeTagsValue(formatTagsValue(target));
      if (metadataValuesEqual(currentTags, nextTags)) {
        tagsInput.value = nextTags.join(", ");
        return;
      }
      ctx.applyDesignChange(
        () => {
          const reservedMetadata = getProtectedMetadata(target, annotationScope);
          const customMetadata = getCustomMetadata(target, annotationScope);
          target.metadata = {
            ...reservedMetadata,
            ...customMetadata,
          };
          if (nextTags.length) {
            target.metadata.tags = nextTags;
          } else {
            delete target.metadata.tags;
          }
        },
        {
          invalidate: metadataInvalidation,
          statusMessage,
        }
      );
      tagsInput.value = formatTagsValue(target);
      if (customMetadataInput) {
        customMetadataInput.value = formatCustomMetadataValue(
          target,
          annotationScope
        );
      }
    });

    bindDebouncedAutosave(
      customMetadataInput,
      customMetadataFieldKey,
      () => {
        let nextCustomMetadata;
        try {
          nextCustomMetadata = parseCustomMetadataValue(
            customMetadataInput.value,
            annotationScope
          );
        } catch (error) {
          ctx.setStatus(error.message, "error");
          return;
        }
        const currentCustomMetadata = getCustomMetadata(target, annotationScope);
        if (metadataValuesEqual(currentCustomMetadata, nextCustomMetadata)) {
          customMetadataInput.value = formatCustomMetadataValue(
            target,
            annotationScope
          );
          return;
        }
        ctx.applyDesignChange(
          () => {
            target.metadata = {
              ...getProtectedMetadata(target, annotationScope),
              ...nextCustomMetadata,
            };
          },
          {
            invalidate: metadataInvalidation,
            statusMessage,
          }
        );
        customMetadataInput.value = formatCustomMetadataValue(
          target,
          annotationScope
        );
        if (tagsInput) {
          tagsInput.value = formatTagsValue(target);
        }
      },
      { commitOnEnter: false }
    );
  }

  function bindSuggestedAnnotationEditors({
    target,
    annotationScope,
    inputForKey,
    fieldKeyForKey,
    suggestionButtonForValue,
    customMetadataInput = null,
    statusMessage,
    invalidate = null,
  }) {
    const definitions = getAnnotationDefinitions(annotationScope);
    if (!definitions.length) {
      return;
    }
    const metadataInvalidation = invalidate || propertyInvalidation(ctx);

    definitions.forEach((definition) => {
      const input = inputForKey(definition.key);
      if (!input) {
        return;
      }

      const commitAnnotationValue = (
        rawValue = input.value,
        options = {}
      ) => {
        const nextValue = normalizeAnnotationValue(rawValue);
        const currentValue = normalizeAnnotationValue(
          formatAnnotationValue(target, definition.key)
        );
        if (currentValue === nextValue) {
          input.value = formatAnnotationValue(target, definition.key);
          if (customMetadataInput) {
            customMetadataInput.value = formatCustomMetadataValue(
              target,
              annotationScope
            );
          }
          return;
        }
        ctx.applyDesignChange(
          () => {
            const metadata = normalizeMetadataTarget(target);
            if (!metadata) {
              return;
            }
            if (nextValue) {
              metadata[definition.key] = nextValue;
            } else {
              delete metadata[definition.key];
            }
          },
          {
            invalidate: metadataInvalidation,
            statusMessage,
            ...options,
          }
        );
        input.value = formatAnnotationValue(target, definition.key);
        if (customMetadataInput) {
          customMetadataInput.value = formatCustomMetadataValue(
            target,
            annotationScope
          );
        }
      };

      bindDebouncedAutosave(
        input,
        fieldKeyForKey(definition.key),
        () => commitAnnotationValue()
      );

      definition.suggestions.forEach((suggestion) => {
        const suggestionButton = suggestionButtonForValue(
          definition.key,
          suggestion
        );
        if (!suggestionButton) {
          return;
        }
        suggestionButton.addEventListener("click", () => {
          input.value = suggestion;
          commitAutosave(fieldKeyForKey(definition.key), () =>
            commitAnnotationValue(suggestion)
          );
        });
      });
    });
  }

  function tensorDisclosureState(tensorId) {
    if (!state.tensorIndexDisclosureState[tensorId]) {
      state.tensorIndexDisclosureState[tensorId] = {};
    }
    return state.tensorIndexDisclosureState[tensorId];
  }

  function isTensorIndexDisclosureOpen(tensorId, indexId) {
    return Boolean(tensorDisclosureState(tensorId)[indexId]);
  }

  function setTensorIndexDisclosureOpen(tensorId, indexId, isOpen) {
    const disclosureState = tensorDisclosureState(tensorId);
    if (isOpen) {
      disclosureState[indexId] = true;
      return;
    }
    delete disclosureState[indexId];
  }

  function syncPendingTensorIndexDisclosure() {
    const pendingIndexId = state.pendingPropertiesIndexFocusId;
    if (!pendingIndexId) {
      return;
    }

    const located = ctx.findIndexOwner(pendingIndexId);
    state.pendingPropertiesIndexFocusId = null;
    if (!located) {
      return;
    }

    const wasOpen = isTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId);
    setTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId, true);
    state.autoExpandedTensorIndex = {
      tensorId: located.tensor.id,
      indexId: pendingIndexId,
      wasOpen,
    };
  }

  function toggleTensorIndexDisclosure(tensorId, indexId) {
    const nextOpen = !isTensorIndexDisclosureOpen(tensorId, indexId);
    setTensorIndexDisclosureOpen(tensorId, indexId, nextOpen);
    if (
      state.autoExpandedTensorIndex &&
      state.autoExpandedTensorIndex.tensorId === tensorId &&
      state.autoExpandedTensorIndex.indexId === indexId
    ) {
      state.autoExpandedTensorIndex = null;
    }
    ctx.renderProperties();
  }

  return {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    buildMetadataEditorMarkup,
    buildSuggestedAnnotationsMarkup,
    bindMetadataEditors,
    bindSuggestedAnnotationEditors,
    propertyInvalidation: (overrides = {}) =>
      propertyInvalidation(ctx, overrides),
    selectionColorInvalidation: (selectedEntries) =>
      selectionColorInvalidation(ctx, selectedEntries),
    renderTrashIcon,
    getTensorTotalElementCount: (tensor) =>
      getTensorTotalElementCount(tensor, ctx.asFiniteNumber),
    getTotalElementCountForTensorIds: (tensorIds) =>
      getTotalElementCountForTensorIds(
        tensorIds,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    getSelectionEntryTensorIds,
    getSelectionTotalElementCount: (selectedEntries) =>
      getSelectionTotalElementCount(
        selectedEntries,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    getContractionTensorTotalElementCount: (tensor) =>
      getContractionTensorTotalElementCount(
        tensor,
        ctx.findTensorById,
        ctx.asFiniteNumber
      ),
    formatTotalElementCount,
    tensorDisclosureState,
    isTensorIndexDisclosureOpen,
    setTensorIndexDisclosureOpen,
    syncPendingTensorIndexDisclosure,
    toggleTensorIndexDisclosure,
  };
}
