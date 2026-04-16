export const RESERVED_METADATA_KEYS = new Set(["color", "collapsed", "tags"]);

export function createMetadataEditorSupport({
  annotationDefinitionsByScope = {},
  escapeHtml,
  isObject,
}) {
  function resolveAnnotationDefinitionsByScope() {
    return typeof annotationDefinitionsByScope === "function"
      ? annotationDefinitionsByScope() || {}
      : annotationDefinitionsByScope || {};
  }

  function normalizeMetadataTarget(target) {
    if (!target || typeof target !== "object") {
      return null;
    }
    if (!isObject(target.metadata)) {
      target.metadata = {};
    }
    return target.metadata;
  }

  function getReservedMetadata(target) {
    return normalizeMetadataTarget(target) || {};
  }

  function getAnnotationDefinitions(annotationScope) {
    const definitions =
      resolveAnnotationDefinitionsByScope()[annotationScope] || [];
    return Array.isArray(definitions) ? definitions : [];
  }

  function getGuidedAnnotationKeys(annotationScope) {
    return new Set(
      getAnnotationDefinitions(annotationScope).map((definition) => definition.key)
    );
  }

  function getProtectedMetadata(target, annotationScope = null) {
    const metadata = getReservedMetadata(target);
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    const protectedMetadata = {};
    Object.keys(metadata).forEach((key) => {
      if (RESERVED_METADATA_KEYS.has(key) || guidedAnnotationKeys.has(key)) {
        protectedMetadata[key] = metadata[key];
      }
    });
    return protectedMetadata;
  }

  function getCustomMetadata(target, annotationScope = null) {
    const metadata = getReservedMetadata(target);
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    const customMetadata = {};
    Object.keys(metadata).forEach((key) => {
      if (!RESERVED_METADATA_KEYS.has(key) && !guidedAnnotationKeys.has(key)) {
        customMetadata[key] = metadata[key];
      }
    });
    return customMetadata;
  }

  function formatTagsValue(target) {
    const metadata = getReservedMetadata(target);
    return Array.isArray(metadata.tags) ? metadata.tags.join(", ") : "";
  }

  function normalizeTagsValue(rawValue) {
    return String(rawValue || "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .filter((value, index, values) => values.indexOf(value) === index);
  }

  function formatAnnotationValue(target, key) {
    const metadata = getReservedMetadata(target);
    return typeof metadata[key] === "string" ? metadata[key] : "";
  }

  function normalizeAnnotationValue(rawValue) {
    return String(rawValue || "").trim();
  }

  function formatCustomMetadataValue(target, annotationScope = null) {
    return JSON.stringify(getCustomMetadata(target, annotationScope), null, 2);
  }

  function parseCustomMetadataValue(rawValue, annotationScope = null) {
    if (!rawValue || !String(rawValue).trim()) {
      return {};
    }
    let parsedValue;
    try {
      parsedValue = JSON.parse(rawValue);
    } catch (error) {
      throw new Error("Custom metadata must be valid JSON.");
    }
    if (!isObject(parsedValue)) {
      throw new Error("Custom metadata must be a JSON object.");
    }
    const sanitizedMetadata = {};
    const guidedAnnotationKeys = getGuidedAnnotationKeys(annotationScope);
    Object.keys(parsedValue).forEach((key) => {
      if (!RESERVED_METADATA_KEYS.has(key) && !guidedAnnotationKeys.has(key)) {
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
          value="${escapeHtml(formatTagsValue(target))}"
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
        >${escapeHtml(formatCustomMetadataValue(target, annotationScope))}</textarea>
      </div>
    `;
    if (!collapsible) {
      return metadataEditorMarkup;
    }
    return `
      <details class="metadata-editor-disclosure properties-disclosure">
        <summary class="properties-disclosure-summary">${escapeHtml(summaryLabel)}</summary>
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
                <label for="${inputIdForKey(definition.key)}">${escapeHtml(definition.label)}</label>
                <input
                  id="${inputIdForKey(definition.key)}"
                  data-focus-key="${focusKeyForKey(definition.key)}"
                  value="${escapeHtml(formatAnnotationValue(target, definition.key))}"
                  placeholder="${escapeHtml(definition.placeholder)}"
                />
                ${
                  definition.suggestions.length
                    ? `
                      <div class="suggested-annotation-chips">
                        ${definition.suggestions
                          .map(
                            (suggestion) => `
                              <button
                                id="${suggestionButtonIdForValue(definition.key, suggestion)}"
                                type="button"
                                class="annotation-chip"
                              >
                                ${escapeHtml(suggestion)}
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
    invalidate,
    annotationScope = null,
    bindDebouncedAutosave,
    applyDesignChange,
    setStatus,
  }) {
    bindDebouncedAutosave(tagsInput, tagsFieldKey, () => {
      const nextTags = normalizeTagsValue(tagsInput.value);
      const currentTags = normalizeTagsValue(formatTagsValue(target));
      if (metadataValuesEqual(currentTags, nextTags)) {
        tagsInput.value = nextTags.join(", ");
        return;
      }
      applyDesignChange(
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
          invalidate,
          statusMessage,
        }
      );
      tagsInput.value = formatTagsValue(target);
      if (customMetadataInput) {
        customMetadataInput.value = formatCustomMetadataValue(target, annotationScope);
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
          setStatus(error.message, "error");
          return;
        }
        const currentCustomMetadata = getCustomMetadata(target, annotationScope);
        if (metadataValuesEqual(currentCustomMetadata, nextCustomMetadata)) {
          customMetadataInput.value = formatCustomMetadataValue(target, annotationScope);
          return;
        }
        applyDesignChange(
          () => {
            target.metadata = {
              ...getProtectedMetadata(target, annotationScope),
              ...nextCustomMetadata,
            };
          },
          {
            invalidate,
            statusMessage,
          }
        );
        customMetadataInput.value = formatCustomMetadataValue(target, annotationScope);
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
    invalidate,
    bindDebouncedAutosave,
    commitAutosave,
    applyDesignChange,
  }) {
    const definitions = getAnnotationDefinitions(annotationScope);
    if (!definitions.length) {
      return;
    }

    definitions.forEach((definition) => {
      const input = inputForKey(definition.key);
      if (!input) {
        return;
      }

      const commitAnnotationValue = (rawValue = input.value, options = {}) => {
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
        applyDesignChange(
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
            invalidate,
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

  return {
    bindMetadataEditors,
    bindSuggestedAnnotationEditors,
    buildMetadataEditorMarkup,
    buildSuggestedAnnotationsMarkup,
    formatAnnotationValue,
    formatCustomMetadataValue,
    formatTagsValue,
    getAnnotationDefinitions,
    getCustomMetadata,
    getProtectedMetadata,
    metadataValuesEqual,
    normalizeAnnotationValue,
    normalizeMetadataTarget,
    normalizeTagsValue,
    parseCustomMetadataValue,
  };
}
