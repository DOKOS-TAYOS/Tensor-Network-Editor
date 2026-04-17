export const RESERVED_METADATA_KEYS = new Set(["color", "collapsed", "tags"]);

export function createMetadataEditorSupport({
  annotationDefinitionsByScope = {},
  tagSuggestionsByScope = {},
  escapeHtml,
  isObject,
  isMetadataDisclosureOpen = null,
  setMetadataDisclosureOpen = null,
}) {
  function resolveAnnotationDefinitionsByScope() {
    return typeof annotationDefinitionsByScope === "function"
      ? annotationDefinitionsByScope() || {}
      : annotationDefinitionsByScope || {};
  }

  function resolveTagSuggestionsByScope() {
    return typeof tagSuggestionsByScope === "function"
      ? tagSuggestionsByScope() || {}
      : tagSuggestionsByScope || {};
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

  function getProtectedMetadata(target) {
    const metadata = getReservedMetadata(target);
    const protectedMetadata = {};
    Object.keys(metadata).forEach((key) => {
      if (RESERVED_METADATA_KEYS.has(key)) {
        protectedMetadata[key] = metadata[key];
      }
    });
    return protectedMetadata;
  }

  function getCustomMetadata(target) {
    const metadata = getReservedMetadata(target);
    const customMetadata = {};
    Object.keys(metadata).forEach((key) => {
      if (!RESERVED_METADATA_KEYS.has(key)) {
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
      .filter((value, index, values) => {
        const normalizedValue = value.toLowerCase();
        return (
          values.findIndex(
            (candidate) => String(candidate || "").trim().toLowerCase() === normalizedValue
          ) === index
        );
      });
  }

  function formatCustomMetadataValue(target) {
    return JSON.stringify(getCustomMetadata(target), null, 2);
  }

  function parseCustomMetadataValue(rawValue) {
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
    Object.keys(parsedValue).forEach((key) => {
      if (!RESERVED_METADATA_KEYS.has(key)) {
        sanitizedMetadata[key] = parsedValue[key];
      }
    });
    return sanitizedMetadata;
  }

  function metadataValuesEqual(leftValue, rightValue) {
    return JSON.stringify(leftValue) === JSON.stringify(rightValue);
  }

  function uniqueCaseInsensitive(values) {
    const seen = new Set();
    return values.filter((value) => {
      const normalizedValue = String(value || "").trim().toLowerCase();
      if (!normalizedValue || seen.has(normalizedValue)) {
        return false;
      }
      seen.add(normalizedValue);
      return true;
    });
  }

  function getExistingTagSuggestions(annotationScope = null) {
    const scopedSuggestions = resolveTagSuggestionsByScope()[annotationScope] || [];
    return Array.isArray(scopedSuggestions)
      ? scopedSuggestions.filter((value) => typeof value === "string" && value.trim())
      : [];
  }

  function getGuidedSuggestionValues(annotationScope = null) {
    return getAnnotationDefinitions(annotationScope).flatMap((definition) =>
      Array.isArray(definition.suggestions)
        ? definition.suggestions.filter(
            (suggestion) => typeof suggestion === "string" && suggestion.trim()
          )
        : []
    );
  }

  function buildTagAutocompleteSuggestions(annotationScope = null, rawQuery = "") {
    const normalizedQuery = String(rawQuery || "").trim().toLowerCase();
    const combinedSuggestions = uniqueCaseInsensitive([
      ...getExistingTagSuggestions(annotationScope),
      ...getGuidedSuggestionValues(annotationScope),
    ]);
    if (!normalizedQuery) {
      return combinedSuggestions;
    }
    return combinedSuggestions.filter((suggestion) =>
      suggestion.toLowerCase().startsWith(normalizedQuery)
    );
  }

  function replaceActiveTagToken(rawValue, suggestion, cursorPosition = null) {
    const value = String(rawValue || "");
    const normalizedSuggestion = String(suggestion || "").trim();
    if (!normalizedSuggestion) {
      return value;
    }
    const safeCursor =
      typeof cursorPosition === "number" && Number.isFinite(cursorPosition)
        ? cursorPosition
        : value.length;
    const previousComma = value.lastIndexOf(",", safeCursor - 1);
    const nextComma = value.indexOf(",", safeCursor);
    const tokenStart = previousComma >= 0 ? previousComma + 1 : 0;
    const tokenEnd = nextComma >= 0 ? nextComma : value.length;
    const before = value
      .slice(0, tokenStart)
      .replace(/\s*$/, "")
      .replace(/\s*,\s*$/, ",");
    const after = value
      .slice(tokenEnd)
      .replace(/^\s*/, "")
      .replace(/^,\s*/, ", ");
    const prefix = before ? `${before} ` : "";
    return `${prefix}${normalizedSuggestion}${after}`.trim();
  }

  function deriveDisclosureKey(baseFieldKey) {
    if (typeof baseFieldKey !== "string" || !baseFieldKey) {
      return null;
    }
    if (baseFieldKey.endsWith(":tags")) {
      return `${baseFieldKey.slice(0, -5)}:metadata`;
    }
    if (baseFieldKey.endsWith(":custom-metadata")) {
      return `${baseFieldKey.slice(0, -16)}:metadata`;
    }
    return null;
  }

  function buildMetadataEditorMarkup({
    tagsInputId,
    tagsFocusKey,
    customMetadataInputId,
    customMetadataFocusKey,
    target,
    annotationScope = null,
    collapsible = false,
    summaryLabel = "Metadata",
    disclosureKey = null,
  }) {
    const tagSuggestionsId = `${tagsInputId}-suggestions`;
    const resolvedDisclosureKey = disclosureKey || deriveDisclosureKey(tagsFocusKey);
    const disclosureId = `${tagsInputId}-disclosure`;
    const disclosureOpen =
      collapsible &&
      resolvedDisclosureKey &&
      typeof isMetadataDisclosureOpen === "function"
        ? Boolean(isMetadataDisclosureOpen(resolvedDisclosureKey))
        : false;
    const metadataEditorMarkup = `
      <div class="field-group">
        <label for="${tagsInputId}">Tags</label>
        <input
          id="${tagsInputId}"
          data-focus-key="${tagsFocusKey}"
          value="${escapeHtml(formatTagsValue(target))}"
          placeholder="physical, observable, left-leg"
        />
        <div
          id="${tagSuggestionsId}"
          class="metadata-tag-suggestions"
          aria-live="polite"
        ></div>
      </div>
      <div class="field-group">
        <label for="${customMetadataInputId}">Custom metadata (JSON)</label>
        <textarea
          id="${customMetadataInputId}"
          data-focus-key="${customMetadataFocusKey}"
          rows="1"
          placeholder='{"role": "physical"}'
        >${escapeHtml(formatCustomMetadataValue(target))}</textarea>
      </div>
    `;
    if (!collapsible) {
      return metadataEditorMarkup;
    }
    return `
      <details
        id="${disclosureId}"
        class="metadata-editor-disclosure properties-disclosure"
        ${disclosureOpen ? "open" : ""}
      >
        <summary class="properties-disclosure-summary properties-disclosure-chevron">${escapeHtml(
          summaryLabel
        )}</summary>
        <div class="properties-disclosure-body metadata-editor-disclosure-body">
          ${metadataEditorMarkup}
        </div>
      </details>
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
    disclosureKey = null,
  }) {
    const documentRef =
      tagsInput &&
      tagsInput.ownerDocument &&
      typeof tagsInput.ownerDocument.getElementById === "function"
        ? tagsInput.ownerDocument
        : customMetadataInput &&
            customMetadataInput.ownerDocument &&
            typeof customMetadataInput.ownerDocument.getElementById === "function"
          ? customMetadataInput.ownerDocument
          : null;
    const suggestionContainer = tagsInput
      ? documentRef && documentRef.getElementById(`${tagsInput.id}-suggestions`)
      : null;
    const resolvedDisclosureKey = disclosureKey || deriveDisclosureKey(tagsFieldKey);
    const disclosureElement =
      resolvedDisclosureKey && tagsInput && documentRef
        ? documentRef.getElementById(`${tagsInput.id}-disclosure`)
        : null;

    if (
      disclosureElement &&
      typeof setMetadataDisclosureOpen === "function"
    ) {
      disclosureElement.addEventListener("toggle", () => {
        setMetadataDisclosureOpen(
          resolvedDisclosureKey,
          Boolean(disclosureElement.open)
        );
      });
    }

    function renderTagSuggestionButtons() {
      if (!tagsInput || !suggestionContainer) {
        return;
      }
      const cursorPosition =
        typeof tagsInput.selectionStart === "number"
          ? tagsInput.selectionStart
          : tagsInput.value.length;
      const rawValue = String(tagsInput.value || "");
      const previousComma = rawValue.lastIndexOf(",", Math.max(0, cursorPosition - 1));
      const nextComma = rawValue.indexOf(",", cursorPosition);
      const tokenStart = previousComma >= 0 ? previousComma + 1 : 0;
      const tokenEnd = nextComma >= 0 ? nextComma : rawValue.length;
      const activeToken = rawValue.slice(tokenStart, tokenEnd).trim();
      const suggestions = buildTagAutocompleteSuggestions(
        annotationScope,
        activeToken
      ).filter(
        (suggestion) => suggestion.toLowerCase() !== activeToken.toLowerCase()
      );
      if (!activeToken || !suggestions.length) {
        suggestionContainer.innerHTML = "";
        return;
      }
      suggestionContainer.innerHTML = suggestions
        .map(
          (suggestion) => `
            <button
              type="button"
              class="metadata-tag-suggestion"
              data-tag-suggestion="${escapeHtml(suggestion)}"
            >
              ${escapeHtml(suggestion)}
            </button>
          `
        )
        .join("");
      suggestionContainer
        .querySelectorAll("[data-tag-suggestion]")
        .forEach((button) => {
          button.addEventListener("mousedown", (event) => {
            event.preventDefault();
            event.stopPropagation();
          });
          button.addEventListener("click", () => {
            const nextValue = replaceActiveTagToken(
              tagsInput.value,
              button.dataset.tagSuggestion || "",
              tagsInput.selectionStart
            );
            tagsInput.value = nextValue;
            commitTagsValue(nextValue);
          });
        });
    }

    function commitTagsValue(rawValue = tagsInput ? tagsInput.value : "") {
      if (!tagsInput) {
        return;
      }
      const nextTags = normalizeTagsValue(rawValue);
      const currentTags = normalizeTagsValue(formatTagsValue(target));
      if (metadataValuesEqual(currentTags, nextTags)) {
        tagsInput.value = nextTags.join(", ");
        renderTagSuggestionButtons();
        return;
      }
      applyDesignChange(
        () => {
          const reservedMetadata = getProtectedMetadata(target);
          const customMetadata = getCustomMetadata(target);
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
      renderTagSuggestionButtons();
      if (customMetadataInput) {
        customMetadataInput.value = formatCustomMetadataValue(target);
      }
    }

    if (tagsInput) {
      bindDebouncedAutosave(tagsInput, tagsFieldKey, () => commitTagsValue());
      tagsInput.addEventListener("focus", () => {
        renderTagSuggestionButtons();
      });
      tagsInput.addEventListener("input", () => {
        renderTagSuggestionButtons();
      });
    }

    bindDebouncedAutosave(
      customMetadataInput,
      customMetadataFieldKey,
      () => {
        let nextCustomMetadata;
        try {
          nextCustomMetadata = parseCustomMetadataValue(customMetadataInput.value);
        } catch (error) {
          setStatus(error.message, "error");
          return;
        }
        const currentCustomMetadata = getCustomMetadata(target);
        if (metadataValuesEqual(currentCustomMetadata, nextCustomMetadata)) {
          customMetadataInput.value = formatCustomMetadataValue(target);
          return;
        }
        applyDesignChange(
          () => {
            target.metadata = {
              ...getProtectedMetadata(target),
              ...nextCustomMetadata,
            };
          },
          {
            invalidate,
            statusMessage,
          }
        );
        customMetadataInput.value = formatCustomMetadataValue(target);
        if (tagsInput) {
          tagsInput.value = formatTagsValue(target);
          renderTagSuggestionButtons();
        }
      },
      { commitOnEnter: false }
    );
  }

  return {
    bindMetadataEditors,
    buildMetadataEditorMarkup,
    buildTagAutocompleteSuggestions,
    formatCustomMetadataValue,
    formatTagsValue,
    getCustomMetadata,
    getProtectedMetadata,
    metadataValuesEqual,
    normalizeMetadataTarget,
    normalizeTagsValue,
    parseCustomMetadataValue,
    replaceActiveTagToken,
  };
}
