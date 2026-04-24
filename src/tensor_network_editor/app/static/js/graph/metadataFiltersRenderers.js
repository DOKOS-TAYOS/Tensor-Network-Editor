import {
  collectFilterOptionsForScope,
  NOT_SPECIFIED_FILTER_LABEL,
  NOT_SPECIFIED_FILTER_VALUE,
} from "./metadataFiltersState.js";

export function createMetadataFilterRendererSupport({
  escapeHtml,
  sanitizeFilename,
  collectTagsForScope,
}) {
  function escapeTooltipText(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function buildTooltipAttributes(label, description, shortcut = "") {
    const attributes = [
      'data-tooltip-enabled="true"',
      `data-shortcut-label="${escapeTooltipText(label)}"`,
      `data-shortcut-description="${escapeTooltipText(description)}"`,
    ];
    if (shortcut) {
      attributes.push(`data-shortcut="${escapeTooltipText(shortcut)}"`);
    }
    return attributes.join(" ");
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
    const sanitized =
      typeof sanitizeFilename === "function"
        ? sanitizeFilename(tag).toLowerCase()
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
                          data-filter-tag="${escapeHtml(tag)}"
                          ${isChecked ? "checked" : ""}
                        />
                        <span>${escapeHtml(label)}</span>
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
            value="${escapeHtml(search.query)}"
            placeholder="Exact name"
            aria-label="Search by exact name"
          />
        </div>
      </div>
    `;
  }

  function buildMetadataFiltersMarkup({ filters, search, openCanvasToolPopover }) {
    return `
      <div class="canvas-tool-tray">
        <div class="canvas-tool">
          <button
            id="canvas-metadata-filter-button"
            type="button"
            class="canvas-tool-button${
              openCanvasToolPopover === "filter" ? " is-open" : ""
            }${filters.enabled ? " is-active" : ""}"
            aria-label="Filter by metadata tags"
            aria-pressed="${openCanvasToolPopover === "filter"}"
            ${buildTooltipAttributes(
              "Filter",
              "Highlight tensors, indices, or bonds by metadata tags without hiding anything.",
              "Ctrl/Cmd+Shift+F"
            )}
          >
            ${filterButtonIcon()}
          </button>
          ${
            openCanvasToolPopover === "filter"
              ? renderFilterPopover(filters)
              : ""
          }
        </div>
        <div class="canvas-tool">
          <button
            id="canvas-name-search-button"
            type="button"
            class="canvas-tool-button${
              openCanvasToolPopover === "search" ? " is-open" : ""
            }${search.enabled ? " is-active" : ""}"
            aria-label="Search by exact name"
            aria-pressed="${openCanvasToolPopover === "search"}"
            ${buildTooltipAttributes(
              "Search",
              "Highlight tensors, indices, or bonds by exact name without changing the selection.",
              "Ctrl/Cmd+F"
            )}
          >
            ${searchButtonIcon()}
          </button>
          ${
            openCanvasToolPopover === "search"
              ? renderSearchPopover(search)
              : ""
          }
        </div>
      </div>
    `;
  }

  return {
    buildMetadataFiltersMarkup,
    getCheckboxIdForTag,
  };
}
