export function createCanvasContextMenuMarkup({
  escapeHtml,
  buildMetadataEditorMarkup,
  findTensorById,
  asFiniteNumber,
}) {
  function buildMenuPositionStyle(menuState, rootElement) {
    const rootRect =
      rootElement && typeof rootElement.getBoundingClientRect === "function"
        ? rootElement.getBoundingClientRect()
        : { left: 0, top: 0 };
    const left = Number.isFinite(menuState && menuState.clientX)
      ? menuState.clientX - rootRect.left
      : 0;
    const top = Number.isFinite(menuState && menuState.clientY)
      ? menuState.clientY - rootRect.top
      : 0;
    return `left: ${left}px; top: ${top}px;`;
  }

  function normalizeElementDimension(value) {
    return BigInt(Math.max(1, Math.round(asFiniteNumber(value, 1))));
  }

  function formatTotalElementCount(totalElementCount) {
    return totalElementCount === null ? "" : totalElementCount.toString();
  }

  function getTensorTotalElementCount(tensor) {
    const indices = Array.isArray(tensor && tensor.indices) ? tensor.indices : [];
    return indices.reduce(
      (product, index) => product * normalizeElementDimension(index.dimension),
      1n
    );
  }

  function getTotalElementCountForTensorIds(tensorIds) {
    const uniqueTensorIds = [...new Set(Array.isArray(tensorIds) ? tensorIds : [])];
    let resolvedTensorCount = 0;
    const totalElementCount = uniqueTensorIds.reduce((sum, tensorId) => {
      const tensor =
        typeof findTensorById === "function" ? findTensorById(tensorId) : null;
      if (!tensor) {
        return sum;
      }
      resolvedTensorCount += 1;
      return sum + getTensorTotalElementCount(tensor);
    }, 0n);
    return resolvedTensorCount ? totalElementCount : null;
  }

  function getIndexCountForTensorIds(tensorIds) {
    return [...new Set(Array.isArray(tensorIds) ? tensorIds : [])].reduce(
      (sum, tensorId) => {
        const tensor =
          typeof findTensorById === "function" ? findTensorById(tensorId) : null;
        return sum + (Array.isArray(tensor && tensor.indices) ? tensor.indices.length : 0);
      },
      0
    );
  }

  function renderTrashIcon() {
    return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
      </svg>
    `;
  }

  function buildTooltipAttributes(label, description = "") {
    const safeLabel = String(label || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    const safeDescription = String(description || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    return `data-tooltip-enabled="true" data-shortcut-label="${safeLabel}"${
      safeDescription ? ` data-shortcut-description="${safeDescription}"` : ""
    }`;
  }

  function buildInlineMetadataEditor({ target, annotationScope, inputPrefix }) {
    if (typeof buildMetadataEditorMarkup !== "function") {
      return "";
    }
    return `
      <div class="canvas-context-menu-section canvas-context-menu-metadata">
        ${buildMetadataEditorMarkup({
          annotationScope,
          collapsible: false,
          customMetadataFocusKey: `${annotationScope}:${target.id}:custom-metadata`,
          customMetadataInputId: `${inputPrefix}-custom-metadata-input`,
          tagsFocusKey: `${annotationScope}:${target.id}:tags`,
          tagsInputId: `${inputPrefix}-tags-input`,
          target,
        })}
      </div>
    `;
  }

  function renderSelectionMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="properties-chip-wrap canvas-context-menu-stats">
          <div class="properties-chip">
            <span>Tensors</span>
            <strong>${resolvedTarget.tensorCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Indices</span>
            <strong>${resolvedTarget.indexCount}</strong>
          </div>
          ${
            resolvedTarget.totalElementCount !== null
              ? `
                <div class="properties-chip">
                  <span>Total elements</span>
                  <strong>${formatTotalElementCount(resolvedTarget.totalElementCount)}</strong>
                </div>
              `
              : ""
          }
        </div>
        <div class="button-row canvas-context-menu-actions">
          <button
            id="context-menu-add-index-to-selection-button"
            type="button"
            ${buildTooltipAttributes(
              "Add index",
              "Add one new open index to each selected tensor."
            )}
          >
            Add index
          </button>
          <button
            id="context-menu-extract-selection-button"
            type="button"
          >
            Extract
          </button>
          <button
            id="context-menu-promote-selection-template-button"
            type="button"
          >
            To Template
          </button>
        </div>
        <div class="button-row canvas-context-menu-actions">
          <label
            class="control-inline-color"
            for="context-menu-selection-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-selection-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.selectionColor)}"
            />
          </label>
          <button
            id="context-menu-group-selection-button"
            type="button"
          >
            Group
          </button>
          <button
            id="context-menu-delete-selection-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete selection"
            ${buildTooltipAttributes(
              "Delete selection",
              "Remove the current selection from the network."
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
    `;
  }

  function renderTensorMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-group">
          <input id="context-menu-name-input" value="${escapeHtml(resolvedTarget.target.name)}" />
        </div>
        <div class="properties-chip-wrap canvas-context-menu-stats">
          <div class="properties-chip">
            <span>Indices</span>
            <strong>${Array.isArray(resolvedTarget.target.indices) ? resolvedTarget.target.indices.length : 0}</strong>
          </div>
          <div class="properties-chip">
            <span>Total elements</span>
            <strong>${formatTotalElementCount(resolvedTarget.totalElementCount)}</strong>
          </div>
        </div>
        <div class="button-row canvas-context-menu-actions">
          <button
            id="context-menu-add-index-button"
            type="button"
            class="icon-button button-accent-insert"
            aria-label="Add index"
            ${buildTooltipAttributes(
              "Add index",
              "Create a new open index on this tensor."
            )}
          >
            +
          </button>
          <label
            class="control-inline-color"
            for="context-menu-tensor-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-tensor-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.tensorColor)}"
            />
          </label>
          <button
            id="context-menu-delete-tensor-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete tensor"
            ${buildTooltipAttributes(
              "Delete tensor",
              "Remove this tensor from the network."
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
      ${buildInlineMetadataEditor({
        annotationScope: "tensor",
        inputPrefix: "context-menu-tensor",
        target: resolvedTarget.target,
      })}
    `;
  }

  function renderIndexMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-row canvas-context-menu-index-fields">
          <div class="field-group">
            <input id="context-menu-name-input" value="${escapeHtml(resolvedTarget.index.name)}" />
          </div>
          <div class="field-group compact-number-field">
            <input
              id="context-menu-dimension-input"
              type="number"
              min="1"
              step="1"
              value="${resolvedTarget.index.dimension}"
              aria-label="Index dimension"
              ${buildTooltipAttributes(
                "Index dimension",
                "Set the size of this index. Connected indices should share the same dimension."
              )}
            />
          </div>
        </div>
        <div class="button-row canvas-context-menu-actions">
          <label
            class="control-inline-color"
            for="context-menu-index-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-index-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.indexColor)}"
            />
          </label>
          <button
            id="context-menu-move-up-button"
            type="button"
            class="icon-button index-action-button"
            aria-label="Move index up"
            ${buildTooltipAttributes(
              "Move index up",
              "Move this index one position earlier in the tensor index order."
            )}
            ${resolvedTarget.indexPosition === 0 ? "disabled" : ""}
          >
            <span aria-hidden="true">&#8593;</span>
          </button>
          <button
            id="context-menu-move-down-button"
            type="button"
            class="icon-button index-action-button"
            aria-label="Move index down"
            ${buildTooltipAttributes(
              "Move index down",
              "Move this index one position later in the tensor index order."
            )}
            ${
              resolvedTarget.indexPosition === resolvedTarget.indices.length - 1
                ? "disabled"
                : ""
            }
          >
            <span aria-hidden="true">&#8595;</span>
          </button>
          <button
            id="context-menu-delete-index-button"
            type="button"
            class="icon-button index-action-button danger"
            aria-label="Delete index"
            ${buildTooltipAttributes(
              "Delete index",
              "Remove this index from the tensor."
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
      ${buildInlineMetadataEditor({
        annotationScope: "index",
        inputPrefix: "context-menu-index",
        target: resolvedTarget.index,
      })}
    `;
  }

  function renderEdgeMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-group">
          <input id="context-menu-name-input" value="${escapeHtml(resolvedTarget.target.name || "")}" />
        </div>
        <div class="button-row canvas-context-menu-actions">
          <label
            class="control-inline-color"
            for="context-menu-edge-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-edge-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.edgeColor)}"
            />
          </label>
          <button
            id="context-menu-delete-edge-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete connection"
            ${buildTooltipAttributes(
              "Delete connection",
              "Remove this connection from the network."
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
      ${buildInlineMetadataEditor({
        annotationScope: "edge",
        inputPrefix: "context-menu-edge",
        target: resolvedTarget.target,
      })}
    `;
  }

  function renderGroupMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-group">
          <input id="context-menu-name-input" value="${escapeHtml(resolvedTarget.target.name)}" />
        </div>
        <div class="properties-chip-wrap canvas-context-menu-stats">
          <div class="properties-chip">
            <span>Member tensors</span>
            <strong>${resolvedTarget.memberTensorCount}</strong>
          </div>
          ${
            resolvedTarget.totalElementCount !== null
              ? `
                <div class="properties-chip">
                  <span>Total elements</span>
                  <strong>${formatTotalElementCount(resolvedTarget.totalElementCount)}</strong>
                </div>
              `
              : ""
          }
        </div>
        <div class="button-row canvas-context-menu-actions">
          <label
            class="control-inline-color"
            for="context-menu-group-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-group-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.groupColor)}"
            />
          </label>
          <button
            id="context-menu-add-index-to-group-button"
            type="button"
            ${buildTooltipAttributes(
              "Add index",
              "Add one new open index to each tensor inside this group."
            )}
          >
            Add index
          </button>
          <button
            id="context-menu-extract-group-button"
            type="button"
          >
            Extract
          </button>
          <button
            id="context-menu-promote-group-template-button"
            type="button"
          >
            To Template
          </button>
          <button id="context-menu-toggle-group-button" type="button">
            ${resolvedTarget.isCollapsed ? "Expand" : "Collapse"}
          </button>
          <button
            id="context-menu-delete-group-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete group"
            ${buildTooltipAttributes(
              "Delete group",
              "Remove this group from the network."
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
      ${buildInlineMetadataEditor({
        annotationScope: "group",
        inputPrefix: "context-menu-group",
        target: resolvedTarget.target,
      })}
    `;
  }

  function renderResolvedContextMenu(resolvedTarget) {
    if (!resolvedTarget) {
      return "";
    }
    if (resolvedTarget.kind === "selection") {
      return renderSelectionMarkup(resolvedTarget);
    }
    if (resolvedTarget.kind === "tensor") {
      return renderTensorMarkup(resolvedTarget);
    }
    if (resolvedTarget.kind === "index") {
      return renderIndexMarkup(resolvedTarget);
    }
    if (resolvedTarget.kind === "edge") {
      return renderEdgeMarkup(resolvedTarget);
    }
    if (resolvedTarget.kind === "group") {
      return renderGroupMarkup(resolvedTarget);
    }
    return "";
  }

  return {
    buildMenuPositionStyle,
    getIndexCountForTensorIds,
    getTensorTotalElementCount,
    getTotalElementCountForTensorIds,
    renderResolvedContextMenu,
  };
}
