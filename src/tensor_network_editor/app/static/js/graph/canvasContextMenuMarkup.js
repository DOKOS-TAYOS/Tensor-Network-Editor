export function createCanvasContextMenuMarkup({
  escapeHtml,
  buildMetadataEditorMarkup,
  findTensorById,
  asFiniteNumber,
}) {
  const CONTEXT_MENU_EDGE_MARGIN = 8;
  const CONTEXT_MENU_MAX_WIDTH = 320;

  function getBoundaryRoleDetails(tensor) {
    if (tensor.grid_periodic_role === "up") {
      return { roleKey: "up", roleLabel: "Upper cell" };
    }
    if (tensor.grid_periodic_role === "right") {
      return { roleKey: "right", roleLabel: "Right cell" };
    }
    if (tensor.grid_periodic_role === "down") {
      return { roleKey: "down", roleLabel: "Lower cell" };
    }
    if (tensor.grid_periodic_role === "left") {
      return { roleKey: "left", roleLabel: "Left cell" };
    }
    if (tensor.tree_periodic_role === "parent") {
      return { roleKey: "parent", roleLabel: "Parent cell" };
    }
    if (tensor.tree_periodic_role === "child") {
      return { roleKey: "child", roleLabel: "Child cell" };
    }
    return {
      roleKey: tensor.linear_periodic_role === "previous" ? "previous" : "next",
      roleLabel:
        tensor.linear_periodic_role === "previous" ? "Previous cell" : "Next cell",
    };
  }

  function clampMenuAnchor(offset, extent) {
    if (!Number.isFinite(offset)) {
      return CONTEXT_MENU_EDGE_MARGIN;
    }
    return Math.min(
      Math.max(offset, CONTEXT_MENU_EDGE_MARGIN),
      Math.max(CONTEXT_MENU_EDGE_MARGIN, extent - CONTEXT_MENU_EDGE_MARGIN)
    );
  }

  function resolveRootRect(rootElement) {
    const rawRect =
      rootElement && typeof rootElement.getBoundingClientRect === "function"
        ? rootElement.getBoundingClientRect()
        : null;
    const left = Number.isFinite(rawRect?.left) ? rawRect.left : 0;
    const top = Number.isFinite(rawRect?.top) ? rawRect.top : 0;
    const width =
      Number.isFinite(rawRect?.width) && rawRect.width > 0
        ? rawRect.width
        : Number.isFinite(rawRect?.right) && Number.isFinite(rawRect?.left)
          ? Math.max(rawRect.right - rawRect.left, 0)
          : 0;
    const height =
      Number.isFinite(rawRect?.height) && rawRect.height > 0
        ? rawRect.height
        : Number.isFinite(rawRect?.bottom) && Number.isFinite(rawRect?.top)
          ? Math.max(rawRect.bottom - rawRect.top, 0)
          : 0;

    return {
      left,
      top,
      width,
      height,
      right: Number.isFinite(rawRect?.right) ? rawRect.right : left + width,
      bottom: Number.isFinite(rawRect?.bottom) ? rawRect.bottom : top + height,
    };
  }

  function buildMenuPositionStyle(menuState, rootElement) {
    const rootRect = resolveRootRect(rootElement);
    const anchorX = clampMenuAnchor(
      Number.isFinite(menuState?.clientX) ? menuState.clientX - rootRect.left : 0,
      rootRect.width
    );
    const anchorY = clampMenuAnchor(
      Number.isFinite(menuState?.clientY) ? menuState.clientY - rootRect.top : 0,
      rootRect.height
    );
    const spaceLeft = anchorX;
    const spaceRight = Math.max(rootRect.width - anchorX, CONTEXT_MENU_EDGE_MARGIN);
    const spaceTop = anchorY;
    const spaceBottom = Math.max(rootRect.height - anchorY, CONTEXT_MENU_EDGE_MARGIN);
    const openToRight =
      spaceRight >= CONTEXT_MENU_MAX_WIDTH || spaceRight >= spaceLeft;
    const openDownward = spaceBottom >= spaceTop;
    const maxWidth = Math.max(
      1,
      Math.min(
        CONTEXT_MENU_MAX_WIDTH,
        Math.floor(
          (openToRight ? spaceRight : spaceLeft) - CONTEXT_MENU_EDGE_MARGIN
        )
      )
    );
    const horizontalStyle = openToRight
      ? `left: ${Math.round(anchorX)}px;`
      : `right: ${Math.round(Math.max(rootRect.width - anchorX, 0))}px;`;
    const verticalStyle = openDownward
      ? `top: ${Math.round(anchorY)}px;`
      : `bottom: ${Math.round(Math.max(rootRect.height - anchorY, 0))}px;`;

    return `${horizontalStyle} ${verticalStyle} max-width: ${maxWidth}px;`;
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

  function buildTooltipAttributes(label, description = "", shortcut = "") {
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
    const safeShortcut = String(shortcut || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    return `data-tooltip-enabled="true" data-shortcut-label="${safeLabel}"${
      safeShortcut ? ` data-shortcut="${safeShortcut}"` : ""
    }${
      safeDescription ? ` data-shortcut-description="${safeDescription}"` : ""
    }`;
  }

  function buildInlineMetadataEditor({
    target,
    annotationScope,
    inputPrefix,
    focusKeyPrefix = annotationScope,
  }) {
    if (typeof buildMetadataEditorMarkup !== "function") {
      return "";
    }
    return `
      <div class="canvas-context-menu-section canvas-context-menu-metadata">
        ${buildMetadataEditorMarkup({
          annotationScope,
          collapsible: false,
          customMetadataFocusKey: `${focusKeyPrefix}:${target.id}:custom-metadata`,
          customMetadataInputId: `${inputPrefix}-custom-metadata-input`,
          tagsFocusKey: `${focusKeyPrefix}:${target.id}:tags`,
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
          ${
            Array.isArray(resolvedTarget.editableTensorIds) &&
            resolvedTarget.editableTensorIds.length
              ? `
                <button
                  id="context-menu-add-index-to-selection-button"
                  type="button"
                  class="button-accent-insert"
                  ${buildTooltipAttributes(
                    "Add index",
                    "Add one new open index to each selected tensor.",
                    "I"
                  )}
                >
                  Add index
                </button>
              `
              : ""
          }
          <button
            id="context-menu-extract-selection-button"
            type="button"
            class="button-accent-positive"
            ${buildTooltipAttributes(
              "Extract",
              "Extract the selected tensors as a reusable subnetwork.",
              "Shift+E"
            )}
          >
            Extract
          </button>
          <button
            id="context-menu-save-selection-subnetwork-library-button"
            type="button"
            class="button-accent-template"
            ${buildTooltipAttributes(
              "To Library",
              "Save the selected tensors to the subnetwork library."
            )}
          >
            To Library
          </button>
          <button
            id="context-menu-promote-selection-template-button"
            type="button"
            class="button-accent-template"
            ${buildTooltipAttributes(
              "To Template",
              "Promote the selected tensors to a reusable template."
            )}
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
            class="button-accent-insert"
            ${buildTooltipAttributes(
              "Group",
              "Create a visual group from the selected tensors.",
              "G"
            )}
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
              "Remove the current selection from the network.",
              "Delete"
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
    `;
  }

  function renderTensorMarkup(resolvedTarget) {
    if (resolvedTarget.isStructuralBoundaryTensor) {
      const { roleKey, roleLabel } = getBoundaryRoleDetails(resolvedTarget.target);
      return `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Virtual tensor</span>
              <strong>${escapeHtml(roleLabel)}</strong>
            </div>
            <div class="properties-chip">
              <span>Ports</span>
              <strong>${Array.isArray(resolvedTarget.target.indices) ? resolvedTarget.target.indices.length : 0}</strong>
            </div>
            <div class="properties-chip">
              <span>Role</span>
              <strong>${escapeHtml(roleKey)}</strong>
            </div>
          </div>
          <div class="button-row canvas-context-menu-actions">
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
          </div>
        </div>
        ${buildInlineMetadataEditor({
          annotationScope: "tensor",
          inputPrefix: "context-menu-tensor",
          target: resolvedTarget.target,
        })}
      `;
    }
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
              "Create a new open index on this tensor.",
              "I"
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
              "Remove this tensor from the network.",
              "Delete"
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

  function renderIndexSelectionMarkup(resolvedTarget) {
    const candidate = resolvedTarget.hyperedgeCreationCandidate || {
      canCreate: false,
      message: "This selection cannot form a hyperedge.",
    };
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-row canvas-context-menu-index-fields canvas-context-menu-index-selection-fields">
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Indices</span>
              <strong>${resolvedTarget.indexCount}</strong>
            </div>
          </div>
          <div class="field-group compact-number-field">
            <input
              id="context-menu-selection-dimension-input"
              type="number"
              min="1"
              step="1"
              value="${escapeHtml(resolvedTarget.indexDimensionValue || "")}"
              ${resolvedTarget.hasMixedIndexDimensions ? 'placeholder="Mixed"' : ""}
              aria-label="Selected index dimension"
              ${buildTooltipAttributes(
                "Selected index dimension",
                "Update the dimension of every selected index at once."
              )}
            />
          </div>
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
            id="context-menu-create-hyperedge-button"
            type="button"
            ${candidate.canCreate ? "" : "disabled"}
            ${buildTooltipAttributes(
              "Create hyperedge",
              candidate.canCreate
                ? "Create a hyperedge from the selected open indices."
                : candidate.message || "This selection cannot form a hyperedge.",
              "H"
            )}
          >
            Create hyperedge
          </button>
          <button
            id="context-menu-delete-selection-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete selection"
            ${buildTooltipAttributes(
              "Delete selection",
              "Remove the current selection from the network.",
              "Delete"
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
    `;
  }

  function renderIndexMarkup(resolvedTarget) {
    if (resolvedTarget.isStructuralBoundaryTensor) {
      return `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Port</span>
              <strong>${escapeHtml(resolvedTarget.index.name)}</strong>
            </div>
            <div class="properties-chip">
              <span>Dimension</span>
              <strong>${resolvedTarget.index.dimension}</strong>
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
          </div>
        </div>
        ${buildInlineMetadataEditor({
          annotationScope: "index",
          inputPrefix: "context-menu-index",
          target: resolvedTarget.index,
        })}
      `;
    }
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
                "Remove this index from the tensor.",
                "Delete"
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
              "Remove this connection from the network.",
              "Delete"
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

  function renderHyperedgeMarkup(resolvedTarget) {
    return `
      <div class="canvas-context-menu-section canvas-context-menu-input-stack">
        <div class="field-group">
          <input id="context-menu-name-input" value="${escapeHtml(resolvedTarget.target.name || "")}" />
        </div>
        <div class="button-row canvas-context-menu-actions">
          <label
            class="control-inline-color"
            for="context-menu-hyperedge-color-input"
            ${buildTooltipAttributes(
              "Choose color",
              "Set the display color for this item."
            )}
          >
            <input
              id="context-menu-hyperedge-color-input"
              type="color"
              aria-label="Choose color"
              value="${escapeHtml(resolvedTarget.hyperedgeColor)}"
            />
          </label>
          <button
            id="context-menu-delete-hyperedge-button"
            type="button"
            class="icon-button danger"
            aria-label="Delete hyperedge"
            ${buildTooltipAttributes(
              "Delete hyperedge",
              "Remove this hyperedge from the network.",
              "Delete"
            )}
          >
            ${renderTrashIcon()}
          </button>
        </div>
      </div>
      ${buildInlineMetadataEditor({
        annotationScope: "edge",
        focusKeyPrefix: "hyperedge",
        inputPrefix: "context-menu-hyperedge",
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
          ${
            Array.isArray(resolvedTarget.editableTensorIds) &&
            resolvedTarget.editableTensorIds.length
              ? `
                <button
                  id="context-menu-add-index-to-group-button"
                  type="button"
                  class="button-accent-insert"
                  ${buildTooltipAttributes(
                    "Add index",
                    "Add one new open index to each tensor inside this group.",
                    "I"
                  )}
                >
                  Add index
                </button>
              `
              : ""
          }
          <button
            id="context-menu-extract-group-button"
            type="button"
            class="button-accent-positive"
            ${buildTooltipAttributes(
              "Extract",
              "Extract the tensors inside this group as a reusable subnetwork.",
              "Shift+E"
            )}
          >
            Extract
          </button>
          <button
            id="context-menu-save-group-subnetwork-library-button"
            type="button"
            class="button-accent-template"
            ${buildTooltipAttributes(
              "To Library",
              "Save the tensors inside this group to the subnetwork library."
            )}
          >
            To Library
          </button>
          <button
            id="context-menu-promote-group-template-button"
            type="button"
            class="button-accent-template"
            ${buildTooltipAttributes(
              "To Template",
              "Promote the tensors inside this group to a reusable template."
            )}
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
              "Remove this group from the network.",
              "Delete"
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
    if (resolvedTarget.kind === "index-selection") {
      return renderIndexSelectionMarkup(resolvedTarget);
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
    if (resolvedTarget.kind === "hyperedge") {
      return renderHyperedgeMarkup(resolvedTarget);
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
