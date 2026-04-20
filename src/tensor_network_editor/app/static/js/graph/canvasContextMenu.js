import { GRAPH_THEME } from "../core/theme.js";

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

function asFiniteNumber(ctx, value, fallback = 1) {
  if (typeof ctx.asFiniteNumber === "function") {
    return ctx.asFiniteNumber(value, fallback);
  }
  const candidate = Number(value);
  return Number.isFinite(candidate) ? candidate : fallback;
}

function normalizeElementDimension(ctx, value) {
  return BigInt(Math.max(1, Math.round(asFiniteNumber(ctx, value, 1))));
}

function formatTotalElementCount(totalElementCount) {
  return totalElementCount === null ? "" : totalElementCount.toString();
}

function getTensorTotalElementCount(ctx, tensor) {
  const indices = Array.isArray(tensor && tensor.indices) ? tensor.indices : [];
  return indices.reduce(
    (product, index) =>
      product * normalizeElementDimension(ctx, index.dimension),
    1n
  );
}

function getTotalElementCountForTensorIds(ctx, tensorIds) {
  const uniqueTensorIds = [...new Set(Array.isArray(tensorIds) ? tensorIds : [])];
  let resolvedTensorCount = 0;
  const totalElementCount = uniqueTensorIds.reduce((sum, tensorId) => {
    const tensor =
      typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null;
    if (!tensor) {
      return sum;
    }
    resolvedTensorCount += 1;
    return sum + getTensorTotalElementCount(ctx, tensor);
  }, 0n);
  return resolvedTensorCount ? totalElementCount : null;
}

function getIndexCountForTensorIds(ctx, tensorIds) {
  return [...new Set(Array.isArray(tensorIds) ? tensorIds : [])].reduce(
    (sum, tensorId) => {
      const tensor =
        typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null;
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

function buildInlineMetadataEditor(
  ctx,
  { target, annotationScope, inputPrefix }
) {
  if (typeof ctx.buildMetadataEditorMarkup !== "function") {
    return "";
  }
  return `
    <div class="canvas-context-menu-section canvas-context-menu-metadata">
      ${ctx.buildMetadataEditorMarkup({
        tagsInputId: `${inputPrefix}-tags-input`,
        tagsFocusKey: `${annotationScope}:${target.id}:tags`,
        customMetadataInputId: `${inputPrefix}-custom-metadata-input`,
        customMetadataFocusKey: `${annotationScope}:${target.id}:custom-metadata`,
        target,
        annotationScope,
        collapsible: false,
      })}
    </div>
  `;
}

export function registerCanvasContextMenu(ctx) {
  const state = ctx.state;
  const { document, window } = ctx;
  const { canvasContextMenuRoot } = ctx.dom;

  function closeCanvasContextMenu() {
    state.canvasContextMenu = null;
    if (canvasContextMenuRoot) {
      canvasContextMenuRoot.innerHTML = "";
    }
  }

  function bindCommitOnBlurAndEnter(element, commit) {
    if (!element) {
      return;
    }
    element.addEventListener("blur", commit);
    element.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      event.preventDefault();
      commit();
      closeCanvasContextMenu();
    });
  }

  function bindCloseOnEnter(element) {
    if (!element) {
      return;
    }
    element.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      closeCanvasContextMenu();
    });
  }

  function bindColorInput(element, { target, statusMessage }) {
    if (!element) {
      return;
    }
    element.addEventListener("input", () => {
      ctx.propertyCommands.updateTargetColor({
        target,
        nextColor: element.value,
        invalidate: ctx.propertyInvalidation({ graph: true, minimap: true }),
        statusMessage,
      });
    });
  }

  function bindSelectionColorInput(element, { statusMessage }) {
    if (!element) {
      return;
    }
    element.addEventListener("input", () => {
      if (!ctx.propertyCommands || typeof ctx.propertyCommands.applySelectionColor !== "function") {
        return;
      }
      ctx.propertyCommands.applySelectionColor({
        nextColor: element.value,
        invalidate: ctx.propertyInvalidation({
          graph: true,
          overlays: true,
          minimap: true,
        }),
        statusMessage,
      });
    });
  }

  function bindInlineMetadataEditor({
    target,
    annotationScope,
    inputPrefix,
    statusMessage,
    invalidate,
  }) {
    if (typeof ctx.bindMetadataEditors !== "function") {
      return;
    }
    const tagsInput = document.getElementById(`${inputPrefix}-tags-input`);
    const customMetadataInput = document.getElementById(
      `${inputPrefix}-custom-metadata-input`
    );
    ctx.bindMetadataEditors({
      target,
      tagsInput,
      tagsFieldKey: `${annotationScope}:${target.id}:tags`,
      customMetadataInput,
      customMetadataFieldKey: `${annotationScope}:${target.id}:custom-metadata`,
      statusMessage,
      invalidate,
      annotationScope,
    });
    bindCloseOnEnter(tagsInput);
  }

  function getSelectedTensorIdsForContext() {
    return typeof ctx.getSelectedIdsByKind === "function"
      ? ctx.getSelectedIdsByKind("tensor")
      : [];
  }

  function isMultiTensorSelectionContext(tensorId) {
    const selectedTensorIds = getSelectedTensorIdsForContext();
    return (
      Array.isArray(state.selectionIds) &&
      selectedTensorIds.length >= 2 &&
      selectedTensorIds.includes(tensorId)
    );
  }

  function getSelectionContextTarget(anchorTensorId) {
    const selectedTensorIds = getSelectedTensorIdsForContext();
    if (selectedTensorIds.length < 2 || !selectedTensorIds.includes(anchorTensorId)) {
      return null;
    }
    const tensorCount = selectedTensorIds.length;
    const indexCount = getIndexCountForTensorIds(ctx, selectedTensorIds);
    const totalElementCount = getTotalElementCountForTensorIds(ctx, selectedTensorIds);
    const selectedEntries =
      typeof ctx.getSelectedEntries === "function" ? ctx.getSelectedEntries() : [];
    const selectionColor =
      typeof ctx.getBatchColorValue === "function"
        ? ctx.getBatchColorValue(selectedEntries) || "#456cbf"
        : "#456cbf";
    return {
      kind: "selection",
      id: anchorTensorId,
      target: null,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Tensors</span>
              <strong>${tensorCount}</strong>
            </div>
            <div class="properties-chip">
              <span>Indices</span>
              <strong>${indexCount}</strong>
            </div>
            ${
              totalElementCount !== null
                ? `
                  <div class="properties-chip">
                    <span>Total elements</span>
                    <strong>${formatTotalElementCount(totalElementCount)}</strong>
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
                value="${ctx.escapeHtml(selectionColor)}"
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
      `,
      bind() {
        const addIndexButton = document.getElementById(
          "context-menu-add-index-to-selection-button"
        );
        const extractButton = document.getElementById(
          "context-menu-extract-selection-button"
        );
        const promoteButton = document.getElementById(
          "context-menu-promote-selection-template-button"
        );
        const colorInput = document.getElementById(
          "context-menu-selection-color-input"
        );
        const groupButton = document.getElementById(
          "context-menu-group-selection-button"
        );
        const deleteButton = document.getElementById(
          "context-menu-delete-selection-button"
        );

        bindSelectionColorInput(colorInput, {
          statusMessage: "Updated the selection color.",
        });

        if (addIndexButton) {
          addIndexButton.addEventListener("click", () => {
            ctx.propertyCommands.addIndexToSelectedTensors({
              selectionIds: [...state.selectionIds],
              primaryId: state.primarySelectionId,
              statusMessage: "Added one index to each selected tensor.",
            });
            closeCanvasContextMenu();
          });
        }

        if (extractButton) {
          extractButton.addEventListener("click", () => {
            if (typeof ctx.exportSelectedSubnetwork === "function") {
              ctx.exportSelectedSubnetwork();
            }
            closeCanvasContextMenu();
          });
        }

        if (promoteButton) {
          promoteButton.addEventListener("click", () => {
            if (typeof ctx.promoteSelectedSubnetworkToTemplate === "function") {
              ctx.promoteSelectedSubnetworkToTemplate();
            }
            closeCanvasContextMenu();
          });
        }

        if (groupButton) {
          groupButton.addEventListener("click", () => {
            if (typeof ctx.createGroupFromSelection === "function") {
              ctx.createGroupFromSelection();
            }
            closeCanvasContextMenu();
          });
        }

        if (deleteButton) {
          deleteButton.addEventListener("click", () => {
            if (
              ctx.propertyCommands &&
              typeof ctx.propertyCommands.deleteCurrentSelection === "function"
            ) {
              ctx.propertyCommands.deleteCurrentSelection();
            }
            closeCanvasContextMenu();
          });
        }
      },
    };
  }

  function getTensorContextTarget(tensorId) {
    const tensor =
      typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null;
    if (!tensor) {
      return null;
    }
    const totalElementCount = getTensorTotalElementCount(ctx, tensor);
    const tensorColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(tensor.metadata, GRAPH_THEME.tensorFallback)
        : GRAPH_THEME.tensorFallback;
    return {
      kind: "tensor",
      id: tensor.id,
      target: tensor,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="field-group">
            <input id="context-menu-name-input" value="${ctx.escapeHtml(tensor.name)}" />
          </div>
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Indices</span>
              <strong>${Array.isArray(tensor.indices) ? tensor.indices.length : 0}</strong>
            </div>
            <div class="properties-chip">
              <span>Total elements</span>
              <strong>${formatTotalElementCount(totalElementCount)}</strong>
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
                value="${ctx.escapeHtml(tensorColor)}"
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
        ${buildInlineMetadataEditor(ctx, {
          target: tensor,
          annotationScope: "tensor",
          inputPrefix: "context-menu-tensor",
        })}
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const addIndexButton = document.getElementById("context-menu-add-index-button");
        const colorInput = document.getElementById("context-menu-tensor-color-input");
        const deleteTensorButton = document.getElementById(
          "context-menu-delete-tensor-button"
        );

        bindCommitOnBlurAndEnter(nameInput, () => {
          ctx.propertyCommands.renameTensor({
            tensor,
            proposedName: nameInput.value,
            invalidate: ctx.propertyInvalidation({ graph: true }),
            statusMessage: `Updated tensor ${nameInput.value.trim()}.`,
          });
        });

        if (addIndexButton) {
          addIndexButton.addEventListener("click", () => {
            ctx.propertyCommands.addTensorIndex({
              tensor,
              selectionIds: [tensor.id],
              primaryId: tensor.id,
              statusMessage: `Added one index to ${tensor.name}.`,
            });
            renderCanvasContextMenu();
          });
        }

        bindColorInput(colorInput, {
          target: tensor,
          statusMessage: `Updated tensor ${tensor.name}.`,
        });

        if (deleteTensorButton) {
          deleteTensorButton.addEventListener("click", () => {
            ctx.propertyCommands.deleteTensor({
              tensorId: tensor.id,
              selectionIds: [],
              statusMessage: `Deleted tensor ${tensor.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        bindInlineMetadataEditor({
          target: tensor,
          annotationScope: "tensor",
          inputPrefix: "context-menu-tensor",
          statusMessage: `Updated tensor ${tensor.name}.`,
          invalidate: ctx.propertyInvalidation(),
        });
      },
    };
  }

  function getIndexContextTarget(indexId) {
    const located =
      typeof ctx.findIndexOwner === "function" ? ctx.findIndexOwner(indexId) : null;
    if (!located || !located.tensor || !located.index) {
      return null;
    }
    const { tensor, index } = located;
    const indices = Array.isArray(tensor.indices) ? tensor.indices : [];
    const indexPosition = indices.findIndex((candidate) => candidate.id === index.id);
    const indexColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(
            index.metadata,
            typeof ctx.getIndexColor === "function"
              ? ctx.getIndexColor(
                  index,
                  Boolean(
                    typeof ctx.findEdgeByIndexId === "function" &&
                      ctx.findEdgeByIndexId(index.id)
                  )
                )
              : "#456cbf"
          )
        : "#456cbf";

    return {
      kind: "index",
      id: index.id,
      target: index,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="field-row canvas-context-menu-index-fields">
            <div class="field-group">
              <input id="context-menu-name-input" value="${ctx.escapeHtml(index.name)}" />
            </div>
            <div class="field-group compact-number-field">
              <input
                id="context-menu-dimension-input"
                type="number"
                min="1"
                step="1"
                value="${index.dimension}"
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
                value="${ctx.escapeHtml(indexColor)}"
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
              ${indexPosition === 0 ? "disabled" : ""}
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
              ${indexPosition === indices.length - 1 ? "disabled" : ""}
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
        ${buildInlineMetadataEditor(ctx, {
          target: index,
          annotationScope: "index",
          inputPrefix: "context-menu-index",
        })}
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const dimensionInput = document.getElementById("context-menu-dimension-input");
        const colorInput = document.getElementById("context-menu-index-color-input");
        const moveUpButton = document.getElementById("context-menu-move-up-button");
        const moveDownButton = document.getElementById("context-menu-move-down-button");
        const deleteIndexButton = document.getElementById(
          "context-menu-delete-index-button"
        );

        bindCommitOnBlurAndEnter(nameInput, () => {
          ctx.propertyCommands.renameIndex({
            tensor,
            index,
            proposedName: nameInput.value,
            invalidate: ctx.propertyInvalidation({ graph: true }),
            statusMessage: `Updated index ${nameInput.value.trim()}.`,
          });
        });

        bindCommitOnBlurAndEnter(dimensionInput, () => {
          ctx.propertyCommands.updateIndexDimension({
            indexId: index.id,
            rawValue: dimensionInput.value,
            invalidate: ctx.propertyInvalidation({
              analysis: true,
              graph: true,
            }),
            statusMessage: `Updated index ${index.name}.`,
          });
        });

        bindColorInput(colorInput, {
          target: index,
          statusMessage: `Updated index ${index.name}.`,
        });

        if (moveUpButton) {
          moveUpButton.addEventListener("click", () => {
            ctx.propertyCommands.moveTensorIndex({
              tensorId: tensor.id,
              indexPosition,
              direction: -1,
              invalidate: ctx.propertyInvalidation({
                graph: true,
                lookups: true,
                properties: true,
              }),
              selectionIds: [index.id],
              primaryId: index.id,
              statusMessage: `Moved index ${index.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        if (moveDownButton) {
          moveDownButton.addEventListener("click", () => {
            ctx.propertyCommands.moveTensorIndex({
              tensorId: tensor.id,
              indexPosition,
              direction: 1,
              invalidate: ctx.propertyInvalidation({
                graph: true,
                lookups: true,
                properties: true,
              }),
              selectionIds: [index.id],
              primaryId: index.id,
              statusMessage: `Moved index ${index.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        if (deleteIndexButton) {
          deleteIndexButton.addEventListener("click", () => {
            ctx.propertyCommands.deleteTensorIndex({
              tensorId: tensor.id,
              indexId: index.id,
              primaryId: tensor.id,
              selectionIds: [tensor.id],
              statusMessage: `Deleted index ${index.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        bindInlineMetadataEditor({
          target: index,
          annotationScope: "index",
          inputPrefix: "context-menu-index",
          statusMessage: `Updated index ${index.name}.`,
          invalidate: ctx.propertyInvalidation(),
        });
      },
    };
  }

  function getEdgeContextTarget(edgeId) {
    const edge =
      typeof ctx.findEdgeById === "function" ? ctx.findEdgeById(edgeId) : null;
    if (!edge) {
      return null;
    }
    const edgeColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(edge.metadata, GRAPH_THEME.edge)
        : GRAPH_THEME.edge;
    return {
      kind: "edge",
      id: edge.id,
      target: edge,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="field-group">
            <input id="context-menu-name-input" value="${ctx.escapeHtml(edge.name || "")}" />
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
                value="${ctx.escapeHtml(edgeColor)}"
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
        ${buildInlineMetadataEditor(ctx, {
          target: edge,
          annotationScope: "edge",
          inputPrefix: "context-menu-edge",
        })}
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const colorInput = document.getElementById("context-menu-edge-color-input");
        const deleteEdgeButton = document.getElementById(
          "context-menu-delete-edge-button"
        );

        bindCommitOnBlurAndEnter(nameInput, () => {
          ctx.propertyCommands.renameEdge({
            edge,
            proposedName: nameInput.value,
            invalidate: ctx.propertyInvalidation({ graph: true }),
            statusMessage: `Updated connection ${nameInput.value.trim()}.`,
          });
        });

        bindColorInput(colorInput, {
          target: edge,
          statusMessage: `Updated connection ${edge.name}.`,
        });

        if (deleteEdgeButton) {
          deleteEdgeButton.addEventListener("click", () => {
            ctx.propertyCommands.deleteEdge({
              edgeId: edge.id,
              selectionIds: [],
              statusMessage: `Deleted connection ${edge.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        bindInlineMetadataEditor({
          target: edge,
          annotationScope: "edge",
          inputPrefix: "context-menu-edge",
          statusMessage: `Updated connection ${edge.name}.`,
          invalidate: ctx.propertyInvalidation({ graph: false, minimap: false }),
        });
      },
    };
  }

  function getGroupContextTarget(groupId) {
    const group =
      typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
    if (!group) {
      return null;
    }
    const isCollapsed = Boolean(group.metadata && group.metadata.collapsed);
    const groupColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(group.metadata, GRAPH_THEME.groupDefault)
        : GRAPH_THEME.groupDefault;
    const memberTensorCount = Array.isArray(group.tensor_ids)
      ? group.tensor_ids.length
      : 0;
    const totalElementCount = getTotalElementCountForTensorIds(
      ctx,
      Array.isArray(group.tensor_ids) ? group.tensor_ids : []
    );
    return {
      kind: "group",
      id: group.id,
      target: group,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="field-group">
            <input id="context-menu-name-input" value="${ctx.escapeHtml(group.name)}" />
          </div>
          <div class="properties-chip-wrap canvas-context-menu-stats">
            <div class="properties-chip">
              <span>Member tensors</span>
              <strong>${memberTensorCount}</strong>
            </div>
            ${
              totalElementCount !== null
                ? `
                  <div class="properties-chip">
                    <span>Total elements</span>
                    <strong>${formatTotalElementCount(totalElementCount)}</strong>
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
                value="${ctx.escapeHtml(groupColor)}"
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
              ${isCollapsed ? "Expand" : "Collapse"}
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
        ${buildInlineMetadataEditor(ctx, {
          target: group,
          annotationScope: "group",
          inputPrefix: "context-menu-group",
        })}
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const colorInput = document.getElementById("context-menu-group-color-input");
        const addIndexToGroupButton = document.getElementById(
          "context-menu-add-index-to-group-button"
        );
        const extractGroupButton = document.getElementById(
          "context-menu-extract-group-button"
        );
        const promoteGroupTemplateButton = document.getElementById(
          "context-menu-promote-group-template-button"
        );
        const toggleGroupButton = document.getElementById(
          "context-menu-toggle-group-button"
        );
        const deleteGroupButton = document.getElementById(
          "context-menu-delete-group-button"
        );

        bindCommitOnBlurAndEnter(nameInput, () => {
          ctx.propertyCommands.renameGroup({
            group,
            proposedName: nameInput.value,
            invalidate: ctx.propertyInvalidation({ overlays: true }),
            statusMessage: `Updated group ${nameInput.value.trim()}.`,
          });
        });

        bindColorInput(colorInput, {
          target: group,
          statusMessage: `Updated group ${group.name}.`,
        });

        if (addIndexToGroupButton) {
          addIndexToGroupButton.addEventListener("click", () => {
            ctx.propertyCommands.addIndexToSelectedTensors({
              tensorIds: [...group.tensor_ids],
              selectionIds: [group.id],
              primaryId: group.id,
              statusMessage: "Added one index to each group tensor.",
            });
            closeCanvasContextMenu();
          });
        }

        if (extractGroupButton) {
          extractGroupButton.addEventListener("click", () => {
            if (typeof ctx.exportGroupSubnetwork === "function") {
              ctx.exportGroupSubnetwork(group.id);
            }
            closeCanvasContextMenu();
          });
        }

        if (promoteGroupTemplateButton) {
          promoteGroupTemplateButton.addEventListener("click", () => {
            if (typeof ctx.promoteGroupToTemplate === "function") {
              ctx.promoteGroupToTemplate(group.id);
            }
            closeCanvasContextMenu();
          });
        }

        if (toggleGroupButton) {
          toggleGroupButton.addEventListener("click", () => {
            if (typeof ctx.toggleGroupCollapse === "function") {
              ctx.toggleGroupCollapse(group.id);
            }
            closeCanvasContextMenu();
          });
        }

        if (deleteGroupButton) {
          deleteGroupButton.addEventListener("click", () => {
            ctx.propertyCommands.deleteGroup({
              groupId: group.id,
              invalidate: ctx.propertyInvalidation({
                lookups: true,
                overlays: true,
              }),
              selectionIds: [],
              statusMessage: `Deleted group ${group.name}.`,
            });
            closeCanvasContextMenu();
          });
        }

        bindInlineMetadataEditor({
          target: group,
          annotationScope: "group",
          inputPrefix: "context-menu-group",
          statusMessage: `Updated group ${group.name}.`,
          invalidate: ctx.propertyInvalidation({ overlays: false }),
        });
      },
    };
  }

  function resolveContextTarget(menuState) {
    if (!menuState || !menuState.kind || !menuState.id) {
      return null;
    }
    if (menuState.kind === "tensor") {
      if (isMultiTensorSelectionContext(menuState.id)) {
        return getSelectionContextTarget(menuState.id);
      }
      return getTensorContextTarget(menuState.id);
    }
    if (menuState.kind === "selection") {
      return getSelectionContextTarget(menuState.id);
    }
    if (menuState.kind === "index") {
      return getIndexContextTarget(menuState.id);
    }
    if (menuState.kind === "edge") {
      return getEdgeContextTarget(menuState.id);
    }
    if (menuState.kind === "group") {
      return getGroupContextTarget(menuState.id);
    }
    return null;
  }

  function renderCanvasContextMenu() {
    if (!canvasContextMenuRoot) {
      return;
    }
    const menuState = state.canvasContextMenu;
    const resolvedTarget = resolveContextTarget(menuState);
    if (!resolvedTarget) {
      canvasContextMenuRoot.innerHTML = "";
      return;
    }

    canvasContextMenuRoot.innerHTML = `
      <div class="canvas-context-menu-scrim"></div>
      <div class="canvas-context-menu" style="${buildMenuPositionStyle(
        menuState,
        canvasContextMenuRoot
      )}">
        ${resolvedTarget.markup}
      </div>
    `;

    resolvedTarget.bind();
  }

  function openCanvasContextMenu(menuState) {
    const resolvedTarget = resolveContextTarget(menuState);
    if (!resolvedTarget) {
      closeCanvasContextMenu();
      return;
    }
    if (
      resolvedTarget.kind !== "selection" &&
      typeof ctx.setSelection === "function"
    ) {
      ctx.setSelection([resolvedTarget.id], {
        primaryId: resolvedTarget.id,
      });
    }
    state.canvasContextMenu = {
      kind: resolvedTarget.kind,
      id: resolvedTarget.id,
      clientX: Number(menuState.clientX || 0),
      clientY: Number(menuState.clientY || 0),
    };
    renderCanvasContextMenu();
  }

  if (document && typeof document.addEventListener === "function") {
    document.addEventListener("click", (event) => {
      if (!state.canvasContextMenu) {
        return;
      }
      if (
        event &&
        event.target &&
        typeof event.target.closest === "function" &&
        event.target.closest(".canvas-context-menu")
      ) {
        return;
      }
      closeCanvasContextMenu();
    });
  }
  if (window && typeof window.addEventListener === "function") {
    window.addEventListener("resize", () => {
      if (state.canvasContextMenu) {
        closeCanvasContextMenu();
      }
    });
  }

  Object.assign(ctx, {
    openCanvasContextMenu,
    closeCanvasContextMenu,
    renderCanvasContextMenu,
  });
}
