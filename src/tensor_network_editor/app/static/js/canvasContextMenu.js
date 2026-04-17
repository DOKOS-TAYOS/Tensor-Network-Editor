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

function renderTrashIcon() {
  return `
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
    </svg>
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

  function getTensorContextTarget(tensorId) {
    const tensor = typeof ctx.findTensorById === "function" ? ctx.findTensorById(tensorId) : null;
    if (!tensor) {
      return null;
    }
    const totalElementCount = getTensorTotalElementCount(ctx, tensor);
    const tensorColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(tensor.metadata, "#18212c")
        : "#18212c";
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
              title="Add index"
            >
              +
            </button>
            <label class="control-inline-color" for="context-menu-tensor-color-input">
              <input
                id="context-menu-tensor-color-input"
                type="color"
                aria-label="Choose tensor tint"
                title="Choose tensor tint"
                value="${ctx.escapeHtml(tensorColor)}"
              />
            </label>
            <button
              id="context-menu-delete-tensor-button"
              type="button"
              class="icon-button danger"
              aria-label="Delete tensor"
              title="Delete tensor"
            >
              ${renderTrashIcon()}
            </button>
          </div>
        </div>
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const addIndexButton = document.getElementById("context-menu-add-index-button");
        const colorInput = document.getElementById("context-menu-tensor-color-input");
        const deleteTensorButton = document.getElementById(
          "context-menu-delete-tensor-button"
        );
        if (nameInput) {
          const commitRename = () => {
            ctx.propertyCommands.renameTensor({
              tensor,
              proposedName: nameInput.value,
              invalidate: ctx.propertyInvalidation({ graph: true }),
              statusMessage: `Updated tensor ${nameInput.value.trim()}.`,
            });
          };
          nameInput.addEventListener("blur", commitRename);
          nameInput.addEventListener("keydown", (event) => {
            if (event.key !== "Enter") {
              return;
            }
            event.preventDefault();
            commitRename();
          });
        }
        if (addIndexButton) {
          addIndexButton.addEventListener("click", () => {
            ctx.propertyCommands.addTensorIndex({
              tensor,
              selectionIds: [tensor.id],
              primaryId: tensor.id,
              statusMessage: `Added one index to ${tensor.name}.`,
            });
            closeCanvasContextMenu();
          });
        }
        if (colorInput) {
          colorInput.addEventListener("input", () => {
            ctx.propertyCommands.updateTargetColor({
              target: tensor,
              nextColor: colorInput.value,
              invalidate: ctx.propertyInvalidation({ graph: true, minimap: true }),
              statusMessage: `Updated tensor ${tensor.name}.`,
            });
          });
        }
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
      },
    };
  }

  function getIndexContextTarget(indexId) {
    const located = typeof ctx.findIndexOwner === "function" ? ctx.findIndexOwner(indexId) : null;
    if (!located || !located.tensor || !located.index) {
      return null;
    }
    const { tensor, index } = located;
    const indexPosition = Array.isArray(tensor.indices)
      ? tensor.indices.findIndex((candidate) => candidate.id === index.id)
      : -1;
    const indexColor =
      typeof ctx.getMetadataColor === "function"
        ? ctx.getMetadataColor(
            index.metadata,
            typeof ctx.getIndexColor === "function"
              ? ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId && ctx.findEdgeByIndexId(index.id)))
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
              />
            </div>
          </div>
          <div class="button-row canvas-context-menu-actions">
            <label class="control-inline-color" for="context-menu-index-color-input">
              <input
                id="context-menu-index-color-input"
                type="color"
                aria-label="Choose index tint"
                title="Choose index tint"
                value="${ctx.escapeHtml(indexColor)}"
              />
            </label>
            <button
              id="context-menu-move-up-button"
              type="button"
              class="icon-button index-action-button"
              aria-label="Move index up"
              title="Move index up"
              ${indexPosition === 0 ? "disabled" : ""}
            >
              <span aria-hidden="true">&#8593;</span>
            </button>
            <button
              id="context-menu-move-down-button"
              type="button"
              class="icon-button index-action-button"
              aria-label="Move index down"
              title="Move index down"
              ${
                indexPosition === tensor.indices.length - 1 ? "disabled" : ""
              }
            >
              <span aria-hidden="true">&#8595;</span>
            </button>
            <button
              id="context-menu-delete-index-button"
              type="button"
              class="icon-button index-action-button danger"
              aria-label="Delete index"
              title="Delete index"
            >
              ${renderTrashIcon()}
            </button>
          </div>
        </div>
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

        if (nameInput) {
          const commitRename = () => {
            ctx.propertyCommands.renameIndex({
              tensor,
              index,
              proposedName: nameInput.value,
              invalidate: ctx.propertyInvalidation({ graph: true }),
              statusMessage: `Updated index ${nameInput.value.trim()}.`,
            });
          };
          nameInput.addEventListener("blur", commitRename);
          nameInput.addEventListener("keydown", (event) => {
            if (event.key !== "Enter") {
              return;
            }
            event.preventDefault();
            commitRename();
          });
        }

        if (dimensionInput) {
          const commitDimension = () => {
            ctx.propertyCommands.updateIndexDimension({
              indexId: index.id,
              rawValue: dimensionInput.value,
              invalidate: ctx.propertyInvalidation({
                analysis: true,
                graph: true,
              }),
              statusMessage: `Updated index ${index.name}.`,
            });
          };
          dimensionInput.addEventListener("blur", commitDimension);
          dimensionInput.addEventListener("keydown", (event) => {
            if (event.key !== "Enter") {
              return;
            }
            event.preventDefault();
            commitDimension();
          });
        }

        if (colorInput) {
          colorInput.addEventListener("input", () => {
            ctx.propertyCommands.updateTargetColor({
              target: index,
              nextColor: colorInput.value,
              invalidate: ctx.propertyInvalidation({ graph: true, minimap: true }),
              statusMessage: `Updated index ${index.name}.`,
            });
          });
        }

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
      },
    };
  }

  function getGroupContextTarget(groupId) {
    const group = typeof ctx.findGroupById === "function" ? ctx.findGroupById(groupId) : null;
    if (!group) {
      return null;
    }
    const isCollapsed = Boolean(group.metadata && group.metadata.collapsed);
    return {
      kind: "group",
      id: group.id,
      target: group,
      markup: `
        <div class="canvas-context-menu-section canvas-context-menu-input-stack">
          <div class="field-group">
            <input id="context-menu-name-input" value="${ctx.escapeHtml(group.name)}" />
          </div>
          <button id="context-menu-toggle-group-button" type="button">
            ${isCollapsed ? "Expand" : "Collapse"}
          </button>
        </div>
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const toggleGroupButton = document.getElementById(
          "context-menu-toggle-group-button"
        );

        if (nameInput) {
          const commitRename = () => {
            ctx.propertyCommands.renameGroup({
              group,
              proposedName: nameInput.value,
              invalidate: ctx.propertyInvalidation({ overlays: true }),
              statusMessage: `Updated group ${nameInput.value.trim()}.`,
            });
          };
          nameInput.addEventListener("blur", commitRename);
          nameInput.addEventListener("keydown", (event) => {
            if (event.key !== "Enter") {
              return;
            }
            event.preventDefault();
            commitRename();
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
      },
    };
  }

  function resolveContextTarget(menuState) {
    if (!menuState || !menuState.kind || !menuState.id) {
      return null;
    }
    if (menuState.kind === "tensor") {
      return getTensorContextTarget(menuState.id);
    }
    if (menuState.kind === "index") {
      return getIndexContextTarget(menuState.id);
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
    if (typeof ctx.setSelection === "function") {
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
