function buildMenuPositionStyle(menuState) {
  const left = Number.isFinite(menuState && menuState.clientX) ? menuState.clientX : 0;
  const top = Number.isFinite(menuState && menuState.clientY) ? menuState.clientY : 0;
  return `left: ${left}px; top: ${top}px;`;
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
    return {
      kind: "tensor",
      id: tensor.id,
      target: tensor,
      title: tensor.name,
      markup: `
        <div class="canvas-context-menu-section">
          <label class="field-group" for="context-menu-name-input">
            <span>Name</span>
            <input id="context-menu-name-input" value="${ctx.escapeHtml(tensor.name)}" />
          </label>
          <button id="context-menu-add-index-button" type="button">Add index</button>
        </div>
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const addIndexButton = document.getElementById("context-menu-add-index-button");
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
            closeCanvasContextMenu();
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

    return {
      kind: "index",
      id: index.id,
      target: index,
      title: index.name,
      markup: `
        <div class="canvas-context-menu-section">
          <label class="field-group" for="context-menu-name-input">
            <span>Name</span>
            <input id="context-menu-name-input" value="${ctx.escapeHtml(index.name)}" />
          </label>
          <label class="field-group" for="context-menu-dimension-input">
            <span>Dimension</span>
            <input
              id="context-menu-dimension-input"
              type="number"
              min="1"
              step="1"
              value="${index.dimension}"
            />
          </label>
          <div class="canvas-context-menu-actions">
            <button id="context-menu-move-up-button" type="button">Move up</button>
            <button id="context-menu-move-down-button" type="button">Move down</button>
          </div>
        </div>
      `,
      bind() {
        const nameInput = document.getElementById("context-menu-name-input");
        const dimensionInput = document.getElementById("context-menu-dimension-input");
        const moveUpButton = document.getElementById("context-menu-move-up-button");
        const moveDownButton = document.getElementById("context-menu-move-down-button");

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
            closeCanvasContextMenu();
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
            closeCanvasContextMenu();
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
      title: group.name,
      markup: `
        <div class="canvas-context-menu-section">
          <label class="field-group" for="context-menu-name-input">
            <span>Name</span>
            <input id="context-menu-name-input" value="${ctx.escapeHtml(group.name)}" />
          </label>
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
      <div class="canvas-context-menu" style="${buildMenuPositionStyle(menuState)}">
        <div class="canvas-context-menu-title">${ctx.escapeHtml(
          resolvedTarget.title
        )}</div>
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
