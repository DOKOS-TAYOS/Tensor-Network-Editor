import { createCanvasContextMenuBindings } from "./canvasContextMenuBindings.js";
import { createCanvasContextMenuMarkup } from "./canvasContextMenuMarkup.js";
import { createCanvasContextMenuTargetResolver } from "./canvasContextMenuTargets.js";

export function registerCanvasContextMenu(ctx) {
  const state = ctx.state;
  const { document, window } = ctx;
  const { canvasContextMenuRoot } = ctx.dom;

  function resolveOptionalAction(actionName) {
    return (...args) => {
      if (typeof ctx[actionName] === "function") {
        return ctx[actionName](...args);
      }
      return undefined;
    };
  }

  const markupSupport = createCanvasContextMenuMarkup({
    asFiniteNumber: (value, fallback = 1) =>
      typeof ctx.asFiniteNumber === "function"
        ? ctx.asFiniteNumber(value, fallback)
        : Number.isFinite(Number(value))
          ? Number(value)
          : fallback,
    buildMetadataEditorMarkup:
      typeof ctx.buildMetadataEditorMarkup === "function"
        ? (options) => ctx.buildMetadataEditorMarkup(options)
        : null,
    escapeHtml: (value) => ctx.escapeHtml(value),
    findTensorById:
      typeof ctx.findTensorById === "function"
        ? (tensorId) => ctx.findTensorById(tensorId)
        : null,
  });
  const {
    buildMenuPositionStyle,
    getIndexCountForTensorIds,
    getTensorTotalElementCount,
    getTotalElementCountForTensorIds,
    renderResolvedContextMenu,
  } = markupSupport;
  const targetResolver = createCanvasContextMenuTargetResolver({
    state,
    findEdgeById:
      typeof ctx.findEdgeById === "function" ? (edgeId) => ctx.findEdgeById(edgeId) : null,
    findEdgeByIndexId:
      typeof ctx.findConnectionByIndexId === "function"
        ? (indexId) => ctx.findConnectionByIndexId(indexId)
        : typeof ctx.findEdgeByIndexId === "function"
          ? (indexId) => ctx.findEdgeByIndexId(indexId)
        : null,
    findGroupById:
      typeof ctx.findGroupById === "function"
        ? (groupId) => ctx.findGroupById(groupId)
        : null,
    findIndexOwner:
      typeof ctx.findIndexOwner === "function"
        ? (indexId) => ctx.findIndexOwner(indexId)
        : null,
    findTensorById:
      typeof ctx.findTensorById === "function"
        ? (tensorId) => ctx.findTensorById(tensorId)
        : null,
    getBatchColorValue:
      typeof ctx.getBatchColorValue === "function"
        ? (entries) => ctx.getBatchColorValue(entries)
        : null,
    describeHyperedgeCandidate:
      typeof ctx.describeHyperedgeCandidate === "function"
        ? (indexIds) => ctx.describeHyperedgeCandidate(indexIds)
        : null,
    getIndexColor:
      typeof ctx.getIndexColor === "function"
        ? (index, isConnected) => ctx.getIndexColor(index, isConnected)
        : null,
    getIndexCountForTensorIds,
    getMetadataColor:
      typeof ctx.getMetadataColor === "function"
        ? (metadata, fallbackColor) => ctx.getMetadataColor(metadata, fallbackColor)
        : null,
    getSelectedEntries:
      typeof ctx.getSelectedEntries === "function" ? () => ctx.getSelectedEntries() : null,
    getSelectedIdsByKind:
      typeof ctx.getSelectedIdsByKind === "function"
        ? (kind) => ctx.getSelectedIdsByKind(kind)
        : null,
    getTensorTotalElementCount,
    getTotalElementCountForTensorIds,
  });

  function closeCanvasContextMenu() {
    state.canvasContextMenu = null;
    if (canvasContextMenuRoot) {
      canvasContextMenuRoot.innerHTML = "";
    }
  }

  function renderCanvasContextMenu() {
    if (!canvasContextMenuRoot) {
      return;
    }
    const menuState = state.canvasContextMenu;
    const resolvedTarget = targetResolver.resolveContextTarget(menuState);
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
        ${renderResolvedContextMenu(resolvedTarget)}
      </div>
    `;

    bindingsSupport.bindResolvedTarget(resolvedTarget);
  }

  const bindingsSupport = createCanvasContextMenuBindings({
    bindMetadataEditors:
      typeof ctx.bindMetadataEditors === "function"
        ? (options) => ctx.bindMetadataEditors(options)
        : null,
    closeCanvasContextMenu,
    createHyperedgeFromSelection: resolveOptionalAction("createHyperedgeFromSelection"),
    createGroupFromSelection: resolveOptionalAction("createGroupFromSelection"),
    document,
    exportGroupSubnetwork: resolveOptionalAction("exportGroupSubnetwork"),
    exportSelectedSubnetwork: resolveOptionalAction("exportSelectedSubnetwork"),
    saveGroupToSubnetworkLibrary: resolveOptionalAction(
      "saveGroupToSubnetworkLibrary"
    ),
    saveSelectionToSubnetworkLibrary: resolveOptionalAction(
      "saveSelectionToSubnetworkLibrary"
    ),
    promoteGroupToTemplate: resolveOptionalAction("promoteGroupToTemplate"),
    promoteSelectedSubnetworkToTemplate: resolveOptionalAction(
      "promoteSelectedSubnetworkToTemplate"
    ),
    propertyCommands: ctx.propertyCommands,
    propertyInvalidation: (invalidate) => ctx.propertyInvalidation(invalidate),
    renderCanvasContextMenu,
    state,
    toggleGroupCollapse: resolveOptionalAction("toggleGroupCollapse"),
    window,
  });

  function openCanvasContextMenu(menuState) {
    const resolvedTarget = targetResolver.resolveContextTarget(menuState);
    if (!resolvedTarget) {
      closeCanvasContextMenu();
      return;
    }
    if (
      resolvedTarget.kind !== "selection" &&
      resolvedTarget.kind !== "index-selection" &&
      typeof ctx.setSelection === "function"
    ) {
      ctx.setSelection([resolvedTarget.id], {
        primaryId: resolvedTarget.id,
      });
    }
    state.canvasContextMenu = {
      clientX: Number(menuState.clientX || 0),
      clientY: Number(menuState.clientY || 0),
      id: resolvedTarget.id,
      kind: resolvedTarget.kind,
    };
    renderCanvasContextMenu();
  }

  bindingsSupport.installCanvasContextMenuGlobalListeners();

  Object.assign(ctx, {
    closeCanvasContextMenu,
    openCanvasContextMenu,
    renderCanvasContextMenu,
  });
}
