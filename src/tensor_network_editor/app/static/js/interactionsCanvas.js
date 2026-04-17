export function createInteractionCanvasBindings({ ctx, state, dom }) {
  const { minimapCanvas, selectionBox } = dom;
  const BOX_SELECTION_DRAG_THRESHOLD = 4;

  function handleCanvasContextMenu(event) {
    event.preventDefault();
  }

  function handleCanvasWheel(event) {
    if (!state.cy || state.isHelpOpen || state.isTemplateManagerOpen) {
      return;
    }
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      event.stopPropagation();
      const container = state.cy.container();
      const rect = container.getBoundingClientRect();
      const renderedPosition = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      const zoomFactor = Math.exp(-event.deltaY * 0.0025);
      const nextZoom = ctx.clamp(
        state.cy.zoom() * zoomFactor,
        state.cy.minZoom(),
        state.cy.maxZoom()
      );
      state.cy.zoom({
        level: nextZoom,
        renderedPosition,
      });
      ctx.renderOverlayDecorations();
      ctx.renderMinimap();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    state.cy.panBy({
      x: -event.deltaX,
      y: -event.deltaY,
    });
    ctx.renderOverlayDecorations();
    ctx.renderMinimap();
  }

  function handleCanvasMouseDown(event) {
    if (state.isHelpOpen || state.isTemplateManagerOpen) {
      return;
    }
    const closestTarget =
      event &&
      event.target &&
      typeof event.target.closest === "function"
        ? (selector) => event.target.closest(selector)
        : () => null;
    if (
      closestTarget(".canvas-context-menu") ||
      closestTarget(".canvas-tools") ||
      closestTarget(".resize-handle") ||
      closestTarget(".group-overlay") ||
      closestTarget(".canvas-note") ||
      closestTarget(".minimap-shell")
    ) {
      return;
    }
    if (event.button === 2) {
      state.pendingBoxSelection = {
        additive: Boolean(event.shiftKey),
        startClientX: event.clientX,
        startClientY: event.clientY,
        startPoint: ctx.clientPointToCanvasPoint(event.clientX, event.clientY),
      };
      if (typeof ctx.closeCanvasContextMenu === "function") {
        ctx.closeCanvasContextMenu();
      }
      return;
    }
    if (typeof ctx.closeCanvasContextMenu === "function") {
      ctx.closeCanvasContextMenu();
    }
  }

  function handleGlobalMouseMove(event) {
    if (state.pendingBoxSelection) {
      const deltaX = event.clientX - state.pendingBoxSelection.startClientX;
      const deltaY = event.clientY - state.pendingBoxSelection.startClientY;
      if (
        Math.hypot(deltaX, deltaY) >= BOX_SELECTION_DRAG_THRESHOLD
      ) {
        startBoxSelectionFromPoint(
          state.pendingBoxSelection.startPoint,
          state.pendingBoxSelection.additive
        );
        state.pendingBoxSelection = null;
      }
    }
    if (state.boxSelection) {
      updateBoxSelection(event);
      return;
    }
    if (state.activeResize) {
      ctx.updateActiveResize(event);
      return;
    }
    if (state.activeGroupDrag) {
      ctx.updateActiveGroupDrag(event);
      return;
    }
    if (state.noteDragState) {
      ctx.updateActiveNoteDrag(event);
      return;
    }
    if (state.activeNoteResize) {
      ctx.updateActiveNoteResize(event);
      return;
    }
    if (state.minimapDrag) {
      ctx.updateViewportFromMinimapClientPoint(event.clientX, event.clientY);
    }
  }

  function handleGlobalMouseUp(event) {
    if (state.pendingBoxSelection && event.button === 2) {
      cancelPendingBoxSelection();
      return;
    }
    if (state.boxSelection && event.button === 2) {
      finishBoxSelection(false);
      return;
    }
    if (state.activeResize && event.button === 0) {
      ctx.finishActiveResize();
      return;
    }
    if (state.activeGroupDrag && event.button === 0) {
      ctx.finishActiveGroupDrag();
      return;
    }
    if (state.noteDragState && event.button === 0) {
      ctx.finishActiveNoteDrag();
      return;
    }
    if (state.activeNoteResize && event.button === 0) {
      ctx.finishActiveNoteResize();
      return;
    }
    if (state.minimapDrag && event.button === 0) {
      state.minimapDrag = null;
      minimapCanvas.classList.remove("is-dragging");
    }
  }

  function startBoxSelection(event) {
    const point = ctx.clientPointToCanvasPoint(event.clientX, event.clientY);
    startBoxSelectionFromPoint(point, Boolean(event.shiftKey));
  }

  function startBoxSelectionFromPoint(point, additive) {
    state.boxSelection = {
      start: point,
      current: point,
      additive: Boolean(additive),
    };
    updateSelectionBoxElement();
  }

  function updateBoxSelection(event) {
    state.boxSelection.current = ctx.clientPointToCanvasPoint(event.clientX, event.clientY);
    updateSelectionBoxElement();
  }

  function collectBoxSelectedNoteIds(box) {
    if (
      !state.spec ||
      !Array.isArray(state.spec.notes) ||
      typeof ctx.noteCanvasBounds !== "function"
    ) {
      return [];
    }
    return state.spec.notes
      .filter((note) => ctx.boxesIntersect(box, ctx.noteCanvasBounds(note)))
      .map((note) => note.id);
  }

  function finishBoxSelection(cancelled) {
    if (!state.boxSelection) {
      return;
    }
    const boxSelectionState = state.boxSelection;
    state.boxSelection = null;
    selectionBox.classList.add("is-hidden");
    if (cancelled || !state.cy) {
      return;
    }
    const box = ctx.normalizedBox(boxSelectionState.start, boxSelectionState.current);
    const hitIds = state.cy
      .elements("node, edge")
      .toArray()
      .filter((element) => element.data("kind") !== "index-label")
      .filter((element) => ctx.boxesIntersect(box, element.renderedBoundingBox()))
      .map((element) => element.id());
    const hitNoteIds = collectBoxSelectedNoteIds(box);
    const selectionIds = [...hitIds, ...hitNoteIds];
    if (boxSelectionState.additive) {
      ctx.setSelection([...state.selectionIds, ...selectionIds], {
        primaryId: selectionIds.length
          ? selectionIds[selectionIds.length - 1]
          : state.primarySelectionId,
      });
      return;
    }
    ctx.setSelection(selectionIds, {
      primaryId: selectionIds.length ? selectionIds[selectionIds.length - 1] : null,
    });
  }

  function updateSelectionBoxElement() {
    if (!state.boxSelection) {
      selectionBox.classList.add("is-hidden");
      return;
    }
    const box = ctx.normalizedBox(state.boxSelection.start, state.boxSelection.current);
    selectionBox.classList.remove("is-hidden");
    selectionBox.style.left = `${box.left}px`;
    selectionBox.style.top = `${box.top}px`;
    selectionBox.style.width = `${Math.max(1, box.width)}px`;
    selectionBox.style.height = `${Math.max(1, box.height)}px`;
  }

  function cancelPendingBoxSelection() {
    state.pendingBoxSelection = null;
  }

  return {
    handleCanvasContextMenu,
    handleCanvasWheel,
    handleCanvasMouseDown,
    handleGlobalMouseMove,
    handleGlobalMouseUp,
    startBoxSelection,
    startBoxSelectionFromPoint,
    updateBoxSelection,
    finishBoxSelection,
    updateSelectionBoxElement,
    cancelPendingBoxSelection,
  };
}
