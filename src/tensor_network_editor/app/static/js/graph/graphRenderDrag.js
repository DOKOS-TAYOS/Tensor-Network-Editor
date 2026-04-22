export function createGraphRenderDragSupport({ ctx, state }) {
  function createTensorDragState(anchorId) {
    const dragSelection = ctx.buildCanvasSelectionDragState(anchorId);
    return {
      anchorId,
      ...dragSelection,
    };
  }

  function createHyperedgeDragState(hyperedgeId) {
    const hyperedge = ctx.findHyperedgeById(hyperedgeId);
    return {
      hyperedgeId,
      startOffset: hyperedge?.hub_offset
        ? {
          x: hyperedge.hub_offset.x,
          y: hyperedge.hub_offset.y,
        }
        : { x: 0, y: 0 },
      snapshot: ctx.createHistorySnapshot(),
    };
  }

  function moveCompanionTensorsDuringDrag() {
    if (!state.activeTensorDrag || !state.cy) {
      return;
    }
    const anchor =
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(state.activeTensorDrag.anchorId)
        : ctx.findTensorById(state.activeTensorDrag.anchorId);
    const anchorStartPosition =
      state.activeTensorDrag.tensorStartPositions[state.activeTensorDrag.anchorId];
    if (!anchor || !anchorStartPosition) {
      return;
    }
    const deltaX = anchor.position.x - anchorStartPosition.x;
    const deltaY = anchor.position.y - anchorStartPosition.y;
    ctx.runWithTensorSync(() => {
      state.activeTensorDrag.tensorIds.forEach((tensorId) => {
        if (tensorId === anchor.id) {
          return;
        }
        const tensor =
          typeof ctx.findVisibleTensorById === "function"
            ? ctx.findVisibleTensorById(tensorId)
            : ctx.findTensorById(tensorId);
        const startPosition = state.activeTensorDrag.tensorStartPositions[tensorId];
        if (!tensor || !startPosition) {
          return;
        }
        const nextPosition = {
          x: Math.round(startPosition.x + deltaX),
          y: Math.round(startPosition.y + deltaY),
        };
        if (
          typeof ctx.canEditCurrentContractionStage === "function" &&
          ctx.canEditCurrentContractionStage() &&
          typeof ctx.updateCurrentStageOperandLayout === "function"
        ) {
          ctx.updateCurrentStageOperandLayout(tensor.id, { position: nextPosition });
          tensor.position = nextPosition;
        } else {
          tensor.position.x = nextPosition.x;
          tensor.position.y = nextPosition.y;
        }
        const tensorElement = state.cy.getElementById(tensor.id);
        if (tensorElement && tensorElement.length) {
          tensorElement.position(tensor.position);
        }
        ctx.syncIndexNodePositions(tensor);
      });
    });
    state.activeTensorDrag.noteIds.forEach((noteId) => {
      const note = ctx.findNoteById(noteId);
      const startPosition = state.activeTensorDrag.noteStartPositions[noteId];
      if (!note || !startPosition) {
        return;
      }
      note.position.x = Math.round(startPosition.x + deltaX);
      note.position.y = Math.round(startPosition.y + deltaY);
    });
  }

  function finishTensorDrag(anchorId) {
    if (!state.activeTensorDrag || state.activeTensorDrag.anchorId !== anchorId) {
      return;
    }
    const shouldResyncSelection = Boolean(state.activeTensorDrag.addedSelectionOnGrab);
    const changed =
      state.activeTensorDrag.tensorIds.some((tensorId) => {
        const tensor =
          typeof ctx.findVisibleTensorById === "function"
            ? ctx.findVisibleTensorById(tensorId)
            : ctx.findTensorById(tensorId);
        const startPosition = state.activeTensorDrag.tensorStartPositions[tensorId];
        return (
          tensor &&
          startPosition &&
          (tensor.position.x !== startPosition.x || tensor.position.y !== startPosition.y)
        );
      }) ||
      state.activeTensorDrag.noteIds.some((noteId) => {
        const note = ctx.findNoteById(noteId);
        const startPosition = state.activeTensorDrag.noteStartPositions[noteId];
        return (
          note &&
          startPosition &&
          (note.position.x !== startPosition.x || note.position.y !== startPosition.y)
        );
      });
    if (changed) {
      ctx.commitHistorySnapshot(state.activeTensorDrag.snapshot);
    }
    state.activeTensorDrag = null;
    ctx.updateToolbarState();
    if (
      shouldResyncSelection &&
      typeof ctx.syncCySelection === "function"
    ) {
      ctx.syncCySelection();
    }
  }

  function finishIndexDrag(indexId) {
    if (!state.activeIndexDrag || state.activeIndexDrag.indexId !== indexId) {
      return;
    }
    const shouldResyncSelection = Boolean(state.activeIndexDrag.addedSelectionOnGrab);
    const located = ctx.findIndexOwner(indexId);
    const changed =
      located &&
      state.activeIndexDrag.startOffset &&
      (located.index.offset.x !== state.activeIndexDrag.startOffset.x ||
        located.index.offset.y !== state.activeIndexDrag.startOffset.y);
    if (changed) {
      ctx.commitHistorySnapshot(state.activeIndexDrag.snapshot);
    }
    state.activeIndexDrag = null;
    ctx.updateToolbarState();
    if (
      shouldResyncSelection &&
      typeof ctx.syncCySelection === "function"
    ) {
      ctx.syncCySelection();
    }
  }

  function finishHyperedgeDrag(hyperedgeId) {
    if (
      !state.activeHyperedgeDrag ||
      state.activeHyperedgeDrag.hyperedgeId !== hyperedgeId
    ) {
      return;
    }
    const shouldResyncSelection = Boolean(
      state.activeHyperedgeDrag.addedSelectionOnGrab
    );
    const hyperedge = ctx.findHyperedgeById(hyperedgeId);
    const startOffset = state.activeHyperedgeDrag.startOffset;
    const changed =
      hyperedge &&
      startOffset &&
      ((hyperedge.hub_offset?.x ?? 0) !== startOffset.x ||
        (hyperedge.hub_offset?.y ?? 0) !== startOffset.y);
    if (changed) {
      ctx.commitHistorySnapshot(state.activeHyperedgeDrag.snapshot);
    }
    state.activeHyperedgeDrag = null;
    ctx.updateToolbarState();
    if (
      shouldResyncSelection &&
      typeof ctx.syncCySelection === "function"
    ) {
      ctx.syncCySelection();
    }
  }

  return {
    createTensorDragState,
    createHyperedgeDragState,
    moveCompanionTensorsDuringDrag,
    finishTensorDrag,
    finishIndexDrag,
    finishHyperedgeDrag,
  };
}
