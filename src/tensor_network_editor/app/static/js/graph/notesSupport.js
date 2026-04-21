export function createNotesSupport({ ctx, state, constants }) {
  const {
    NOTE_WIDTH,
    NOTE_HEIGHT,
    NOTE_MIN_WIDTH,
    NOTE_MIN_HEIGHT,
    NOTE_COLLAPSED_SIZE,
  } = constants;

  function noteInvalidation(overrides = {}) {
    return {
      graph: false,
      lookups: false,
      analysis: false,
      overlays: true,
      planner: false,
      minimap: false,
      ...overrides,
    };
  }

  function formatNoteColorAlpha(hexColor, alpha) {
    const { red, green, blue } = ctx.parseHexColor(hexColor);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function getCanvasZoom() {
    return state.cy ? Math.max(0.1, state.cy.zoom()) : 1;
  }

  function createNote(x, y) {
    const zoom = getCanvasZoom();
    return {
      id: ctx.makeId("note"),
      text: "New note",
      position: { x, y },
      size: {
        width: NOTE_WIDTH / zoom,
        height: NOTE_HEIGHT / zoom,
      },
      metadata: {},
    };
  }

  function findNoteById(noteId) {
    if (typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    return state.noteById[noteId] || null;
  }

  function removeNote(noteId) {
    state.spec.notes = state.spec.notes.filter((note) => note.id !== noteId);
  }

  function addNoteAtCenter() {
    const center = ctx.viewportCenterPosition();
    const zoom = getCanvasZoom();
    const worldWidth = NOTE_WIDTH / zoom;
    const worldHeight = NOTE_HEIGHT / zoom;
    const note = createNote(
      center.x - worldWidth / 2,
      center.y - worldHeight / 2
    );
    ctx.applyDesignChange(
      () => {
        state.spec.notes.push(note);
      },
      {
        selectionIds: [note.id],
        primaryId: note.id,
        invalidate: noteInvalidation({ lookups: true }),
        statusMessage: "Added a canvas note.",
      }
    );
  }

  function getRenderableNoteSize(note) {
    if (Boolean(note.metadata && note.metadata.collapsed)) {
      return {
        width: NOTE_COLLAPSED_SIZE,
        height: NOTE_COLLAPSED_SIZE,
      };
    }
    return {
      width: Math.max(
        NOTE_MIN_WIDTH,
        Number(note.size && note.size.width) || NOTE_WIDTH
      ),
      height: Math.max(
        NOTE_MIN_HEIGHT,
        Number(note.size && note.size.height) || NOTE_HEIGHT
      ),
    };
  }

  function noteCanvasBounds(note) {
    const canvasPoint = ctx.worldToCanvasPoint(note.position);
    const noteSize = getRenderableNoteSize(note);
    const zoom = getCanvasZoom();
    return {
      x1: canvasPoint.x,
      y1: canvasPoint.y,
      x2: canvasPoint.x + noteSize.width * zoom,
      y2: canvasPoint.y + noteSize.height * zoom,
      width: noteSize.width * zoom,
      height: noteSize.height * zoom,
    };
  }

  function preserveSelectionForCanvasDrag(selectionId, options = {}) {
    if (options.additive && !state.selectionIds.includes(selectionId)) {
      ctx.setSelection([...state.selectionIds, selectionId], {
        primaryId: selectionId,
      });
      return;
    }
    if (!state.selectionIds.includes(selectionId)) {
      ctx.setSelection([selectionId], { primaryId: selectionId });
    }
  }

  function selectNoteIfNeeded(noteId, options = {}) {
    if (options.additive) {
      ctx.selectElement("note", noteId, { additive: true });
      return true;
    }
    if (state.selectionIds.length === 1 && state.selectionIds[0] === noteId) {
      return false;
    }
    ctx.setSelection([noteId], { primaryId: noteId });
    return true;
  }

  function buildCanvasSelectionDragState(anchorSelectionId) {
    const tensorIds = [];
    const noteIds = [];
    const selectedEntries = ctx.getSelectedEntries();

    selectedEntries.forEach((entry) => {
      if (
        (entry.kind === "tensor" || entry.kind === "contraction-tensor") &&
        !tensorIds.includes(entry.tensor.id)
      ) {
        tensorIds.push(entry.tensor.id);
        return;
      }
      if (entry.kind === "note" && !noteIds.includes(entry.note.id)) {
        noteIds.push(entry.note.id);
        return;
      }
      if (entry.kind === "group") {
        entry.group.tensor_ids.forEach((tensorId) => {
          if (!tensorIds.includes(tensorId)) {
            tensorIds.push(tensorId);
          }
        });
      }
    });

    if (!tensorIds.length && !noteIds.length) {
      const anchorEntry = ctx.getSelectionEntry(anchorSelectionId);
      if (
        anchorEntry &&
        (anchorEntry.kind === "tensor" || anchorEntry.kind === "contraction-tensor")
      ) {
        tensorIds.push(anchorEntry.tensor.id);
      } else if (anchorEntry && anchorEntry.kind === "note") {
        noteIds.push(anchorEntry.note.id);
      } else if (anchorEntry && anchorEntry.kind === "group") {
        anchorEntry.group.tensor_ids.forEach((tensorId) => {
          if (!tensorIds.includes(tensorId)) {
            tensorIds.push(tensorId);
          }
        });
      }
    }

    return {
      snapshot: ctx.createHistorySnapshot(),
      tensorIds,
      noteIds,
      tensorStartPositions: Object.fromEntries(
        tensorIds
          .map((tensorId) =>
            typeof ctx.findVisibleTensorById === "function"
              ? ctx.findVisibleTensorById(tensorId)
              : ctx.findTensorById(tensorId)
          )
          .filter(Boolean)
          .map((tensor) => [tensor.id, { x: tensor.position.x, y: tensor.position.y }])
      ),
      noteStartPositions: Object.fromEntries(
        noteIds
          .map((noteId) => findNoteById(noteId))
          .filter(Boolean)
          .map((note) => [note.id, { x: note.position.x, y: note.position.y }])
      ),
    };
  }

  function applyCanvasSelectionDragDelta(dragState, deltaX, deltaY, options = {}) {
    const excludedTensorIds = Array.isArray(options.excludedTensorIds)
      ? options.excludedTensorIds
      : [];
    const excludedNoteIds = Array.isArray(options.excludedNoteIds)
      ? options.excludedNoteIds
      : [];

    const updateTensorPositions = () => {
      dragState.tensorIds.forEach((tensorId) => {
        if (excludedTensorIds.includes(tensorId)) {
          return;
        }
        const tensor =
          typeof ctx.findVisibleTensorById === "function"
            ? ctx.findVisibleTensorById(tensorId)
            : ctx.findTensorById(tensorId);
        const startPosition = dragState.tensorStartPositions[tensorId];
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
        } else if (ctx.findTensorById(tensorId)) {
          tensor.position.x = nextPosition.x;
          tensor.position.y = nextPosition.y;
        } else {
          return;
        }
        if (!state.cy) {
          return;
        }
        const tensorElement = state.cy.getElementById(tensor.id);
        if (tensorElement && tensorElement.length) {
          tensorElement.position(tensor.position);
        }
        ctx.syncIndexNodePositions(tensor);
      });
    };

    if (typeof ctx.runWithTensorSync === "function") {
      ctx.runWithTensorSync(updateTensorPositions);
    } else {
      updateTensorPositions();
    }

    dragState.noteIds.forEach((noteId) => {
      if (excludedNoteIds.includes(noteId)) {
        return;
      }
      const note = findNoteById(noteId);
      const startPosition = dragState.noteStartPositions[noteId];
      if (!note || !startPosition) {
        return;
      }
      note.position.x = Math.round(startPosition.x + deltaX);
      note.position.y = Math.round(startPosition.y + deltaY);
    });
  }

  function startNoteDrag(event, noteId) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const note = findNoteById(noteId);
    if (!note) {
      return;
    }
    preserveSelectionForCanvasDrag(noteId, {
      additive:
        typeof ctx.isAdditiveSelectionModifier === "function" &&
        ctx.isAdditiveSelectionModifier(event),
    });
    const dragSelection = buildCanvasSelectionDragState(noteId);
    state.noteDragState = {
      noteId,
      startPointer: ctx.clientPointToWorldPoint(event.clientX, event.clientY),
      ...dragSelection,
    };
  }

  function updateActiveNoteDrag(event) {
    if (!state.noteDragState) {
      return;
    }
    const note = findNoteById(state.noteDragState.noteId);
    if (!note) {
      return;
    }
    const worldPoint = ctx.clientPointToWorldPoint(event.clientX, event.clientY);
    const deltaX = worldPoint.x - state.noteDragState.startPointer.x;
    const deltaY = worldPoint.y - state.noteDragState.startPointer.y;
    applyCanvasSelectionDragDelta(state.noteDragState, deltaX, deltaY);
    ctx.renderOverlayDecorations();
    ctx.renderMinimap();
  }

  function finishActiveNoteDrag() {
    if (!state.noteDragState) {
      return;
    }
    const changed = state.noteDragState.noteIds.some((noteId) => {
      const note = findNoteById(noteId);
      const startPosition = state.noteDragState.noteStartPositions[noteId];
      return (
        note &&
        startPosition &&
        (note.position.x !== startPosition.x || note.position.y !== startPosition.y)
      );
    });
    if (changed) {
      ctx.commitHistorySnapshot(state.noteDragState.snapshot);
    }
    state.noteDragState = null;
    ctx.renderOverlayDecorations();
    ctx.updateToolbarState();
  }

  function startNoteResize(event, noteId) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const note = findNoteById(noteId);
    if (!note || (note.metadata && note.metadata.collapsed)) {
      return;
    }
    preserveSelectionForCanvasDrag(noteId);
    state.activeNoteResize = {
      noteId,
      snapshot: ctx.createHistorySnapshot(),
      startPointer: ctx.clientPointToWorldPoint(event.clientX, event.clientY),
      startSize: {
        width: Number(note.size && note.size.width) || NOTE_WIDTH,
        height: Number(note.size && note.size.height) || NOTE_HEIGHT,
      },
    };
  }

  function updateActiveNoteResize(event) {
    if (!state.activeNoteResize) {
      return;
    }
    const note = findNoteById(state.activeNoteResize.noteId);
    if (!note) {
      return;
    }
    const worldPoint = ctx.clientPointToWorldPoint(event.clientX, event.clientY);
    const minimumWorldWidth = NOTE_MIN_WIDTH / getCanvasZoom();
    const minimumWorldHeight = NOTE_MIN_HEIGHT / getCanvasZoom();
    note.size.width = Math.max(
      minimumWorldWidth,
      Math.round(
        state.activeNoteResize.startSize.width +
          worldPoint.x -
          state.activeNoteResize.startPointer.x
      )
    );
    note.size.height = Math.max(
      minimumWorldHeight,
      Math.round(
        state.activeNoteResize.startSize.height +
          worldPoint.y -
          state.activeNoteResize.startPointer.y
      )
    );
    ctx.renderOverlayDecorations();
  }

  function finishActiveNoteResize() {
    if (!state.activeNoteResize) {
      return;
    }
    const note = findNoteById(state.activeNoteResize.noteId);
    const changed =
      note &&
      (
        note.size.width !== state.activeNoteResize.startSize.width ||
        note.size.height !== state.activeNoteResize.startSize.height
      );
    if (changed) {
      ctx.commitHistorySnapshot(state.activeNoteResize.snapshot);
    }
    state.activeNoteResize = null;
    ctx.renderOverlayDecorations();
    ctx.updateToolbarState();
  }

  function toggleNoteCollapse(noteId) {
    const note = findNoteById(noteId);
    if (!note) {
      return;
    }
    ctx.applyDesignChange(
      () => {
        note.metadata.collapsed = !Boolean(note.metadata && note.metadata.collapsed);
      },
      {
        selectionIds: [note.id],
        primaryId: note.id,
        invalidate: noteInvalidation(),
        statusMessage: note.metadata && note.metadata.collapsed
          ? "Expanded the note."
          : "Collapsed the note.",
      }
    );
  }

  return {
    addNoteAtCenter,
    applyCanvasSelectionDragDelta,
    buildCanvasSelectionDragState,
    createNote,
    findNoteById,
    finishActiveNoteDrag,
    finishActiveNoteResize,
    formatNoteColorAlpha,
    getCanvasZoom,
    getRenderableNoteSize,
    noteCanvasBounds,
    noteInvalidation,
    removeNote,
    selectNoteIfNeeded,
    startNoteDrag,
    startNoteResize,
    toggleNoteCollapse,
    updateActiveNoteDrag,
    updateActiveNoteResize,
  };
}
