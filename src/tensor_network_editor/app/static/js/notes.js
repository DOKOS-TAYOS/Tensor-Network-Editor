export function registerNotesFeature(ctx) {
  const state = ctx.state;
  const {
    NOTE_WIDTH,
    NOTE_HEIGHT,
    NOTE_MIN_WIDTH,
    NOTE_MIN_HEIGHT,
    NOTE_COLLAPSED_SIZE,
  } = ctx.constants;
  const { addNoteButton, notesLayer } = ctx.dom;

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

  function getCanvasZoom() {
    return state.cy ? Math.max(0.1, state.cy.zoom()) : 1;
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
      width: Number(note.size && note.size.width) || NOTE_WIDTH,
      height: Number(note.size && note.size.height) || NOTE_HEIGHT,
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

    if (state.cy) {
      ctx.runWithTensorSync(() => {
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
          const tensorElement = state.cy.getElementById(tensor.id);
          if (tensorElement && tensorElement.length) {
            tensorElement.position(tensor.position);
          }
          ctx.syncIndexNodePositions(tensor);
        });
      });
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

  function renderNotes() {
    if (!notesLayer) {
      return;
    }
    notesLayer.innerHTML = "";
    const noteZoom = getCanvasZoom();
    state.spec.notes.forEach((note) => {
      const isCollapsed = Boolean(note.metadata && note.metadata.collapsed);
      const bounds = noteCanvasBounds(note);
      const noteSize = getRenderableNoteSize(note);
      const noteElement = document.createElement("article");
      noteElement.className = "canvas-note";
      noteElement.dataset.noteId = note.id;
      noteElement.style.left = `${bounds.x1}px`;
      noteElement.style.top = `${bounds.y1}px`;
      noteElement.style.width = `${bounds.width}px`;
      noteElement.style.height = `${bounds.height}px`;
      const frame = document.createElement("div");
      frame.className = "canvas-note-frame";
      if (state.selectionIds.includes(note.id)) {
        frame.classList.add("is-selected");
      }
      if (isCollapsed) {
        frame.classList.add("is-collapsed");
      }
      frame.style.width = `${noteSize.width}px`;
      frame.style.height = `${noteSize.height}px`;
      frame.style.transform = `scale(${noteZoom})`;
      frame.style.transformOrigin = "top left";
      const noteColor = ctx.getMetadataColor(note.metadata, "#5f95ff");
      frame.style.borderColor = noteColor;
      frame.style.setProperty("--note-accent-color", noteColor);
      frame.style.setProperty(
        "--note-surface-color",
        formatNoteColorAlpha(ctx.shiftColor(noteColor, -18), 0.96)
      );
      frame.style.setProperty(
        "--note-surface-color-strong",
        formatNoteColorAlpha(ctx.shiftColor(noteColor, -46), 0.98)
      );
      frame.style.setProperty(
        "--note-header-color",
        ctx.readableTextColor(ctx.shiftColor(noteColor, -6))
      );

      if (isCollapsed) {
        const collapsedToggle = createNoteCollapseButton(note);
        collapsedToggle.classList.add("canvas-note-collapsed-toggle");
        frame.appendChild(collapsedToggle);
        frame.addEventListener("mousedown", (event) => {
          if (event.target.closest(".toggle-note-collapse")) {
            return;
          }
          startNoteDrag(event, note.id);
        });
      } else {
        const header = document.createElement("div");
        header.className = "canvas-note-header";
        header.textContent = "Note";
        header.addEventListener("mousedown", (event) => startNoteDrag(event, note.id));
        header.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          selectNoteIfNeeded(note.id, { additive: Boolean(event.shiftKey) });
        });

        const actions = document.createElement("div");
        actions.className = "canvas-note-actions";

        const { colorButton, colorInput } = createNoteColorControl(note);
        actions.appendChild(colorButton);
        actions.appendChild(colorInput);

        const collapseButton = createNoteCollapseButton(note);
        actions.appendChild(collapseButton);

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "canvas-note-delete danger";
        deleteButton.setAttribute("aria-label", "Delete note");
        deleteButton.setAttribute("title", "Delete note");
        deleteButton.textContent = "×";
        deleteButton.innerHTML = `
          <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
            <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
          </svg>
        `;
        deleteButton.addEventListener("mousedown", (event) => {
          event.stopPropagation();
        });
        deleteButton.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          ctx.applyDesignChange(
            () => {
              removeNote(note.id);
            },
            {
              selectionIds: [],
              invalidate: noteInvalidation({ lookups: true }),
              statusMessage: "Deleted a canvas note.",
            }
          );
        });
        actions.appendChild(deleteButton);
        header.appendChild(actions);

        const textarea = document.createElement("textarea");
        textarea.className = "canvas-note-body";
        textarea.value = note.text;
        textarea.spellcheck = false;
        textarea.addEventListener("mousedown", (event) => {
          event.stopPropagation();
        });
        textarea.addEventListener("keydown", (event) => {
          event.stopPropagation();
        });
        textarea.addEventListener("click", (event) => {
          event.stopPropagation();
          selectNoteIfNeeded(note.id, { additive: Boolean(event.shiftKey) });
        });
        textarea.addEventListener("focus", () => {
          if (state.selectionIds.length === 1 && state.selectionIds[0] === note.id) {
            return;
          }
          ctx.setSelection([note.id], { primaryId: note.id });
        });
        ctx.bindDebouncedAutosave(
          textarea,
          `note:${note.id}:canvas-text`,
          () => {
            const proposedText = textarea.value.trim();
            if (!proposedText) {
              textarea.value = note.text;
              ctx.setStatus("Notes cannot be empty.", "error");
              return;
            }
            if (proposedText === note.text) {
              return;
            }
            ctx.applyDesignChange(
              () => {
                note.text = proposedText;
              },
              {
                selectionIds: [note.id],
                primaryId: note.id,
                invalidate: noteInvalidation({ overlays: false }),
                statusMessage: "Updated the note text.",
              }
            );
          },
          { commitOnEnter: false }
        );

        const resizeHandle = document.createElement("div");
        resizeHandle.className = "canvas-note-resize-handle";
        resizeHandle.addEventListener("mousedown", (event) => startNoteResize(event, note.id));

        frame.appendChild(header);
        frame.appendChild(textarea);
        frame.appendChild(resizeHandle);
      }
      noteElement.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectNoteIfNeeded(note.id, { additive: Boolean(event.shiftKey) });
      });
      noteElement.appendChild(frame);
      notesLayer.appendChild(noteElement);
    });
  }

  function createNoteColorControl(note) {
    const colorButton = document.createElement("button");
    colorButton.type = "button";
    colorButton.className = "canvas-note-color-button";
    colorButton.setAttribute("aria-label", "Change note color");
    colorButton.setAttribute("title", "Change note color");
    colorButton.innerHTML = `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M11.6 1.5a1.9 1.9 0 0 1 2.7 2.7l-1 1-2.7-2.7zm-1.7 1.7L2.2 10.9a2.5 2.5 0 0 0-.6 1l-.7 2.5a.7.7 0 0 0 .9.9l2.5-.7a2.5 2.5 0 0 0 1-.6L13 6.2z"/>
      </svg>
    `;
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.className = "canvas-note-color-input";
    colorInput.value = ctx.getMetadataColor(note.metadata, "#5f95ff");
    colorInput.setAttribute("tabindex", "-1");
    colorInput.setAttribute("aria-hidden", "true");

    colorButton.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    colorButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (typeof colorInput.showPicker === "function") {
        colorInput.showPicker();
      } else {
        colorInput.click();
      }
    });
    ctx.bindImmediateAutosave(
      colorInput,
      `note:${note.id}:canvas-color`,
      () => {
        if (
          colorInput.value ===
          ctx.getMetadataColor(note.metadata, "#5f95ff")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.metadata.color = colorInput.value;
          },
          {
            selectionIds: [note.id],
            primaryId: note.id,
            invalidate: noteInvalidation(),
            statusMessage: "Updated the note.",
          }
        );
      },
      "input"
    );
    return { colorButton, colorInput };
  }

  function createNoteCollapseButton(note) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "toggle-note-collapse";
    button.setAttribute("aria-label", note.metadata && note.metadata.collapsed ? "Expand note" : "Collapse note");
    button.setAttribute("title", note.metadata && note.metadata.collapsed ? "Expand note" : "Collapse note");
    button.innerHTML = `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M3 2.5h10A1.5 1.5 0 0 1 14.5 4v6A1.5 1.5 0 0 1 13 11.5H8.6L5 14v-2.5H3A1.5 1.5 0 0 1 1.5 10V4A1.5 1.5 0 0 1 3 2.5Zm1 3.25a.75.75 0 0 0 0 1.5h8a.75.75 0 0 0 0-1.5Zm0 2.75a.75.75 0 0 0 0 1.5h5.5a.75.75 0 0 0 0-1.5Z"/>
      </svg>
    `;
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleNoteCollapse(note.id);
    });
    return button;
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
      additive: Boolean(event.shiftKey),
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
      Math.round(state.activeNoteResize.startSize.width + worldPoint.x - state.activeNoteResize.startPointer.x)
    );
    note.size.height = Math.max(
      minimumWorldHeight,
      Math.round(state.activeNoteResize.startSize.height + worldPoint.y - state.activeNoteResize.startPointer.y)
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

  function copySelectedSubgraphToClipboard() {
    const tensorIds = ctx.getSelectedIdsByKind("tensor");
    if (!tensorIds.length) {
      ctx.setStatus("Select one or more tensors to copy.");
      return;
    }
    const tensorIdSet = new Set(tensorIds);
    const clipboardPayload = {
      tensors: ctx.deepClone(
        state.spec.tensors.filter((tensor) => tensorIdSet.has(tensor.id))
      ),
      edges: ctx.deepClone(
        state.spec.edges.filter(
          (edge) =>
            tensorIdSet.has(edge.left.tensor_id) &&
            tensorIdSet.has(edge.right.tensor_id)
        )
      ),
      groups: ctx.deepClone(
        state.spec.groups.filter(
          (group) =>
            group.tensor_ids.length &&
            group.tensor_ids.every((tensorId) => tensorIdSet.has(tensorId))
        )
      ),
      pasteCount: 0,
    };
    state.clipboard = clipboardPayload;
    ctx.setStatus(
      `Copied ${clipboardPayload.tensors.length} tensor${clipboardPayload.tensors.length === 1 ? "" : "s"} to the editor clipboard.`,
      "success"
    );
  }

  function pasteClipboardToCanvas() {
    if (!state.clipboard || !Array.isArray(state.clipboard.tensors) || !state.clipboard.tensors.length) {
      ctx.setStatus("There is no copied tensor subgraph to paste.");
      return;
    }
    const pasteCount = (state.clipboard.pasteCount || 0) + 1;
    const offset = 40 * pasteCount;
    const tensorIdMap = {};
    const indexIdMap = {};
    const groupIdMap = {};
    const clipboard = ctx.deepClone(state.clipboard);

    clipboard.tensors.forEach((tensor) => {
      const nextTensorId = ctx.makeId("tensor");
      tensorIdMap[tensor.id] = nextTensorId;
      tensor.id = nextTensorId;
      tensor.position.x += offset;
      tensor.position.y += offset;
      tensor.indices.forEach((index) => {
        const nextIndexId = ctx.makeId("index");
        indexIdMap[index.id] = nextIndexId;
        index.id = nextIndexId;
      });
    });
    clipboard.edges.forEach((edge) => {
      edge.id = ctx.makeId("edge");
      edge.left.tensor_id = tensorIdMap[edge.left.tensor_id];
      edge.right.tensor_id = tensorIdMap[edge.right.tensor_id];
      edge.left.index_id = indexIdMap[edge.left.index_id];
      edge.right.index_id = indexIdMap[edge.right.index_id];
    });
    clipboard.groups.forEach((group) => {
      const nextGroupId = ctx.makeId("group");
      groupIdMap[group.id] = nextGroupId;
      group.id = nextGroupId;
      group.tensor_ids = group.tensor_ids.map((tensorId) => tensorIdMap[tensorId]);
    });

    state.clipboard.pasteCount = pasteCount;
    ctx.applyDesignChange(
      () => {
        state.spec.tensors.push(...clipboard.tensors);
        state.spec.edges.push(...clipboard.edges);
        state.spec.groups.push(...clipboard.groups);
        clipboard.tensors.forEach((tensor) => {
          ctx.bringTensorToFront(tensor.id);
        });
      },
      {
        selectionIds: clipboard.tensors.map((tensor) => tensor.id),
        primaryId: clipboard.tensors.length
          ? clipboard.tensors[clipboard.tensors.length - 1].id
          : null,
        statusMessage: `Pasted ${clipboard.tensors.length} tensor${clipboard.tensors.length === 1 ? "" : "s"} from the editor clipboard.`,
      }
    );
  }


  if (addNoteButton) {
    addNoteButton.addEventListener("click", addNoteAtCenter);
  }

  Object.assign(ctx, {
    addNoteAtCenter,
    createNote,
    findNoteById,
    getRenderableNoteSize,
    noteCanvasBounds,
    buildCanvasSelectionDragState,
    applyCanvasSelectionDragDelta,
    removeNote,
    renderNotes,
    startNoteDrag,
    updateActiveNoteDrag,
    finishActiveNoteDrag,
    startNoteResize,
    updateActiveNoteResize,
    finishActiveNoteResize,
    toggleNoteCollapse,
    copySelectedSubgraphToClipboard,
    pasteClipboardToCanvas,
  });
}
