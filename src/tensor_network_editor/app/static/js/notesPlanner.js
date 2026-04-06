export function registerNotesPlanner(ctx) {
  const state = ctx.state;
  const {
    NOTE_WIDTH,
    NOTE_HEIGHT,
    NOTE_MIN_WIDTH,
    NOTE_MIN_HEIGHT,
    NOTE_COLLAPSED_SIZE,
  } = ctx.constants;
  const { addNoteButton, notesLayer, plannerPanel } = ctx.dom;

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
    return state.spec.notes.find((note) => note.id === noteId) || null;
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
      frame.style.borderColor = ctx.getMetadataColor(note.metadata, "#5f95ff");

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
          ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
        });

        const actions = document.createElement("div");
        actions.className = "canvas-note-actions";

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
        textarea.addEventListener("click", (event) => {
          event.stopPropagation();
          ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
        });
        textarea.addEventListener("focus", () => {
          ctx.setSelection([note.id], { primaryId: note.id });
        });
        textarea.addEventListener("change", () => {
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
              statusMessage: "Updated the note text.",
            }
          );
        });

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
        ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
      });
      noteElement.appendChild(frame);
      notesLayer.appendChild(noteElement);
    });
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
    ctx.commitHistorySnapshot(state.noteDragState.snapshot);
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
    renderNotes();
  }

  function finishActiveNoteResize() {
    if (!state.activeNoteResize) {
      return;
    }
    ctx.commitHistorySnapshot(state.activeNoteResize.snapshot);
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

  function ensureContractionPlan() {
    if (!state.spec.contraction_plan) {
      state.spec.contraction_plan = {
        id: ctx.makeId("plan"),
        name: "Manual path",
        steps: [],
        metadata: {},
      };
    }
    return state.spec.contraction_plan;
  }

  function getPlannerStepId(step) {
    if (!step || typeof step !== "object") {
      return null;
    }
    if (typeof step.id === "string" && step.id) {
      return step.id;
    }
    if (typeof step.step_id === "string" && step.step_id) {
      return step.step_id;
    }
    return null;
  }

  function buildPlannerOperandState(tensors, steps) {
    const activeOperands = new Map();
    const representativeByTensorId = {};
    const representativeByOperandId = {};
    const sourceTensorIdsByOperandId = {};
    const validSteps = [];
    const reservedOperandIds = new Set();
    const stepOrdersByTensorId = {};

    tensors.forEach((tensor) => {
      const sourceTensorIds = [tensor.id];
      activeOperands.set(tensor.id, { sourceTensorIds });
      representativeByTensorId[tensor.id] = tensor.id;
      representativeByOperandId[tensor.id] = tensor.id;
      sourceTensorIdsByOperandId[tensor.id] = sourceTensorIds;
      reservedOperandIds.add(tensor.id);
    });

    for (const step of steps) {
      const stepId = getPlannerStepId(step);
      if (
        !step ||
        !stepId ||
        step.left_operand_id === step.right_operand_id ||
        !activeOperands.has(step.left_operand_id) ||
        !activeOperands.has(step.right_operand_id) ||
        reservedOperandIds.has(stepId)
      ) {
        break;
      }
      const leftOperand = activeOperands.get(step.left_operand_id);
      const rightOperand = activeOperands.get(step.right_operand_id);
      if (!leftOperand || !rightOperand) {
        break;
      }
      const sourceTensorIds = [...new Set([
        ...leftOperand.sourceTensorIds,
        ...rightOperand.sourceTensorIds,
      ])];

      activeOperands.delete(step.left_operand_id);
      activeOperands.delete(step.right_operand_id);
      activeOperands.set(stepId, { sourceTensorIds });
      reservedOperandIds.add(stepId);
      validSteps.push(step);
      sourceTensorIdsByOperandId[stepId] = sourceTensorIds;

      sourceTensorIds.forEach((tensorId) => {
        representativeByTensorId[tensorId] = stepId;
        representativeByOperandId[tensorId] = stepId;
        if (!Array.isArray(stepOrdersByTensorId[tensorId])) {
          stepOrdersByTensorId[tensorId] = [];
        }
        stepOrdersByTensorId[tensorId].push(validSteps.length);
      });
      Object.keys(sourceTensorIdsByOperandId).forEach((operandId) => {
        const operandSourceTensorIds = sourceTensorIdsByOperandId[operandId] || [];
        if (operandSourceTensorIds.some((tensorId) => sourceTensorIds.includes(tensorId))) {
          representativeByOperandId[operandId] = stepId;
        }
      });
    }

    return {
      activeOperandIds: [...activeOperands.keys()],
      representativeByTensorId,
      representativeByOperandId,
      sourceTensorIdsByOperandId,
      validSteps,
      stepOrdersByTensorId,
    };
  }

  function getPlannerOperandState() {
    const planSteps = state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
    return buildPlannerOperandState(state.spec.tensors, planSteps);
  }

  function buildStepOrdersByTensorId(steps) {
    return buildPlannerOperandState(state.spec.tensors, steps || []).stepOrdersByTensorId;
  }

  function buildPreviewOrderByVisibleTensorId(steps) {
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : state.spec.tensors;
    const previewOrderByTensorId = Object.fromEntries(
      visibleTensors.map((tensor) => [tensor.id, []])
    );
    const sourceTensorIdsByOperandId = {};

    visibleTensors.forEach((tensor) => {
      sourceTensorIdsByOperandId[tensor.id] =
        Array.isArray(tensor.sourceTensorIds) && tensor.sourceTensorIds.length
          ? [...tensor.sourceTensorIds]
          : [tensor.id];
    });

    (Array.isArray(steps) ? steps : []).forEach((step, index) => {
      const leftSourceTensorIds = sourceTensorIdsByOperandId[step.left_operand_id] || [
        step.left_operand_id,
      ];
      const rightSourceTensorIds = sourceTensorIdsByOperandId[step.right_operand_id] || [
        step.right_operand_id,
      ];
      const resultSourceTensorIds = [...new Set([...leftSourceTensorIds, ...rightSourceTensorIds])];
      sourceTensorIdsByOperandId[step.result_operand_id] = resultSourceTensorIds;

      visibleTensors.forEach((tensor) => {
        const visibleSourceTensorIds = sourceTensorIdsByOperandId[tensor.id] || [tensor.id];
        if (
          resultSourceTensorIds.some((tensorId) => visibleSourceTensorIds.includes(tensorId))
        ) {
          previewOrderByTensorId[tensor.id].push(index + 1);
        }
      });
    });

    return previewOrderByTensorId;
  }

  function syncPlannerOrderBadges() {
    state.plannerManualOrderByTensorId = {};
    if (
      state.plannerPreviewMode &&
      state.contractionAnalysis &&
      state.contractionAnalysis.status === "ready"
    ) {
      const previewAnalysis = getAutomaticAnalysisByMode(
        state.contractionAnalysis.payload,
        state.plannerPreviewMode
      );
      state.plannerPreviewOrderByTensorId = previewAnalysis
        ? buildPreviewOrderByVisibleTensorId(previewAnalysis.steps)
        : {};
      return;
    }
    state.plannerPreviewOrderByTensorId = {};
  }

  function resolvePlannerOperandId(operandId) {
    if (typeof operandId !== "string" || !operandId) {
      return null;
    }
    const plannerOperandState = getPlannerOperandState();
    return plannerOperandState.representativeByOperandId[operandId]
      || plannerOperandState.representativeByTensorId[operandId]
      || null;
  }

  function repairContractionPlan() {
    const plan = state.spec.contraction_plan;
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      if (plan) {
        plan.view_snapshots = [];
      }
      state.plannerInspectionStepCount = null;
      state.plannerFutureBadgeDisclosure = {};
      return;
    }
    const plannerOperandState = getPlannerOperandState();
    if (!plannerOperandState.validSteps.length) {
      state.spec.contraction_plan = null;
      state.plannerInspectionStepCount = null;
      state.plannerFutureBadgeDisclosure = {};
      return;
    }
    plan.steps = plannerOperandState.validSteps;
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
    const latestAppliedStepCount =
      typeof ctx.getLatestAppliedStepCount === "function"
        ? ctx.getLatestAppliedStepCount()
        : plannerOperandState.validSteps.length;
    if (
      Number.isInteger(state.plannerInspectionStepCount) &&
      state.plannerInspectionStepCount >= latestAppliedStepCount
    ) {
      state.plannerInspectionStepCount = null;
    }
    state.plannerFutureBadgeDisclosure = {};
  }

  function getPlannerRemainingOperandIds() {
    return getPlannerOperandState().activeOperandIds;
  }

  function isPlannerOperandAvailable(operandId) {
    return resolvePlannerOperandId(operandId) !== null;
  }

  function getPlannerOperandSourceTensorIds(operandId) {
    const representativeOperandId = resolvePlannerOperandId(operandId) || operandId;
    const plannerOperandState = getPlannerOperandState();
    return plannerOperandState.sourceTensorIdsByOperandId[representativeOperandId]
      ? [...plannerOperandState.sourceTensorIdsByOperandId[representativeOperandId]]
      : [];
  }

  function getPlannerOperandLabel(operandId) {
    const tensor = ctx.findTensorById(operandId);
    if (tensor) {
      return tensor.name;
    }
    const planSteps = state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
    const stepIndex = planSteps.findIndex((step) => step.id === operandId);
    if (stepIndex >= 0) {
      return `Result ${stepIndex + 1}`;
    }
    if (/^auto_future_step_\d+$/.test(operandId)) {
      return `Auto future ${operandId.replace("auto_future_step_", "step ")}`;
    }
    if (/__auto_past_\d+$/.test(operandId)) {
      return `Auto past ${operandId.split("__auto_past_")[1]}`;
    }
    return operandId;
  }

  function handlePlannerOperandClick(operandId) {
    if (!state.plannerMode) {
      return;
    }
    if (
      typeof ctx.isInspectingPastStage === "function" &&
      ctx.isInspectingPastStage()
    ) {
      ctx.setStatus(
        "Past contraction steps are read-only. Return to the latest step before adding a new contraction.",
        "error"
      );
      return;
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    const resolvedOperandId = resolvePlannerOperandId(operandId);
    if (!resolvedOperandId) {
      ctx.setStatus("That operand is not available for the next manual contraction step.", "error");
      return;
    }
    if (!state.pendingPlannerOperandId) {
      state.pendingPlannerOperandId = resolvedOperandId;
      state.pendingPlannerSelectionId = operandId;
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
      renderPlanner();
      ctx.renderOverlayDecorations();
      ctx.setStatus(`Selected ${getPlannerOperandLabel(resolvedOperandId)} as the first manual operand.`);
      return;
    }
    if (state.pendingPlannerOperandId === resolvedOperandId) {
      ctx.setStatus(
        "Choose a different tensor or intermediate; both selections refer to the same contracted operand.",
        "error"
      );
      return;
    }
    const leftOperandId = state.pendingPlannerOperandId;
    const rightOperandId = resolvedOperandId;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    const leftLabel = getPlannerOperandLabel(leftOperandId);
    const rightLabel = getPlannerOperandLabel(rightOperandId);
    ctx.applyDesignChange(
      () => {
        if (typeof ctx.applyManualContractionStep === "function") {
          ctx.applyManualContractionStep(leftOperandId, rightOperandId);
        } else {
          const plan = ensureContractionPlan();
          plan.steps.push({
            id: ctx.makeId("step"),
            left_operand_id: leftOperandId,
            right_operand_id: rightOperandId,
            metadata: {},
          });
        }
      },
      {
        statusMessage: `Added manual contraction step ${leftLabel} × ${rightLabel}.`,
      }
    );
  }

  function trimContractionPlan(stepCount) {
    const plan = state.spec.contraction_plan;
    if (!plan) {
      return;
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    ctx.applyDesignChange(
      () => {
        if (stepCount <= 0) {
          state.spec.contraction_plan = null;
        } else {
          plan.steps = plan.steps.slice(0, stepCount);
        }
        state.plannerPreviewMode = null;
        state.plannerFutureBadgeDisclosure = {};
        state.plannerInspectionStepCount =
          stepCount <= 0
            ? null
            : Number.isInteger(state.plannerInspectionStepCount)
            ? Math.min(state.plannerInspectionStepCount, stepCount - 1)
            : null;
      },
      {
        statusMessage:
          stepCount <= 0 ? "Reset the manual contraction path." : "Trimmed the manual contraction path.",
      }
    );
  }

  function togglePlannerMode() {
    state.plannerMode = !state.plannerMode;
    if (!state.plannerMode) {
      state.pendingPlannerOperandId = null;
      state.pendingPlannerSelectionId = null;
    }
    if (typeof ctx.syncPendingInteractionClasses === "function") {
      ctx.syncPendingInteractionClasses();
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
    ctx.setStatus(
      state.plannerMode
        ? "Manual planner mode active. Click visible tensors or result tensors to define the next contraction step."
        : "Manual planner mode disabled."
    );
  }

  async function refreshContractionAnalysis(options = {}) {
    if (options.focusTab && typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    const requestId = state.contractionAnalysisRequestId + 1;
    state.contractionAnalysisRequestId = requestId;
    state.contractionAnalysis = { status: "loading" };
    renderPlanner();
    try {
      const payload = await ctx.apiPost("/api/analyze-contraction", {
        spec: ctx.serializeCurrentSpec(),
      });
      if (state.contractionAnalysisRequestId !== requestId) {
        return;
      }
      if (!payload.ok) {
        state.contractionAnalysis = {
          status: "issues",
          issues: payload.issues || [],
        };
      } else {
        state.contractionAnalysis = {
          status: "ready",
          payload,
        };
      }
    } catch (error) {
      if (state.contractionAnalysisRequestId !== requestId) {
        return;
      }
      state.contractionAnalysis = {
        status: "error",
        message: error.message,
      };
    }
    renderPlanner();
    ctx.renderOverlayDecorations();
  }

  function formatShape(shape) {
    if (!Array.isArray(shape) || !shape.length) {
      return "scalar";
    }
    return shape.join(" × ");
  }

  function formatNumber(value) {
    return Number(value || 0).toLocaleString();
  }

  function getAutomaticAnalysisByMode(payload, mode) {
    if (!payload) {
      return null;
    }
    if (mode === "automaticFuture") {
      return payload.automatic_future || null;
    }
    if (mode === "automaticPast") {
      return payload.automatic_past || null;
    }
    return null;
  }

  function togglePlannerDisclosure(disclosureKey) {
    state.plannerDisclosureState[disclosureKey] = !state.plannerDisclosureState[disclosureKey];
    renderPlanner();
  }

  function buildAutomaticPastRootGroups(steps) {
    const plannerOperandState = getPlannerOperandState();
    const planSteps = state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
    const sourceTensorIdsByOperandId = plannerOperandState.sourceTensorIdsByOperandId || {};
    const stepOrderById = Object.fromEntries(
      planSteps.map((step, index) => [step.id, index + 1])
    );
    const groups = {};

    (Array.isArray(steps) ? steps : []).forEach((step) => {
      const rootId = typeof step.result_operand_id === "string" &&
        Object.prototype.hasOwnProperty.call(stepOrderById, step.result_operand_id)
        ? step.result_operand_id
        : step.result_operand_id.split("__auto_past_")[0];
      if (!groups[rootId]) {
        const rootSourceTensorIds = sourceTensorIdsByOperandId[rootId] || [];
        const earliestStepCount = planSteps.reduce((minimum, candidate, index) => {
          const candidateSourceTensorIds = sourceTensorIdsByOperandId[candidate.id] || [];
          const belongsToRoot =
            rootSourceTensorIds.length &&
            candidateSourceTensorIds.every((tensorId) => rootSourceTensorIds.includes(tensorId));
          if (!belongsToRoot) {
            return minimum;
          }
          return Math.min(minimum, index);
        }, Number.POSITIVE_INFINITY);
        groups[rootId] = {
          rootId,
          steps: [],
          earliestStepCount: Number.isFinite(earliestStepCount) ? earliestStepCount : 0,
          originalStepOrder: stepOrderById[rootId] || 0,
        };
      }
      groups[rootId].steps.push(step);
    });

    return Object.values(groups).sort(
      (left, right) => left.originalStepOrder - right.originalStepOrder
    );
  }

  function clearAutomaticPreview(options = {}) {
    const previousPreviewMode = state.plannerPreviewMode;
    state.plannerPreviewMode = null;
    state.plannerPreviewOrderByTensorId = {};
    if (
      previousPreviewMode === "automaticPast" &&
      !options.preservePastInspection &&
      typeof ctx.clearPastInspection === "function"
    ) {
      ctx.clearPastInspection();
    }
  }

  function startAutomaticPreview(mode) {
    if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {
      return;
    }
    if (state.plannerPreviewMode === mode) {
      clearAutomaticPreview();
      renderPlanner();
      ctx.render();
      ctx.setStatus("Automatic preview cleared.");
      return;
    }
    const analysis = getAutomaticAnalysisByMode(state.contractionAnalysis.payload, mode);
    if (!analysis || analysis.status === "unavailable" || !Array.isArray(analysis.steps)) {
      ctx.setStatus("That automatic preview is not available yet.", "error");
      return;
    }
    clearAutomaticPreview();
    state.plannerPreviewMode = mode;
    if (mode === "automaticPast") {
      const rootGroups = buildAutomaticPastRootGroups(analysis.steps);
      if (rootGroups.length && typeof ctx.beginPastInspection === "function") {
        ctx.beginPastInspection(rootGroups[0].earliestStepCount);
      }
      renderPlanner();
      ctx.render();
      ctx.setStatus("Showing the auto past preview from the first affected contraction step.");
      return;
    }
    if (typeof ctx.clearPastInspection === "function") {
      ctx.clearPastInspection();
    }
    renderPlanner();
    ctx.render();
    ctx.setStatus("Showing the auto future preview.");
  }

  function appendAutomaticFutureSteps(steps) {
    const plan = ensureContractionPlan();
    const stepIdMap = {};
    steps.forEach((step) => {
      const nextStepId = ctx.makeId("step");
      stepIdMap[step.result_operand_id] = nextStepId;
      plan.steps.push({
        id: nextStepId,
        left_operand_id: stepIdMap[step.left_operand_id] || step.left_operand_id,
        right_operand_id: stepIdMap[step.right_operand_id] || step.right_operand_id,
        metadata: {},
      });
    });
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
  }

  function rewriteAutomaticPastSteps(steps) {
    const plan = ensureContractionPlan();
    const previousVisibleLayoutMap =
      typeof ctx.captureVisibleOperandLayoutMap === "function"
        ? ctx.captureVisibleOperandLayoutMap(
            typeof ctx.getLatestAppliedStepCount === "function"
              ? ctx.getLatestAppliedStepCount()
              : null
          )
        : {};
    const rootGroups = buildAutomaticPastRootGroups(steps);
    const plannerOperandState = getPlannerOperandState();
    const sourceTensorIdsByOperandId = plannerOperandState.sourceTensorIdsByOperandId || {};
    const sourceTensorIdsByRootId = Object.fromEntries(
      rootGroups.map((group) => [group.rootId, sourceTensorIdsByOperandId[group.rootId] || []])
    );
    const rewrittenSteps = [];

    plan.steps.forEach((step) => {
      const rootMatch = rootGroups.find((group) => {
        const rootSourceTensorIds = sourceTensorIdsByRootId[group.rootId] || [];
        const stepSourceTensorIds = sourceTensorIdsByOperandId[step.id] || [];
        return (
          rootSourceTensorIds.length &&
          stepSourceTensorIds.length &&
          stepSourceTensorIds.every((tensorId) => rootSourceTensorIds.includes(tensorId))
        );
      });
      if (!rootMatch) {
        rewrittenSteps.push(step);
        return;
      }
      if (step.id !== rootMatch.rootId) {
        return;
      }

      const existingRootStep = plan.steps.find((candidate) => candidate.id === rootMatch.rootId);
      const autoOperandIdMap = {};
      rootMatch.steps.forEach((autoStep) => {
        const isRootResult = autoStep.result_operand_id === rootMatch.rootId;
        const nextStepId = isRootResult ? rootMatch.rootId : ctx.makeId("step");
        autoOperandIdMap[autoStep.result_operand_id] = nextStepId;
        rewrittenSteps.push({
          id: nextStepId,
          left_operand_id: autoOperandIdMap[autoStep.left_operand_id] || autoStep.left_operand_id,
          right_operand_id: autoOperandIdMap[autoStep.right_operand_id] || autoStep.right_operand_id,
          metadata:
            isRootResult && existingRootStep && existingRootStep.metadata
              ? ctx.deepClone(existingRootStep.metadata)
              : {},
        });
      });
    });

    plan.steps = rewrittenSteps;
    if (typeof ctx.ensureContractionViewSnapshots === "function") {
      ctx.ensureContractionViewSnapshots();
    }
    if (
      typeof ctx.getLatestAppliedStepCount === "function" &&
      typeof ctx.applySnapshotLayoutMap === "function"
    ) {
      ctx.applySnapshotLayoutMap(ctx.getLatestAppliedStepCount(), previousVisibleLayoutMap);
    }
    if (
      state.plannerPreviewMode === "automaticPast" &&
      rootGroups.length &&
      typeof ctx.beginPastInspection === "function"
    ) {
      ctx.beginPastInspection(rootGroups[0].earliestStepCount);
    }
  }

  function acceptAutomaticPlan(mode) {
    if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {
      return;
    }
    const analysis = getAutomaticAnalysisByMode(state.contractionAnalysis.payload, mode);
    if (!analysis || analysis.status === "unavailable" || !Array.isArray(analysis.steps) || !analysis.steps.length) {
      ctx.setStatus("That automatic path is not available to accept.", "error");
      return;
    }
    ctx.applyDesignChange(
      () => {
        if (mode === "automaticFuture") {
          appendAutomaticFutureSteps(analysis.steps);
          if (typeof ctx.clearPastInspection === "function") {
            ctx.clearPastInspection();
          }
        } else {
          rewriteAutomaticPastSteps(analysis.steps);
        }
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
        state.plannerFutureBadgeDisclosure = {};
        clearAutomaticPreview();
      },
      {
        statusMessage:
          mode === "automaticFuture"
            ? "Assigned the remaining contraction steps from the auto future path."
            : "Rewired the contracted history with the auto past path.",
      }
    );
  }

  function renderMetricChips(items) {
    return `
      <div class="planner-chip-grid">
        ${items
          .map(
            (item) => `
              <div class="planner-chip">
                <span>${ctx.escapeHtml(item.label)}</span>
                <strong>${ctx.escapeHtml(String(item.value))}</strong>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function renderAutomaticPreviewSteps(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return "";
    }
    const previewLabelByOperandId = {};
    const resolvePreviewLabel = (operandId) => {
      if (previewLabelByOperandId[operandId]) {
        return previewLabelByOperandId[operandId];
      }
      const autoFutureMatch =
        typeof operandId === "string" ? operandId.match(/^auto_future_step_(\d+)$/) : null;
      if (autoFutureMatch) {
        return `Result ${autoFutureMatch[1]}`;
      }
      const autoPastMatch =
        typeof operandId === "string" ? operandId.match(/__auto_past_(\d+)$/) : null;
      if (autoPastMatch) {
        return `Result ${autoPastMatch[1]}`;
      }
      return getPlannerOperandLabel(operandId);
    };
    return `
      <div class="planner-step-list planner-preview-step-list">
        ${steps
          .map(
            (step, index) => {
              const leftLabel = resolvePreviewLabel(step.left_operand_id);
              const rightLabel = resolvePreviewLabel(step.right_operand_id);
              previewLabelByOperandId[step.step_id] = `Result ${index + 1}`;
              previewLabelByOperandId[step.result_operand_id] = `Result ${index + 1}`;
              return `
              <article class="planner-step planner-preview-step">
                <div class="planner-step-header">
                  <strong>Step ${index + 1}</strong>
                </div>
                <p>${ctx.escapeHtml(leftLabel)} Ã— ${ctx.escapeHtml(rightLabel)}</p>
              </article>
            `;
            }
          )
          .join("")}
      </div>
    `;
  }

  function renderAutomaticPreviewStepList(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return "";
    }
    const previewLabelByOperandId = {};
    const resolvePreviewLabel = (operandId) => {
      if (previewLabelByOperandId[operandId]) {
        return previewLabelByOperandId[operandId];
      }
      const autoFutureMatch =
        typeof operandId === "string" ? operandId.match(/^auto_future_step_(\d+)$/) : null;
      if (autoFutureMatch) {
        return `Result ${autoFutureMatch[1]}`;
      }
      const autoPastMatch =
        typeof operandId === "string" ? operandId.match(/__auto_past_(\d+)$/) : null;
      if (autoPastMatch) {
        return `Result ${autoPastMatch[1]}`;
      }
      return getPlannerOperandLabel(operandId);
    };
    return `
      <div class="planner-step-list planner-preview-step-list">
        ${steps
          .map((step, index) => {
            const leftLabel = resolvePreviewLabel(step.left_operand_id);
            const rightLabel = resolvePreviewLabel(step.right_operand_id);
            previewLabelByOperandId[step.step_id] = `Result ${index + 1}`;
            previewLabelByOperandId[step.result_operand_id] = `Result ${index + 1}`;
            return `
              <article class="planner-step planner-preview-step">
                <div class="planner-step-header">
                  <strong>Step ${index + 1}</strong>
                </div>
                <p>${ctx.escapeHtml(leftLabel)} &times; ${ctx.escapeHtml(rightLabel)}</p>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function renderAutomaticSection(title, disclosureKey, mode, analysis) {
    const isOpen = Boolean(state.plannerDisclosureState[disclosureKey]);
    const canAct = Boolean(analysis && analysis.status !== "unavailable");
    const summary = analysis && analysis.summary ? analysis.summary : {};
    const isPreviewing = state.plannerPreviewMode === mode;
    const previewShortcut = mode === "automaticFuture" ? "A" : "Shift+A";
    const acceptShortcut = mode === "automaticFuture" ? "Ctrl+A" : "Ctrl+Shift+A";
    const meta = analysis && analysis.message
      ? `<p class="planner-inline-meta">${ctx.escapeHtml(analysis.message)}</p>`
      : "";
    return `
      <section class="planner-section planner-disclosure">
        <button
          type="button"
          class="planner-disclosure-toggle button-accent-cool${isOpen ? " is-open" : ""}"
          data-disclosure="${ctx.escapeHtml(disclosureKey)}"
        >
          <span>${ctx.escapeHtml(title)}</span>
          <strong>${isOpen ? "Hide" : "Show"}</strong>
        </button>
        ${isOpen ? `
          <div class="planner-disclosure-body">
            ${renderMetricChips([
              { label: "FLOP", value: formatNumber(summary.total_estimated_flops) },
              { label: "MAC", value: formatNumber(summary.total_estimated_macs) },
              { label: "Peak", value: formatNumber(summary.peak_intermediate_size) },
            ])}
            ${
              isPreviewing
                ? `<p class="planner-inline-meta">Preview active.</p>${renderAutomaticPreviewStepList(
                    analysis && analysis.steps
                  )}`
                : ""
            }
            ${meta}
            <div class="button-row">
              <button
                type="button"
                class="button-accent-cool${isPreviewing ? " is-active" : ""}"
                data-preview-mode="${ctx.escapeHtml(mode)}"
                data-shortcut="${ctx.escapeHtml(previewShortcut)}"
                data-shortcut-label="${ctx.escapeHtml(isPreviewing ? "Deactivate preview" : "Preview")}"
                aria-pressed="${isPreviewing}"
                ${canAct ? "" : " disabled"}
              >
                ${isPreviewing ? "Deactivate preview" : "Preview"}
              </button>
              <button
                type="button"
                class="apply-button"
                data-accept-mode="${ctx.escapeHtml(mode)}"
                data-shortcut="${ctx.escapeHtml(acceptShortcut)}"
                data-shortcut-label="Accept"
                ${canAct ? "" : " disabled"}
              >
                Accept
              </button>
            </div>
          </div>
        ` : ""}
      </section>
    `;
  }

  function renderManualSection(manualAnalysis) {
    if (!manualAnalysis) {
      return `<section class="planner-section"><h3>Manual</h3><p class="planner-inline-meta">Waiting for analysis.</p></section>`;
    }
    return `
      <section class="planner-section">
        <h3>Manual</h3>
        ${renderMetricChips([
          { label: "Status", value: manualAnalysis.status || "unknown" },
          { label: "FLOP", value: formatNumber(manualAnalysis.summary && manualAnalysis.summary.total_estimated_flops) },
          { label: "MAC", value: formatNumber(manualAnalysis.summary && manualAnalysis.summary.total_estimated_macs) },
          { label: "Peak", value: formatNumber(manualAnalysis.summary && manualAnalysis.summary.peak_intermediate_size) },
          { label: "Shape", value: formatShape(manualAnalysis.summary && manualAnalysis.summary.final_shape) },
        ])}
        <div class="planner-step-list">
          ${renderManualSteps(manualAnalysis.steps)}
        </div>
      </section>
    `;
  }

  function renderManualSteps(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return `<p class="planner-inline-meta">No manual steps yet. Turn on manual mode and click two tensors to create the first contraction.</p>`;
    }
    const inspectedStepCount = Number.isInteger(state.plannerInspectionStepCount)
      ? state.plannerInspectionStepCount
      : null;
    return steps
      .map(
        (step, index) => `
          <article class="planner-step${inspectedStepCount === index ? " is-active" : ""}">
            <div class="planner-step-header">
              <button
                type="button"
                class="planner-step-toggle"
                data-inspect-step="${index}"
                aria-pressed="${inspectedStepCount === index}"
              >
                Step ${index + 1}
              </button>
              <button type="button" class="planner-trim-button" data-trim-step="${index}">Trim Here</button>
            </div>
            <p>${ctx.escapeHtml(getPlannerOperandLabel(step.left_operand_id))} × ${ctx.escapeHtml(getPlannerOperandLabel(step.right_operand_id))}</p>
            <div class="planner-step-meta">
              <span>Shape ${ctx.escapeHtml(formatShape(step.result_shape))}</span>
              <span>FLOP ${formatNumber(step.estimated_flops)}</span>
              <span>MAC ${formatNumber(step.estimated_macs)}</span>
            </div>
          </article>
        `
      )
      .join("");
  }

  function renderPlannerAnalysis() {
    if (!state.contractionAnalysis || state.contractionAnalysis.status === "loading") {
      return `<p class="planner-inline-meta">Analyzing contraction paths...</p>`;
    }
    if (state.contractionAnalysis.status === "issues") {
      return `<p class="planner-inline-meta planner-error">${ctx.escapeHtml(ctx.formatIssues(state.contractionAnalysis.issues || []))}</p>`;
    }
    if (state.contractionAnalysis.status === "error") {
      return `<p class="planner-inline-meta planner-error">${ctx.escapeHtml(state.contractionAnalysis.message || "Could not analyze contraction paths.")}</p>`;
    }
    const payload = state.contractionAnalysis.payload;
    const inspectionMeta =
      Number.isInteger(state.plannerInspectionStepCount)
        ? `<section class="planner-section"><p class="planner-inline-meta">Viewing the scene before step ${state.plannerInspectionStepCount + 1}. Click that step again to return to the latest contracted view.</p></section>`
        : "";
    return `
      <section class="planner-section">
        <p class="planner-network-output-label">Network output shape</p>
        <p class="planner-network-output">${ctx.escapeHtml(formatShape(payload.network_output_shape))}</p>
      </section>
      ${inspectionMeta}
      <div class="planner-summary-grid">
        ${renderAutomaticSection(
          "Auto future",
          "automaticFuture",
          "automaticFuture",
          payload.automatic_future
        )}
        ${renderAutomaticSection(
          "Auto past",
          "automaticPast",
          "automaticPast",
          payload.automatic_past
        )}
      </div>
      ${renderManualSection(payload.manual)}
    `;
  }

  function renderPlanner() {
    if (!plannerPanel) {
      return;
    }
    syncPlannerOrderBadges();
    const planSteps = state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
    const pendingLabel = state.pendingPlannerOperandId
      ? getPlannerOperandLabel(state.pendingPlannerOperandId)
      : null;

    plannerPanel.innerHTML = `
      <div class="planner-toolbar">
        <button
          id="toggle-planner-mode-button"
          type="button"
          class="button-accent-cool${state.plannerMode ? " is-active" : ""}"
          data-shortcut="M"
          data-shortcut-label="Manual scheme"
        >
          Contract
        </button>
        <button
          id="planner-reset-button"
          type="button"
          class="icon-button planner-icon-button danger"
          data-shortcut="Shift+R"
          data-shortcut-label="Reset path"
          aria-label="Reset path"
          title="Reset path"
          ${planSteps.length ? "" : " disabled"}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
            <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
          </svg>
        </button>
      </div>
      ${pendingLabel ? `<p class="planner-inline-meta">Pending operand: ${ctx.escapeHtml(pendingLabel)}.</p>` : ""}
      ${renderPlannerAnalysis()}
    `;

    document
      .getElementById("toggle-planner-mode-button")
      .addEventListener("click", togglePlannerMode);
    document
      .getElementById("planner-reset-button")
      .addEventListener("click", () => trimContractionPlan(0));
    plannerPanel.querySelectorAll("[data-trim-step]").forEach((button) => {
      button.addEventListener("click", () => {
        trimContractionPlan(Number(button.dataset.trimStep));
      });
    });
    plannerPanel.querySelectorAll("[data-inspect-step]").forEach((button) => {
      button.addEventListener("click", () => {
        if (typeof ctx.togglePastInspection === "function") {
          ctx.togglePastInspection(Number(button.dataset.inspectStep));
        }
        clearAutomaticPreview({ preservePastInspection: true });
        renderPlanner();
        ctx.render();
      });
    });
    plannerPanel.querySelectorAll("[data-disclosure]").forEach((button) => {
      button.addEventListener("click", () => {
        togglePlannerDisclosure(button.dataset.disclosure);
      });
    });
    plannerPanel.querySelectorAll("[data-preview-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        startAutomaticPreview(button.dataset.previewMode);
      });
    });
    plannerPanel.querySelectorAll("[data-accept-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        acceptAutomaticPlan(button.dataset.acceptMode);
      });
    });
    ctx.renderOverlayDecorations();
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
    repairContractionPlan,
    ensureContractionPlan,
    getPlannerRemainingOperandIds,
    isPlannerOperandAvailable,
    getPlannerOperandSourceTensorIds,
    getPlannerOperandLabel,
    resolvePlannerOperandId,
    handlePlannerOperandClick,
    trimContractionPlan,
    togglePlannerMode,
    refreshContractionAnalysis,
    renderPlanner,
    buildPlannerOperandState,
    buildStepOrdersByTensorId,
    syncPlannerOrderBadges,
    startAutomaticPreview,
    acceptAutomaticPlan,
    clearAutomaticPreview,
  });
}
