export function registerNotesPlanner(ctx) {
  const state = ctx.state;
  const { addNoteButton, notesLayer, plannerPanel } = ctx.dom;

  function createNote(x, y) {
    return {
      id: ctx.makeId("note"),
      text: "New note",
      position: { x, y },
      size: { width: 220, height: 112 },
      metadata: {},
    };
  }

  function findNoteById(noteId) {
    return state.spec.notes.find((note) => note.id === noteId) || null;
  }

  function removeNote(noteId) {
    state.spec.notes = state.spec.notes.filter((note) => note.id !== noteId);
  }

  function addNoteAtCenter() {
    const center = ctx.viewportCenterPosition();
    const note = createNote(center.x - 110, center.y - 56);
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

  function renderNotes() {
    if (!notesLayer) {
      return;
    }
    notesLayer.innerHTML = "";
    state.spec.notes.forEach((note) => {
      const canvasPoint = ctx.worldToCanvasPoint(note.position);
      const isCollapsed = Boolean(note.metadata && note.metadata.collapsed);
      const noteWidth = isCollapsed ? 48 : Math.max(140, Number(note.size && note.size.width) || 220);
      const noteHeight = isCollapsed ? 48 : Math.max(96, Number(note.size && note.size.height) || 112);
      const noteElement = document.createElement("article");
      noteElement.className = "canvas-note";
      noteElement.dataset.noteId = note.id;
      if (state.selectionIds.includes(note.id)) {
        noteElement.classList.add("is-selected");
      }
      if (isCollapsed) {
        noteElement.classList.add("is-collapsed");
      }
      noteElement.style.left = `${canvasPoint.x}px`;
      noteElement.style.top = `${canvasPoint.y}px`;
      noteElement.style.width = `${noteWidth}px`;
      noteElement.style.height = `${noteHeight}px`;
      noteElement.style.borderColor = ctx.getMetadataColor(note.metadata, "#5f95ff");

      if (isCollapsed) {
        const collapsedToggle = createNoteCollapseButton(note);
        collapsedToggle.classList.add("canvas-note-collapsed-toggle");
        noteElement.appendChild(collapsedToggle);
        noteElement.addEventListener("mousedown", (event) => {
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
        deleteButton.className = "canvas-note-delete";
        deleteButton.textContent = "×";
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
        textarea.style.height = `${Math.max(54, noteHeight - 52)}px`;
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

        noteElement.appendChild(header);
        noteElement.appendChild(textarea);
        noteElement.appendChild(resizeHandle);
      }
      noteElement.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
      });
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
    ctx.setSelection([noteId], { primaryId: noteId });
    state.noteDragState = {
      noteId,
      snapshot: ctx.createHistorySnapshot(),
      startPointer: ctx.clientPointToWorldPoint(event.clientX, event.clientY),
      startPosition: { x: note.position.x, y: note.position.y },
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
    note.position.x = Math.round(
      state.noteDragState.startPosition.x + worldPoint.x - state.noteDragState.startPointer.x
    );
    note.position.y = Math.round(
      state.noteDragState.startPosition.y + worldPoint.y - state.noteDragState.startPointer.y
    );
    renderNotes();
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
    ctx.setSelection([noteId], { primaryId: noteId });
    state.activeNoteResize = {
      noteId,
      snapshot: ctx.createHistorySnapshot(),
      startPointer: ctx.clientPointToCanvasPoint(event.clientX, event.clientY),
      startSize: {
        width: Math.max(140, Number(note.size && note.size.width) || 220),
        height: Math.max(96, Number(note.size && note.size.height) || 112),
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
    const canvasPoint = ctx.clientPointToCanvasPoint(event.clientX, event.clientY);
    note.size.width = Math.max(
      140,
      Math.round(state.activeNoteResize.startSize.width + canvasPoint.x - state.activeNoteResize.startPointer.x)
    );
    note.size.height = Math.max(
      96,
      Math.round(state.activeNoteResize.startSize.height + canvasPoint.y - state.activeNoteResize.startPointer.y)
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

  function syncPlannerOrderBadges() {
    state.plannerManualOrderByTensorId = getPlannerOperandState().stepOrdersByTensorId;
    if (state.plannerPreviewMode && state.contractionAnalysis && state.contractionAnalysis.status === "ready") {
      const previewAnalysis = getAutomaticAnalysisByMode(
        state.contractionAnalysis.payload,
        state.plannerPreviewMode
      );
      state.plannerPreviewOrderByTensorId = previewAnalysis
        ? buildStepOrdersByTensorId(previewAnalysis.steps)
        : {};
    } else {
      state.plannerPreviewOrderByTensorId = {};
    }
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
      syncPlannerOrderBadges();
      return;
    }
    const plannerOperandState = getPlannerOperandState();
    if (!plannerOperandState.validSteps.length) {
      state.spec.contraction_plan = null;
      syncPlannerOrderBadges();
      return;
    }
    plan.steps = plannerOperandState.validSteps;
    syncPlannerOrderBadges();
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
      return `Intermediate ${stepIndex + 1}`;
    }
    if (/^auto_step_\d+$/.test(operandId)) {
      return `Automatic ${operandId.replace("auto_step_", "step ")}`;
    }
    return operandId;
  }

  function handlePlannerOperandClick(operandId) {
    if (!state.plannerMode) {
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
      state.pendingPlannerSelectionId = ctx.findTensorById(operandId) ? operandId : null;
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
        const plan = ensureContractionPlan();
        plan.steps.push({
          id: ctx.makeId("step"),
          left_operand_id: leftOperandId,
          right_operand_id: rightOperandId,
          metadata: {},
        });
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
        ? "Manual planner mode active. Click tensors or intermediate cards to define the next contraction step."
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
    syncPlannerOrderBadges();
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
    syncPlannerOrderBadges();
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
    if (mode === "automatic_global") {
      return payload.automatic_global || null;
    }
    if (mode === "automatic_local") {
      return payload.automatic_local || null;
    }
    return null;
  }

  function togglePlannerDisclosure(disclosureKey) {
    state.plannerDisclosureState[disclosureKey] = !state.plannerDisclosureState[disclosureKey];
    renderPlanner();
  }

  function startAutomaticPreview(mode) {
    if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {
      return;
    }
    const analysis = getAutomaticAnalysisByMode(state.contractionAnalysis.payload, mode);
    if (!analysis || analysis.status === "unavailable" || !Array.isArray(analysis.steps)) {
      ctx.setStatus("That automatic preview is not available yet.", "error");
      return;
    }
    state.plannerPreviewMode = mode;
    syncPlannerOrderBadges();
    ctx.render();
    ctx.setStatus(
      mode === "automatic_global"
        ? "Showing the global automatic preview."
        : "Showing the local automatic preview."
    );
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
        const acceptedSteps = [];
        const stepIdMap = {};
        analysis.steps.forEach((step) => {
          const nextStepId = ctx.makeId("step");
          stepIdMap[step.result_operand_id] = nextStepId;
          acceptedSteps.push({
            id: nextStepId,
            left_operand_id: stepIdMap[step.left_operand_id] || step.left_operand_id,
            right_operand_id: stepIdMap[step.right_operand_id] || step.right_operand_id,
            metadata: {},
          });
        });
        state.spec.contraction_plan = {
          id: state.spec.contraction_plan ? state.spec.contraction_plan.id : ctx.makeId("plan"),
          name: "Manual path",
          steps: acceptedSteps,
          metadata: state.spec.contraction_plan && state.spec.contraction_plan.metadata
            ? ctx.deepClone(state.spec.contraction_plan.metadata)
            : {},
        };
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
      },
      {
        statusMessage:
          mode === "automatic_global"
            ? "Replaced the manual path with the global automatic path."
            : "Replaced the manual path with the local automatic path.",
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

  function renderAutomaticSection(title, disclosureKey, mode, analysis) {
    const isOpen = Boolean(state.plannerDisclosureState[disclosureKey]);
    const canAct = Boolean(analysis && analysis.status !== "unavailable");
    const summary = analysis && analysis.summary ? analysis.summary : {};
    const meta = analysis && analysis.message
      ? `<p class="planner-inline-meta">${ctx.escapeHtml(analysis.message)}</p>`
      : "";
    return `
      <section class="planner-section planner-disclosure">
        <button
          type="button"
          class="planner-disclosure-toggle${isOpen ? " is-open" : ""}"
          data-disclosure="${ctx.escapeHtml(disclosureKey)}"
        >
          <span>${ctx.escapeHtml(title)}</span>
          <strong>${isOpen ? "Hide" : "Show"}</strong>
        </button>
        ${isOpen ? `
          <div class="planner-disclosure-body">
            ${renderMetricChips([
              { label: "FLOPs", value: formatNumber(summary.total_estimated_flops) },
              { label: "MACs", value: formatNumber(summary.total_estimated_macs) },
              { label: "Peak", value: formatNumber(summary.peak_intermediate_size) },
            ])}
            ${meta}
            <div class="button-row">
              <button type="button" data-preview-mode="${ctx.escapeHtml(mode)}"${canAct ? "" : " disabled"}>Preview</button>
              <button type="button" class="apply-button" data-accept-mode="${ctx.escapeHtml(mode)}"${canAct ? "" : " disabled"}>Accept</button>
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
          { label: "FLOPs", value: formatNumber(manualAnalysis.summary && manualAnalysis.summary.total_estimated_flops) },
          { label: "MACs", value: formatNumber(manualAnalysis.summary && manualAnalysis.summary.total_estimated_macs) },
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
    return steps
      .map(
        (step, index) => `
          <article class="planner-step">
            <div class="planner-step-header">
              <strong>Step ${index + 1}</strong>
              <button type="button" class="planner-trim-button" data-trim-step="${index}">Trim Here</button>
            </div>
            <p>${ctx.escapeHtml(getPlannerOperandLabel(step.left_operand_id))} × ${ctx.escapeHtml(getPlannerOperandLabel(step.right_operand_id))}</p>
            <div class="planner-step-meta">
              <span>Shape ${ctx.escapeHtml(formatShape(step.result_shape))}</span>
              <span>FLOPs ${formatNumber(step.estimated_flops)}</span>
              <span>MACs ${formatNumber(step.estimated_macs)}</span>
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
    return `
      <section class="planner-section">
        <p class="planner-network-output-label">Network output shape</p>
        <p class="planner-network-output">${ctx.escapeHtml(formatShape(payload.network_output_shape))}</p>
      </section>
      <div class="planner-summary-grid">
        ${renderAutomaticSection(
          "Automatic global",
          "automaticGlobal",
          "automatic_global",
          payload.automatic_global
        )}
        ${renderAutomaticSection(
          "Automatic local",
          "automaticLocal",
          "automatic_local",
          payload.automatic_local
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
        <button id="toggle-planner-mode-button" type="button"${state.plannerMode ? ' class="is-active"' : ""}>
          Contract
        </button>
        <button
          id="planner-reset-button"
          type="button"
          class="icon-button planner-icon-button"
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
  });
}
