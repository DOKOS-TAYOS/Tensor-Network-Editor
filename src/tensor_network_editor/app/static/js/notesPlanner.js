export function registerNotesPlanner(ctx) {
  const state = ctx.state;
  const {
    addNoteButton,
    notesLayer,
    plannerPanel,
  } = ctx.dom;

  function createNote(x, y) {
    return {
      id: ctx.makeId("note"),
      text: "New note",
      position: { x, y },
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
      const noteElement = document.createElement("article");
      noteElement.className = "canvas-note";
      noteElement.dataset.noteId = note.id;
      if (state.selectionIds.includes(note.id)) {
        noteElement.classList.add("is-selected");
      }
      noteElement.style.left = `${canvasPoint.x}px`;
      noteElement.style.top = `${canvasPoint.y}px`;
      noteElement.style.borderColor = ctx.getMetadataColor(note.metadata, "#5f95ff");

      const header = document.createElement("div");
      header.className = "canvas-note-header";
      header.textContent = "Note";
      header.addEventListener("mousedown", (event) => startNoteDrag(event, note.id));
      header.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
      });

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "canvas-note-delete";
      deleteButton.textContent = "×";
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
      header.appendChild(deleteButton);

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

      noteElement.appendChild(header);
      noteElement.appendChild(textarea);
      noteElement.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        ctx.selectElement("note", note.id, { additive: Boolean(event.shiftKey) });
      });
      notesLayer.appendChild(noteElement);
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

  function repairContractionPlan() {
    const plan = state.spec.contraction_plan;
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      return;
    }
    const availableOperandIds = new Set(state.spec.tensors.map((tensor) => tensor.id));
    const repairedSteps = [];
    for (const step of plan.steps) {
      if (
        !step ||
        !step.id ||
        step.left_operand_id === step.right_operand_id ||
        !availableOperandIds.has(step.left_operand_id) ||
        !availableOperandIds.has(step.right_operand_id) ||
        availableOperandIds.has(step.id)
      ) {
        break;
      }
      repairedSteps.push(step);
      availableOperandIds.delete(step.left_operand_id);
      availableOperandIds.delete(step.right_operand_id);
      availableOperandIds.add(step.id);
    }
    if (!repairedSteps.length) {
      state.spec.contraction_plan = null;
      return;
    }
    plan.steps = repairedSteps;
  }

  function getPlannerRemainingOperandIds() {
    if (
      state.contractionAnalysis &&
      state.contractionAnalysis.status === "ready" &&
      state.contractionAnalysis.payload &&
      state.contractionAnalysis.payload.manual &&
      state.contractionAnalysis.payload.manual.summary
    ) {
      return Array.isArray(state.contractionAnalysis.payload.manual.summary.remaining_operand_ids)
        ? [...state.contractionAnalysis.payload.manual.summary.remaining_operand_ids]
        : [];
    }
    return state.spec.tensors.map((tensor) => tensor.id);
  }

  function isPlannerOperandAvailable(operandId) {
    return getPlannerRemainingOperandIds().includes(operandId);
  }

  function getPlannerOperandLabel(operandId) {
    const tensor = ctx.findTensorById(operandId);
    if (tensor) {
      return tensor.name;
    }
    const manualSteps =
      state.contractionAnalysis &&
      state.contractionAnalysis.status === "ready" &&
      state.contractionAnalysis.payload &&
      state.contractionAnalysis.payload.manual
        ? state.contractionAnalysis.payload.manual.steps
        : [];
    const stepIndex = manualSteps.findIndex(
      (step) => step.result_operand_id === operandId
    );
    if (stepIndex >= 0) {
      return `Intermediate ${stepIndex + 1}`;
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
    if (!isPlannerOperandAvailable(operandId)) {
      ctx.setStatus("That operand is not available for the next manual contraction step.", "error");
      return;
    }
    if (!state.pendingPlannerOperandId) {
      state.pendingPlannerOperandId = operandId;
      renderPlanner();
      ctx.setStatus(`Selected ${getPlannerOperandLabel(operandId)} as the first manual operand.`);
      return;
    }
    if (state.pendingPlannerOperandId === operandId) {
      state.pendingPlannerOperandId = null;
      renderPlanner();
      ctx.setStatus("Manual planner operand selection cleared.");
      return;
    }
    const leftOperandId = state.pendingPlannerOperandId;
    const rightOperandId = operandId;
    state.pendingPlannerOperandId = null;
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
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("planner");
    }
    renderPlanner();
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
  }

  function formatShape(shape) {
    if (!Array.isArray(shape) || !shape.length) {
      return "scalar";
    }
    return shape.join(" × ");
  }

  function renderSummaryCard(title, analysis) {
    if (!analysis) {
      return `<section class="planner-summary-card"><h3>${ctx.escapeHtml(title)}</h3><p class="planner-inline-meta">Waiting for analysis.</p></section>`;
    }
    if (analysis.status === "unavailable") {
      return `
        <section class="planner-summary-card">
          <h3>${ctx.escapeHtml(title)}</h3>
          <p class="planner-inline-meta">${ctx.escapeHtml(analysis.message || "Unavailable")}</p>
        </section>
      `;
    }
    const summary = analysis.summary || {};
    return `
      <section class="planner-summary-card">
        <h3>${ctx.escapeHtml(title)}</h3>
        <div class="planner-chip-grid">
          <div class="planner-chip"><span>Status</span><strong>${ctx.escapeHtml(analysis.status || summary.completion_status || "unknown")}</strong></div>
          <div class="planner-chip"><span>FLOPs</span><strong>${summary.total_estimated_flops ?? 0}</strong></div>
          <div class="planner-chip"><span>Peak</span><strong>${summary.peak_intermediate_size ?? 0}</strong></div>
          <div class="planner-chip"><span>Final shape</span><strong>${ctx.escapeHtml(formatShape(summary.final_shape))}</strong></div>
        </div>
      </section>
    `;
  }

  function renderManualSteps(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return `<p class="planner-inline-meta">No manual steps yet. Turn on manual mode and click two tensors to create the first contraction.</p>`;
    }
    return `
      <div class="planner-step-list">
        ${steps
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
                  <span>Cost ${step.estimated_flops}</span>
                  <span>Intermediate ${step.intermediate_size}</span>
                </div>
              </article>
            `
          )
          .join("")}
      </div>
    `;
  }

  function renderIntermediateCards(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return "";
    }
    return `
      <section class="planner-section">
        <h3>Intermediates</h3>
        <div class="planner-intermediate-list">
          ${steps
            .map(
              (step, index) => `
                <button
                  type="button"
                  class="planner-intermediate-card${state.pendingPlannerOperandId === step.result_operand_id ? " is-selected" : ""}"
                  data-operand-id="${ctx.escapeHtml(step.result_operand_id)}"
                >
                  <strong>Intermediate ${index + 1}</strong>
                  <span>${ctx.escapeHtml(formatShape(step.result_shape))}</span>
                </button>
              `
            )
            .join("")}
        </div>
      </section>
    `;
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
      <div class="planner-summary-grid">
        ${renderSummaryCard("Manual", payload.manual)}
        ${renderSummaryCard("Automatic", payload.automatic)}
      </div>
      <section class="planner-section">
        <h3>Manual Steps</h3>
        ${renderManualSteps(payload.manual.steps)}
      </section>
      ${renderIntermediateCards(payload.manual.steps)}
    `;
  }

  function renderPlanner() {
    if (!plannerPanel) {
      return;
    }
    const planSteps = state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
      ? state.spec.contraction_plan.steps
      : [];
    const pendingLabel = state.pendingPlannerOperandId
      ? getPlannerOperandLabel(state.pendingPlannerOperandId)
      : null;
    plannerPanel.innerHTML = `
      <div class="planner-toolbar">
        <button id="toggle-planner-mode-button" type="button"${state.plannerMode ? ' class="is-active"' : ""}>
          ${state.plannerMode ? "Exit Manual Mode" : "Enter Manual Mode"}
        </button>
        <button id="refresh-planner-button" type="button">Refresh</button>
      </div>
      <div class="planner-toolbar">
        <button id="planner-remove-last-button" type="button"${planSteps.length ? "" : " disabled"}>Remove Last</button>
        <button id="planner-reset-button" type="button"${planSteps.length ? "" : " disabled"}>Reset Path</button>
      </div>
      <p class="planner-inline-meta">
        ${pendingLabel ? `Pending operand: ${ctx.escapeHtml(pendingLabel)}.` : "Manual mode uses clicks on tensors and intermediate cards."}
      </p>
      ${renderPlannerAnalysis()}
    `;

    document
      .getElementById("toggle-planner-mode-button")
      .addEventListener("click", togglePlannerMode);
    document
      .getElementById("refresh-planner-button")
      .addEventListener("click", () => refreshContractionAnalysis({ focusTab: true }));
    document
      .getElementById("planner-remove-last-button")
      .addEventListener("click", () => trimContractionPlan(planSteps.length - 1));
    document
      .getElementById("planner-reset-button")
      .addEventListener("click", () => trimContractionPlan(0));
    plannerPanel.querySelectorAll("[data-trim-step]").forEach((button) => {
      button.addEventListener("click", () => {
        trimContractionPlan(Number(button.dataset.trimStep));
      });
    });
    plannerPanel.querySelectorAll("[data-operand-id]").forEach((button) => {
      button.addEventListener("click", () => {
        handlePlannerOperandClick(button.dataset.operandId);
      });
    });
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
    copySelectedSubgraphToClipboard,
    pasteClipboardToCanvas,
    repairContractionPlan,
    ensureContractionPlan,
    getPlannerRemainingOperandIds,
    isPlannerOperandAvailable,
    getPlannerOperandLabel,
    handlePlannerOperandClick,
    trimContractionPlan,
    togglePlannerMode,
    refreshContractionAnalysis,
    renderPlanner,
  });
}
