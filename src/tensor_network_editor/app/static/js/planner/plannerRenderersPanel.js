export function createPlannerPanelRendererSupport({
  ctx,
  state,
  plannerPanel,
  plannerPanelBindings,
  actions,
  syncPlannerOrderBadges,
  getPlannerOperandLabel,
  isBenchmarkBasePosition,
  automaticRenderer,
  manualRenderer,
  formatters,
}) {
  const { formatShape, getAnalysisMemoryDtype, renderShapeElementDetail } = formatters;

  function renderPlannerAnalysis() {
    if (
      Array.isArray(state.spec?.hyperedges) &&
      state.spec.hyperedges.length &&
      (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready")
    ) {
      return `<p class="planner-inline-meta">${ctx.escapeHtml(
        state.contractionAnalysis?.message ||
          "Manual contraction planning is unavailable while the design contains hyperedges."
      )}</p>`;
    }
    if (!state.contractionAnalysis || state.contractionAnalysis.status === "loading") {
      return `<p class="planner-inline-meta">Analyzing contraction paths...</p>`;
    }
    if (state.contractionAnalysis.status === "benchmarkBase") {
      return `<p class="planner-inline-meta">Preparing benchmark scheme analysis...</p>`;
    }
    if (state.contractionAnalysis.status === "gridPeriodicDisabled") {
      return `<p class="planner-inline-meta">${ctx.escapeHtml(state.contractionAnalysis.message || "Contractions are disabled in For bidimensional mode.")}</p>`;
    }
    if (state.contractionAnalysis.status === "treePeriodicDisabled") {
      return `<p class="planner-inline-meta">${ctx.escapeHtml(state.contractionAnalysis.message || "Contractions are disabled in For Tree mode.")}</p>`;
    }
    if (state.contractionAnalysis.status === "hyperedgesDisabled") {
      return `<p class="planner-inline-meta">${ctx.escapeHtml(state.contractionAnalysis.message || "Manual contraction planning is unavailable while the design contains hyperedges.")}</p>`;
    }
    if (state.contractionAnalysis.status === "issues") {
      return `<p class="planner-inline-meta planner-error">${ctx.escapeHtml(ctx.formatIssues(state.contractionAnalysis.issues || []))}</p>`;
    }
    if (state.contractionAnalysis.status === "error") {
      return `<p class="planner-inline-meta planner-error">${ctx.escapeHtml(state.contractionAnalysis.message || "Could not analyze contraction paths.")}</p>`;
    }
    const payload = state.contractionAnalysis.payload;
    if (!payload) {
      return `<p class="planner-inline-meta">Analyzing contraction paths...</p>`;
    }
    const memoryDtype = getAnalysisMemoryDtype(payload);
    const missingOptEinsumMessage = automaticRenderer.getMissingOptEinsumMessage(payload);
    return `
      <section class="planner-section">
        <p class="planner-network-output-label">Network output shape</p>
        <p class="planner-network-output">${ctx.escapeHtml(formatShape(payload.network_output_shape))}</p>
        ${
          renderShapeElementDetail(payload.network_output_shape)
            ? `<p class="planner-shape-detail">${ctx.escapeHtml(renderShapeElementDetail(payload.network_output_shape))}</p>`
            : ""
        }
      </section>
      ${
        missingOptEinsumMessage
          ? `<section class="planner-section"><p class="planner-inline-meta planner-error">${ctx.escapeHtml(
              missingOptEinsumMessage
            )}</p></section>`
          : `
            <div class="planner-summary-grid">
              ${automaticRenderer.renderAutomaticSection(
                "Auto full",
                "automaticFull",
                null,
                payload.automatic_full,
                memoryDtype,
                {
                  comparisonTitle: "Manual vs auto full",
                  comparisonDisclosureKey: "automaticFullComparison",
                  comparison:
                    payload.comparisons && payload.comparisons.manual_vs_automatic_full,
                }
              )}
              ${automaticRenderer.renderAutomaticSection(
                "Auto future",
                "automaticFuture",
                "automaticFuture",
                payload.automatic_future,
                memoryDtype
              )}
              ${automaticRenderer.renderAutomaticSection(
                "Auto past",
                "automaticPast",
                "automaticPast",
                payload.automatic_past,
                memoryDtype,
                {
                  comparisonTitle: "Manual contractions vs auto past",
                  comparisonDisclosureKey: "automaticPastComparison",
                  comparison:
                    payload.comparisons &&
                    payload.comparisons.manual_subtrees_vs_automatic_past,
                }
              )}
            </div>
          `
      }
      ${manualRenderer.renderManualSection(payload.manual, memoryDtype)}
    `;
  }

  function renderPlanner() {
    const hasHyperedges = Boolean(
      Array.isArray(state.spec?.hyperedges) && state.spec.hyperedges.length
    );
    if (!plannerPanel) {
      return;
    }
    syncPlannerOrderBadges();
    if (typeof isBenchmarkBasePosition === "function" && isBenchmarkBasePosition()) {
      plannerPanel.innerHTML = `
        <div class="planner-toolbar">
        <button
          id="toggle-planner-mode-button"
          type="button"
          class="button-accent-cool"
          data-shortcut="M"
          data-shortcut-label="Contract"
          data-tooltip-enabled="true"
          data-shortcut-description="Toggle manual contraction mode, then click two tensors or intermediate results to add a step."
          disabled
        >
          Contract
        </button>
          <button
            id="planner-reset-button"
            type="button"
          class="icon-button planner-icon-button danger"
          data-shortcut="Shift+R"
          data-shortcut-label="Reset path"
          data-shortcut-description="Remove all manual steps from the current contraction path."
          aria-label="Reset path"
          disabled
        >
            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
            </svg>
          </button>
        </div>
        <section class="planner-section">
          <h3>Benchmark</h3>
          <p class="planner-inline-meta">
            Move right to open or create a contraction scheme.
          </p>
        </section>
      `;
      plannerPanelBindings.bindPlannerPanelInteractions();
      actions.renderOverlayDecorations();
      return;
    }
    const planSteps =
      state.spec.contraction_plan && Array.isArray(state.spec.contraction_plan.steps)
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
          data-shortcut-label="Contract"
          data-tooltip-enabled="true"
          data-shortcut-description="Toggle manual contraction mode, then click two tensors or intermediate results to add a step."
          ${hasHyperedges ? "disabled" : ""}
        >
          Contract
        </button>
        <button
          id="planner-reset-button"
          type="button"
          class="icon-button planner-icon-button danger"
          data-shortcut="Shift+R"
          data-shortcut-label="Reset path"
          data-shortcut-description="Remove all manual steps from the current contraction path."
          aria-label="Reset path"
          ${planSteps.length && !hasHyperedges ? "" : " disabled"}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
            <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
          </svg>
        </button>
      </div>
      ${pendingLabel ? `<p class="planner-inline-meta">Pending operand: ${ctx.escapeHtml(pendingLabel)}.</p>` : ""}
      ${renderPlannerAnalysis()}
    `;

    plannerPanelBindings.bindPlannerPanelInteractions();
    actions.renderOverlayDecorations();
  }

  return {
    renderPlannerAnalysis,
    renderPlanner,
  };
}
