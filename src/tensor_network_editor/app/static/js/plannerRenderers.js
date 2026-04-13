export function createPlannerRenderers({
  ctx,
  state,
  plannerPanel,
  plannerDocument,
  support,
}) {
  const {
    syncPlannerOrderBadges,
    getPlannerOperandLabel,
    getAutomaticAnalysisByMode,
    togglePlannerDisclosure,
    trimContractionPlan,
    togglePlannerMode,
    startAutomaticPreview,
    acceptAutomaticPlan,
    clearAutomaticPreview,
  } = support;

  function formatShape(shape) {
    if (!Array.isArray(shape) || !shape.length) {
      return "scalar";
    }
    return shape.join(" \u00d7 ");
  }

  function formatNumber(value) {
    return Number(value || 0).toLocaleString();
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
                ${
                  item.detail
                    ? `<small class="planner-chip-detail">${ctx.escapeHtml(String(item.detail))}</small>`
                    : ""
                }
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function normalizeShapeDimension(value) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
      return null;
    }
    return BigInt(Math.max(1, Math.round(numericValue)));
  }

  function getShapeElementCount(shape) {
    if (!Array.isArray(shape)) {
      return null;
    }
    return shape.reduce((product, dimension) => {
      const normalizedDimension = normalizeShapeDimension(dimension);
      if (normalizedDimension === null) {
        return product;
      }
      return product * normalizedDimension;
    }, 1n);
  }

  function formatShapeElementCount(shape) {
    const elementCount = getShapeElementCount(shape);
    return elementCount === null ? "" : elementCount.toString();
  }

  function renderShapeElementDetail(shape) {
    const formattedElementCount = formatShapeElementCount(shape);
    return formattedElementCount
      ? `Total elements ${formattedElementCount}`
      : "";
  }

  function getAnalysisMemoryDtype(payload) {
    if (payload && typeof payload.memory_dtype === "string" && payload.memory_dtype) {
      return payload.memory_dtype;
    }
    return "float64";
  }

  function getMemoryBytesPerElement(memoryDtype) {
    switch (memoryDtype) {
      case "float16":
        return 2;
      case "float32":
        return 4;
      case "complex64":
        return 8;
      case "complex128":
        return 16;
      case "float64":
      default:
        return 8;
    }
  }

  function getPeakMemoryBytes(summary, memoryDtype) {
    if (!summary || typeof summary !== "object") {
      return 0;
    }
    if (Number.isFinite(Number(summary.peak_intermediate_bytes))) {
      return Number(summary.peak_intermediate_bytes);
    }
    return (
      Number(summary.peak_intermediate_size || 0) *
      getMemoryBytesPerElement(memoryDtype)
    );
  }

  function formatBytes(value) {
    return `${formatNumber(value)} bytes`;
  }

  function formatSignedDelta(value, unit = "") {
    const numericValue = Number(value || 0);
    if (!Number.isFinite(numericValue)) {
      return unit ? `0 ${unit}` : "0";
    }
    const prefix = numericValue > 0 ? "+" : numericValue < 0 ? "-" : "";
    const suffix = unit ? ` ${unit}` : "";
    return `${prefix}${formatNumber(Math.abs(numericValue))}${suffix}`;
  }

  function renderComparisonBody(comparison) {
    if (!comparison) {
      return "";
    }
    const status = typeof comparison.status === "string" ? comparison.status : "unknown";
    if (status !== "complete") {
      const unavailableMessage =
        typeof comparison.message === "string" && comparison.message
          ? comparison.message
          : "Comparison is not available for the current plan.";
      return `<p class="planner-inline-meta">${ctx.escapeHtml(unavailableMessage)}</p>`;
    }
    return `
      ${renderMetricChips([
        {
          label: "FLOP",
          value: formatSignedDelta(comparison.delta_total_estimated_flops),
          detail: "Auto - Manual",
        },
        {
          label: "MAC",
          value: formatSignedDelta(comparison.delta_total_estimated_macs),
          detail: "Auto - Manual",
        },
        {
          label: "Peak",
          value: formatSignedDelta(comparison.delta_peak_intermediate_size),
          detail: "Auto - Manual",
        },
        {
          label: "Memory",
          value: formatSignedDelta(comparison.delta_peak_intermediate_bytes, "bytes"),
          detail: "Auto - Manual",
        },
      ])}
    `;
  }

  function renderComparisonDisclosure(title, disclosureKey, comparison) {
    if (!title || !disclosureKey || !comparison) {
      return "";
    }
    const isOpen = Boolean(state.plannerDisclosureState[disclosureKey]);
    return `
      <div class="planner-nested-disclosure">
        <button
          type="button"
          class="planner-disclosure-toggle planner-nested-disclosure-toggle${isOpen ? " is-open" : ""}"
          data-disclosure="${ctx.escapeHtml(disclosureKey)}"
        >
          <span>${ctx.escapeHtml(title)}</span>
          <strong>${isOpen ? "Hide" : "Show"}</strong>
        </button>
        ${
          isOpen
            ? `
              <div class="planner-disclosure-body planner-nested-disclosure-body">
                ${renderComparisonBody(comparison)}
              </div>
            `
            : ""
        }
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

  function renderAutomaticSection(
    title,
    disclosureKey,
    mode,
    analysis,
    memoryDtype,
    options = {}
  ) {
    const isOpen = Boolean(state.plannerDisclosureState[disclosureKey]);
    const hasActions = typeof mode === "string" && mode;
    const canAct = Boolean(hasActions && analysis && analysis.status !== "unavailable");
    const summary = analysis && analysis.summary ? analysis.summary : {};
    const isPreviewing = hasActions && state.plannerPreviewMode === mode;
    const previewShortcut = mode === "automaticFuture" ? "A" : "Shift+A";
    const acceptShortcut = mode === "automaticFuture" ? "Ctrl+A" : "Ctrl+Shift+A";
    const meta =
      analysis && analysis.message
        ? `<p class="planner-inline-meta">${ctx.escapeHtml(analysis.message)}</p>`
        : "";
    const comparisonDisclosure = renderComparisonDisclosure(
      options.comparisonTitle,
      options.comparisonDisclosureKey,
      options.comparison
    );
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
              {
                label: "Memory",
                value: formatBytes(getPeakMemoryBytes(summary, memoryDtype)),
                detail: memoryDtype,
              },
            ])}
            ${comparisonDisclosure}
            ${
              isPreviewing
                ? `<p class="planner-inline-meta">Preview active.</p>${renderAutomaticPreviewStepList(
                    analysis && analysis.steps
                  )}`
                : ""
            }
            ${meta}
            ${
              hasActions
                ? `
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
                `
                : ""
            }
          </div>
        ` : ""}
      </section>
    `;
  }

  function renderManualSection(manualAnalysis, memoryDtype) {
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
          {
            label: "Memory",
            value: formatBytes(getPeakMemoryBytes(manualAnalysis.summary, memoryDtype)),
            detail: memoryDtype,
          },
          {
            label: "Shape",
            value: formatShape(manualAnalysis.summary && manualAnalysis.summary.final_shape),
            detail: renderShapeElementDetail(manualAnalysis.summary && manualAnalysis.summary.final_shape),
          },
        ])}
        <div class="planner-step-list planner-manual-step-list">
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
            <p>${ctx.escapeHtml(getPlannerOperandLabel(step.left_operand_id))} &times; ${ctx.escapeHtml(getPlannerOperandLabel(step.right_operand_id))}</p>
            <div class="planner-step-meta">
              <span>Shape ${ctx.escapeHtml(formatShape(step.result_shape))}</span>
              <span>FLOP ${formatNumber(step.estimated_flops)}</span>
              <span>MAC ${formatNumber(step.estimated_macs)}</span>
            </div>
            ${
              renderShapeElementDetail(step.result_shape)
                ? `<div class="planner-step-detail">${ctx.escapeHtml(renderShapeElementDetail(step.result_shape))}</div>`
                : ""
            }
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
    const memoryDtype = getAnalysisMemoryDtype(payload);
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
      <div class="planner-summary-grid">
        ${renderAutomaticSection(
          "Auto full",
          "automaticFull",
          null,
          payload.automatic_full,
          memoryDtype,
          {
            comparisonTitle: "Manual vs auto full",
            comparisonDisclosureKey: "automaticFullComparison",
            comparison: payload.comparisons && payload.comparisons.manual_vs_automatic_full,
          }
        )}
        ${renderAutomaticSection(
          "Auto future",
          "automaticFuture",
          "automaticFuture",
          payload.automatic_future,
          memoryDtype
        )}
        ${renderAutomaticSection(
          "Auto past",
          "automaticPast",
          "automaticPast",
          payload.automatic_past,
          memoryDtype,
          {
            comparisonTitle: "Manual contractions vs auto past",
            comparisonDisclosureKey: "automaticPastComparison",
            comparison:
              payload.comparisons && payload.comparisons.manual_subtrees_vs_automatic_past,
          }
        )}
      </div>
      ${renderManualSection(payload.manual, memoryDtype)}
    `;
  }

  function renderPlanner() {
    if (!plannerPanel) {
      return;
    }
    syncPlannerOrderBadges();
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

    plannerDocument
      ?.getElementById("toggle-planner-mode-button")
      ?.addEventListener("click", togglePlannerMode);
    plannerDocument
      ?.getElementById("planner-reset-button")
      ?.addEventListener("click", () => trimContractionPlan(0));
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

  return {
    renderPlanner,
    formatShape,
    formatNumber,
    getShapeElementCount,
  };
}
