export function createPlannerAutomaticRendererSupport({
  ctx,
  state,
  common,
  getPlannerOperandLabel,
  formatters,
}) {
  const { buildTooltipAttributes, METRIC_DESCRIPTIONS, renderDisclosureState, renderMetricChips } =
    common;
  const { formatBytes, formatNumber, formatSignedDelta, getPeakMemoryBytes } =
    formatters;
  const AUTO_PAST_UNLOCK_MESSAGE =
    "Contract at least one tensor pair to unlock the auto past preview.";

  function isAutoPastUnlockMessage(message) {
    return (
      typeof message === "string" &&
      message.trim() === AUTO_PAST_UNLOCK_MESSAGE
    );
  }

  function renderComparisonBody(comparison, options = {}) {
    if (!comparison) {
      return "";
    }
    const status = typeof comparison.status === "string" ? comparison.status : "unknown";
    if (status !== "complete") {
      if (
        options.hideUnavailableMessage &&
        isAutoPastUnlockMessage(comparison.message)
      ) {
        return "";
      }
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
          description: METRIC_DESCRIPTIONS.FLOP,
        },
        {
          label: "MAC",
          value: formatSignedDelta(comparison.delta_total_estimated_macs),
          detail: "Auto - Manual",
          description: METRIC_DESCRIPTIONS.MAC,
        },
        {
          label: "Peak",
          value: formatSignedDelta(comparison.delta_peak_intermediate_size),
          detail: "Auto - Manual",
          description: METRIC_DESCRIPTIONS.Peak,
        },
        {
          label: "Memory",
          value: formatSignedDelta(comparison.delta_peak_intermediate_bytes, "bytes"),
          detail: "Auto - Manual",
          description: METRIC_DESCRIPTIONS.Memory,
        },
      ])}
    `;
  }

  function renderComparisonDisclosure(title, disclosureKey, comparison, options = {}) {
    if (!title || !disclosureKey || !comparison) {
      return "";
    }
    const isOpen = Boolean(state.plannerDisclosureState[disclosureKey]);
    const description =
      disclosureKey === "automaticFullComparison"
        ? "Compares the current manual path against the full automatic contraction path."
        : "Compares the already contracted manual subtrees against the automatic replanning of that past work.";
    return `
      <div class="planner-nested-disclosure">
        <button
          type="button"
          class="planner-disclosure-toggle planner-nested-disclosure-toggle${isOpen ? " is-open" : ""}"
          data-disclosure="${ctx.escapeHtml(disclosureKey)}"
          ${buildTooltipAttributes(title, description)}
        >
          <span>${ctx.escapeHtml(title)}</span>
          ${renderDisclosureState(isOpen)}
        </button>
        ${
          isOpen
            ? `
              <div class="planner-disclosure-body planner-nested-disclosure-body">
                ${renderComparisonBody(comparison, options)}
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
    const previewShortcut = mode === "automaticFuture" ? "Alt+A" : "Shift+A";
    const acceptShortcut =
      mode === "automaticFuture" ? "Ctrl/Cmd+Alt+A" : "Ctrl/Cmd+Shift+A";
    const hideUnavailableMessage = Boolean(
      options.hideUnavailableMessage &&
      analysis &&
      analysis.status === "unavailable" &&
      isAutoPastUnlockMessage(analysis.message)
    );
    const meta =
      analysis && analysis.message && !hideUnavailableMessage
        ? `<p class="planner-inline-meta">${ctx.escapeHtml(analysis.message)}</p>`
        : "";
    const comparisonDisclosure = renderComparisonDisclosure(
      options.comparisonTitle,
      options.comparisonDisclosureKey,
      options.comparison,
      {
        hideUnavailableMessage: Boolean(options.hideUnavailableComparisonMessage),
      }
    );
    const sectionDescription =
      mode === "automaticFuture"
        ? "Plans the remaining visible operands from the current manual path onward."
        : mode === "automaticPast"
          ? "Replans tensors that are already merged inside the current manual contractions."
          : "Computes a full automatic contraction path for the whole visible network.";
    return `
      <section class="planner-section planner-disclosure">
        <button
          type="button"
          class="planner-disclosure-toggle button-accent-contraction${isOpen ? " is-open" : ""}"
          data-disclosure="${ctx.escapeHtml(disclosureKey)}"
          ${buildTooltipAttributes(title, sectionDescription)}
        >
          <span>${ctx.escapeHtml(title)}</span>
          ${renderDisclosureState(isOpen)}
        </button>
        ${isOpen ? `
          <div class="planner-disclosure-body">
            ${renderMetricChips([
              {
                label: "FLOP",
                value: formatNumber(summary.total_estimated_flops),
                description: METRIC_DESCRIPTIONS.FLOP,
              },
              {
                label: "MAC",
                value: formatNumber(summary.total_estimated_macs),
                description: METRIC_DESCRIPTIONS.MAC,
              },
              {
                label: "Peak",
                value: formatNumber(summary.peak_intermediate_size),
                description: METRIC_DESCRIPTIONS.Peak,
              },
              {
                label: "Memory",
                value: formatBytes(getPeakMemoryBytes(summary, memoryDtype)),
                detail: memoryDtype,
                description: METRIC_DESCRIPTIONS.Memory,
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
                      class="button-accent-contraction${isPreviewing ? " is-active" : ""}"
                      data-preview-mode="${ctx.escapeHtml(mode)}"
                      data-shortcut="${ctx.escapeHtml(previewShortcut)}"
                      data-shortcut-label="${ctx.escapeHtml(isPreviewing ? "Deactivate preview" : "Preview")}"
                      data-tooltip-enabled="true"
                      data-shortcut-description="Toggle a non-destructive preview of this automatic path on the canvas."
                      aria-pressed="${isPreviewing}"
                      ${canAct ? "" : " disabled"}
                    >
                      ${isPreviewing ? "Deactivate preview" : "Preview"}
                    </button>
                    <button
                      type="button"
                      class="button-accent-contraction"
                      data-accept-mode="${ctx.escapeHtml(mode)}"
                      data-shortcut="${ctx.escapeHtml(acceptShortcut)}"
                      data-shortcut-label="Accept"
                      data-tooltip-enabled="true"
                      data-shortcut-description="Replace the current manual path with this automatic contraction plan."
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

  function getMissingOptEinsumMessage(payload) {
    if (!payload) {
      return "";
    }
    const automaticAnalyses = [
      payload.automatic_full,
      payload.automatic_future,
      payload.automatic_past,
    ];
    const matchingAnalysis = automaticAnalyses.find((analysis) => {
      const message = typeof analysis?.message === "string" ? analysis.message : "";
      return /opt_einsum/i.test(message);
    });
    return matchingAnalysis && typeof matchingAnalysis.message === "string"
      ? matchingAnalysis.message
      : "";
  }

  return {
    renderComparisonBody,
    renderComparisonDisclosure,
    renderAutomaticPreviewStepList,
    renderAutomaticSection,
    getMissingOptEinsumMessage,
  };
}
