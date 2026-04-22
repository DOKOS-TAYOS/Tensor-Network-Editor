export function createPlannerManualRendererSupport({
  ctx,
  state,
  common,
  getPlannerOperandLabel,
  formatters,
}) {
  const { METRIC_DESCRIPTIONS, renderMetricChips } = common;
  const {
    formatBytes,
    formatNumber,
    formatShape,
    getPeakMemoryBytes,
    renderShapeElementDetail,
  } = formatters;

  function renderManualSection(manualAnalysis, memoryDtype) {
    if (!manualAnalysis) {
      return `<section class="planner-section"><h3>Manual</h3><p class="planner-inline-meta">Waiting for analysis.</p></section>`;
    }
    return `
      <section class="planner-section">
        <h3>Manual</h3>
        ${renderMetricChips([
          { label: "Status", value: manualAnalysis.status || "unknown" },
          {
            label: "FLOP",
            value: formatNumber(
              manualAnalysis.summary && manualAnalysis.summary.total_estimated_flops
            ),
            description: METRIC_DESCRIPTIONS.FLOP,
          },
          {
            label: "MAC",
            value: formatNumber(
              manualAnalysis.summary && manualAnalysis.summary.total_estimated_macs
            ),
            description: METRIC_DESCRIPTIONS.MAC,
          },
          {
            label: "Peak",
            value: formatNumber(
              manualAnalysis.summary && manualAnalysis.summary.peak_intermediate_size
            ),
            description: METRIC_DESCRIPTIONS.Peak,
          },
          {
            label: "Memory",
            value: formatBytes(getPeakMemoryBytes(manualAnalysis.summary, memoryDtype)),
            detail: memoryDtype,
            description: METRIC_DESCRIPTIONS.Memory,
          },
          {
            label: "Shape",
            value: formatShape(manualAnalysis.summary && manualAnalysis.summary.final_shape),
            detail: renderShapeElementDetail(
              manualAnalysis.summary && manualAnalysis.summary.final_shape
            ),
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

  return {
    renderManualSection,
    renderManualSteps,
  };
}
