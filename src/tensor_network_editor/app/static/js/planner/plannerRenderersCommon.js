export function createPlannerRendererCommonSupport({ ctx }) {
  const METRIC_DESCRIPTIONS = {
    FLOP: "Estimated floating-point operations across the full contraction path.",
    MAC: "Estimated multiply-accumulate operations across the full contraction path.",
    Peak: "Largest intermediate tensor reached during the path, measured in elements.",
    Memory:
      "Estimated memory used by the largest intermediate tensor for the reported dtype.",
  };

  function buildTooltipAriaLabel(label, shortcut = "", description = "") {
    if (!label) {
      return description;
    }
    const header = shortcut ? `${label} (${shortcut})` : label;
    return description ? `${header}. ${description}` : header;
  }

  function buildTooltipAttributes(label, description = "", shortcut = "") {
    const attributes = ['data-tooltip-enabled="true"'];
    if (label) {
      attributes.push(`data-shortcut-label="${ctx.escapeHtml(label)}"`);
    }
    if (shortcut) {
      attributes.push(`data-shortcut="${ctx.escapeHtml(shortcut)}"`);
    }
    if (description) {
      attributes.push(
        `data-shortcut-description="${ctx.escapeHtml(description)}"`
      );
    }
    const ariaLabel = buildTooltipAriaLabel(label, shortcut, description);
    if (ariaLabel) {
      attributes.push(`aria-label="${ctx.escapeHtml(ariaLabel)}"`);
    }
    return attributes.join(" ");
  }

  function renderDisclosureState(isOpen) {
    return `
      <strong class="planner-disclosure-state ${
        isOpen
          ? "planner-disclosure-state-hide"
          : "planner-disclosure-state-show"
      }">${isOpen ? "Hide" : "Show"}</strong>
    `;
  }

  function renderMetricLabel(label, description = "") {
    return `
      <span class="planner-chip-label">
        <span>${ctx.escapeHtml(label)}</span>
        ${
          description
            ? `
              <span
                class="planner-chip-info"
                tabindex="0"
                ${buildTooltipAttributes(label, description)}
              >?</span>
            `
            : ""
        }
      </span>
    `;
  }

  function getPlannerChipValueClass(valueTone = "") {
    switch (valueTone) {
      case "better":
        return "planner-chip-value planner-chip-value-better";
      case "worse":
        return "planner-chip-value planner-chip-value-worse";
      default:
        return "planner-chip-value";
    }
  }

  function renderMetricChips(items) {
    return `
      <div class="planner-chip-grid">
        ${items
          .map(
            (item) => `
              <div class="planner-chip">
                ${renderMetricLabel(item.label, item.description)}
                <strong class="${getPlannerChipValueClass(item.valueTone)}">${ctx.escapeHtml(
                  String(item.value)
                )}</strong>
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

  return {
    METRIC_DESCRIPTIONS,
    buildTooltipAttributes,
    renderDisclosureState,
    renderMetricChips,
  };
}
