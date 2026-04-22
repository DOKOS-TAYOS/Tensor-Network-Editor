function resolveTooltipTarget(eventTarget) {
  return eventTarget && typeof eventTarget.closest === "function"
    ? eventTarget.closest('[data-tooltip-enabled="true"]')
    : null;
}

export function createShortcutTooltip({ documentRef, windowRef }) {
  let tooltipNode = null;
  let activeButton = null;

  function readAttribute(element, attributeName) {
    if (!element) {
      return "";
    }
    if (typeof element.getAttribute === "function") {
      return String(element.getAttribute(attributeName) || "");
    }
    if (
      element.attributes
      && Object.prototype.hasOwnProperty.call(element.attributes, attributeName)
    ) {
      return String(element.attributes[attributeName] || "");
    }
    return "";
  }

  function readElementText(element, selector) {
    if (!element || typeof element.querySelector !== "function") {
      return "";
    }
    const target = element.querySelector(selector);
    return target && typeof target.textContent === "string"
      ? target.textContent.trim()
      : "";
  }

  function buildAriaLabel(label, shortcut, description) {
    if (!label) {
      return description;
    }
    const header = shortcut ? `${label} (${shortcut})` : label;
    return description ? `${header}. ${description}` : header;
  }

  function setTooltipData(button, label, shortcut = "", description = "") {
    if (!button || !button.dataset) {
      return;
    }
    button.dataset.tooltipEnabled = "true";
    if (label) {
      button.dataset.shortcutLabel = label;
    } else {
      delete button.dataset.shortcutLabel;
    }
    if (shortcut) {
      button.dataset.shortcut = shortcut;
    } else {
      delete button.dataset.shortcut;
    }
    if (description) {
      button.dataset.shortcutDescription = description;
    } else {
      delete button.dataset.shortcutDescription;
    }
    const ariaLabel = buildAriaLabel(label, shortcut, description);
    if (ariaLabel && typeof button.setAttribute === "function") {
      button.setAttribute("aria-label", ariaLabel);
    }
    if (typeof button.removeAttribute === "function") {
      button.removeAttribute("title");
    }
  }

  function applyShortcutHint(buttonId, label, shortcut = "", description = "") {
    const button = documentRef.getElementById(buttonId);
    if (!button) {
      return;
    }
    setTooltipData(button, label, shortcut, description);
  }

  function applyTitleHint(buttonId, { label = "", shortcut = "", description } = {}) {
    const button = documentRef.getElementById(buttonId);
    if (!button) {
      return;
    }
    const resolvedLabel =
      label
      || readElementText(button, ".toolbar-menu-item-label")
      || readAttribute(button, "aria-label").trim()
      || String(button.textContent || "").trim().replace(/\s+/g, " ");
    const resolvedShortcut =
      shortcut
      || readElementText(button, ".toolbar-menu-item-shortcut")
      || (button.dataset && typeof button.dataset.shortcut === "string"
        ? button.dataset.shortcut.trim()
        : "");
    const resolvedDescription =
      typeof description === "string"
        ? description
        : readElementText(button, ".toolbar-menu-item-description")
          || readAttribute(button, "title").trim()
          || (button.dataset && typeof button.dataset.shortcutDescription === "string"
            ? button.dataset.shortcutDescription.trim()
            : "");
    const normalizedDescription =
      resolvedDescription === resolvedLabel ? "" : resolvedDescription;
    setTooltipData(
      button,
      resolvedLabel,
      resolvedShortcut,
      normalizedDescription
    );
  }

  function ensureTooltipNode() {
    if (tooltipNode) {
      return tooltipNode;
    }
    tooltipNode = documentRef.createElement("div");
    tooltipNode.className = "shortcut-tooltip is-hidden";
    tooltipNode.setAttribute("aria-hidden", "true");
    documentRef.body.appendChild(tooltipNode);
    return tooltipNode;
  }

  function escapeTooltipText(value) {
    return String(value)
      .replaceAll("\r\n", "\n")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function getMetricValueToneClass(value) {
    if (typeof value !== "string" || !value) {
      return "";
    }
    if (value.startsWith("-")) {
      return " shortcut-tooltip-metric-value-better";
    }
    if (value.startsWith("+")) {
      return " shortcut-tooltip-metric-value-worse";
    }
    return "";
  }

  function buildTooltipDescriptionLineMarkup(line) {
    const text = String(line || "");
    const metricMatch = text.match(/^(.*?)([+-]\d[\d,]*(?: bytes)?)$/);
    if (!metricMatch) {
      return `<span class="shortcut-tooltip-description-line">${escapeTooltipText(
        text
      )}</span>`;
    }
    const label = metricMatch[1].trimEnd();
    const value = metricMatch[2];
    return `
      <span class="shortcut-tooltip-description-line">
        <span class="shortcut-tooltip-description-label">${escapeTooltipText(label)}</span>
        <span class="shortcut-tooltip-metric-value${getMetricValueToneClass(value)}">${escapeTooltipText(
          value
        )}</span>
      </span>
    `;
  }

  function buildTooltipDescriptionMarkup(description) {
    return String(description)
      .split("\n")
      .map((line) => buildTooltipDescriptionLineMarkup(line))
      .join("");
  }

  function buildTooltipMarkup(button) {
    const label = button.dataset.shortcutLabel || String(button.textContent || "").trim();
    const shortcut = button.dataset.shortcut || "";
    const description = button.dataset.shortcutDescription || "";
    const headerParts = [];
    if (label) {
      headerParts.push(
        `<span class="shortcut-tooltip-label">${escapeTooltipText(label)}</span>`
      );
    }
    if (shortcut) {
      headerParts.push(
        `<span class="shortcut-tooltip-shortcut">${escapeTooltipText(shortcut)}</span>`
      );
    }
    const sections = [];
    if (headerParts.length) {
      sections.push(
        `<span class="shortcut-tooltip-header">${headerParts.join("")}</span>`
      );
    }
    if (description) {
      sections.push(
        `<span class="shortcut-tooltip-description">${buildTooltipDescriptionMarkup(
          description
        )}</span>`
      );
    }
    return sections.join("");
  }

  function positionTooltip(button) {
    const tooltip = ensureTooltipNode();
    const rect = button.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const margin = 8;
    let left = rect.right - tooltipRect.width;
    let top = rect.bottom + margin;

    if (top + tooltipRect.height > windowRef.innerHeight - margin) {
      top = rect.top - tooltipRect.height - margin;
    }
    left = Math.min(
      Math.max(margin, left),
      Math.max(margin, windowRef.innerWidth - tooltipRect.width - margin)
    );
    top = Math.min(
      Math.max(margin, top),
      Math.max(margin, windowRef.innerHeight - tooltipRect.height - margin)
    );

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function normalizeVirtualRect(rect) {
    const left = Number.isFinite(rect?.left) ? rect.left : 0;
    const top = Number.isFinite(rect?.top) ? rect.top : 0;
    const width = Number.isFinite(rect?.width)
      ? rect.width
      : Number.isFinite(rect?.right)
        ? Math.max(rect.right - left, 0)
        : 0;
    const height = Number.isFinite(rect?.height)
      ? rect.height
      : Number.isFinite(rect?.bottom)
        ? Math.max(rect.bottom - top, 0)
        : 0;
    const right = Number.isFinite(rect?.right) ? rect.right : left + width;
    const bottom = Number.isFinite(rect?.bottom) ? rect.bottom : top + height;
    return {
      left,
      top,
      right,
      bottom,
      width,
      height,
    };
  }

  function createVirtualTooltipTarget({
    label = "",
    shortcut = "",
    description = "",
    rect = null,
  }) {
    const resolvedRect = normalizeVirtualRect(rect);
    return {
      disabled: false,
      dataset: {
        shortcutLabel: label,
        shortcut,
        shortcutDescription: description,
      },
      getBoundingClientRect() {
        return resolvedRect;
      },
    };
  }

  function showTooltip(button) {
    if (
      !button
      || button.disabled
      || !button.dataset
      || !(
        button.dataset.shortcut
        || button.dataset.shortcutLabel
        || button.dataset.shortcutDescription
      )
    ) {
      return;
    }
    const tooltip = ensureTooltipNode();
    tooltip.innerHTML = buildTooltipMarkup(button);
    tooltip.classList.remove("is-hidden");
    activeButton = button;
    positionTooltip(button);
  }

  function hideTooltip(button = null) {
    if (button && activeButton && button !== activeButton) {
      return;
    }
    if (!tooltipNode) {
      return;
    }
    tooltipNode.classList.add("is-hidden");
    activeButton = null;
  }

  function showVirtualTooltip({
    label = "",
    shortcut = "",
    description = "",
    rect = null,
  }) {
    if (!rect) {
      return;
    }
    showTooltip(
      createVirtualTooltipTarget({
        label,
        shortcut,
        description,
        rect,
      })
    );
  }

  function hideActiveTooltip() {
    hideTooltip();
  }

  function attachShortcutTooltipHandlers() {
    documentRef.addEventListener("mouseover", (event) => {
      const button = resolveTooltipTarget(event.target);
      if (button) {
        showTooltip(button);
      }
    });
    documentRef.addEventListener("mouseout", (event) => {
      const button = resolveTooltipTarget(event.target);
      const relatedButton = resolveTooltipTarget(event.relatedTarget);
      if (button && relatedButton !== button) {
        hideTooltip(button);
      }
    });
    documentRef.addEventListener("focusin", (event) => {
      const button = resolveTooltipTarget(event.target);
      if (button) {
        showTooltip(button);
      }
    });
    documentRef.addEventListener("focusout", (event) => {
      const button = resolveTooltipTarget(event.target);
      if (button) {
        hideTooltip(button);
      }
    });
    windowRef.addEventListener("resize", () => {
      if (activeButton) {
        positionTooltip(activeButton);
      }
    });
    windowRef.addEventListener(
      "scroll",
      () => {
        if (activeButton) {
          positionTooltip(activeButton);
        }
      },
      true
    );
  }

  return {
    applyShortcutHint,
    applyTitleHint,
    attachShortcutTooltipHandlers,
    showVirtualTooltip,
    hideActiveTooltip,
  };
}
