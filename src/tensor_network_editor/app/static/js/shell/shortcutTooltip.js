function resolveShortcutButton(eventTarget) {
  return eventTarget && typeof eventTarget.closest === "function"
    ? eventTarget.closest("button[data-shortcut]")
    : null;
}

export function createShortcutTooltip({ documentRef, windowRef }) {
  let tooltipNode = null;
  let activeButton = null;

  function applyShortcutHint(buttonId, label, shortcut) {
    const button = documentRef.getElementById(buttonId);
    if (!button) {
      return;
    }
    button.dataset.shortcut = shortcut;
    button.dataset.shortcutLabel = label;
    button.setAttribute("aria-label", `${label} (${shortcut})`);
    button.removeAttribute("title");
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

  function formatTooltipText(button) {
    const label = button.dataset.shortcutLabel || String(button.textContent || "").trim();
    const shortcut = button.dataset.shortcut || "";
    return label ? `${label} (${shortcut})` : shortcut;
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

  function showTooltip(button) {
    if (!button || !button.dataset.shortcut || button.disabled) {
      return;
    }
    const tooltip = ensureTooltipNode();
    tooltip.textContent = formatTooltipText(button);
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

  function attachShortcutTooltipHandlers() {
    documentRef.addEventListener("mouseover", (event) => {
      const button = resolveShortcutButton(event.target);
      if (button) {
        showTooltip(button);
      }
    });
    documentRef.addEventListener("mouseout", (event) => {
      const button = resolveShortcutButton(event.target);
      const relatedButton = resolveShortcutButton(event.relatedTarget);
      if (button && relatedButton !== button) {
        hideTooltip(button);
      }
    });
    documentRef.addEventListener("focusin", (event) => {
      const button = resolveShortcutButton(event.target);
      if (button) {
        showTooltip(button);
      }
    });
    documentRef.addEventListener("focusout", (event) => {
      const button = resolveShortcutButton(event.target);
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
    attachShortcutTooltipHandlers,
  };
}
