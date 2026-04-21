export function createUtilityUiDomSupport({ windowRef }) {
  const FLOATING_PANEL_MARGIN = 8;
  const FLOATING_PANEL_GAP = 4;

  function toggleElementClass(element, className, isActive) {
    if (
      !element
      || !element.classList
      || typeof element.classList.toggle !== "function"
    ) {
      return;
    }
    element.classList.toggle(className, isActive);
  }

  function setExpandedState(button, isExpanded) {
    if (!button || typeof button.setAttribute !== "function") {
      return;
    }
    button.setAttribute("aria-expanded", String(isExpanded));
  }

  function setElementHidden(element, isHidden) {
    if (!element) {
      return;
    }
    element.hidden = Boolean(isHidden);
  }

  function setTooltipDescription(button, description) {
    if (!button) {
      return;
    }
    if (!button.dataset) {
      button.dataset = {};
    }
    if (typeof description === "string" && description) {
      button.dataset.shortcutDescription = description;
    } else {
      delete button.dataset.shortcutDescription;
    }
    const label =
      typeof button.dataset.shortcutLabel === "string"
        ? button.dataset.shortcutLabel.trim()
        : "";
    const shortcut =
      typeof button.dataset.shortcut === "string"
        ? button.dataset.shortcut.trim()
        : "";
    const header = shortcut ? `${label} (${shortcut})` : label;
    if (header && typeof button.setAttribute === "function") {
      button.setAttribute(
        "aria-label",
        description ? `${header}. ${description}` : header
      );
    }
    if (typeof button.removeAttribute === "function") {
      button.removeAttribute("title");
    }
  }

  function setButtonGroupDisabled(buttons, isDisabled, title) {
    buttons.forEach((button) => {
      if (!button) {
        return;
      }
      button.disabled = isDisabled;
      if (typeof title === "string") {
        setTooltipDescription(button, title);
      }
    });
  }

  function setMenuItemChecked(menuItem, checked) {
    if (!menuItem) {
      return;
    }
    toggleElementClass(menuItem, "is-checked", checked);
    if (typeof menuItem.setAttribute === "function") {
      menuItem.setAttribute("aria-checked", String(checked));
    }
  }

  function setStyleVariable(element, propertyName, value) {
    if (!element || !element.style) {
      return;
    }
    if (typeof element.style.setProperty === "function") {
      element.style.setProperty(propertyName, value);
      return;
    }
    element.style[propertyName] = value;
  }

  function normalizeRect(rect, fallbackWidth = 0, fallbackHeight = 0) {
    const left = Number.isFinite(rect?.left) ? rect.left : 0;
    const top = Number.isFinite(rect?.top) ? rect.top : 0;
    const widthFromEdges =
      Number.isFinite(rect?.right) && Number.isFinite(rect?.left)
        ? Math.max(rect.right - rect.left, 0)
        : 0;
    const heightFromEdges =
      Number.isFinite(rect?.bottom) && Number.isFinite(rect?.top)
        ? Math.max(rect.bottom - rect.top, 0)
        : 0;
    const width =
      Number.isFinite(rect?.width) && rect.width > 0
        ? rect.width
        : Math.max(widthFromEdges, fallbackWidth);
    const height =
      Number.isFinite(rect?.height) && rect.height > 0
        ? rect.height
        : Math.max(heightFromEdges, fallbackHeight);

    return {
      left,
      top,
      width,
      height,
      right: Number.isFinite(rect?.right) ? rect.right : left + width,
      bottom: Number.isFinite(rect?.bottom) ? rect.bottom : top + height,
    };
  }

  function getElementRect(element, fallbackWidth = 0, fallbackHeight = 0) {
    const rawRect =
      element && typeof element.getBoundingClientRect === "function"
        ? element.getBoundingClientRect()
        : null;
    return normalizeRect(rawRect, fallbackWidth, fallbackHeight);
  }

  function clampFloatingOffset(offset, panelSize, viewportSize) {
    const minOffset = FLOATING_PANEL_MARGIN;
    const maxOffset = Math.max(
      FLOATING_PANEL_MARGIN,
      viewportSize - panelSize - FLOATING_PANEL_MARGIN
    );
    return Math.min(Math.max(offset, minOffset), maxOffset);
  }

  function positionFloatingPanel(
    panel,
    anchor,
    {
      align = "left",
      leftVariable,
      topVariable,
      fallbackWidth = 0,
      fallbackHeight = 0,
    }
  ) {
    if (!panel || !anchor) {
      return;
    }

    const resolvedWindowRef =
      windowRef && typeof windowRef === "object" ? windowRef : globalThis;
    const anchorRect = getElementRect(anchor);
    const panelRect = getElementRect(panel, fallbackWidth, fallbackHeight);
    const viewportWidth = Number.isFinite(resolvedWindowRef.innerWidth)
      ? resolvedWindowRef.innerWidth
      : anchorRect.right + panelRect.width + FLOATING_PANEL_MARGIN;
    const viewportHeight = Number.isFinite(resolvedWindowRef.innerHeight)
      ? resolvedWindowRef.innerHeight
      : anchorRect.bottom + panelRect.height + FLOATING_PANEL_MARGIN;
    const rawLeft =
      align === "right" ? anchorRect.right - panelRect.width : anchorRect.left;
    const left = clampFloatingOffset(rawLeft, panelRect.width, viewportWidth);
    const top = clampFloatingOffset(
      anchorRect.bottom + FLOATING_PANEL_GAP,
      panelRect.height,
      viewportHeight
    );

    setStyleVariable(panel, leftVariable, `${Math.round(left)}px`);
    setStyleVariable(panel, topVariable, `${Math.round(top)}px`);
  }

  return {
    toggleElementClass,
    setExpandedState,
    setElementHidden,
    setTooltipDescription,
    setButtonGroupDisabled,
    setMenuItemChecked,
    setStyleVariable,
    normalizeRect,
    getElementRect,
    clampFloatingOffset,
    positionFloatingPanel,
  };
}
