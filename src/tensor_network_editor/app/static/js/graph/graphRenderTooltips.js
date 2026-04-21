export function createGraphRenderTooltipSupport({ ctx, canvasShell }) {
  function getForBoundaryTensorTooltip(tensor) {
    if (
      typeof ctx.isLinearPeriodicBoundaryTensor === "function" &&
      ctx.isLinearPeriodicBoundaryTensor(tensor)
    ) {
      if (tensor.linear_periodic_role === "previous") {
        return {
          label: "Previous cell",
          description:
            "Virtual boundary tensor for the previous cell in For unidimensional mode. Connect open indices here when the bond should continue into the cell on the left.",
        };
      }
      return {
        label: "Next cell",
        description:
          "Virtual boundary tensor for the next cell in For unidimensional mode. Connect open indices here when the bond should continue into the cell on the right.",
      };
    }
    if (
      typeof ctx.isGridPeriodicBoundaryTensor === "function" &&
      ctx.isGridPeriodicBoundaryTensor(tensor)
    ) {
      if (tensor.grid_periodic_role === "up") {
        return {
          label: "Upper cell",
          description:
            "Virtual boundary tensor for the cell above in For bidimensional mode. Connect open indices here when the bond should continue into the upper neighboring cell.",
        };
      }
      if (tensor.grid_periodic_role === "right") {
        return {
          label: "Right cell",
          description:
            "Virtual boundary tensor for the cell on the right in For bidimensional mode. Connect open indices here when the bond should continue into the right neighboring cell.",
        };
      }
      if (tensor.grid_periodic_role === "down") {
        return {
          label: "Lower cell",
          description:
            "Virtual boundary tensor for the cell below in For bidimensional mode. Connect open indices here when the bond should continue into the lower neighboring cell.",
        };
      }
      if (tensor.grid_periodic_role === "left") {
        return {
          label: "Left cell",
          description:
            "Virtual boundary tensor for the cell on the left in For bidimensional mode. Connect open indices here when the bond should continue into the left neighboring cell.",
        };
      }
    }
    if (
      typeof ctx.isTreePeriodicBoundaryTensor === "function" &&
      ctx.isTreePeriodicBoundaryTensor(tensor)
    ) {
      if (tensor.tree_periodic_role === "parent") {
        return {
          label: "Parent cell",
          description:
            "Virtual boundary tensor for the parent node in For Tree mode. Connect open indices here when the bond should continue upward to the parent branch.",
        };
      }
      return {
        label: `Child ${Number(tensor.tree_periodic_child_index) + 1}`,
        description:
          "Virtual boundary tensor for one child branch in For Tree mode. Connect open indices here when the bond should continue downward into that child slot.",
      };
    }
    return null;
  }

  function hideBoundaryTensorTooltip() {
    if (
      !ctx.shortcutTooltip ||
      typeof ctx.shortcutTooltip.hideActiveTooltip !== "function"
    ) {
      return;
    }
    ctx.shortcutTooltip.hideActiveTooltip();
  }

  function showBoundaryTensorTooltip(node) {
    if (
      !ctx.shortcutTooltip ||
      typeof ctx.shortcutTooltip.showVirtualTooltip !== "function" ||
      !node ||
      typeof node.id !== "function"
    ) {
      return;
    }
    const tensor =
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(node.id())
        : ctx.findTensorById(node.id());
    const tooltip = getForBoundaryTensorTooltip(tensor);
    if (!tooltip) {
      hideBoundaryTensorTooltip();
      return;
    }
    const renderedBounds =
      typeof node.renderedBoundingBox === "function"
        ? node.renderedBoundingBox({
            includeLabels: false,
            includeNodes: true,
            includeOverlays: false,
          })
        : null;
    if (!renderedBounds) {
      return;
    }
    const canvasBounds =
      canvasShell && typeof canvasShell.getBoundingClientRect === "function"
        ? canvasShell.getBoundingClientRect()
        : { left: 0, top: 0 };
    ctx.shortcutTooltip.showVirtualTooltip({
      label: tooltip.label,
      description: tooltip.description,
      rect: {
        left: canvasBounds.left + renderedBounds.x1,
        top: canvasBounds.top + renderedBounds.y1,
        right: canvasBounds.left + renderedBounds.x2,
        bottom: canvasBounds.top + renderedBounds.y2,
        width: renderedBounds.w,
        height: renderedBounds.h,
      },
    });
  }

  return {
    getForBoundaryTensorTooltip,
    hideBoundaryTensorTooltip,
    showBoundaryTensorTooltip,
  };
}
