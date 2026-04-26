import {
  GRID_PERIODIC_CELL_LABELS,
} from "./utilitiesGridPeriodicState.js";
import {
  LINEAR_PERIODIC_CELL_LABELS,
} from "./utilitiesLinearPeriodicState.js";
import {
  TREE_PERIODIC_CELL_LABELS,
} from "./utilitiesTreePeriodicState.js";

export function createUiToolbarModeControlSupport({
  dom,
  runtime,
  setTooltipDescription,
  setMenuItemChecked,
}) {
  const {
    singleModeMenuItem,
    linearPeriodicModeMenuItem,
    gridPeriodicModeMenuItem,
    treeModeMenuItem,
    benchmarkModeMenuItem,
    toolbarModeControls,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    gridPeriodicUpCellButton,
    gridPeriodicDownCellButton,
    linearPeriodicNextCellButton,
    benchmarkSchemeNameInput,
    benchmarkCompareButton,
  } = dom;

  function syncToolbarModeControls(derivedState) {
    const {
      linearPeriodicMode,
      activeLinearPeriodicCell,
      gridPeriodicMode,
      activeGridPeriodicCell,
      treePeriodicMode,
      activeTreePeriodicCell,
      forMode,
      canSwitchGridPeriodicCell,
      canSwitchTreePeriodicCell,
      benchmarkMode,
      benchmarkActivePosition,
      activeBenchmarkScheme,
    } = derivedState;

    setMenuItemChecked(singleModeMenuItem, !forMode && !benchmarkMode);
    setMenuItemChecked(linearPeriodicModeMenuItem, linearPeriodicMode);
    if (gridPeriodicModeMenuItem) {
      setMenuItemChecked(gridPeriodicModeMenuItem, gridPeriodicMode);
    }
    if (treeModeMenuItem) {
      setMenuItemChecked(treeModeMenuItem, treePeriodicMode);
    }
    if (benchmarkModeMenuItem) {
      setMenuItemChecked(benchmarkModeMenuItem, benchmarkMode);
      benchmarkModeMenuItem.disabled = !benchmarkMode && forMode;
      setTooltipDescription(
        benchmarkModeMenuItem,
        forMode
          ? "Benchmark mode is unavailable while a For mode is active."
          : "Compare manual contraction schemes on the current tensor network."
      );
    }
    if (toolbarModeControls) {
      toolbarModeControls.hidden = !(forMode || benchmarkMode);
    }
    if (linearPeriodicCellLabel && !benchmarkMode) {
      linearPeriodicCellLabel.hidden = false;
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : gridPeriodicMode
          ? GRID_PERIODIC_CELL_LABELS[activeGridPeriodicCell] || "Grid cell"
          : treePeriodicMode
            ? TREE_PERIODIC_CELL_LABELS[activeTreePeriodicCell] || "Tree cell"
            : "Single";
    }
    if (benchmarkSchemeNameInput && !benchmarkMode) {
      benchmarkSchemeNameInput.hidden = true;
      benchmarkSchemeNameInput.disabled = true;
      benchmarkSchemeNameInput.value = "";
    }
    if (benchmarkCompareButton && !benchmarkMode) {
      benchmarkCompareButton.hidden = true;
      benchmarkCompareButton.disabled = true;
      setTooltipDescription(
        benchmarkCompareButton,
        "Compare the saved contraction schemes."
      );
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicPreviousCellButton.disabled = benchmarkMode
        ? benchmarkActivePosition === 0
        : gridPeriodicMode
          ? !canSwitchGridPeriodicCell("left")
          : !linearPeriodicMode || activeLinearPeriodicCell === "initial";
      setTooltipDescription(
        linearPeriodicPreviousCellButton,
        benchmarkMode
          ? benchmarkActivePosition === 0
            ? "You are already at the tensor network view. In benchmark mode, use Previous and Next to move between the base network and the saved contraction schemes."
            : "Open the previous saved benchmark scheme. Use Previous and Next to move between the tensor network view and each saved scheme."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("left")
              ? "Move to the cell on the left. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the left edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
              : !linearPeriodicMode
                ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
                : activeLinearPeriodicCell === "initial"
                  ? "You are already at the initial cell of the three-cell unidimensional workflow."
                  : "Move to the previous cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.hidden = treePeriodicMode && !benchmarkMode;
      linearPeriodicNextCellButton.disabled =
        !benchmarkMode &&
        (gridPeriodicMode
          ? !canSwitchGridPeriodicCell("right")
          : !linearPeriodicMode || activeLinearPeriodicCell === "final");
      linearPeriodicNextCellButton.textContent = benchmarkMode
        ? typeof runtime.getBenchmarkNextButtonLabel === "function"
          ? runtime.getBenchmarkNextButtonLabel()
          : ">"
        : ">";
      setTooltipDescription(
        linearPeriodicNextCellButton,
        benchmarkMode
          ? linearPeriodicNextCellButton.textContent === "+"
            ? "Create a new benchmark scheme after the current one. Use Next repeatedly to add schemes and then compare them in Planner."
            : "Open the next saved benchmark scheme. Use Previous and Next to move through the benchmark chain."
          : gridPeriodicMode
            ? canSwitchGridPeriodicCell("right")
              ? "Move to the cell on the right. Use the cell arrows to edit each representative cell of the bidimensional layout."
              : "You are already at the right edge of the bidimensional layout."
            : treePeriodicMode
              ? "For Tree mode uses only the vertical cell arrows."
              : !linearPeriodicMode
                ? "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
                : activeLinearPeriodicCell === "final"
                  ? "You are already at the final cell of the three-cell unidimensional workflow."
                  : "Move to the next cell in the three-cell unidimensional workflow: initial, periodic, final."
      );
    }
    if (gridPeriodicUpCellButton) {
      gridPeriodicUpCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicUpCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("up")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("up");
      setTooltipDescription(
        gridPeriodicUpCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("up")
            ? "Move to the upper cell."
            : "You are already at the top edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("up")
              ? "Move to the parent-facing cell above."
              : "You are already at the root cell."
      );
    }
    if (gridPeriodicDownCellButton) {
      gridPeriodicDownCellButton.hidden = !(gridPeriodicMode || treePeriodicMode) || benchmarkMode;
      gridPeriodicDownCellButton.disabled =
        gridPeriodicMode
          ? !canSwitchGridPeriodicCell("down")
          : !treePeriodicMode || !canSwitchTreePeriodicCell("down");
      setTooltipDescription(
        gridPeriodicDownCellButton,
        gridPeriodicMode
          ? canSwitchGridPeriodicCell("down")
            ? "Move to the lower cell."
            : "You are already at the bottom edge."
          : !treePeriodicMode
            ? "For Tree mode is not active."
            : canSwitchTreePeriodicCell("down")
              ? "Move to the child-facing cell below."
              : "You are already at the leaf cell."
      );
    }
    if (benchmarkMode) {
      if (linearPeriodicCellLabel) {
        linearPeriodicCellLabel.hidden = true;
        linearPeriodicCellLabel.textContent =
          typeof runtime.getBenchmarkBaseLabel === "function"
            ? runtime.getBenchmarkBaseLabel()
            : "Tensor network";
      }
      if (benchmarkSchemeNameInput) {
        benchmarkSchemeNameInput.hidden = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.disabled = benchmarkActivePosition === 0;
        benchmarkSchemeNameInput.value =
          benchmarkActivePosition > 0
            ? activeBenchmarkScheme && typeof activeBenchmarkScheme.name === "string"
              ? activeBenchmarkScheme.name
              : typeof runtime.getBenchmarkSchemeName === "function"
                ? runtime.getBenchmarkSchemeName(benchmarkActivePosition - 1)
                : `Scheme ${benchmarkActivePosition}`
            : "";
      }
      if (benchmarkCompareButton) {
        benchmarkCompareButton.hidden = false;
        benchmarkCompareButton.disabled = !(
          typeof runtime.canOpenBenchmarkCompare === "function" &&
          runtime.canOpenBenchmarkCompare()
        );
        setTooltipDescription(
          benchmarkCompareButton,
          benchmarkCompareButton.disabled
            ? "Create at least one scheme first."
            : "Compare the saved contraction schemes."
        );
      }
    }
  }

  return {
    syncToolbarModeControls,
  };
}
