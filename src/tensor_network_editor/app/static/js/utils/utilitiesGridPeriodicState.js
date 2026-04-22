export const GRID_PERIODIC_CELL_ORDER = [
  "top_left",
  "top",
  "top_right",
  "left",
  "center",
  "right",
  "bottom_left",
  "bottom",
  "bottom_right",
];

export const GRID_PERIODIC_CELL_LABELS = {
  top_left: "Top-left cell",
  top: "Top cell",
  top_right: "Top-right cell",
  left: "Left cell",
  center: "Center cell",
  right: "Right cell",
  bottom_left: "Bottom-left cell",
  bottom: "Bottom cell",
  bottom_right: "Bottom-right cell",
};

const GRID_PERIODIC_NAVIGATION = {
  top_left: { up: null, right: "top", down: "left", left: null },
  top: { up: null, right: "top_right", down: "center", left: "top_left" },
  top_right: { up: null, right: null, down: "right", left: "top" },
  left: { up: "top_left", right: "center", down: "bottom_left", left: null },
  center: { up: "top", right: "right", down: "bottom", left: "left" },
  right: { up: "top_right", right: null, down: "bottom_right", left: "center" },
  bottom_left: { up: "left", right: "bottom", down: null, left: null },
  bottom: { up: "center", right: "bottom_right", down: null, left: "bottom_left" },
  bottom_right: { up: "right", right: null, down: null, left: "bottom" },
};

export const GRID_PERIODIC_BOUNDARY_SETTINGS = {
  up: {
    name: "Upper cell",
    color: "#456cbf",
  },
  right: {
    name: "Right cell",
    color: "#2f9b8f",
  },
  down: {
    name: "Lower cell",
    color: "#d38a37",
  },
  left: {
    name: "Left cell",
    color: "#8e5bcc",
  },
};

const GRID_PERIODIC_CELL_KEYS = Object.fromEntries(
  GRID_PERIODIC_CELL_ORDER.map((cellName) => [cellName, `${cellName}_cell`])
);

export const GRID_PERIODIC_FAMILY_BY_CELL_ROLE = {
  top_left: {
    right: "row_top",
    down: "column_left",
  },
  top: {
    left: "row_top",
    right: "row_top",
    down: "column_center",
  },
  top_right: {
    left: "row_top",
    down: "column_right",
  },
  left: {
    up: "column_left",
    right: "row_middle",
    down: "column_left",
  },
  center: {
    up: "column_center",
    right: "row_middle",
    down: "column_center",
    left: "row_middle",
  },
  right: {
    up: "column_right",
    down: "column_right",
    left: "row_middle",
  },
  bottom_left: {
    up: "column_left",
    right: "row_bottom",
  },
  bottom: {
    up: "column_center",
    left: "row_bottom",
    right: "row_bottom",
  },
  bottom_right: {
    up: "column_right",
    left: "row_bottom",
  },
};

const GRID_PERIODIC_EXPECTED_ROLES = {
  top_left: ["right", "down"],
  top: ["left", "right", "down"],
  top_right: ["left", "down"],
  left: ["up", "right", "down"],
  center: ["up", "right", "down", "left"],
  right: ["up", "down", "left"],
  bottom_left: ["up", "right"],
  bottom: ["up", "left", "right"],
  bottom_right: ["up", "left"],
};

export function createGridPeriodicStateSupport({ state, runtime }) {
  function getGridPeriodicCellKey(cellName) {
    return Object.prototype.hasOwnProperty.call(GRID_PERIODIC_CELL_KEYS, cellName)
      ? GRID_PERIODIC_CELL_KEYS[cellName]
      : null;
  }

  function normalizeGridPeriodicGridInPlace(grid) {
    grid.metadata = runtime.isObject(grid.metadata) ? grid.metadata : {};
    grid.active_cell = GRID_PERIODIC_CELL_ORDER.includes(grid.active_cell)
      ? grid.active_cell
      : "center";
    GRID_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellKey = getGridPeriodicCellKey(cellName);
      grid[cellKey] = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          runtime.isObject(grid[cellKey])
            ? grid[cellKey]
            : runtime.buildEmptyGraphSection()
        )
      );
      grid[cellKey].contraction_plan = null;
    });
    return grid;
  }

  function getGridPeriodicGrid(spec = state.spec) {
    return spec && runtime.isObject(spec.grid_periodic_grid)
      ? spec.grid_periodic_grid
      : null;
  }

  function isGridPeriodicMode(spec = state.spec) {
    return Boolean(getGridPeriodicGrid(spec));
  }

  function isForMode(spec = state.spec) {
    return (
      (typeof runtime.isLinearPeriodicMode === "function" &&
        runtime.isLinearPeriodicMode(spec)) ||
      (typeof runtime.isTreePeriodicMode === "function" &&
        runtime.isTreePeriodicMode(spec)) ||
      isGridPeriodicMode(spec)
    );
  }

  function getActiveGridPeriodicCellName(spec = state.spec) {
    const grid = getGridPeriodicGrid(spec);
    return grid ? grid.active_cell : null;
  }

  function getGridPeriodicCell(
    spec = state.spec,
    cellName = getActiveGridPeriodicCellName(spec)
  ) {
    const grid = getGridPeriodicGrid(spec);
    const cellKey = getGridPeriodicCellKey(cellName);
    if (!grid || !cellKey) {
      return null;
    }
    return grid[cellKey] || null;
  }

  function getGridPeriodicCellLabel(cellName) {
    return GRID_PERIODIC_CELL_LABELS[cellName] || "Grid cell";
  }

  function getGridPeriodicNeighborCellName(cellName, direction) {
    return (
      (GRID_PERIODIC_NAVIGATION[cellName] &&
        GRID_PERIODIC_NAVIGATION[cellName][direction]) ||
      null
    );
  }

  function canSwitchGridPeriodicCell(direction, spec = state.spec) {
    const activeCellName = getActiveGridPeriodicCellName(spec);
    return Boolean(
      activeCellName && getGridPeriodicNeighborCellName(activeCellName, direction)
    );
  }

  function getGridPeriodicBoundaryTensorByRole(role, spec = state.spec) {
    return (
      Array.isArray(spec && spec.tensors) ? spec.tensors : []
    ).find((tensor) => tensor.grid_periodic_role === role) || null;
  }

  function isGridPeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.grid_periodic_role === "up" ||
          tensor.grid_periodic_role === "right" ||
          tensor.grid_periodic_role === "down" ||
          tensor.grid_periodic_role === "left")
    );
  }

  function isForBoundaryTensor(tensor) {
    return (
      isGridPeriodicBoundaryTensor(tensor) ||
      (typeof runtime.isTreePeriodicBoundaryTensor === "function" &&
        runtime.isTreePeriodicBoundaryTensor(tensor)) ||
      (typeof runtime.isLinearPeriodicBoundaryTensor === "function" &&
        runtime.isLinearPeriodicBoundaryTensor(tensor))
    );
  }

  function getExpectedGridPeriodicRoles(cellName) {
    return Object.prototype.hasOwnProperty.call(
      GRID_PERIODIC_EXPECTED_ROLES,
      cellName
    )
      ? [...GRID_PERIODIC_EXPECTED_ROLES[cellName]]
      : [];
  }

  return {
    getGridPeriodicCellKey,
    normalizeGridPeriodicGridInPlace,
    getGridPeriodicGrid,
    isGridPeriodicMode,
    isForMode,
    getActiveGridPeriodicCellName,
    getGridPeriodicCell,
    getGridPeriodicCellLabel,
    getGridPeriodicNeighborCellName,
    canSwitchGridPeriodicCell,
    getGridPeriodicBoundaryTensorByRole,
    isGridPeriodicBoundaryTensor,
    isForBoundaryTensor,
    getExpectedGridPeriodicRoles,
  };
}
