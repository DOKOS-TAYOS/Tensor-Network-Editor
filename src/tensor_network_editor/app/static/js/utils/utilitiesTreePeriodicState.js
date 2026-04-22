export const TREE_PERIODIC_CELL_ORDER = ["root", "branch", "leaf"];

export const TREE_PERIODIC_CELL_LABELS = {
  root: "Root cell",
  branch: "Branch cell",
  leaf: "Leaf cell",
};

export const TREE_PERIODIC_NAVIGATION = {
  root: { up: null, down: "branch" },
  branch: { up: "root", down: "leaf" },
  leaf: { up: "branch", down: null },
};

export const TREE_PERIODIC_PARENT_SETTINGS = {
  name: "Parent cell",
  color: "#456cbf",
};

export const TREE_PERIODIC_CHILD_SETTINGS = {
  color: "#2f9b8f",
};

export function createTreePeriodicStateSupport({ state, runtime }) {
  function getTreePeriodicCellKey(cellName) {
    return TREE_PERIODIC_CELL_ORDER.includes(cellName) ? `${cellName}_cell` : null;
  }

  function normalizeTreePeriodicTreeInPlace(tree) {
    tree.metadata = runtime.isObject(tree.metadata) ? tree.metadata : {};
    tree.active_cell = TREE_PERIODIC_CELL_ORDER.includes(tree.active_cell)
      ? tree.active_cell
      : "root";
    tree.branching_factor = Math.max(
      2,
      Math.round(runtime.asFiniteNumber(tree.branching_factor, 2))
    );
    TREE_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellKey = getTreePeriodicCellKey(cellName);
      tree[cellKey] = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          runtime.isObject(tree[cellKey])
            ? tree[cellKey]
            : runtime.buildEmptyGraphSection()
        )
      );
      tree[cellKey].contraction_plan = null;
    });
    return tree;
  }

  function getTreePeriodicTree(spec = state.spec) {
    return spec && runtime.isObject(spec.tree_periodic_tree)
      ? spec.tree_periodic_tree
      : null;
  }

  function isTreePeriodicMode(spec = state.spec) {
    return Boolean(getTreePeriodicTree(spec));
  }

  function getTreePeriodicBranchingFactor(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    return tree
      ? Math.max(2, Math.round(runtime.asFiniteNumber(tree.branching_factor, 2)))
      : 2;
  }

  function getActiveTreePeriodicCellName(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    return tree ? tree.active_cell : null;
  }

  function getTreePeriodicCell(
    spec = state.spec,
    cellName = getActiveTreePeriodicCellName(spec)
  ) {
    const tree = getTreePeriodicTree(spec);
    const cellKey = getTreePeriodicCellKey(cellName);
    if (!tree || !cellKey) {
      return null;
    }
    return tree[cellKey] || null;
  }

  function getTreePeriodicCellLabel(cellName) {
    return TREE_PERIODIC_CELL_LABELS[cellName] || "Tree cell";
  }

  function getTreePeriodicNeighborCellName(cellName, direction) {
    return (
      (TREE_PERIODIC_NAVIGATION[cellName] &&
        TREE_PERIODIC_NAVIGATION[cellName][direction]) ||
      null
    );
  }

  function canSwitchTreePeriodicCell(direction, spec = state.spec) {
    const activeCellName = getActiveTreePeriodicCellName(spec);
    return Boolean(
      activeCellName && getTreePeriodicNeighborCellName(activeCellName, direction)
    );
  }

  function isTreePeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.tree_periodic_role === "parent" ||
          tensor.tree_periodic_role === "child")
    );
  }

  return {
    getTreePeriodicCellKey,
    normalizeTreePeriodicTreeInPlace,
    getTreePeriodicTree,
    isTreePeriodicMode,
    getTreePeriodicBranchingFactor,
    getActiveTreePeriodicCellName,
    getTreePeriodicCell,
    getTreePeriodicCellLabel,
    getTreePeriodicNeighborCellName,
    canSwitchTreePeriodicCell,
    isTreePeriodicBoundaryTensor,
  };
}
