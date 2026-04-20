export function createUtilityTreePeriodicBindings({
  ctx,
  state,
  constants,
  runtime,
}) {
  const { TENSOR_WIDTH, TENSOR_HEIGHT } = constants;
  const { window } = ctx;

  const TREE_PERIODIC_CELL_ORDER = ["root", "branch", "leaf"];
  const TREE_PERIODIC_CELL_LABELS = {
    root: "Root cell",
    branch: "Branch cell",
    leaf: "Leaf cell",
  };
  const TREE_PERIODIC_NAVIGATION = {
    root: { up: null, down: "branch" },
    branch: { up: "root", down: "leaf" },
    leaf: { up: "branch", down: null },
  };
  const TREE_PERIODIC_PARENT_SETTINGS = {
    name: "Parent cell",
    color: "#456cbf",
  };
  const TREE_PERIODIC_CHILD_SETTINGS = {
    color: "#2f9b8f",
  };

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

  function buildTreePeriodicBoundaryKey(role, childIndex = null) {
    return role === "child" ? `child:${childIndex}` : "parent";
  }

  function getTreePeriodicBoundaryDescriptor(role, childIndex = null) {
    return {
      role,
      childIndex: role === "child" ? childIndex : null,
      key: buildTreePeriodicBoundaryKey(role, childIndex),
    };
  }

  function getExpectedTreePeriodicBoundaryDescriptors(
    cellName,
    branchingFactor = getTreePeriodicBranchingFactor()
  ) {
    if (cellName === "root") {
      return Array.from({ length: branchingFactor }, (_, childIndex) =>
        getTreePeriodicBoundaryDescriptor("child", childIndex)
      );
    }
    if (cellName === "branch") {
      return [
        getTreePeriodicBoundaryDescriptor("parent"),
        ...Array.from({ length: branchingFactor }, (_, childIndex) =>
          getTreePeriodicBoundaryDescriptor("child", childIndex)
        ),
      ];
    }
    if (cellName === "leaf") {
      return [getTreePeriodicBoundaryDescriptor("parent")];
    }
    return [];
  }

  function getRealTensorBounds(spec = state.spec) {
    const tensors = (Array.isArray(spec && spec.tensors) ? spec.tensors : []).filter(
      (tensor) => !isTreePeriodicBoundaryTensor(tensor)
    );
    if (!tensors.length) {
      return {
        minX: 120,
        maxX: 320,
        minY: 120,
        maxY: 260,
        centerX: 220,
        centerY: 190,
      };
    }
    const leftEdges = tensors.map(
      (tensor) => tensor.position.x - runtime.tensorWidth(tensor) / 2
    );
    const rightEdges = tensors.map(
      (tensor) => tensor.position.x + runtime.tensorWidth(tensor) / 2
    );
    const topEdges = tensors.map(
      (tensor) => tensor.position.y - runtime.tensorHeight(tensor) / 2
    );
    const bottomEdges = tensors.map(
      (tensor) => tensor.position.y + runtime.tensorHeight(tensor) / 2
    );
    const centersX = tensors.map((tensor) => tensor.position.x);
    const centersY = tensors.map((tensor) => tensor.position.y);
    return {
      minX: Math.min(...leftEdges),
      maxX: Math.max(...rightEdges),
      minY: Math.min(...topEdges),
      maxY: Math.max(...bottomEdges),
      centerX: centersX.reduce((sum, value) => sum + value, 0) / centersX.length,
      centerY: centersY.reduce((sum, value) => sum + value, 0) / centersY.length,
    };
  }

  function getTreePeriodicChildName(childIndex) {
    return `Child ${childIndex + 1}`;
  }

  function positionTreePeriodicBoundaryTensor(
    tensor,
    descriptor,
    spec = state.spec,
    branchingFactor = getTreePeriodicBranchingFactor(spec)
  ) {
    const bounds = getRealTensorBounds(spec);
    if (descriptor.role === "parent") {
      tensor.position = { x: bounds.centerX, y: bounds.minY - 180 };
      return;
    }
    const childSpacing = 220;
    const centerOffset = (branchingFactor - 1) / 2;
    tensor.position = {
      x: bounds.centerX + (descriptor.childIndex - centerOffset) * childSpacing,
      y: bounds.maxY + 180,
    };
  }

  function createTreePeriodicBoundaryTensor(
    descriptor,
    spec = state.spec,
    branchingFactor = getTreePeriodicBranchingFactor(spec)
  ) {
    const isParent = descriptor.role === "parent";
    const tensor = {
      id: runtime.makeId("tensor"),
      name: isParent
        ? TREE_PERIODIC_PARENT_SETTINGS.name
        : getTreePeriodicChildName(descriptor.childIndex),
      position: { x: 0, y: 0 },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      tree_periodic_role: descriptor.role,
      tree_periodic_child_index: isParent ? null : descriptor.childIndex,
      metadata: {
        color: isParent
          ? TREE_PERIODIC_PARENT_SETTINGS.color
          : TREE_PERIODIC_CHILD_SETTINGS.color,
      },
    };
    positionTreePeriodicBoundaryTensor(tensor, descriptor, spec, branchingFactor);
    return tensor;
  }

  function ensureActiveTreePeriodicBoundaryTensors(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    const activeCellName = getActiveTreePeriodicCellName(spec);
    if (!tree || !activeCellName) {
      return;
    }
    const branchingFactor = getTreePeriodicBranchingFactor(spec);
    const expectedDescriptors = getExpectedTreePeriodicBoundaryDescriptors(
      activeCellName,
      branchingFactor
    );
    const expectedKeys = new Set(
      expectedDescriptors.map((descriptor) => descriptor.key)
    );
    const retainedBoundaryByKey = {};
    const removedBoundaryIds = new Set();
    const nonBoundaryTensors = [];

    (Array.isArray(spec.tensors) ? spec.tensors : []).forEach((tensor) => {
      if (!isTreePeriodicBoundaryTensor(tensor)) {
        nonBoundaryTensors.push(tensor);
        return;
      }
      const boundaryKey = buildTreePeriodicBoundaryKey(
        tensor.tree_periodic_role,
        tensor.tree_periodic_child_index
      );
      if (
        !expectedKeys.has(boundaryKey) ||
        Object.prototype.hasOwnProperty.call(retainedBoundaryByKey, boundaryKey)
      ) {
        removedBoundaryIds.add(tensor.id);
        return;
      }
      retainedBoundaryByKey[boundaryKey] = tensor;
    });

    const orderedBoundaryTensors = expectedDescriptors.map((descriptor) => {
      const boundaryTensor = Object.prototype.hasOwnProperty.call(
        retainedBoundaryByKey,
        descriptor.key
      )
        ? retainedBoundaryByKey[descriptor.key]
        : createTreePeriodicBoundaryTensor(descriptor, spec, branchingFactor);
      boundaryTensor.tree_periodic_role = descriptor.role;
      boundaryTensor.tree_periodic_child_index =
        descriptor.role === "child" ? descriptor.childIndex : null;
      boundaryTensor.name =
        descriptor.role === "parent"
          ? TREE_PERIODIC_PARENT_SETTINGS.name
          : getTreePeriodicChildName(descriptor.childIndex);
      boundaryTensor.metadata = runtime.isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = runtime.getMetadataColor(
        boundaryTensor.metadata,
        descriptor.role === "parent"
          ? TREE_PERIODIC_PARENT_SETTINGS.color
          : TREE_PERIODIC_CHILD_SETTINGS.color
      );
      boundaryTensor.indices = Array.isArray(boundaryTensor.indices)
        ? boundaryTensor.indices
        : [];
      boundaryTensor.indices.forEach((index, indexPosition) => {
        index.metadata = runtime.isObject(index.metadata) ? index.metadata : {};
        if (!index.id) {
          index.id = runtime.makeId("index");
        }
        if (!index.name) {
          index.name = `slot_${indexPosition + 1}`;
        }
      });
      if (typeof runtime.ensureTensorIndexOffsets === "function") {
        runtime.ensureTensorIndexOffsets(boundaryTensor);
      }
      positionTreePeriodicBoundaryTensor(
        boundaryTensor,
        descriptor,
        spec,
        branchingFactor
      );
      return boundaryTensor;
    });

    spec.tensors = [...nonBoundaryTensors, ...orderedBoundaryTensors];
    if (removedBoundaryIds.size) {
      spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
        (edge) =>
          !removedBoundaryIds.has(edge.left && edge.left.tensor_id) &&
          !removedBoundaryIds.has(edge.right && edge.right.tensor_id)
      );
    }
  }

  function getTreePeriodicChildInterfaceOwners(spec = state.spec) {
    const tensors = Array.isArray(spec && spec.tensors) ? spec.tensors : [];
    const edges = Array.isArray(spec && spec.edges) ? spec.edges : [];
    const tensorById = Object.fromEntries(
      tensors.map((tensor) => [tensor.id, tensor])
    );
    const internallyConnectedIndexIds = new Set();
    const parentConnectedIndexIds = new Set();

    edges.forEach((edge) => {
      const leftTensor = tensorById[edge.left && edge.left.tensor_id];
      const rightTensor = tensorById[edge.right && edge.right.tensor_id];
      if (!leftTensor || !rightTensor) {
        return;
      }
      const leftIsBoundary = isTreePeriodicBoundaryTensor(leftTensor);
      const rightIsBoundary = isTreePeriodicBoundaryTensor(rightTensor);
      if (!leftIsBoundary && !rightIsBoundary) {
        internallyConnectedIndexIds.add(edge.left.index_id);
        internallyConnectedIndexIds.add(edge.right.index_id);
        return;
      }
      if (leftTensor.tree_periodic_role === "parent" && !rightIsBoundary) {
        parentConnectedIndexIds.add(edge.right.index_id);
        return;
      }
      if (rightTensor.tree_periodic_role === "parent" && !leftIsBoundary) {
        parentConnectedIndexIds.add(edge.left.index_id);
      }
    });

    const owners = [];
    tensors.forEach((tensor) => {
      if (isTreePeriodicBoundaryTensor(tensor)) {
        return;
      }
      (Array.isArray(tensor.indices) ? tensor.indices : []).forEach(
        (index, indexPosition) => {
          if (
            !internallyConnectedIndexIds.has(index.id) &&
            !parentConnectedIndexIds.has(index.id)
          ) {
            owners.push({ tensor, index, indexPosition });
          }
        }
      );
    });
    return owners;
  }

  function getTreePeriodicBoundaryInterfaceDimensions(
    graphSection,
    role,
    childIndex = null
  ) {
    const boundaryTensors = (
      Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : []
    ).filter((tensor) => isTreePeriodicBoundaryTensor(tensor));
    const preferredTensor = boundaryTensors.find(
      (tensor) =>
        tensor.tree_periodic_role === role &&
        (role !== "child" || childIndex === null
          ? true
          : tensor.tree_periodic_child_index === childIndex) &&
        Array.isArray(tensor.indices) &&
        tensor.indices.length
    );
    const fallbackTensor =
      preferredTensor ||
      boundaryTensors.find(
        (tensor) =>
          tensor.tree_periodic_role === role &&
          Array.isArray(tensor.indices) &&
          tensor.indices.length
      );
    return fallbackTensor
      ? fallbackTensor.indices.map((index) => index.dimension)
      : [];
  }

  function getStoredTreePeriodicChildInterfaceDimensions(
    cellName,
    spec = state.spec
  ) {
    const cell = getTreePeriodicCell(spec, cellName);
    if (!cell) {
      return [];
    }
    const boundaryDimensions = getTreePeriodicBoundaryInterfaceDimensions(
      cell,
      "child",
      0
    );
    return boundaryDimensions.length
      ? boundaryDimensions
      : getTreePeriodicChildInterfaceOwners(cell).map(
          (owner) => owner.index.dimension
        );
  }

  function getCanonicalTreePeriodicInterfaceDimensions(spec = state.spec) {
    const activeCellName = getActiveTreePeriodicCellName(spec);
    const rootChildDimensions =
      activeCellName === "root"
        ? getTreePeriodicChildInterfaceOwners(spec).map(
            (owner) => owner.index.dimension
          )
        : getStoredTreePeriodicChildInterfaceDimensions("root", spec);
    const branchChildDimensions =
      activeCellName === "branch"
        ? getTreePeriodicChildInterfaceOwners(spec).map(
            (owner) => owner.index.dimension
          )
        : getStoredTreePeriodicChildInterfaceDimensions("branch", spec);
    return {
      rootChildDimensions,
      branchChildDimensions,
    };
  }

  function getTreePeriodicBoundaryTensorDimensions(
    boundaryTensor,
    activeCellName,
    interfaceDimensions
  ) {
    if (boundaryTensor.tree_periodic_role === "parent") {
      if (activeCellName === "branch") {
        return interfaceDimensions.rootChildDimensions;
      }
      if (activeCellName === "leaf") {
        return interfaceDimensions.branchChildDimensions;
      }
      return [];
    }
    if (activeCellName === "root") {
      return interfaceDimensions.rootChildDimensions;
    }
    if (activeCellName === "branch") {
      return interfaceDimensions.branchChildDimensions;
    }
    return [];
  }

  function syncTreePeriodicBoundaryTensors(
    spec = state.spec,
    interfaceDimensions = null
  ) {
    if (!isTreePeriodicMode(spec)) {
      return;
    }
    ensureActiveTreePeriodicBoundaryTensors(spec);
    const activeCellName = getActiveTreePeriodicCellName(spec);
    if (!activeCellName) {
      return;
    }
    const resolvedInterfaceDimensions =
      interfaceDimensions &&
      runtime.isObject(interfaceDimensions) &&
      Array.isArray(interfaceDimensions.rootChildDimensions) &&
      Array.isArray(interfaceDimensions.branchChildDimensions)
        ? interfaceDimensions
        : getCanonicalTreePeriodicInterfaceDimensions(spec);
    const boundaryTensors = (Array.isArray(spec.tensors) ? spec.tensors : []).filter(
      (tensor) => isTreePeriodicBoundaryTensor(tensor)
    );
    boundaryTensors.forEach((boundaryTensor) => {
      const resolvedDimensions = getTreePeriodicBoundaryTensorDimensions(
        boundaryTensor,
        activeCellName,
        resolvedInterfaceDimensions
      );
      const existingIndices = Array.isArray(boundaryTensor.indices)
        ? boundaryTensor.indices
        : [];
      const keptIndices = existingIndices.slice(0, resolvedDimensions.length);
      const removedIndexIds = new Set(
        existingIndices.slice(resolvedDimensions.length).map((index) => index.id)
      );
      if (removedIndexIds.size) {
        spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
          (edge) =>
            !removedIndexIds.has(edge.left && edge.left.index_id) &&
            !removedIndexIds.has(edge.right && edge.right.index_id)
        );
      }
      boundaryTensor.indices = resolvedDimensions.map((dimension, indexPosition) => {
        const existingIndex = keptIndices[indexPosition];
        return {
          id:
            existingIndex && existingIndex.id
              ? existingIndex.id
              : runtime.makeId("index"),
          name: `slot_${indexPosition + 1}`,
          dimension,
          offset:
            existingIndex && existingIndex.offset
              ? existingIndex.offset
              : runtime.defaultIndexOffsetForOrder(indexPosition, boundaryTensor),
          metadata:
            existingIndex && runtime.isObject(existingIndex.metadata)
              ? existingIndex.metadata
              : {},
        };
      });
      runtime.ensureTensorIndexOffsets(boundaryTensor);
    });
  }

  function seedTreePeriodicCell(
    cellName,
    graphSection,
    branchingFactor = getTreePeriodicBranchingFactor(),
    interfaceDimensions = null
  ) {
    const runtimeSpec = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    runtimeSpec.contraction_plan = null;
    runtimeSpec.tree_periodic_tree = {
      active_cell: cellName,
      branching_factor: branchingFactor,
      metadata: {},
    };
    ensureActiveTreePeriodicBoundaryTensors(runtimeSpec);
    syncTreePeriodicBoundaryTensors(runtimeSpec, interfaceDimensions);
    return runtime.buildGraphSectionFromSpec(runtimeSpec);
  }

  function syncTreePeriodicTreeInterfaceDimensions(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    const activeCellName = getActiveTreePeriodicCellName(spec);
    const cellKey = getTreePeriodicCellKey(activeCellName);
    if (!tree || !activeCellName || !cellKey) {
      return spec;
    }
    const branchingFactor = getTreePeriodicBranchingFactor(spec);
    const activeCell = getTreePeriodicCell(spec, activeCellName);
    const interfaceDimensions = getCanonicalTreePeriodicInterfaceDimensions(spec);

    tree[cellKey] = runtime.buildGraphSectionFromSpec(spec, activeCell);
    tree[cellKey].contraction_plan = null;

    TREE_PERIODIC_CELL_ORDER.forEach((cellName) => {
      const cellSpec = runtime.normalizeGraphSectionInPlace(
        runtime.deepClone(
          getTreePeriodicCell(spec, cellName) || runtime.buildEmptyGraphSection()
        )
      );
      cellSpec.contraction_plan = null;
      tree[getTreePeriodicCellKey(cellName)] = seedTreePeriodicCell(
        cellName,
        cellSpec,
        branchingFactor,
        interfaceDimensions
      );
    });

    runtime.replaceGraphSectionOnSpec(
      spec,
      getTreePeriodicCell(spec, activeCellName) || runtime.buildEmptyGraphSection()
    );

    return spec;
  }

  function syncCurrentGraphIntoTreePeriodicTree(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    if (!tree) {
      return spec;
    }
    spec.contraction_plan = null;
    syncTreePeriodicBoundaryTensors(spec);
    syncTreePeriodicTreeInterfaceDimensions(spec);
    return spec;
  }

  function invalidateActiveTreePeriodicLookups(spec = state.spec) {
    if (spec !== state.spec) {
      return;
    }
    state.lookupRevision = -1;
    if (typeof ctx.resetDerivedStateCaches === "function") {
      ctx.resetDerivedStateCaches();
    }
  }

  function hydrateActiveTreePeriodicCell(spec = state.spec) {
    const tree = getTreePeriodicTree(spec);
    if (!tree) {
      return spec;
    }
    tree.active_cell = TREE_PERIODIC_CELL_ORDER.includes(tree.active_cell)
      ? tree.active_cell
      : "root";
    const activeCell = getTreePeriodicCell(spec, tree.active_cell);
    runtime.replaceGraphSectionOnSpec(
      spec,
      activeCell || runtime.buildEmptyGraphSection()
    );
    spec.contraction_plan = null;
    invalidateActiveTreePeriodicLookups(spec);
    syncTreePeriodicBoundaryTensors(spec);
    return spec;
  }

  function stripTreePeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    stripped.contraction_plan = null;
    const boundaryTensorIds = new Set(
      stripped.tensors
        .filter((tensor) => isTreePeriodicBoundaryTensor(tensor))
        .map((tensor) => tensor.id)
    );
    stripped.tensors = stripped.tensors.filter(
      (tensor) => !boundaryTensorIds.has(tensor.id)
    );
    stripped.edges = stripped.edges.filter(
      (edge) =>
        !boundaryTensorIds.has(edge.left && edge.left.tensor_id) &&
        !boundaryTensorIds.has(edge.right && edge.right.tensor_id)
    );
    stripped.groups = stripped.groups
      .map((group) => ({
        ...group,
        tensor_ids: group.tensor_ids.filter(
          (tensorId) => !boundaryTensorIds.has(tensorId)
        ),
      }))
      .filter((group) => group.tensor_ids.length > 0);
    return stripped;
  }

  function resetTransientEditorStateForCellSwitch() {
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerInspectionStepCount = null;
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    state.activeTensorDrag = null;
    state.activeIndexDrag = null;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.boxSelection = null;
    state.connectMode = false;
  }

  function switchTreePeriodicCell(direction) {
    if (!isTreePeriodicMode()) {
      return;
    }
    const activeCellName = getActiveTreePeriodicCellName();
    const nextCellName = getTreePeriodicNeighborCellName(activeCellName, direction);
    if (!nextCellName) {
      return;
    }
    syncCurrentGraphIntoTreePeriodicTree();
    state.spec.tree_periodic_tree.active_cell = nextCellName;
    hydrateActiveTreePeriodicCell();
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    runtime.reconcileTensorOrder();
    resetTransientEditorStateForCellSwitch();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.setStatus(
      `Editing ${getTreePeriodicCellLabel(nextCellName).toLowerCase()}.`,
      "success"
    );
  }

  function buildTreePeriodicSeedGraphSection() {
    if (
      typeof runtime.isLinearPeriodicMode === "function" &&
      runtime.isLinearPeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain();
      const strippedLinearCell =
        typeof runtime.stripLinearPeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripLinearPeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.linear_periodic_chain = null;
      runtime.replaceGraphSectionOnSpec(state.spec, strippedLinearCell);
      return strippedLinearCell;
    }
    if (
      typeof runtime.isGridPeriodicMode === "function" &&
      runtime.isGridPeriodicMode()
    ) {
      runtime.syncCurrentGraphIntoGridPeriodicGrid();
      const strippedGridCell =
        typeof runtime.stripGridPeriodicBoundaryTensorsFromGraphSection === "function"
          ? runtime.stripGridPeriodicBoundaryTensorsFromGraphSection(
              runtime.buildGraphSectionFromSpec(state.spec)
            )
          : runtime.buildGraphSectionFromSpec(state.spec);
      state.spec.grid_periodic_grid = null;
      runtime.replaceGraphSectionOnSpec(state.spec, strippedGridCell);
      return strippedGridCell;
    }
    return runtime.buildGraphSectionFromSpec(state.spec);
  }

  function readTreePeriodicBranchingFactor() {
    if (!window || typeof window.prompt !== "function") {
      return null;
    }
    const response = window.prompt(
      "How many child branches should each tree node have?",
      "2"
    );
    if (response === null) {
      return null;
    }
    const trimmedResponse = String(response).trim();
    const branchingFactor = Number(trimmedResponse);
    if (!Number.isInteger(branchingFactor) || branchingFactor < 2) {
      ctx.setStatus(
        "Enter an integer branching factor of 2 or more to enable For Tree mode.",
        "error"
      );
      return null;
    }
    return branchingFactor;
  }

  function toggleTreePeriodicMode() {
    if (!state.spec) {
      return;
    }
    if (
      !isTreePeriodicMode() &&
      typeof runtime.isBenchmarkMode === "function" &&
      runtime.isBenchmarkMode()
    ) {
      ctx.setStatus(
        "Leave Benchmark mode before enabling For Tree mode.",
        "error"
      );
      return;
    }
    if (isTreePeriodicMode()) {
      if (
        !window.confirm(
          "Leave For Tree mode and keep the current cell as a normal network?"
        )
      ) {
        return;
      }
      syncCurrentGraphIntoTreePeriodicTree();
      const plainActiveCell = stripTreePeriodicBoundaryTensorsFromGraphSection(
        runtime.buildGraphSectionFromSpec(state.spec)
      );
      state.spec.tree_periodic_tree = null;
      runtime.replaceGraphSectionOnSpec(state.spec, plainActiveCell);
      if (typeof ctx.bumpSpecRevision === "function") {
        ctx.bumpSpecRevision();
      }
      runtime.reconcileTensorOrder();
      resetTransientEditorStateForCellSwitch();
      if (typeof ctx.clearGeneratedCodePreview === "function") {
        ctx.clearGeneratedCodePreview();
      }
      ctx.render();
      if (typeof ctx.refreshContractionAnalysis === "function") {
        ctx.refreshContractionAnalysis();
      }
      ctx.setStatus(
        "For Tree mode disabled. Restored the active cell as a normal network.",
        "success"
      );
      return;
    }

    const branchingFactor = readTreePeriodicBranchingFactor();
    if (branchingFactor === null) {
      return;
    }
    const rootCell = seedTreePeriodicCell(
      "root",
      buildTreePeriodicSeedGraphSection(),
      branchingFactor
    );
    const interfaceDimensions = {
      rootChildDimensions: getTreePeriodicBoundaryInterfaceDimensions(
        rootCell,
        "child",
        0
      ),
      branchChildDimensions: [],
    };
    const branchCell = seedTreePeriodicCell(
      "branch",
      runtime.buildEmptyGraphSection(),
      branchingFactor,
      interfaceDimensions
    );
    const leafCell = seedTreePeriodicCell(
      "leaf",
      runtime.buildEmptyGraphSection(),
      branchingFactor,
      interfaceDimensions
    );
    state.spec.linear_periodic_chain = null;
    state.spec.grid_periodic_grid = null;
    state.spec.tree_periodic_tree = {
      active_cell: "root",
      branching_factor: branchingFactor,
      root_cell: rootCell,
      branch_cell: branchCell,
      leaf_cell: leafCell,
      metadata: {},
    };
    hydrateActiveTreePeriodicCell();
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    runtime.reconcileTensorOrder();
    resetTransientEditorStateForCellSwitch();
    if (typeof ctx.clearGeneratedCodePreview === "function") {
      ctx.clearGeneratedCodePreview();
    }
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.setStatus(
      "For Tree mode enabled. You are editing the root cell.",
      "success"
    );
  }

  function setTreePeriodicMode(enabled) {
    const shouldEnable = Boolean(enabled);
    if (shouldEnable === isTreePeriodicMode()) {
      return;
    }
    toggleTreePeriodicMode();
  }

  return {
    normalizeTreePeriodicTreeInPlace,
    getTreePeriodicTree,
    isTreePeriodicMode,
    getTreePeriodicBranchingFactor,
    getActiveTreePeriodicCellName,
    getTreePeriodicCellKey,
    getTreePeriodicCell,
    getTreePeriodicCellLabel,
    getTreePeriodicNeighborCellName,
    canSwitchTreePeriodicCell,
    isTreePeriodicBoundaryTensor,
    syncTreePeriodicBoundaryTensors,
    syncCurrentGraphIntoTreePeriodicTree,
    hydrateActiveTreePeriodicCell,
    stripTreePeriodicBoundaryTensorsFromGraphSection,
    switchTreePeriodicCell,
    toggleTreePeriodicMode,
    setTreePeriodicMode,
  };
}
