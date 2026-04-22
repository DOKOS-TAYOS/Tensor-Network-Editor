import {
  GRID_PERIODIC_BOUNDARY_SETTINGS,
  GRID_PERIODIC_CELL_ORDER,
  GRID_PERIODIC_FAMILY_BY_CELL_ROLE,
} from "./utilitiesGridPeriodicState.js";

export function createGridPeriodicBoundarySupport({
  ctx,
  state,
  constants,
  runtime,
  gridState,
}) {
  const { TENSOR_WIDTH, TENSOR_HEIGHT } = constants;
  const {
    getGridPeriodicGrid,
    isGridPeriodicMode,
    getActiveGridPeriodicCellName,
    getGridPeriodicCell,
    isGridPeriodicBoundaryTensor,
    isForBoundaryTensor,
    getExpectedGridPeriodicRoles,
  } = gridState;

  function getRealTensorBounds(spec = state.spec) {
    const tensors = (Array.isArray(spec && spec.tensors) ? spec.tensors : []).filter(
      (tensor) => !isForBoundaryTensor(tensor)
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

  function positionGridPeriodicBoundaryTensor(tensor, role, spec = state.spec) {
    const bounds = getRealTensorBounds(spec);
    if (role === "up") {
      tensor.position = { x: bounds.centerX, y: bounds.minY - 180 };
      return;
    }
    if (role === "right") {
      tensor.position = { x: bounds.maxX + 220, y: bounds.centerY };
      return;
    }
    if (role === "down") {
      tensor.position = { x: bounds.centerX, y: bounds.maxY + 180 };
      return;
    }
    tensor.position = { x: bounds.minX - 220, y: bounds.centerY };
  }

  function createGridPeriodicBoundaryTensor(role, spec = state.spec) {
    const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[role];
    const tensor = {
      id: runtime.makeId("tensor"),
      name: settings.name,
      position: { x: 0, y: 0 },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      grid_periodic_role: role,
      metadata: { color: settings.color },
    };
    positionGridPeriodicBoundaryTensor(tensor, role, spec);
    return tensor;
  }

  function ensureActiveGridPeriodicBoundaryTensors(spec = state.spec) {
    const activeCellName = getActiveGridPeriodicCellName(spec);
    if (!activeCellName) {
      return;
    }
    const expectedRoles = new Set(getExpectedGridPeriodicRoles(activeCellName));
    const nextTensors = [];
    const removedBoundaryIds = new Set();
    const seenRoles = new Set();
    (Array.isArray(spec.tensors) ? spec.tensors : []).forEach((tensor) => {
      if (!isGridPeriodicBoundaryTensor(tensor)) {
        nextTensors.push(tensor);
        return;
      }
      if (
        !expectedRoles.has(tensor.grid_periodic_role) ||
        seenRoles.has(tensor.grid_periodic_role)
      ) {
        removedBoundaryIds.add(tensor.id);
        return;
      }
      const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[tensor.grid_periodic_role];
      tensor.name = settings.name;
      tensor.metadata = runtime.isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.metadata.color = runtime.getMetadataColor(tensor.metadata, settings.color);
      seenRoles.add(tensor.grid_periodic_role);
      positionGridPeriodicBoundaryTensor(tensor, tensor.grid_periodic_role, spec);
      nextTensors.push(tensor);
    });
    getExpectedGridPeriodicRoles(activeCellName).forEach((role) => {
      if (!seenRoles.has(role)) {
        nextTensors.push(createGridPeriodicBoundaryTensor(role, spec));
      }
    });
    spec.tensors = nextTensors;
    if (removedBoundaryIds.size) {
      spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
        (edge) =>
          !removedBoundaryIds.has(edge.left && edge.left.tensor_id) &&
          !removedBoundaryIds.has(edge.right && edge.right.tensor_id)
      );
    }
  }

  function getGridPeriodicCandidateOwners(spec = state.spec) {
    const isActiveSpec = spec === state.spec;
    const tensors = Array.isArray(spec && spec.tensors) ? spec.tensors : [];
    const edges = Array.isArray(spec && spec.edges) ? spec.edges : [];
    if (isActiveSpec && typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    const tensorById = isActiveSpec
      ? state.tensorById
      : Object.fromEntries(tensors.map((tensor) => [tensor.id, tensor]));
    const internallyConnectedIndexIds = new Set();
    edges.forEach((edge) => {
      const leftTensor = tensorById[edge.left && edge.left.tensor_id];
      const rightTensor = tensorById[edge.right && edge.right.tensor_id];
      if (
        leftTensor &&
        rightTensor &&
        !isForBoundaryTensor(leftTensor) &&
        !isForBoundaryTensor(rightTensor)
      ) {
        internallyConnectedIndexIds.add(edge.left.index_id);
        internallyConnectedIndexIds.add(edge.right.index_id);
      }
    });
    const owners = [];
    tensors.forEach((tensor) => {
      if (isForBoundaryTensor(tensor)) {
        return;
      }
      tensor.indices.forEach((index, indexPosition) => {
        if (!internallyConnectedIndexIds.has(index.id)) {
          owners.push({ tensor, index, indexPosition });
        }
      });
    });
    return owners;
  }

  function getGridPeriodicBoundaryInterfaceDimensions(
    graphSection,
    preferredRole = "right"
  ) {
    const boundaryTensors = (
      Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : []
    ).filter((tensor) => isGridPeriodicBoundaryTensor(tensor));
    const preferredTensor = boundaryTensors.find(
      (tensor) =>
        tensor.grid_periodic_role === preferredRole &&
        Array.isArray(tensor.indices) &&
        tensor.indices.length
    );
    const fallbackTensor =
      preferredTensor ||
      boundaryTensors.find(
        (tensor) => Array.isArray(tensor.indices) && tensor.indices.length
      );
    return fallbackTensor
      ? fallbackTensor.indices.map((index) => index.dimension)
      : [];
  }

  function buildEmptyGridPeriodicFamilyDimensions() {
    return {
      row_top: [],
      row_middle: [],
      row_bottom: [],
      column_left: [],
      column_center: [],
      column_right: [],
    };
  }

  function getGridPeriodicRoleFamily(cellName, role) {
    if (
      !Object.prototype.hasOwnProperty.call(GRID_PERIODIC_FAMILY_BY_CELL_ROLE, cellName)
    ) {
      return null;
    }
    const roleFamilies = GRID_PERIODIC_FAMILY_BY_CELL_ROLE[cellName];
    return Object.prototype.hasOwnProperty.call(roleFamilies, role)
      ? roleFamilies[role]
      : null;
  }

  function assignGridPeriodicFamilyDimensionsFromCell(
    familyDimensions,
    cellName,
    graphSection
  ) {
    getExpectedGridPeriodicRoles(cellName).forEach((role) => {
      const familyKey = getGridPeriodicRoleFamily(cellName, role);
      if (!familyKey || familyDimensions[familyKey].length) {
        return;
      }
      const candidateDimensions = getGridPeriodicBoundaryInterfaceDimensions(
        graphSection,
        role
      );
      if (candidateDimensions.length) {
        familyDimensions[familyKey] = [...candidateDimensions];
      }
    });
  }

  function getCanonicalGridPeriodicFamilyDimensions(spec = state.spec) {
    const familyDimensions = buildEmptyGridPeriodicFamilyDimensions();
    const activeCellName = getActiveGridPeriodicCellName(spec);
    if (activeCellName) {
      assignGridPeriodicFamilyDimensionsFromCell(familyDimensions, activeCellName, spec);
    }
    const grid = getGridPeriodicGrid(spec);
    if (grid) {
      GRID_PERIODIC_CELL_ORDER.forEach((cellName) => {
        assignGridPeriodicFamilyDimensionsFromCell(
          familyDimensions,
          cellName,
          getGridPeriodicCell(spec, cellName)
        );
      });
    }
    if (
      !Object.values(familyDimensions).some((dimensions) => dimensions.length) &&
      activeCellName
    ) {
      const ownerDimensions = getGridPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
      const activeRoles = getExpectedGridPeriodicRoles(activeCellName);
      activeRoles.forEach((role) => {
        const familyKey = getGridPeriodicRoleFamily(activeCellName, role);
        if (familyKey && !familyDimensions[familyKey].length) {
          familyDimensions[familyKey] = [...ownerDimensions];
        }
      });
    }
    return familyDimensions;
  }

  function syncGridPeriodicBoundaryTensors(
    spec = state.spec,
    familyDimensions = null
  ) {
    if (!isGridPeriodicMode(spec)) {
      return;
    }
    ensureActiveGridPeriodicBoundaryTensors(spec);
    const activeCellName = getActiveGridPeriodicCellName(spec);
    const resolvedFamilyDimensions = runtime.isObject(familyDimensions)
      ? familyDimensions
      : getCanonicalGridPeriodicFamilyDimensions(spec);
    const boundaryTensors = (Array.isArray(spec.tensors) ? spec.tensors : []).filter(
      (tensor) => isGridPeriodicBoundaryTensor(tensor)
    );
    boundaryTensors.forEach((boundaryTensor) => {
      const familyKey = getGridPeriodicRoleFamily(
        activeCellName,
        boundaryTensor.grid_periodic_role
      );
      const resolvedInterfaceDimensions =
        familyKey && Array.isArray(resolvedFamilyDimensions[familyKey])
          ? resolvedFamilyDimensions[familyKey]
          : [];
      const existingIndices = Array.isArray(boundaryTensor.indices)
        ? boundaryTensor.indices
        : [];
      const keptIndices = existingIndices.slice(0, resolvedInterfaceDimensions.length);
      const removedIndexIds = new Set(
        existingIndices.slice(resolvedInterfaceDimensions.length).map((index) => index.id)
      );
      if (removedIndexIds.size) {
        spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter(
          (edge) =>
            !removedIndexIds.has(edge.left && edge.left.index_id) &&
            !removedIndexIds.has(edge.right && edge.right.index_id)
        );
      }
      boundaryTensor.indices = resolvedInterfaceDimensions.map(
        (dimension, indexPosition) => {
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
        }
      );
      const settings = GRID_PERIODIC_BOUNDARY_SETTINGS[boundaryTensor.grid_periodic_role];
      boundaryTensor.name = settings.name;
      boundaryTensor.metadata = runtime.isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = runtime.getMetadataColor(
        boundaryTensor.metadata,
        settings.color
      );
      runtime.ensureTensorIndexOffsets(boundaryTensor);
      positionGridPeriodicBoundaryTensor(
        boundaryTensor,
        boundaryTensor.grid_periodic_role,
        spec
      );
    });
  }

  function seedGridPeriodicCell(cellName, graphSection, familyDimensions = null) {
    const runtimeSpec = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    runtimeSpec.contraction_plan = null;
    runtimeSpec.grid_periodic_grid = { active_cell: cellName };
    ensureActiveGridPeriodicBoundaryTensors(runtimeSpec);
    syncGridPeriodicBoundaryTensors(runtimeSpec, familyDimensions);
    return runtime.buildGraphSectionFromSpec(runtimeSpec);
  }

  function buildActiveGridPeriodicCellSnapshot(spec = state.spec) {
    const graphSection = runtime.buildGraphSectionFromSpec(spec);
    graphSection.contraction_plan = null;
    return graphSection;
  }

  return {
    getRealTensorBounds,
    positionGridPeriodicBoundaryTensor,
    createGridPeriodicBoundaryTensor,
    ensureActiveGridPeriodicBoundaryTensors,
    getGridPeriodicCandidateOwners,
    getGridPeriodicBoundaryInterfaceDimensions,
    buildEmptyGridPeriodicFamilyDimensions,
    getGridPeriodicRoleFamily,
    assignGridPeriodicFamilyDimensionsFromCell,
    getCanonicalGridPeriodicFamilyDimensions,
    syncGridPeriodicBoundaryTensors,
    seedGridPeriodicCell,
    buildActiveGridPeriodicCellSnapshot,
  };
}
