import { LINEAR_PERIODIC_BOUNDARY_SETTINGS } from "./utilitiesLinearPeriodicState.js";

export function createLinearPeriodicBoundarySupport({
  ctx,
  state,
  constants,
  runtime,
  linearState,
}) {
  const { TENSOR_WIDTH, TENSOR_HEIGHT } = constants;
  const {
    getActiveLinearPeriodicCellName,
    getLinearPeriodicChain,
    isLinearPeriodicMode,
    isLinearPeriodicBoundaryTensor,
    isContractibleBoundaryTensor,
    getContractibleTensors,
    getExpectedLinearPeriodicRoles,
  } = linearState;

  function getRealTensorBounds(spec = state.spec) {
    const tensors = getContractibleTensors(spec);
    if (!tensors.length) {
      return {
        minX: 120,
        maxX: 320,
        centerY: 140,
      };
    }
    const leftEdges = tensors.map(
      (tensor) => tensor.position.x - runtime.tensorWidth(tensor) / 2
    );
    const rightEdges = tensors.map(
      (tensor) => tensor.position.x + runtime.tensorWidth(tensor) / 2
    );
    const centersY = tensors.map((tensor) => tensor.position.y);
    return {
      minX: Math.min(...leftEdges),
      maxX: Math.max(...rightEdges),
      centerY: centersY.reduce((sum, value) => sum + value, 0) / centersY.length,
    };
  }

  function positionLinearPeriodicBoundaryTensor(tensor, role, spec = state.spec) {
    const bounds = getRealTensorBounds(spec);
    tensor.position = {
      x: role === "previous" ? bounds.minX - 220 : bounds.maxX + 220,
      y: bounds.centerY,
    };
  }

  function createLinearPeriodicBoundaryTensor(role, spec = state.spec) {
    const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[role];
    const tensor = {
      id: runtime.makeId("tensor"),
      name: settings.name,
      position: { x: 0, y: 0 },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      linear_periodic_role: role,
      metadata: { color: settings.color },
    };
    positionLinearPeriodicBoundaryTensor(tensor, role, spec);
    return tensor;
  }

  function ensureActiveLinearPeriodicBoundaryTensors(spec = state.spec) {
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!activeCellName) {
      return;
    }
    const expectedRoles = new Set(getExpectedLinearPeriodicRoles(activeCellName));
    const nextTensors = [];
    const seenRoles = new Set();
    (Array.isArray(spec.tensors) ? spec.tensors : []).forEach((tensor) => {
      if (!isLinearPeriodicBoundaryTensor(tensor)) {
        nextTensors.push(tensor);
        return;
      }
      if (
        !expectedRoles.has(tensor.linear_periodic_role) ||
        seenRoles.has(tensor.linear_periodic_role)
      ) {
        return;
      }
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[tensor.linear_periodic_role];
      tensor.name = settings.name;
      tensor.metadata = runtime.isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.metadata.color = runtime.getMetadataColor(tensor.metadata, settings.color);
      seenRoles.add(tensor.linear_periodic_role);
      nextTensors.push(tensor);
    });
    getExpectedLinearPeriodicRoles(activeCellName).forEach((role) => {
      if (!seenRoles.has(role)) {
        nextTensors.push(createLinearPeriodicBoundaryTensor(role, spec));
      }
    });
    spec.tensors = nextTensors;
    const allowedBoundaryIds = new Set(
      spec.tensors
        .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
        .map((tensor) => tensor.id)
    );
    spec.edges = (Array.isArray(spec.edges) ? spec.edges : []).filter((edge) => {
      const touchesBoundary =
        allowedBoundaryIds.has(edge.left && edge.left.tensor_id) ||
        allowedBoundaryIds.has(edge.right && edge.right.tensor_id);
      if (!touchesBoundary) {
        return true;
      }
      return (
        allowedBoundaryIds.has(edge.left && edge.left.tensor_id) ||
        allowedBoundaryIds.has(edge.right && edge.right.tensor_id)
      );
    });
  }

  function getLinearPeriodicCandidateOwners(spec = state.spec) {
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
        !isContractibleBoundaryTensor(leftTensor) &&
        !isContractibleBoundaryTensor(rightTensor)
      ) {
        internallyConnectedIndexIds.add(edge.left.index_id);
        internallyConnectedIndexIds.add(edge.right.index_id);
      }
    });
    const owners = [];
    getContractibleTensors(spec).forEach((tensor) => {
      tensor.indices.forEach((index, indexPosition) => {
        if (!internallyConnectedIndexIds.has(index.id)) {
          owners.push({ tensor, index, indexPosition });
        }
      });
    });
    return owners;
  }

  function getBoundaryInterfaceDimensions(graphSection, preferredRole = "next") {
    const boundaryTensors = (
      Array.isArray(graphSection && graphSection.tensors) ? graphSection.tensors : []
    ).filter((tensor) => isLinearPeriodicBoundaryTensor(tensor));
    const preferredTensor = boundaryTensors.find(
      (tensor) =>
        tensor.linear_periodic_role === preferredRole &&
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

  function getCanonicalLinearPeriodicInterfaceDimensions(spec = state.spec) {
    const activeCellName = getActiveLinearPeriodicCellName(spec);
    if (!activeCellName) {
      return [];
    }
    if (activeCellName === "initial") {
      return getLinearPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
    }
    const activePreferredRole = activeCellName === "final" ? "previous" : "next";
    const activeDimensions = getBoundaryInterfaceDimensions(spec, activePreferredRole);
    if (activeDimensions.length) {
      return activeDimensions;
    }
    const chain = getLinearPeriodicChain(spec);
    if (!chain) {
      return getLinearPeriodicCandidateOwners(spec).map(
        (owner) => owner.index.dimension
      );
    }
    const initialDimensions = getBoundaryInterfaceDimensions(chain.initial_cell, "next");
    return initialDimensions.length
      ? initialDimensions
      : getLinearPeriodicCandidateOwners(spec).map(
          (owner) => owner.index.dimension
        );
  }

  function syncLinearPeriodicBoundaryTensors(
    spec = state.spec,
    interfaceDimensions = null
  ) {
    if (!isLinearPeriodicMode(spec)) {
      return;
    }
    ensureActiveLinearPeriodicBoundaryTensors(spec);
    const resolvedInterfaceDimensions = Array.isArray(interfaceDimensions)
      ? interfaceDimensions
      : getCanonicalLinearPeriodicInterfaceDimensions(spec);
    const boundaryTensors = (Array.isArray(spec.tensors) ? spec.tensors : []).filter(
      (tensor) => isLinearPeriodicBoundaryTensor(tensor)
    );
    boundaryTensors.forEach((boundaryTensor) => {
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
      const settings = LINEAR_PERIODIC_BOUNDARY_SETTINGS[boundaryTensor.linear_periodic_role];
      boundaryTensor.name = settings.name;
      boundaryTensor.metadata = runtime.isObject(boundaryTensor.metadata)
        ? boundaryTensor.metadata
        : {};
      boundaryTensor.metadata.color = runtime.getMetadataColor(
        boundaryTensor.metadata,
        settings.color
      );
      runtime.ensureTensorIndexOffsets(boundaryTensor);
    });
  }

  function stripLinearPeriodicBoundaryTensorsFromGraphSection(graphSection) {
    const stripped = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || runtime.buildEmptyGraphSection())
    );
    const boundaryTensorIds = new Set(
      stripped.tensors
        .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
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
        tensor_ids: group.tensor_ids.filter((tensorId) => !boundaryTensorIds.has(tensorId)),
      }))
      .filter((group) => group.tensor_ids.length > 0);
    return stripped;
  }

  return {
    getRealTensorBounds,
    positionLinearPeriodicBoundaryTensor,
    createLinearPeriodicBoundaryTensor,
    ensureActiveLinearPeriodicBoundaryTensors,
    getLinearPeriodicCandidateOwners,
    getBoundaryInterfaceDimensions,
    getCanonicalLinearPeriodicInterfaceDimensions,
    syncLinearPeriodicBoundaryTensors,
    stripLinearPeriodicBoundaryTensorsFromGraphSection,
  };
}
