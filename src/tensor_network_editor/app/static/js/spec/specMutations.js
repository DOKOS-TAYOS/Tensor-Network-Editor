import { GRAPH_THEME } from "../core/theme.js";

export function createSpecMutationBindings({
  ctx,
  state,
  constants,
  runtime,
  findTensorById,
  findEdgeById,
  findHyperedgeById,
  findIndexOwner,
  findEdgeByIndexId,
  findHyperedgeByIndexId,
  findConnectionByIndexId,
  resolveBaseEdgeId,
  resolveBaseHyperedgeId,
}) {
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
  } = constants;
  const INDEX_RADIUS =
    Number.isFinite(constants && constants.INDEX_RADIUS) &&
    constants.INDEX_RADIUS >= 0
      ? constants.INDEX_RADIUS
      : 15;
  const INDEX_PADDING =
    Number.isFinite(constants && constants.INDEX_PADDING) &&
    constants.INDEX_PADDING >= 0
      ? constants.INDEX_PADDING
      : 8;
  const INDEX_PLACEMENT_STEP = Math.max(INDEX_RADIUS * 2 + INDEX_PADDING, 18);

  function tensorWidth(tensor) {
    return Math.max(
      TENSOR_WIDTH,
      runtime.asFiniteNumber(tensor && tensor.size && tensor.size.width, TENSOR_WIDTH)
    );
  }

  function tensorHeight(tensor) {
    return Math.max(
      TENSOR_HEIGHT,
      runtime.asFiniteNumber(tensor && tensor.size && tensor.size.height, TENSOR_HEIGHT)
    );
  }

  function buildDistributedAxisOffsets(start, end) {
    const span = Math.abs(end - start);
    const count = Math.max(1, Math.floor(span / INDEX_PLACEMENT_STEP) + 1);
    if (count === 1) {
      return [(start + end) / 2];
    }
    const step = (end - start) / (count - 1);
    return Array.from({ length: count }, (_, index) => start + step * index);
  }

  function normalizeOffsetKey(offset) {
    return [
      Math.round(runtime.asFiniteNumber(offset && offset.x, 0) * 1000),
      Math.round(runtime.asFiniteNumber(offset && offset.y, 0) * 1000),
    ].join(":");
  }

  function buildBoundaryIndexOffsetCandidates(tensor) {
    const horizontalExtent =
      tensorWidth(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;
    const verticalExtent =
      tensorHeight(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;
    const xOffsets = buildDistributedAxisOffsets(
      -horizontalExtent,
      horizontalExtent
    );
    const yOffsets = buildDistributedAxisOffsets(-verticalExtent, verticalExtent);
    const candidates = [];
    const candidateKeys = new Set();
    function pushCandidate(offset) {
      const key = normalizeOffsetKey(offset);
      if (candidateKeys.has(key)) {
        return;
      }
      candidateKeys.add(key);
      candidates.push(offset);
    }

    xOffsets.forEach((xOffset) => {
      pushCandidate({ x: xOffset, y: -verticalExtent });
      pushCandidate({ x: xOffset, y: verticalExtent });
    });
    yOffsets.forEach((yOffset) => {
      pushCandidate({ x: -horizontalExtent, y: yOffset });
      pushCandidate({ x: horizontalExtent, y: yOffset });
    });

    return candidates;
  }

  function findAvailableIndexOffset(tensor, indexPosition) {
    const occupiedOffsetKeys = new Set(
      (Array.isArray(tensor && tensor.indices) ? tensor.indices : []).map((index) =>
        normalizeOffsetKey(index.offset)
      )
    );
    const maxDefaultOrder = Math.max(
      indexPosition + 12,
      (Array.isArray(tensor && tensor.indices) ? tensor.indices.length : 0) + 12
    );

    for (let order = 0; order <= maxDefaultOrder; order += 1) {
      const candidate = runtime.defaultIndexOffsetForOrder(order, tensor);
      if (!occupiedOffsetKeys.has(normalizeOffsetKey(candidate))) {
        return candidate;
      }
    }

    const boundaryCandidates = buildBoundaryIndexOffsetCandidates(tensor);
    for (const candidate of boundaryCandidates) {
      if (!occupiedOffsetKeys.has(normalizeOffsetKey(candidate))) {
        return candidate;
      }
    }

    return runtime.defaultIndexOffsetForOrder(indexPosition, tensor);
  }

  function moveIndex(tensorId, indexPosition, direction) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const targetPosition = indexPosition + direction;
    if (targetPosition < 0 || targetPosition >= tensor.indices.length) {
      return;
    }
    const [index] = tensor.indices.splice(indexPosition, 1);
    tensor.indices.splice(targetPosition, 0, index);
  }

  function removeTensor(tensorId) {
    const tensor = findTensorById(tensorId);
    if (
      !tensor ||
      (typeof runtime.isForBoundaryTensor === "function" &&
        runtime.isForBoundaryTensor(tensor)) ||
      runtime.isLinearPeriodicBoundaryTensor(tensor)
    ) {
      return;
    }
    const tensorIndexIds = new Set(tensor.indices.map((index) => index.id));
    state.spec.edges = state.spec.edges.filter(
      (edge) =>
        !tensorIndexIds.has(edge.left.index_id) &&
        !tensorIndexIds.has(edge.right.index_id)
    );
    state.spec.hyperedges = (Array.isArray(state.spec.hyperedges) ? state.spec.hyperedges : [])
      .filter((hyperedge) =>
        !(Array.isArray(hyperedge.endpoints) ? hyperedge.endpoints : []).some((endpoint) =>
          tensorIndexIds.has(endpoint.index_id)
        )
      );
    state.spec.tensors = state.spec.tensors.filter(
      (candidate) => candidate.id !== tensorId
    );
    state.spec.groups = state.spec.groups
      .map((group) => ({
        ...group,
        tensor_ids: group.tensor_ids.filter(
          (candidateId) => candidateId !== tensorId
        ),
      }))
      .filter((group) => group.tensor_ids.length > 0);
    state.tensorOrder = state.tensorOrder.filter(
      (candidateId) => candidateId !== tensorId
    );
  }

  function removeIndex(tensorId, indexId) {
    const tensor = findTensorById(tensorId);
    if (
      !tensor ||
      (typeof runtime.isForBoundaryTensor === "function" &&
        runtime.isForBoundaryTensor(tensor)) ||
      runtime.isLinearPeriodicBoundaryTensor(tensor)
    ) {
      return;
    }
    state.spec.edges = state.spec.edges.filter(
      (edge) => edge.left.index_id !== indexId && edge.right.index_id !== indexId
    );
    state.spec.hyperedges = (Array.isArray(state.spec.hyperedges) ? state.spec.hyperedges : [])
      .filter((hyperedge) =>
        !(Array.isArray(hyperedge.endpoints) ? hyperedge.endpoints : []).some(
          (endpoint) => endpoint.index_id === indexId
        )
      );
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function removeEdge(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId) || edgeId;
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== resolvedEdgeId);
  }

  function removeHyperedge(hyperedgeId) {
    const resolvedHyperedgeId = resolveBaseHyperedgeId(hyperedgeId) || hyperedgeId;
    state.spec.hyperedges = (Array.isArray(state.spec.hyperedges) ? state.spec.hyperedges : [])
      .filter((hyperedge) => hyperedge.id !== resolvedHyperedgeId);
  }

  function syncConnectedIndexDimension(indexId, nextDimension) {
    const connectedEdge = findEdgeByIndexId(indexId);
    if (connectedEdge) {
      const connectedIndexId =
        connectedEdge.left && connectedEdge.left.index_id === indexId
          ? connectedEdge.right && connectedEdge.right.index_id
          : connectedEdge.left && connectedEdge.left.index_id;
      if (!connectedIndexId) {
        return;
      }
      const connectedOwner = findIndexOwner(connectedIndexId);
      if (!connectedOwner || !connectedOwner.index) {
        return;
      }
      connectedOwner.index.dimension = nextDimension;
      return;
    }
    const connectedHyperedge = findHyperedgeByIndexId(indexId);
    if (!connectedHyperedge) {
      return;
    }
    (Array.isArray(connectedHyperedge.endpoints) ? connectedHyperedge.endpoints : []).forEach(
      (endpoint) => {
        if (endpoint.index_id === indexId) {
          return;
        }
        const connectedOwner = findIndexOwner(endpoint.index_id);
        if (connectedOwner && connectedOwner.index) {
          connectedOwner.index.dimension = nextDimension;
        }
      }
    );
  }

  function describeHyperedgeCandidate(indexIds = []) {
    const normalizedIndexIds = [...new Set(
      (Array.isArray(indexIds) ? indexIds : [])
        .map((indexId) => String(indexId || ""))
        .filter(Boolean)
    )];
    if (
      (typeof runtime.isForMode === "function" && runtime.isForMode()) ||
      (typeof runtime.isLinearPeriodicMode === "function" && runtime.isLinearPeriodicMode()) ||
      (typeof runtime.isGridPeriodicMode === "function" && runtime.isGridPeriodicMode()) ||
      (typeof runtime.isTreePeriodicMode === "function" && runtime.isTreePeriodicMode())
    ) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "Hyperedges are available only in normal mode.",
      };
    }
    if (typeof ctx.isBenchmarkMode === "function" && ctx.isBenchmarkMode()) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "Leave benchmark mode before creating hyperedges.",
      };
    }
    if (normalizedIndexIds.length < 3) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "Select at least three open indices to create a hyperedge.",
      };
    }
    const owners = normalizedIndexIds.map((indexId) => findIndexOwner(indexId));
    if (owners.some((owner) => !owner || !owner.tensor || !owner.index)) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "Only base graph indices can be used to create hyperedges.",
      };
    }
    const dimensions = [...new Set(owners.map((owner) => owner.index.dimension))];
    if (dimensions.length !== 1) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "All selected indices must share the same dimension.",
      };
    }
    const connectedIndexId = normalizedIndexIds.find((indexId) =>
      Boolean(findConnectionByIndexId(indexId))
    );
    if (connectedIndexId) {
      return {
        canCreate: false,
        indexIds: normalizedIndexIds,
        message: "All selected indices must be open before creating a hyperedge.",
      };
    }
    return {
      canCreate: true,
      dimension: dimensions[0],
      indexIds: normalizedIndexIds,
      message: `Create a hyperedge with ${normalizedIndexIds.length} endpoints of dimension ${dimensions[0]}.`,
    };
  }

  function createHyperedge(indexIds = []) {
    const candidate = describeHyperedgeCandidate(indexIds);
    if (!candidate.canCreate) {
      return null;
    }
    const hyperedge = {
      id: runtime.makeId("hyperedge"),
      name: runtime.nextName(
        "hyperedge",
        (Array.isArray(state.spec.hyperedges) ? state.spec.hyperedges : []).map(
          (existingHyperedge) => existingHyperedge.name
        )
      ),
      endpoints: candidate.indexIds
        .map((indexId) => findIndexOwner(indexId))
        .filter(Boolean)
        .map((owner) => ({
          tensor_id: owner.tensor.id,
          index_id: owner.index.id,
        })),
      hub_offset: { x: 0, y: 0 },
      metadata: {},
    };
    if (!Array.isArray(state.spec.hyperedges)) {
      state.spec.hyperedges = [];
    }
    state.spec.hyperedges.push(hyperedge);
    return hyperedge;
  }

  function createTensor(x, y) {
    const tensor = {
      id: runtime.makeId("tensor"),
      name: runtime.nextName("T", state.spec.tensors.map((tensor) => tensor.name)),
      position: { x, y },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      metadata: {},
    };
    tensor.indices.push(createIndex(tensor, 0));
    tensor.indices.push(createIndex(tensor, 1));
    return tensor;
  }

  function createIndex(tensor, indexPosition) {
    return {
      id: runtime.makeId("index"),
      name: runtime.nextName("i", tensor.indices.map((index) => index.name)),
      dimension: 2,
      offset: findAvailableIndexOffset(tensor, indexPosition),
      metadata: {},
    };
  }

  function applyColorToSelection(colorValue) {
    ctx.getSelectedEntries().forEach((entry) => {
      if (entry.kind === "tensor") {
        entry.tensor.metadata.color = colorValue;
      } else if (entry.kind === "index") {
        entry.located.index.metadata.color = colorValue;
      } else if (entry.kind === "edge") {
        entry.edge.metadata.color = colorValue;
      } else if (entry.kind === "hyperedge") {
        entry.hyperedge.metadata.color = colorValue;
      } else if (entry.kind === "group") {
        entry.group.metadata.color = colorValue;
      } else if (entry.kind === "note") {
        entry.note.metadata.color = colorValue;
      }
    });
  }

  function getEntryColor(entry) {
    if (entry.kind === "tensor") {
      return runtime.getMetadataColor(
        entry.tensor.metadata,
        GRAPH_THEME.tensorFallback
      );
    }
    if (entry.kind === "index") {
      return runtime.getMetadataColor(
        entry.located.index.metadata,
        runtime.getIndexColor(
          entry.located.index,
          Boolean(findConnectionByIndexId(entry.id))
        )
      );
    }
    if (entry.kind === "group") {
      return runtime.getMetadataColor(
        entry.group.metadata,
        GRAPH_THEME.groupDefault
      );
    }
    if (entry.kind === "note") {
      return runtime.getMetadataColor(
        entry.note.metadata,
        GRAPH_THEME.noteDefault
      );
    }
    if (entry.kind === "hyperedge") {
      return runtime.getMetadataColor(entry.hyperedge.metadata, GRAPH_THEME.edge);
    }
    return runtime.getMetadataColor(entry.edge.metadata, GRAPH_THEME.edge);
  }

  function getBatchColorValue(selectedEntries) {
    if (!selectedEntries.length) {
      return GRAPH_THEME.groupDefault;
    }
    return getEntryColor(selectedEntries[0]);
  }

  return {
    moveIndex,
    removeTensor,
    removeIndex,
    removeEdge,
    removeHyperedge,
    syncConnectedIndexDimension,
    describeHyperedgeCandidate,
    createHyperedge,
    createTensor,
    createIndex,
    applyColorToSelection,
    getEntryColor,
    getBatchColorValue,
  };
}
