import { GRAPH_THEME } from "../core/theme.js";

export function createSpecMutationBindings({
  ctx,
  state,
  constants,
  runtime,
  findTensorById,
  findEdgeById,
  findIndexOwner,
  findEdgeByIndexId,
  resolveBaseEdgeId,
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
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function removeEdge(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId) || edgeId;
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== resolvedEdgeId);
  }

  function syncConnectedIndexDimension(indexId, nextDimension) {
    const connectedEdge = findEdgeByIndexId(indexId);
    if (!connectedEdge) {
      return;
    }
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
        runtime.getIndexColor(entry.located.index, Boolean(findEdgeByIndexId(entry.id)))
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
    syncConnectedIndexDimension,
    createTensor,
    createIndex,
    applyColorToSelection,
    getEntryColor,
    getBatchColorValue,
  };
}
