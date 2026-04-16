export function createSpecLookupBindings({ ctx, state }) {
  function resetDerivedStateCaches() {
    state.contractibleCacheRevision = -1;
    state.contractibleCacheTensorRef = null;
    state.contractibleCacheTensorCount = -1;
    state.contractibleCacheEdgeRef = null;
    state.contractibleCacheEdgeCount = -1;
    state.contractibleTensorsCache = [];
    state.contractibleEdgesCache = [];
    state.contractibleCacheToken += 1;
    state.plannerOperandStateCacheRevision = -1;
    state.plannerOperandStateCacheStepsRef = null;
    state.plannerOperandStateCacheStepCount = -1;
    state.plannerOperandStateCacheContractibleToken = -1;
    state.plannerOperandStateCache = null;
    state.contractionProgressionCacheRevision = -1;
    state.contractionProgressionCacheStepsRef = null;
    state.contractionProgressionCacheStepCount = -1;
    state.contractionProgressionCacheContractibleToken = -1;
    state.contractionProgressionCache = null;
    state.contractionProgressionCacheToken += 1;
    state.contractionSceneCacheRevision = -1;
    state.contractionSceneCacheViewRevision = -1;
    state.contractionSceneCacheProgressionToken = -1;
    state.contractionSceneCacheByAppliedStepCount = {};
  }

  function bumpSpecRevision() {
    state.specRevision += 1;
    state.lookupRevision = -1;
    resetDerivedStateCaches();
  }

  function ensureSpecLookups() {
    if (!state.spec) {
      state.lookupRevision = state.specRevision;
      state.tensorById = {};
      state.edgeById = {};
      state.edgeByIndexId = {};
      state.groupById = {};
      state.indexOwnerById = {};
      state.groupsByTensorId = {};
      state.noteById = {};
      return;
    }
    if (state.lookupRevision === state.specRevision) {
      return;
    }

    const tensorById = {};
    const edgeById = {};
    const edgeByIndexId = {};
    const groupById = {};
    const indexOwnerById = {};
    const groupsByTensorId = {};
    const noteById = {};

    state.spec.tensors.forEach((tensor) => {
      tensorById[tensor.id] = tensor;
      tensor.indices.forEach((index, indexPosition) => {
        indexOwnerById[index.id] = { tensor, index, indexPosition };
      });
    });
    state.spec.edges.forEach((edge) => {
      edgeById[edge.id] = edge;
      edgeByIndexId[edge.left.index_id] = edge;
      edgeByIndexId[edge.right.index_id] = edge;
    });
    state.spec.groups.forEach((group) => {
      groupById[group.id] = group;
      group.tensor_ids.forEach((tensorId) => {
        if (!Array.isArray(groupsByTensorId[tensorId])) {
          groupsByTensorId[tensorId] = [];
        }
        groupsByTensorId[tensorId].push(group);
      });
    });
    state.spec.notes.forEach((note) => {
      noteById[note.id] = note;
    });

    state.tensorById = tensorById;
    state.edgeById = edgeById;
    state.edgeByIndexId = edgeByIndexId;
    state.groupById = groupById;
    state.indexOwnerById = indexOwnerById;
    state.groupsByTensorId = groupsByTensorId;
    state.noteById = noteById;
    state.lookupRevision = state.specRevision;
  }

  function findBaseIndexOwner(indexId) {
    ensureSpecLookups();
    return state.indexOwnerById[indexId] || null;
  }

  function resolveBaseEdgeId(edgeId) {
    if (!edgeId) {
      return null;
    }
    ensureSpecLookups();
    const baseEdge = state.edgeById[edgeId];
    if (baseEdge) {
      return baseEdge.id;
    }
    const visibleEdge =
      typeof ctx.findVisibleEdgeById === "function" ? ctx.findVisibleEdgeById(edgeId) : null;
    if (
      visibleEdge &&
      typeof visibleEdge.baseEdgeId === "string" &&
      visibleEdge.baseEdgeId
    ) {
      return visibleEdge.baseEdgeId;
    }
    return null;
  }

  function findTensorById(tensorId) {
    ensureSpecLookups();
    return state.tensorById[tensorId] || null;
  }

  function findGroupById(groupId) {
    ensureSpecLookups();
    return state.groupById[groupId] || null;
  }

  function findGroupsByTensorId(tensorId) {
    ensureSpecLookups();
    return state.groupsByTensorId[tensorId] || [];
  }

  function findEdgeById(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId);
    if (!resolvedEdgeId) {
      return null;
    }
    ensureSpecLookups();
    return state.edgeById[resolvedEdgeId] || null;
  }

  function findVisibleIndexOwner(indexId) {
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : [];
    for (const tensor of visibleTensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return null;
  }

  function findIndexOwner(indexId) {
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    return findVisibleIndexOwner(indexId);
  }

  function resolveConnectableIndexOwner(indexId) {
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    const visibleOwner = findVisibleIndexOwner(indexId);
    if (
      !visibleOwner ||
      typeof visibleOwner.index.sourceIndexId !== "string" ||
      !visibleOwner.index.sourceIndexId
    ) {
      return null;
    }
    return findBaseIndexOwner(visibleOwner.index.sourceIndexId);
  }

  function findEdgeByIndexId(indexId) {
    ensureSpecLookups();
    const baseEdge = state.edgeByIndexId[indexId];
    if (baseEdge) {
      return baseEdge;
    }
    const visibleEdges =
      typeof ctx.getVisibleEdges === "function" ? ctx.getVisibleEdges() : [];
    return (
      visibleEdges.find(
        (edge) => edge.leftIndexId === indexId || edge.rightIndexId === indexId
      ) || null
    );
  }

  return {
    resetDerivedStateCaches,
    bumpSpecRevision,
    ensureSpecLookups,
    findBaseIndexOwner,
    resolveBaseEdgeId,
    findTensorById,
    findGroupById,
    findGroupsByTensorId,
    findEdgeById,
    findVisibleIndexOwner,
    findIndexOwner,
    resolveConnectableIndexOwner,
    findEdgeByIndexId,
  };
}
