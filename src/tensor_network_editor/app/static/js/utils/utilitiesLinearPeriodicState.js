export const LINEAR_PERIODIC_CELL_ORDER = ["initial", "periodic", "final"];

export const LINEAR_PERIODIC_CELL_LABELS = {
  initial: "Initial cell",
  periodic: "Periodic cell",
  final: "Final cell",
};

export const LINEAR_PERIODIC_PREVIOUS_OPERAND_ID = "__linear_previous__";
export const LINEAR_PERIODIC_NEXT_OPERAND_ID = "__linear_next__";

export const LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE = {
  previous: LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
  next: LINEAR_PERIODIC_NEXT_OPERAND_ID,
};

export const LINEAR_PERIODIC_BOUNDARY_SETTINGS = {
  previous: {
    name: "Previous cell",
    color: "#456cbf",
  },
  next: {
    name: "Next cell",
    color: "#2f9b8f",
  },
};

export function createLinearPeriodicStateSupport({ ctx, state, runtime }) {
  function getLinearPeriodicReservedOperandId(role) {
    return Object.prototype.hasOwnProperty.call(
      LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE,
      role
    )
      ? LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE[role]
      : null;
  }

  function isLinearPeriodicReservedOperandId(operandId) {
    return (
      operandId === LINEAR_PERIODIC_PREVIOUS_OPERAND_ID ||
      operandId === LINEAR_PERIODIC_NEXT_OPERAND_ID
    );
  }

  function normalizeLinearPeriodicChainInPlace(chain) {
    chain.metadata = runtime.isObject(chain.metadata) ? chain.metadata : {};
    chain.active_cell = LINEAR_PERIODIC_CELL_ORDER.includes(chain.active_cell)
      ? chain.active_cell
      : "initial";
    chain.initial_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.initial_cell)
          ? chain.initial_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    chain.periodic_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.periodic_cell)
          ? chain.periodic_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    chain.final_cell = runtime.normalizeGraphSectionInPlace(
      runtime.deepClone(
        runtime.isObject(chain.final_cell)
          ? chain.final_cell
          : runtime.buildEmptyGraphSection()
      )
    );
    return chain;
  }

  function getLinearPeriodicChain(spec = state.spec) {
    return spec && runtime.isObject(spec.linear_periodic_chain)
      ? spec.linear_periodic_chain
      : null;
  }

  function isLinearPeriodicMode(spec = state.spec) {
    return Boolean(getLinearPeriodicChain(spec));
  }

  function getActiveLinearPeriodicCellName(spec = state.spec) {
    const chain = getLinearPeriodicChain(spec);
    return chain ? chain.active_cell : null;
  }

  function getLinearPeriodicCell(
    spec = state.spec,
    cellName = getActiveLinearPeriodicCellName(spec)
  ) {
    const chain = getLinearPeriodicChain(spec);
    if (!chain || !cellName) {
      return null;
    }
    return chain[`${cellName}_cell`] || null;
  }

  function getLinearPeriodicBoundaryTensorByRole(role, spec = state.spec) {
    return (
      Array.isArray(spec && spec.tensors) ? spec.tensors : []
    ).find((tensor) => tensor.linear_periodic_role === role) || null;
  }

  function isLinearPeriodicBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (tensor.linear_periodic_role === "previous" ||
          tensor.linear_periodic_role === "next")
    );
  }

  function getLinearPeriodicReservedOperandIdForTensor(
    tensorOrId,
    spec = state.spec
  ) {
    const tensor =
      typeof tensorOrId === "string"
        ? (
            Array.isArray(spec && spec.tensors) ? spec.tensors : []
          ).find((candidate) => candidate.id === tensorOrId) || null
        : tensorOrId;
    if (!isLinearPeriodicBoundaryTensor(tensor)) {
      return null;
    }
    return getLinearPeriodicReservedOperandId(tensor.linear_periodic_role);
  }

  function isContractibleBoundaryTensor(tensor) {
    return (
      isLinearPeriodicBoundaryTensor(tensor) ||
      (typeof runtime.isTreePeriodicBoundaryTensor === "function" &&
        runtime.isTreePeriodicBoundaryTensor(tensor)) ||
      (typeof runtime.isGridPeriodicBoundaryTensor === "function" &&
        runtime.isGridPeriodicBoundaryTensor(tensor))
    );
  }

  function buildContractibleCollections(spec) {
    const tensors = Array.isArray(spec && spec.tensors) ? spec.tensors : [];
    const edges = Array.isArray(spec && spec.edges) ? spec.edges : [];
    const contractibleTensors = tensors.filter(
      (tensor) => !isContractibleBoundaryTensor(tensor)
    );
    if (!edges.length || !contractibleTensors.length) {
      return {
        tensors: contractibleTensors,
        edges: [],
      };
    }

    let tensorById = {};
    if (spec === state.spec && typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
      tensorById = state.tensorById;
    } else {
      tensorById = Object.fromEntries(tensors.map((tensor) => [tensor.id, tensor]));
    }

    return {
      tensors: contractibleTensors,
      edges: edges.filter((edge) => {
        const leftTensor = tensorById[edge.left && edge.left.tensor_id];
        const rightTensor = tensorById[edge.right && edge.right.tensor_id];
        return (
          leftTensor &&
          rightTensor &&
          !isContractibleBoundaryTensor(leftTensor) &&
          !isContractibleBoundaryTensor(rightTensor)
        );
      }),
    };
  }

  function ensureContractibleCollections() {
    if (!state.spec) {
      state.contractibleCacheRevision = state.specRevision;
      state.contractibleCacheTensorRef = null;
      state.contractibleCacheTensorCount = 0;
      state.contractibleCacheEdgeRef = null;
      state.contractibleCacheEdgeCount = 0;
      state.contractibleTensorsCache = [];
      state.contractibleEdgesCache = [];
      state.contractibleCacheToken += 1;
      return;
    }

    const tensors = Array.isArray(state.spec.tensors) ? state.spec.tensors : [];
    const edges = Array.isArray(state.spec.edges) ? state.spec.edges : [];
    const cacheIsFresh =
      state.contractibleCacheRevision === state.specRevision &&
      state.contractibleCacheTensorRef === tensors &&
      state.contractibleCacheTensorCount === tensors.length &&
      state.contractibleCacheEdgeRef === edges &&
      state.contractibleCacheEdgeCount === edges.length;
    if (cacheIsFresh) {
      return;
    }

    const collections = buildContractibleCollections(state.spec);
    state.contractibleCacheRevision = state.specRevision;
    state.contractibleCacheTensorRef = tensors;
    state.contractibleCacheTensorCount = tensors.length;
    state.contractibleCacheEdgeRef = edges;
    state.contractibleCacheEdgeCount = edges.length;
    state.contractibleTensorsCache = collections.tensors;
    state.contractibleEdgesCache = collections.edges;
    state.contractibleCacheToken += 1;
  }

  function getContractibleTensors(spec = state.spec) {
    if (spec !== state.spec) {
      return buildContractibleCollections(spec).tensors;
    }
    ensureContractibleCollections();
    return state.contractibleTensorsCache;
  }

  function getContractibleEdges(spec = state.spec) {
    if (spec !== state.spec) {
      return buildContractibleCollections(spec).edges;
    }
    ensureContractibleCollections();
    return state.contractibleEdgesCache;
  }

  function getExpectedLinearPeriodicRoles(cellName) {
    if (cellName === "initial") {
      return ["next"];
    }
    if (cellName === "periodic") {
      return ["previous", "next"];
    }
    if (cellName === "final") {
      return ["previous"];
    }
    return [];
  }

  return {
    getLinearPeriodicReservedOperandId,
    isLinearPeriodicReservedOperandId,
    normalizeLinearPeriodicChainInPlace,
    getLinearPeriodicChain,
    isLinearPeriodicMode,
    getActiveLinearPeriodicCellName,
    getLinearPeriodicCell,
    getLinearPeriodicBoundaryTensorByRole,
    isLinearPeriodicBoundaryTensor,
    getLinearPeriodicReservedOperandIdForTensor,
    isContractibleBoundaryTensor,
    getContractibleTensors,
    getContractibleEdges,
    getExpectedLinearPeriodicRoles,
  };
}
