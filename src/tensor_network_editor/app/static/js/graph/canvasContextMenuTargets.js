import { GRAPH_THEME } from "../core/theme.js";

export function createCanvasContextMenuTargetResolver({
  state,
  findTensorById,
  findEdgeById,
  findHyperedgeById,
  findIndexOwner,
  findGroupById,
  findEdgeByIndexId,
  getSelectedIdsByKind,
  getSelectedEntries,
  getHyperedgeSelectionId,
  getBatchColorValue,
  describeHyperedgeCandidate,
  getMetadataColor,
  getIndexColor,
  getTensorTotalElementCount,
  getTotalElementCountForTensorIds,
  getIndexCountForTensorIds,
}) {
  function isStructuralBoundaryTensor(tensor) {
    return Boolean(
      tensor &&
        (
          tensor.linear_periodic_role === "previous" ||
          tensor.linear_periodic_role === "next" ||
          tensor.grid_periodic_role === "up" ||
          tensor.grid_periodic_role === "right" ||
          tensor.grid_periodic_role === "down" ||
          tensor.grid_periodic_role === "left" ||
          tensor.tree_periodic_role === "parent" ||
          tensor.tree_periodic_role === "child"
        )
    );
  }

  function getSelectedTensorIdsForContext() {
    return typeof getSelectedIdsByKind === "function"
      ? getSelectedIdsByKind("tensor")
      : [];
  }

  function getSelectedIndexIdsForContext() {
    return typeof getSelectedIdsByKind === "function"
      ? getSelectedIdsByKind("index")
      : [];
  }

  function selectionContainsOnlyIndicesOrOwningTensors(
    selectedEntries = [],
    selectedIndexIds = []
  ) {
    if (!selectedEntries.length || !selectedIndexIds.length) {
      return false;
    }
    const selectedTensorIds = new Set(
      selectedEntries
        .filter((entry) => entry?.kind === "index")
        .map((entry) => entry?.located?.tensor?.id)
        .filter(Boolean)
    );
    return selectedEntries.every((entry) => {
      if (entry?.kind === "index") {
        return true;
      }
      if (entry?.kind !== "tensor") {
        return false;
      }
      return selectedTensorIds.has(entry?.tensor?.id || entry?.id);
    });
  }

  function isMultiTensorSelectionContext(tensorId) {
    const selectedTensorIds = getSelectedTensorIdsForContext();
    return (
      Array.isArray(state.selectionIds) &&
      selectedTensorIds.length >= 2 &&
      selectedTensorIds.includes(tensorId)
    );
  }

  function isMultiIndexSelectionContext(indexId) {
    const selectedIndexIds = getSelectedIndexIdsForContext();
    return (
      Array.isArray(state.selectionIds) &&
      selectedIndexIds.length >= 2 &&
      selectedIndexIds.includes(indexId)
    );
  }

  function getSelectionContextTarget(anchorTensorId) {
    const selectedTensorIds = getSelectedTensorIdsForContext();
    if (selectedTensorIds.length < 2 || !selectedTensorIds.includes(anchorTensorId)) {
      return null;
    }
    const selectedEntries =
      typeof getSelectedEntries === "function" ? getSelectedEntries() : [];
    const selectionColor =
      typeof getBatchColorValue === "function"
        ? getBatchColorValue(selectedEntries) || "#456cbf"
        : "#456cbf";
    return {
      editableTensorIds: selectedTensorIds.filter((tensorId) => {
        const tensor =
          typeof findTensorById === "function" ? findTensorById(tensorId) : null;
        return tensor && !isStructuralBoundaryTensor(tensor);
      }),
      id: anchorTensorId,
      indexCount: getIndexCountForTensorIds(selectedTensorIds),
      kind: "selection",
      primarySelectionId: state.primarySelectionId,
      selectionColor,
      selectionIds: Array.isArray(state.selectionIds) ? [...state.selectionIds] : [],
      target: null,
      tensorCount: selectedTensorIds.length,
      tensorIds: [...selectedTensorIds],
      totalElementCount: getTotalElementCountForTensorIds(selectedTensorIds),
    };
  }

  function getTensorContextTarget(tensorId) {
    const tensor =
      typeof findTensorById === "function" ? findTensorById(tensorId) : null;
    if (!tensor) {
      return null;
    }
    const tensorColor =
      typeof getMetadataColor === "function"
        ? getMetadataColor(tensor.metadata, GRAPH_THEME.tensorFallback)
        : GRAPH_THEME.tensorFallback;
    return {
      id: tensor.id,
      isStructuralBoundaryTensor: isStructuralBoundaryTensor(tensor),
      kind: "tensor",
      target: tensor,
      tensorColor,
      totalElementCount: getTensorTotalElementCount(tensor),
    };
  }

  function getIndexSelectionContextTarget(anchorIndexId) {
    const selectedEntries =
      typeof getSelectedEntries === "function" ? getSelectedEntries() : [];
    const selectedIndexIds = getSelectedIndexIdsForContext();
    const selectedIndexEntries = selectedEntries.filter(
      (entry) => entry?.kind === "index"
    );
    if (
      selectedIndexIds.length < 2 ||
      !selectedIndexIds.includes(anchorIndexId) ||
      !selectionContainsOnlyIndicesOrOwningTensors(selectedEntries, selectedIndexIds)
    ) {
      return null;
    }
    const selectionColor =
      typeof getBatchColorValue === "function"
        ? getBatchColorValue(selectedEntries) || "#456cbf"
        : "#456cbf";
    const hyperedgeCreationCandidate =
      typeof describeHyperedgeCandidate === "function"
        ? describeHyperedgeCandidate(selectedIndexIds)
        : {
          canCreate: false,
          message: "Hyperedge creation is unavailable in this session.",
        };
    const selectedIndexDimensions = [
      ...new Set(
        selectedIndexEntries
          .map((entry) => entry?.located?.index?.dimension)
          .filter((dimension) => Number.isFinite(dimension))
      ),
    ];
    return {
      hyperedgeCreationCandidate,
      id: anchorIndexId,
      indexCount: selectedIndexIds.length,
      indexDimensionValue:
        selectedIndexDimensions.length === 1
          ? String(selectedIndexDimensions[0])
          : "",
      hasMixedIndexDimensions: selectedIndexDimensions.length > 1,
      indexIds: [...selectedIndexIds],
      kind: "index-selection",
      primarySelectionId: state.primarySelectionId,
      selectionColor,
      selectionIds: Array.isArray(state.selectionIds) ? [...state.selectionIds] : [],
      target: null,
    };
  }

  function getIndexContextTarget(indexId) {
    if (isMultiIndexSelectionContext(indexId)) {
      return getIndexSelectionContextTarget(indexId);
    }
    const located =
      typeof findIndexOwner === "function" ? findIndexOwner(indexId) : null;
    if (!located || !located.tensor || !located.index) {
      return null;
    }
    const { index, tensor } = located;
    const indices = Array.isArray(tensor.indices) ? tensor.indices : [];
    const indexPosition = indices.findIndex((candidate) => candidate.id === index.id);
    const indexColor =
      typeof getMetadataColor === "function"
        ? getMetadataColor(
            index.metadata,
            typeof getIndexColor === "function"
              ? getIndexColor(
                  index,
                  Boolean(
                    typeof findEdgeByIndexId === "function" &&
                      findEdgeByIndexId(index.id)
                  )
                )
              : "#456cbf"
          )
        : "#456cbf";

    return {
      id: index.id,
      index,
      indexColor,
      indexPosition,
      indices,
      isStructuralBoundaryTensor: isStructuralBoundaryTensor(tensor),
      kind: "index",
      target: index,
      tensor,
    };
  }

  function getEdgeContextTarget(edgeId) {
    const edge =
      typeof findEdgeById === "function" ? findEdgeById(edgeId) : null;
    if (!edge) {
      return null;
    }
    const edgeColor =
      typeof getMetadataColor === "function"
        ? getMetadataColor(edge.metadata, GRAPH_THEME.edge)
        : GRAPH_THEME.edge;
    return {
      edgeColor,
      id: edge.id,
      kind: "edge",
      target: edge,
    };
  }

  function getHyperedgeContextTarget(hyperedgeId) {
    const hyperedge =
      typeof findHyperedgeById === "function"
        ? findHyperedgeById(hyperedgeId)
        : null;
    if (!hyperedge) {
      return null;
    }
    const hyperedgeColor =
      typeof getMetadataColor === "function"
        ? getMetadataColor(hyperedge.metadata, GRAPH_THEME.edge)
        : GRAPH_THEME.edge;
    return {
      hyperedgeColor,
      id:
        typeof getHyperedgeSelectionId === "function"
          ? getHyperedgeSelectionId(hyperedge.id)
          : typeof hyperedgeId === "string" &&
              (hyperedgeId.startsWith("hyperedge-hub:")
                || hyperedgeId.startsWith("hyperedge-spoke:"))
            ? `hyperedge-hub:${hyperedge.id}`
          : hyperedge.id,
      kind: "hyperedge",
      target: hyperedge,
    };
  }

  function getGroupContextTarget(groupId) {
    const group =
      typeof findGroupById === "function" ? findGroupById(groupId) : null;
    if (!group) {
      return null;
    }
    const groupColor =
      typeof getMetadataColor === "function"
        ? getMetadataColor(group.metadata, GRAPH_THEME.groupDefault)
        : GRAPH_THEME.groupDefault;
    return {
      editableTensorIds: (Array.isArray(group.tensor_ids) ? group.tensor_ids : []).filter(
        (tensorId) => {
          const tensor =
            typeof findTensorById === "function" ? findTensorById(tensorId) : null;
          return tensor && !isStructuralBoundaryTensor(tensor);
        }
      ),
      groupColor,
      id: group.id,
      isCollapsed: Boolean(group.metadata && group.metadata.collapsed),
      kind: "group",
      memberTensorCount: Array.isArray(group.tensor_ids) ? group.tensor_ids.length : 0,
      target: group,
      totalElementCount: getTotalElementCountForTensorIds(
        Array.isArray(group.tensor_ids) ? group.tensor_ids : []
      ),
    };
  }

  function resolveContextTarget(menuState) {
    if (!menuState || !menuState.kind || !menuState.id) {
      return null;
    }
    if (menuState.kind === "tensor") {
      if (isMultiTensorSelectionContext(menuState.id)) {
        return getSelectionContextTarget(menuState.id);
      }
      return getTensorContextTarget(menuState.id);
    }
    if (menuState.kind === "selection") {
      return getSelectionContextTarget(menuState.id);
    }
    if (menuState.kind === "index-selection") {
      return getIndexSelectionContextTarget(menuState.id);
    }
    if (menuState.kind === "index") {
      return getIndexContextTarget(menuState.id);
    }
    if (menuState.kind === "edge") {
      return getEdgeContextTarget(menuState.id);
    }
    if (menuState.kind === "hyperedge") {
      return getHyperedgeContextTarget(menuState.id);
    }
    if (menuState.kind === "group") {
      return getGroupContextTarget(menuState.id);
    }
    return null;
  }

  return {
    getEdgeContextTarget,
    getGroupContextTarget,
    getHyperedgeContextTarget,
    getIndexContextTarget,
    getIndexSelectionContextTarget,
    getSelectionContextTarget,
    getTensorContextTarget,
    resolveContextTarget,
  };
}
