import { GRAPH_THEME } from "../core/theme.js";

export function createCanvasContextMenuTargetResolver({
  state,
  findTensorById,
  findEdgeById,
  findIndexOwner,
  findGroupById,
  findEdgeByIndexId,
  getSelectedIdsByKind,
  getSelectedEntries,
  getBatchColorValue,
  getMetadataColor,
  getIndexColor,
  getTensorTotalElementCount,
  getTotalElementCountForTensorIds,
  getIndexCountForTensorIds,
}) {
  function getSelectedTensorIdsForContext() {
    return typeof getSelectedIdsByKind === "function"
      ? getSelectedIdsByKind("tensor")
      : [];
  }

  function isMultiTensorSelectionContext(tensorId) {
    const selectedTensorIds = getSelectedTensorIdsForContext();
    return (
      Array.isArray(state.selectionIds) &&
      selectedTensorIds.length >= 2 &&
      selectedTensorIds.includes(tensorId)
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
      kind: "tensor",
      target: tensor,
      tensorColor,
      totalElementCount: getTensorTotalElementCount(tensor),
    };
  }

  function getIndexContextTarget(indexId) {
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
    if (menuState.kind === "index") {
      return getIndexContextTarget(menuState.id);
    }
    if (menuState.kind === "edge") {
      return getEdgeContextTarget(menuState.id);
    }
    if (menuState.kind === "group") {
      return getGroupContextTarget(menuState.id);
    }
    return null;
  }

  return {
    getEdgeContextTarget,
    getGroupContextTarget,
    getIndexContextTarget,
    getSelectionContextTarget,
    getTensorContextTarget,
    resolveContextTarget,
  };
}
