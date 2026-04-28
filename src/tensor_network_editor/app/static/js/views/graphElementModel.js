import { GRAPH_THEME } from "../core/theme.js";

export function createGraphElementModelBuilder({
  state,
  buildContractionScene,
  ensureTensorIndexOffsets,
  findIndexOwner,
  findTensorById,
  getIndexColor,
  getMetadataColor,
  getMetadataFilterEntityState,
  getMetadataFilterHighlight,
  getHyperedgeHubPosition,
  hyperedgeHubNodeId,
  hyperedgeSpokeEdgeId,
  indexAbsolutePosition,
  indexLabelNodeId,
  indexLabelPosition,
  isInspectingPastStage,
  readableTextColor,
  resolveTensorBorderColor,
  shiftColor,
  tensorHeight,
  tensorLayerRank,
  tensorWidth,
  zIndexes,
}) {
  const {
    edge = 100,
    indexLabel = 230,
    port = 200,
    tensor = 10,
  } = zIndexes || {};

  function appendDescriptor(accumulator, descriptor) {
    accumulator.orderedIds.push(descriptor.data.id);
    accumulator.descriptorsById[descriptor.data.id] = descriptor;
  }

  function joinClasses(...classNames) {
    return classNames.filter(Boolean).join(" ");
  }

  function isTensorActiveForPorts(tensorId) {
    return Boolean(
      tensorId &&
        ((Array.isArray(state.selectionIds) && state.selectionIds.includes(tensorId)) ||
          (state.activeTensorDrag &&
            Array.isArray(state.activeTensorDrag.tensorIds) &&
            state.activeTensorDrag.tensorIds.includes(tensorId)))
    );
  }

  function isIndexElevated(indexId, isConnected, tensorId = null) {
    return Boolean(
      isConnected ||
        state.pendingIndexId === indexId ||
        isTensorActiveForPorts(tensorId) ||
        (Array.isArray(state.selectionIds) && state.selectionIds.includes(indexId))
    );
  }

  function getLayeredPortZIndex(
    indexId,
    indexPosition,
    tensorRank,
    isConnected,
    tensorId = null
  ) {
    if (isIndexElevated(indexId, isConnected, tensorId)) {
      return port + tensorRank * 10 + indexPosition;
    }
    return tensor + tensorRank + 0.2 + indexPosition / 1000;
  }

  function getLayeredIndexLabelZIndex(
    indexId,
    indexPosition,
    tensorRank,
    isConnected,
    tensorId = null
  ) {
    if (isIndexElevated(indexId, isConnected, tensorId)) {
      return indexLabel + tensorRank * 10 + indexPosition;
    }
    return tensor + tensorRank + 0.24 + indexPosition / 1000;
  }

  function getMetadataFilterClass(metadataFilterHighlight, entityKind, entityId) {
    if (!metadataFilterHighlight || typeof getMetadataFilterEntityState !== "function") {
      return "";
    }
    const entityState = getMetadataFilterEntityState(
      entityKind,
      entityId,
      metadataFilterHighlight
    );
    if (entityState === "context") {
      return "metadata-filter-context";
    }
    if (entityState === "dim") {
      return "metadata-filter-dim";
    }
    return "";
  }

  function getFallbackHyperedgeHubPosition(hyperedge) {
    const endpointPositions = (Array.isArray(hyperedge?.endpoints) ? hyperedge.endpoints : [])
      .map((endpoint) =>
        typeof findIndexOwner === "function" ? findIndexOwner(endpoint.index_id) : null
      )
      .filter((owner) => owner && owner.tensor && owner.index)
      .map((owner) => indexAbsolutePosition(owner.tensor, owner.index));
    if (!endpointPositions.length) {
      return null;
    }
    const summed = endpointPositions.reduce(
      (accumulator, position) => ({
        x: accumulator.x + position.x,
        y: accumulator.y + position.y,
      }),
      { x: 0, y: 0 }
    );
    return {
      x: Math.round(summed.x / endpointPositions.length),
      y: Math.round(summed.y / endpointPositions.length),
    };
  }

  return function buildGraphElementModel(contractionScene = null) {
    const descriptorsById = {};
    const orderedIds = [];
    const connectedIndexIds = new Set();
    const resolvedContractionScene =
      contractionScene ||
      (typeof buildContractionScene === "function" ? buildContractionScene() : null);
    const visibleTensors = resolvedContractionScene
      ? resolvedContractionScene.tensors
      : state.spec.tensors;
    const visibleEdges = resolvedContractionScene
      ? resolvedContractionScene.edges
      : state.spec.edges;
    const visibleHyperedges = resolvedContractionScene
      ? []
      : Array.isArray(state.spec?.hyperedges)
        ? state.spec.hyperedges
        : [];
    const readOnlyScene = Boolean(
      resolvedContractionScene &&
        typeof isInspectingPastStage === "function" &&
        isInspectingPastStage()
    );
    const indexNodesInteractive = !readOnlyScene;
    const metadataFilterHighlight =
      typeof getMetadataFilterHighlight === "function"
        ? getMetadataFilterHighlight()
        : null;

    const accumulator = {
      descriptorsById,
      orderedIds,
    };

    visibleEdges.forEach((edgeItem) => {
      connectedIndexIds.add(edgeItem.leftIndexId || edgeItem.left.index_id);
      connectedIndexIds.add(edgeItem.rightIndexId || edgeItem.right.index_id);
    });
    visibleHyperedges.forEach((hyperedge) => {
      (Array.isArray(hyperedge.endpoints) ? hyperedge.endpoints : []).forEach((endpoint) => {
        connectedIndexIds.add(endpoint.index_id);
      });
    });

    visibleTensors.forEach((tensorItem) => {
      if (!resolvedContractionScene && typeof ensureTensorIndexOffsets === "function") {
        ensureTensorIndexOffsets(tensorItem);
      }
      const tensorRank = tensorLayerRank(tensorItem.id);
      const anchorTensor =
        tensorItem.isDerived &&
        Array.isArray(tensorItem.sourceTensorIds) &&
        tensorItem.sourceTensorIds.length
          ? findTensorById(tensorItem.sourceTensorIds[0])
          : findTensorById(tensorItem.id);
      const tensorColor = getMetadataColor(
        anchorTensor ? anchorTensor.metadata : null,
        GRAPH_THEME.tensorFallback
      );
      appendDescriptor(accumulator, {
        group: "nodes",
        data: {
          id: tensorItem.id,
          label: tensorItem.name,
          kind: "tensor",
          width: tensorWidth(tensorItem),
          height: tensorHeight(tensorItem),
          resultCount: Number(tensorItem.resultCount || 0),
          backgroundColor: tensorColor,
          borderColor: resolveTensorBorderColor(tensorColor),
          textColor: readableTextColor(tensorColor),
          zIndex: tensor + tensorRank,
        },
        classes: joinClasses(
          state.pendingPlannerSelectionId === tensorItem.id
            ? "planner-pending-tensor"
            : "",
          Array.isArray(state.selectionIds) && state.selectionIds.includes(tensorItem.id)
            ? "is-selection-highlight"
            : "",
          getMetadataFilterClass(metadataFilterHighlight, "tensor", tensorItem.id)
        ),
        position: { x: tensorItem.position.x, y: tensorItem.position.y },
        grabbable: !readOnlyScene,
        selectable: true,
      });

      tensorItem.indices.forEach((indexItem, indexPosition) => {
        const isConnectedIndex = connectedIndexIds.has(indexItem.id);
        const indexColor = getIndexColor(indexItem, isConnectedIndex);
        const indexPositionAbsolute = resolvedContractionScene
          ? {
              x: tensorItem.position.x + indexItem.offset.x,
              y: tensorItem.position.y + indexItem.offset.y,
            }
          : indexAbsolutePosition(tensorItem, indexItem);
        appendDescriptor(accumulator, {
          group: "nodes",
          data: {
            id: indexItem.id,
            kind: "index",
            tensor_id: tensorItem.id,
            orderLabel: String(indexPosition + 1),
            backgroundColor: indexColor,
            borderColor: shiftColor(indexColor, 34),
            textColor: readableTextColor(indexColor),
            zIndex: getLayeredPortZIndex(
              indexItem.id,
              indexPosition,
              tensorRank,
              isConnectedIndex,
              tensorItem.id
            ),
          },
          classes: [
            isConnectedIndex ? "index-connected" : "index-open",
            state.pendingIndexId === indexItem.id ? "planner-pending-index" : "",
            getMetadataFilterClass(metadataFilterHighlight, "index", indexItem.id),
          ]
            .filter(Boolean)
            .join(" "),
          position: indexPositionAbsolute,
          grabbable: indexNodesInteractive,
          selectable: indexNodesInteractive,
        });
        appendDescriptor(accumulator, {
          group: "nodes",
          data: {
            id: indexLabelNodeId(indexItem.id),
            kind: "index-label",
            label: `${indexItem.name} · ${indexItem.dimension}`,
            textColor: shiftColor(indexColor, 64),
            zIndex: getLayeredIndexLabelZIndex(
              indexItem.id,
              indexPosition,
              tensorRank,
              isConnectedIndex,
              tensorItem.id
            ),
          },
          classes: getMetadataFilterClass(metadataFilterHighlight, "index", indexItem.id),
          position: indexLabelPosition(indexPositionAbsolute),
          grabbable: false,
          selectable: false,
        });
      });
    });

    visibleEdges.forEach((edgeItem) => {
      const edgeColor = getMetadataColor(edgeItem.metadata, GRAPH_THEME.edge);
      appendDescriptor(accumulator, {
        group: "edges",
        data: {
          id: edgeItem.id,
          source: edgeItem.leftIndexId || edgeItem.left.index_id,
          target: edgeItem.rightIndexId || edgeItem.right.index_id,
          label: edgeItem.name || edgeItem.label || "",
          kind: "edge",
          lineColor: edgeColor,
          labelBackgroundColor: GRAPH_THEME.edgeLabelBackground,
          labelTextColor: GRAPH_THEME.edgeLabelText,
          zIndex: edge,
        },
        classes: getMetadataFilterClass(metadataFilterHighlight, "edge", edgeItem.id),
        position: null,
        grabbable: false,
        selectable: !readOnlyScene,
      });
    });

    visibleHyperedges.forEach((hyperedge) => {
      const hubId = typeof hyperedgeHubNodeId === "function"
        ? hyperedgeHubNodeId(hyperedge.id)
        : hyperedge.id;
      const hubPosition =
        typeof getHyperedgeHubPosition === "function"
          ? getHyperedgeHubPosition(hyperedge)
          : getFallbackHyperedgeHubPosition(hyperedge);
      if (!hubPosition) {
        return;
      }
      const hyperedgeColor = getMetadataColor(hyperedge.metadata, GRAPH_THEME.edge);
      appendDescriptor(accumulator, {
        group: "nodes",
        data: {
          id: hubId,
          kind: "hyperedge-hub",
          label: hyperedge.name || "",
          baseHyperedgeId: hyperedge.id,
          backgroundColor: hyperedgeColor,
          borderColor: shiftColor(hyperedgeColor, 18),
          textColor: readableTextColor(hyperedgeColor),
          zIndex: edge + 1,
        },
        classes: getMetadataFilterClass(metadataFilterHighlight, "edge", hubId),
        position: hubPosition,
        grabbable: !readOnlyScene,
        selectable: !readOnlyScene,
      });
      (Array.isArray(hyperedge.endpoints) ? hyperedge.endpoints : []).forEach(
        (endpoint, endpointPosition) => {
          appendDescriptor(accumulator, {
            group: "edges",
            data: {
              id:
                typeof hyperedgeSpokeEdgeId === "function"
                  ? hyperedgeSpokeEdgeId(hyperedge.id, endpointPosition)
                  : `${hubId}:${endpointPosition}`,
              source: endpoint.index_id,
              target: hubId,
              kind: "hyperedge-spoke",
              baseHyperedgeId: hyperedge.id,
              lineColor: hyperedgeColor,
              textColor: shiftColor(hyperedgeColor, 64),
              zIndex: edge,
            },
            classes: getMetadataFilterClass(metadataFilterHighlight, "edge", hubId),
            position: null,
            grabbable: false,
            selectable: !readOnlyScene,
          });
        }
      );
    });

    return {
      descriptorsById,
      elements: orderedIds.map((elementId) => descriptorsById[elementId]),
      ephemeralSignature: [
        state.pendingPlannerSelectionId || "",
        state.pendingIndexId || "",
        readOnlyScene ? "readonly" : "editable",
      ].join("|"),
      orderedIds,
      visibleSignature: orderedIds.join("|"),
      visibleTensors,
    };
  };
}
