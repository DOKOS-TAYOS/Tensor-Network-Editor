import { GRAPH_THEME } from "../theme.js";

export function createGraphElementModelBuilder({
  state,
  buildContractionScene,
  ensureTensorIndexOffsets,
  findTensorById,
  getIndexColor,
  getMetadataColor,
  getMetadataFilterEntityState,
  getMetadataFilterHighlight,
  indexAbsolutePosition,
  indexLabelNodeId,
  indexLabelPosition,
  isInspectingPastStage,
  readableTextColor,
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
          borderColor: shiftColor(tensorColor, 26),
          textColor: readableTextColor(tensorColor),
          zIndex: tensor + tensorRank,
        },
        classes: joinClasses(
          state.pendingPlannerSelectionId === tensorItem.id
            ? "planner-pending-tensor"
            : "",
          getMetadataFilterClass(metadataFilterHighlight, "tensor", tensorItem.id)
        ),
        position: { x: tensorItem.position.x, y: tensorItem.position.y },
        grabbable: !readOnlyScene,
        selectable: true,
      });

      tensorItem.indices.forEach((indexItem, indexPosition) => {
        const indexColor = getIndexColor(indexItem, connectedIndexIds.has(indexItem.id));
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
            zIndex: port + tensorRank * 10 + indexPosition,
          },
          classes: [
            connectedIndexIds.has(indexItem.id) ? "index-connected" : "index-open",
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
            zIndex: indexLabel + tensorRank * 10 + indexPosition,
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
          textColor: shiftColor(edgeColor, 72),
          zIndex: edge,
        },
        classes: getMetadataFilterClass(metadataFilterHighlight, "edge", edgeItem.id),
        position: null,
        grabbable: false,
        selectable: !readOnlyScene,
      });
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
