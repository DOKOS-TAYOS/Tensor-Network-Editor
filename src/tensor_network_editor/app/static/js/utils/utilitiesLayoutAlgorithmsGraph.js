export function createLayoutAlgorithmGraphSupport({
  ctx,
  state,
  selection,
  positionSupport,
}) {
  const {
    computeTensorBounds,
    buildGridLocalPositions,
    buildChainLocalPositions,
    buildLayeredLocalPositions,
    buildComponentLayoutFromLocalPositions,
    centerLayoutPositions,
    centerPackedComponentLayouts,
    buildTreeSubtreeLayout,
  } = positionSupport;

  function compareTensorIdsByPosition(leftId, rightId) {
    const leftTensor = ctx.findTensorById(leftId);
    const rightTensor = ctx.findTensorById(rightId);
    if (!leftTensor || !rightTensor) {
      return 0;
    }
    if (leftTensor.position.y !== rightTensor.position.y) {
      return leftTensor.position.y - rightTensor.position.y;
    }
    if (leftTensor.position.x !== rightTensor.position.x) {
      return leftTensor.position.x - rightTensor.position.x;
    }
    return String(leftId).localeCompare(String(rightId));
  }

  function sortTensorIdsByPosition(tensorIds) {
    return [...tensorIds].sort(compareTensorIdsByPosition);
  }

  function buildTensorAdjacency(tensorIds) {
    const tensorIdSet = new Set(tensorIds);
    const adjacency = new Map(tensorIds.map((tensorId) => [tensorId, []]));
    (Array.isArray(state.spec && state.spec.edges) ? state.spec.edges : []).forEach(
      (edge) => {
        const leftTensorId = edge && edge.left ? edge.left.tensor_id : null;
        const rightTensorId = edge && edge.right ? edge.right.tensor_id : null;
        if (!tensorIdSet.has(leftTensorId) || !tensorIdSet.has(rightTensorId)) {
          return;
        }
        adjacency.get(leftTensorId).push(rightTensorId);
        adjacency.get(rightTensorId).push(leftTensorId);
      }
    );
    return adjacency;
  }

  function getInternalEdgeCount(tensorIds, adjacency) {
    return (
      tensorIds.reduce(
        (edgeCount, tensorId) => edgeCount + (adjacency.get(tensorId) || []).length,
        0
      ) / 2
    );
  }

  function buildConnectedComponents(tensorIds, adjacency) {
    const visited = new Set();
    const components = [];
    sortTensorIdsByPosition(tensorIds).forEach((tensorId) => {
      if (visited.has(tensorId)) {
        return;
      }
      const queue = [tensorId];
      const componentIds = [];
      visited.add(tensorId);
      while (queue.length) {
        const currentId = queue.shift();
        componentIds.push(currentId);
        (adjacency.get(currentId) || []).forEach((neighborId) => {
          if (visited.has(neighborId)) {
            return;
          }
          visited.add(neighborId);
          queue.push(neighborId);
        });
      }
      components.push(componentIds);
    });
    return components;
  }

  function buildBreadthFirstLevels(componentIds, adjacency, rootId) {
    const componentIdSet = new Set(componentIds);
    const depthById = new Map([[rootId, 0]]);
    const idsByDepth = new Map([[0, [rootId]]]);
    const queue = [rootId];
    while (queue.length) {
      const currentId = queue.shift();
      const currentDepth = depthById.get(currentId) || 0;
      sortTensorIdsByPosition(adjacency.get(currentId) || []).forEach((neighborId) => {
        if (!componentIdSet.has(neighborId) || depthById.has(neighborId)) {
          return;
        }
        const nextDepth = currentDepth + 1;
        depthById.set(neighborId, nextDepth);
        if (!idsByDepth.has(nextDepth)) {
          idsByDepth.set(nextDepth, []);
        }
        idsByDepth.get(nextDepth).push(neighborId);
        queue.push(neighborId);
      });
    }
    return {
      depthById,
      idsByDepth,
    };
  }

  function buildPathOrder(tensorIds, adjacency) {
    const endpoints = sortTensorIdsByPosition(
      tensorIds.filter((tensorId) => (adjacency.get(tensorId) || []).length <= 1)
    );
    if (!endpoints.length) {
      return sortTensorIdsByPosition(tensorIds);
    }
    const orderedIds = [];
    let previousId = null;
    let currentId = endpoints[0];
    while (currentId) {
      orderedIds.push(currentId);
      const nextId = (adjacency.get(currentId) || []).find(
        (neighborId) => neighborId !== previousId
      );
      previousId = currentId;
      currentId = nextId || null;
    }
    return orderedIds;
  }

  function isPathComponent(tensorIds, adjacency) {
    return (
      tensorIds.length > 1 &&
      getInternalEdgeCount(tensorIds, adjacency) === tensorIds.length - 1 &&
      tensorIds.every((tensorId) => (adjacency.get(tensorId) || []).length <= 2)
    );
  }

  function isTreeComponent(tensorIds, adjacency) {
    return (
      tensorIds.length > 1 &&
      getInternalEdgeCount(tensorIds, adjacency) === tensorIds.length - 1
    );
  }

  function computeLayerAnchorX(tensorId, adjacency, depthById) {
    const tensorDepth = depthById.get(tensorId) || 0;
    const parentXs = (adjacency.get(tensorId) || [])
      .filter((neighborId) => (depthById.get(neighborId) || 0) < tensorDepth)
      .map((neighborId) => {
        const neighborTensor = ctx.findTensorById(neighborId);
        return neighborTensor ? neighborTensor.position.x : 0;
      });
    if (parentXs.length) {
      return (
        parentXs.reduce((sum, currentValue) => sum + currentValue, 0) /
        parentXs.length
      );
    }
    const tensor = ctx.findTensorById(tensorId);
    return tensor ? tensor.position.x : 0;
  }

  function sortLayerTensorIds(layerIds, adjacency, depthById) {
    return [...layerIds].sort((leftId, rightId) => {
      const leftAnchor = computeLayerAnchorX(leftId, adjacency, depthById);
      const rightAnchor = computeLayerAnchorX(rightId, adjacency, depthById);
      if (leftAnchor !== rightAnchor) {
        return leftAnchor - rightAnchor;
      }
      return compareTensorIdsByPosition(leftId, rightId);
    });
  }

  function resolveLayeredRootId(componentIds, adjacency, preferredRootId = null) {
    if (preferredRootId && componentIds.includes(preferredRootId)) {
      return preferredRootId;
    }
    return [...componentIds].sort((leftId, rightId) => {
      const leftDegree = (adjacency.get(leftId) || []).length;
      const rightDegree = (adjacency.get(rightId) || []).length;
      if (leftDegree !== rightDegree) {
        return rightDegree - leftDegree;
      }
      return compareTensorIdsByPosition(leftId, rightId);
    })[0];
  }

  function resolveComponentRootId(componentIds, adjacency, preferredRootId = null) {
    if (preferredRootId && componentIds.includes(preferredRootId)) {
      return preferredRootId;
    }
    const endpointIds = componentIds.filter(
      (tensorId) => (adjacency.get(tensorId) || []).length <= 1
    );
    if (endpointIds.length) {
      return sortTensorIdsByPosition(endpointIds)[0];
    }
    return sortTensorIdsByPosition(componentIds)[0];
  }

  function buildSpanningTree(componentIds, adjacency, preferredRootId = null) {
    const rootId = resolveComponentRootId(componentIds, adjacency, preferredRootId);
    const childrenById = new Map(componentIds.map((tensorId) => [tensorId, []]));
    const visited = new Set([rootId]);
    const queue = [rootId];
    while (queue.length) {
      const currentId = queue.shift();
      sortTensorIdsByPosition(adjacency.get(currentId) || []).forEach((neighborId) => {
        if (!childrenById.has(neighborId) || visited.has(neighborId)) {
          return;
        }
        visited.add(neighborId);
        childrenById.get(currentId).push(neighborId);
        queue.push(neighborId);
      });
    }
    return {
      childrenById,
      rootId,
    };
  }

  function buildTreeComponentLayout(componentIds, adjacency, preferredRootId = null) {
    const { rootId, childrenById } = buildSpanningTree(
      componentIds,
      adjacency,
      preferredRootId
    );
    return buildTreeSubtreeLayout(rootId, childrenById);
  }

  function buildLayeredComponentLayout(
    componentIds,
    adjacency,
    preferredRootId = null
  ) {
    const rootId = resolveLayeredRootId(componentIds, adjacency, preferredRootId);
    const { depthById, idsByDepth } = buildBreadthFirstLevels(
      componentIds,
      adjacency,
      rootId
    );
    const layeredIds = [...idsByDepth.keys()]
      .sort((leftDepth, rightDepth) => leftDepth - rightDepth)
      .map((depth) =>
        sortLayerTensorIds(idsByDepth.get(depth) || [], adjacency, depthById)
      )
      .filter((layerIds) => layerIds.length > 0);
    return buildComponentLayoutFromLocalPositions(
      componentIds,
      buildLayeredLocalPositions(layeredIds)
    );
  }

  function buildComponentTraversalOrder(
    componentIds,
    adjacency,
    preferredRootId = null
  ) {
    if (componentIds.length <= 1) {
      return [...componentIds];
    }
    if (isPathComponent(componentIds, adjacency)) {
      return buildPathOrder(componentIds, adjacency);
    }
    const { rootId, childrenById } = buildSpanningTree(
      componentIds,
      adjacency,
      preferredRootId
    );
    const orderedIds = [];
    function visitSubtree(tensorId) {
      orderedIds.push(tensorId);
      (childrenById.get(tensorId) || []).forEach((childId) => {
        visitSubtree(childId);
      });
    }
    visitSubtree(rootId);
    return orderedIds;
  }

  function buildSelectionComponentLayout(
    componentIds,
    adjacency,
    mode,
    preferredRootId = null
  ) {
    if (mode === "tree") {
      return buildTreeComponentLayout(componentIds, adjacency, preferredRootId);
    }
    const orderedIds = buildComponentTraversalOrder(
      componentIds,
      adjacency,
      preferredRootId
    );
    if (mode === "chain") {
      return buildComponentLayoutFromLocalPositions(
        componentIds,
        buildChainLocalPositions(orderedIds)
      );
    }
    if (mode === "grid") {
      return buildComponentLayoutFromLocalPositions(
        componentIds,
        buildGridLocalPositions(orderedIds)
      );
    }
    return null;
  }

  function buildSyntheticTreeLocalPositions(tensorIds, preferredRootId = null) {
    const sortedIds = sortTensorIdsByPosition(tensorIds);
    const rootId =
      preferredRootId && tensorIds.includes(preferredRootId)
        ? preferredRootId
        : sortedIds[0];
    const remainingIds = sortedIds.filter((tensorId) => tensorId !== rootId);
    const childrenById = new Map(tensorIds.map((tensorId) => [tensorId, []]));
    const parentQueue = [rootId];
    let parentIndex = 0;
    remainingIds.forEach((tensorId) => {
      while (
        parentIndex < parentQueue.length &&
        (childrenById.get(parentQueue[parentIndex]) || []).length >= 2
      ) {
        parentIndex += 1;
      }
      const parentId = parentQueue[parentIndex] || rootId;
      childrenById.get(parentId).push(tensorId);
      parentQueue.push(tensorId);
    });
    return buildTreeSubtreeLayout(rootId, childrenById).positions;
  }

  function buildArrangedSelectionPositions(tensorIds, mode) {
    const tensors = selection.getLayoutTensorsById(tensorIds);
    const targetCenter = computeTensorBounds(tensors);
    const adjacency = buildTensorAdjacency(tensorIds);
    const internalEdgeCount = getInternalEdgeCount(tensorIds, adjacency);
    if (mode === "grid" && internalEdgeCount === 0) {
      return centerLayoutPositions(
        buildGridLocalPositions(sortTensorIdsByPosition(tensorIds)),
        tensorIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    if (mode === "tree" && internalEdgeCount === 0) {
      return centerLayoutPositions(
        buildSyntheticTreeLocalPositions(
          tensorIds,
          tensorIds.includes(state.primarySelectionId) ? state.primarySelectionId : null
        ),
        tensorIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    const componentLayouts = buildConnectedComponents(tensorIds, adjacency).map(
      (componentIds) =>
        buildSelectionComponentLayout(
          componentIds,
          adjacency,
          mode,
          componentIds.includes(state.primarySelectionId)
            ? state.primarySelectionId
            : null
        )
    );
    if (!componentLayouts.every(Boolean)) {
      return null;
    }
    return centerPackedComponentLayouts(componentLayouts, tensorIds, {
      x: (targetCenter.left + targetCenter.right) / 2,
      y: (targetCenter.top + targetCenter.bottom) / 2,
    });
  }

  function buildImportedReflowPositions(tensorIds) {
    return buildAutoLayoutPositions(tensorIds);
  }

  function buildAutoLayoutPositions(tensorIds, preferredRootId = null) {
    const adjacency = buildTensorAdjacency(tensorIds);
    const componentLayouts = buildConnectedComponents(tensorIds, adjacency).map(
      (componentIds) => {
        if (isPathComponent(componentIds, adjacency)) {
          return buildComponentLayoutFromLocalPositions(
            componentIds,
            buildChainLocalPositions(buildPathOrder(componentIds, adjacency))
          );
        }
        if (isTreeComponent(componentIds, adjacency)) {
          return buildTreeComponentLayout(
            componentIds,
            adjacency,
            componentIds.includes(preferredRootId) ? preferredRootId : null
          );
        }
        return buildLayeredComponentLayout(
          componentIds,
          adjacency,
          componentIds.includes(preferredRootId) ? preferredRootId : null
        );
      }
    );
    const importedTensors = selection.getLayoutTensorsById(tensorIds);
    const importedBounds = computeTensorBounds(importedTensors);
    return centerPackedComponentLayouts(componentLayouts, tensorIds, {
      x: (importedBounds.left + importedBounds.right) / 2,
      y: (importedBounds.top + importedBounds.bottom) / 2,
    });
  }

  return {
    buildTensorAdjacency,
    buildConnectedComponents,
    buildBreadthFirstLevels,
    buildSpanningTree,
    buildArrangedSelectionPositions,
    buildImportedReflowPositions,
    buildAutoLayoutPositions,
  };
}
