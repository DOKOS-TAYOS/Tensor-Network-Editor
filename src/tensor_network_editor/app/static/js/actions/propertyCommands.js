export function createPropertyCommands({
  applyDesignChange,
  applyColorToSelection = () => {},
  centerTensor,
  createHyperedge = () => null,
  createIndex,
  describeHyperedgeCandidate = () => ({ canCreate: false, message: "" }),
  deleteSelection = () => {},
  findIndexOwner,
  findTensorById = () => null,
  getSelectedTensorIds = () => [],
  isStructuralBoundaryTensor = (tensor) =>
    Boolean(
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
    ),
  moveIndex,
  removeEdge = () => {},
  removeHyperedge = () => {},
  removeGroup = () => {},
  removeIndex,
  removeNote = () => {},
  removeTensor,
  resolveHyperedgeSelectionId = (hyperedgeId) => hyperedgeId,
  setStatus,
  setSelection = () => {},
  syncConnectedIndexDimension = () => {},
  tensorIndexNameExists = () => false,
}) {
  function renameNetwork({ spec, proposedName, invalidate, statusMessage }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Design name cannot be empty.", "error");
      return false;
    }
    if (normalizedName === spec.name) {
      return false;
    }
    applyDesignChange(
      () => {
        spec.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function renameTensor({ tensor, proposedName, invalidate, statusMessage }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Tensor name cannot be empty.", "error");
      return false;
    }
    if (normalizedName === tensor.name) {
      return false;
    }
    applyDesignChange(
      () => {
        tensor.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function updateTensorData({
    tensorId,
    nextTensorData,
    invalidate,
    statusMessage,
  }) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return false;
    }
    const normalizedTensorData =
      nextTensorData && typeof nextTensorData === "object"
        ? JSON.parse(JSON.stringify(nextTensorData))
        : null;
    const currentPayload = JSON.stringify(tensor.tensor_data ?? null);
    const nextPayload = JSON.stringify(normalizedTensorData);
    if (currentPayload === nextPayload) {
      return false;
    }
    applyDesignChange(
      () => {
        tensor.tensor_data = normalizedTensorData;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function updateTargetColor({ target, nextColor, invalidate, statusMessage }) {
    const currentColor = target?.metadata?.color;
    if (!target || nextColor === currentColor) {
      return false;
    }
    applyDesignChange(
      () => {
        target.metadata = target.metadata || {};
        target.metadata.color = nextColor;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function applySelectionColor({ nextColor, invalidate, statusMessage }) {
    if (!nextColor) {
      return false;
    }
    applyDesignChange(
      () => {
        applyColorToSelection(nextColor);
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function addTensorIndex({ tensor, selectionIds, primaryId, statusMessage }) {
    if (!tensor || isStructuralBoundaryTensor(tensor)) {
      return false;
    }
    applyDesignChange(
      () => {
        tensor.indices.push(createIndex(tensor, tensor.indices.length));
      },
      {
        primaryId,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function centerTensorInView({ tensorId, invalidate, statusMessage }) {
    applyDesignChange(
      () => {
        centerTensor(tensorId);
      },
      {
        invalidate,
        statusMessage,
      }
    );
  }

  function deleteTensor({ tensorId, selectionIds, statusMessage }) {
    const tensor = findTensorById(tensorId);
    if (!tensor || isStructuralBoundaryTensor(tensor)) {
      return false;
    }
    applyDesignChange(
      () => {
        removeTensor(tensorId);
      },
      {
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function deleteCurrentSelection() {
    deleteSelection();
  }

  function renameIndex({
    tensor,
    index,
    proposedName,
    invalidate,
    statusMessage,
  }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Index name cannot be empty.", "error");
      return false;
    }
    if (tensorIndexNameExists(tensor, normalizedName, index.id)) {
      setStatus(
        `Tensor ${tensor.name} already has an index named ${normalizedName}.`,
        "error"
      );
      return false;
    }
    if (normalizedName === index.name) {
      return false;
    }
    applyDesignChange(
      () => {
        index.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function updateIndexDimension({
    indexId,
    rawValue,
    invalidate,
    statusMessage,
  }) {
    const currentOwner = findIndexOwner(indexId);
    const currentIndex = currentOwner ? currentOwner.index : null;
    if (!currentIndex) {
      return false;
    }
    if (
      !currentOwner.tensor ||
      isStructuralBoundaryTensor(currentOwner.tensor)
    ) {
      return false;
    }
    const parsed = Number.parseInt(rawValue, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setStatus("Index dimension must be a positive integer.", "error");
      return false;
    }
    if (parsed === currentIndex.dimension) {
      return false;
    }
    applyDesignChange(
      () => {
        const nextOwner = findIndexOwner(indexId);
        if (
          !nextOwner ||
          !nextOwner.index ||
          !nextOwner.tensor ||
          isStructuralBoundaryTensor(nextOwner.tensor)
        ) {
          return;
        }
        nextOwner.index.dimension = parsed;
        syncConnectedIndexDimension(indexId, parsed);
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function moveTensorIndex({
    tensorId,
    indexPosition,
    direction,
    invalidate,
    primaryId,
    selectionIds,
    statusMessage,
  }) {
    const tensor = findTensorById(tensorId);
    if (!tensor || isStructuralBoundaryTensor(tensor)) {
      return false;
    }
    applyDesignChange(
      () => {
        moveIndex(tensorId, indexPosition, direction);
      },
      {
        invalidate,
        primaryId,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function deleteTensorIndex({
    tensorId,
    indexId,
    primaryId,
    selectionIds,
    statusMessage,
  }) {
    const tensor = findTensorById(tensorId);
    if (!tensor || isStructuralBoundaryTensor(tensor)) {
      return false;
    }
    applyDesignChange(
      () => {
        removeIndex(tensorId, indexId);
      },
      {
        primaryId,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function addIndexToSelectedTensors({
    tensorIds = getSelectedTensorIds(),
    selectionIds = tensorIds,
    primaryId = selectionIds[0] || null,
    invalidate,
    statusMessage,
  }) {
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      return false;
    }
    const editableTensorIds = tensorIds.filter((tensorId) => {
      const tensor = findTensorById(tensorId);
      return tensor && !isStructuralBoundaryTensor(tensor);
    });
    if (!editableTensorIds.length) {
      return false;
    }
    applyDesignChange(
      () => {
        editableTensorIds.forEach((tensorId) => {
          const tensor = findTensorById(tensorId);
          if (!tensor) {
            return;
          }
          tensor.indices.push(createIndex(tensor, tensor.indices.length));
        });
      },
      {
        invalidate,
        selectionIds,
        primaryId,
        statusMessage,
      }
    );
    return true;
  }

  function renameGroup({ group, proposedName, invalidate, statusMessage }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Group name cannot be empty.", "error");
      return false;
    }
    if (normalizedName === group.name) {
      return false;
    }
    applyDesignChange(
      () => {
        group.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function deleteGroup({ groupId, selectionIds, invalidate, statusMessage }) {
    applyDesignChange(
      () => {
        removeGroup(groupId);
      },
      {
        invalidate,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function renameEdge({ edge, proposedName, invalidate, statusMessage }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Connection name cannot be empty.", "error");
      return false;
    }
    if (normalizedName === edge.name) {
      return false;
    }
    applyDesignChange(
      () => {
        edge.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function deleteEdge({ edgeId, selectionIds, invalidate, statusMessage }) {
    applyDesignChange(
      () => {
        removeEdge(edgeId);
      },
      {
        invalidate,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function renameHyperedge({
    hyperedge,
    proposedName,
    invalidate,
    statusMessage,
  }) {
    const normalizedName = String(proposedName || "").trim();
    if (!normalizedName) {
      setStatus("Hyperedge name cannot be empty.", "error");
      return false;
    }
    if (normalizedName === hyperedge.name) {
      return false;
    }
    applyDesignChange(
      () => {
        hyperedge.name = normalizedName;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function deleteHyperedge({
    hyperedgeId,
    selectionIds,
    invalidate,
    statusMessage,
  }) {
    applyDesignChange(
      () => {
        removeHyperedge(hyperedgeId);
      },
      {
        invalidate,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  function createHyperedgeFromIndices({
    indexIds,
    invalidate,
    statusMessage,
  }) {
    const candidate = describeHyperedgeCandidate(indexIds);
    if (!candidate.canCreate) {
      setStatus(candidate.message || "This selection cannot form a hyperedge.", "error");
      return false;
    }
    let nextHyperedge = null;
    applyDesignChange(
      () => {
        nextHyperedge = createHyperedge(candidate.indexIds);
      },
      {
        invalidate,
        statusMessage,
        afterRender: () => {
          if (!nextHyperedge?.id) {
            return;
          }
          const selectionId = resolveHyperedgeSelectionId(nextHyperedge.id);
          setSelection([selectionId], { primaryId: selectionId });
        },
      }
    );
    return Boolean(nextHyperedge);
  }

  function updateNoteText({ note, proposedText, invalidate, statusMessage }) {
    const normalizedText = String(proposedText || "").trim();
    if (!normalizedText) {
      setStatus("Notes cannot be empty.", "error");
      return false;
    }
    if (normalizedText === note.text) {
      return false;
    }
    applyDesignChange(
      () => {
        note.text = normalizedText;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function deleteNote({ noteId, selectionIds, invalidate, statusMessage }) {
    applyDesignChange(
      () => {
        removeNote(noteId);
      },
      {
        invalidate,
        selectionIds,
        statusMessage,
      }
    );
    return true;
  }

  return {
    addIndexToSelectedTensors,
    addTensorIndex,
    applySelectionColor,
    centerTensorInView,
    deleteCurrentSelection,
    deleteEdge,
    deleteHyperedge,
    deleteGroup,
    deleteNote,
    deleteTensor,
    deleteTensorIndex,
    moveTensorIndex,
    renameEdge,
    renameHyperedge,
    renameGroup,
    renameIndex,
    renameNetwork,
    renameTensor,
    createHyperedgeFromIndices,
    updateTensorData,
    updateNoteText,
    updateIndexDimension,
    updateTargetColor,
  };
}
