export function createPropertyCommands({
  applyDesignChange,
  centerTensor,
  createIndex,
  deleteSelection = () => {},
  findIndexOwner,
  moveIndex,
  removeIndex,
  removeTensor,
  setStatus,
  syncConnectedIndexDimension = () => {},
  tensorIndexNameExists = () => false,
}) {
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

  function updateTargetColor({ target, nextColor, invalidate, statusMessage }) {
    if (!target || nextColor === target.metadata.color) {
      return false;
    }
    applyDesignChange(
      () => {
        target.metadata.color = nextColor;
      },
      {
        invalidate,
        statusMessage,
      }
    );
    return true;
  }

  function addTensorIndex({ tensor, selectionIds, primaryId, statusMessage }) {
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
    applyDesignChange(
      () => {
        removeTensor(tensorId);
      },
      {
        selectionIds,
        statusMessage,
      }
    );
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
        if (!nextOwner || !nextOwner.index) {
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
  }

  function deleteTensorIndex({
    tensorId,
    indexId,
    primaryId,
    selectionIds,
    statusMessage,
  }) {
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
  }

  return {
    addTensorIndex,
    centerTensorInView,
    deleteCurrentSelection,
    deleteTensor,
    deleteTensorIndex,
    moveTensorIndex,
    renameIndex,
    renameTensor,
    updateIndexDimension,
    updateTargetColor,
  };
}
