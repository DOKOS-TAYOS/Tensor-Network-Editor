export function createCanvasContextMenuBindings({
  document,
  window,
  state,
  propertyCommands,
  propertyInvalidation,
  bindMetadataEditors,
  renderCanvasContextMenu,
  closeCanvasContextMenu,
  exportSelectedSubnetwork,
  exportGroupSubnetwork,
  saveSelectionToSubnetworkLibrary,
  promoteSelectedSubnetworkToTemplate,
  createGroupFromSelection,
  saveGroupToSubnetworkLibrary,
  promoteGroupToTemplate,
  toggleGroupCollapse,
}) {
  function bindCommitOnBlurAndEnter(element, commit) {
    if (!element) {
      return;
    }
    element.addEventListener("blur", commit);
    element.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      event.preventDefault();
      commit();
      closeCanvasContextMenu();
    });
  }

  function bindCloseOnEnter(element) {
    if (!element) {
      return;
    }
    element.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      closeCanvasContextMenu();
    });
  }

  function bindColorInput(element, { target, statusMessage }) {
    if (
      !element ||
      !propertyCommands ||
      typeof propertyCommands.updateTargetColor !== "function"
    ) {
      return;
    }
    element.addEventListener("input", () => {
      propertyCommands.updateTargetColor({
        invalidate: propertyInvalidation({ graph: true, minimap: true }),
        nextColor: element.value,
        statusMessage,
        target,
      });
    });
  }

  function bindSelectionColorInput(element, { statusMessage }) {
    if (
      !element ||
      !propertyCommands ||
      typeof propertyCommands.applySelectionColor !== "function"
    ) {
      return;
    }
    element.addEventListener("input", () => {
      propertyCommands.applySelectionColor({
        invalidate: propertyInvalidation({
          graph: true,
          minimap: true,
          overlays: true,
        }),
        nextColor: element.value,
        statusMessage,
      });
    });
  }

  function bindInlineMetadataEditor({
    target,
    annotationScope,
    inputPrefix,
    statusMessage,
    invalidate,
  }) {
    if (typeof bindMetadataEditors !== "function") {
      return;
    }
    const tagsInput = document.getElementById(`${inputPrefix}-tags-input`);
    const customMetadataInput = document.getElementById(
      `${inputPrefix}-custom-metadata-input`
    );
    bindMetadataEditors({
      annotationScope,
      customMetadataFieldKey: `${annotationScope}:${target.id}:custom-metadata`,
      customMetadataInput,
      invalidate,
      statusMessage,
      tagsFieldKey: `${annotationScope}:${target.id}:tags`,
      tagsInput,
      target,
    });
    bindCloseOnEnter(tagsInput);
  }

  function bindSelectionContextTarget(resolvedTarget) {
    const addIndexButton = document.getElementById(
      "context-menu-add-index-to-selection-button"
    );
    const extractButton = document.getElementById(
      "context-menu-extract-selection-button"
    );
    const saveLibraryButton = document.getElementById(
      "context-menu-save-selection-subnetwork-library-button"
    );
    const promoteButton = document.getElementById(
      "context-menu-promote-selection-template-button"
    );
    const colorInput = document.getElementById("context-menu-selection-color-input");
    const groupButton = document.getElementById("context-menu-group-selection-button");
    const deleteButton = document.getElementById("context-menu-delete-selection-button");

    bindSelectionColorInput(colorInput, {
      statusMessage: "Updated the selection color.",
    });

    if (
      addIndexButton &&
      propertyCommands &&
      typeof propertyCommands.addIndexToSelectedTensors === "function"
    ) {
      addIndexButton.addEventListener("click", () => {
        propertyCommands.addIndexToSelectedTensors({
          primaryId: state.primarySelectionId,
          selectionIds: [...state.selectionIds],
          statusMessage: "Added one index to each selected tensor.",
        });
        closeCanvasContextMenu();
      });
    }

    if (extractButton) {
      extractButton.addEventListener("click", () => {
        if (typeof exportSelectedSubnetwork === "function") {
          exportSelectedSubnetwork();
        }
        closeCanvasContextMenu();
      });
    }

    if (saveLibraryButton) {
      saveLibraryButton.addEventListener("click", () => {
        if (typeof saveSelectionToSubnetworkLibrary === "function") {
          saveSelectionToSubnetworkLibrary();
        }
        closeCanvasContextMenu();
      });
    }

    if (promoteButton) {
      promoteButton.addEventListener("click", () => {
        if (typeof promoteSelectedSubnetworkToTemplate === "function") {
          promoteSelectedSubnetworkToTemplate();
        }
        closeCanvasContextMenu();
      });
    }

    if (groupButton) {
      groupButton.addEventListener("click", () => {
        if (typeof createGroupFromSelection === "function") {
          createGroupFromSelection();
        }
        closeCanvasContextMenu();
      });
    }

    if (
      deleteButton &&
      propertyCommands &&
      typeof propertyCommands.deleteCurrentSelection === "function"
    ) {
      deleteButton.addEventListener("click", () => {
        propertyCommands.deleteCurrentSelection();
        closeCanvasContextMenu();
      });
    }
  }

  function bindTensorContextTarget(resolvedTarget) {
    const tensor = resolvedTarget.target;
    const nameInput = document.getElementById("context-menu-name-input");
    const addIndexButton = document.getElementById("context-menu-add-index-button");
    const colorInput = document.getElementById("context-menu-tensor-color-input");
    const deleteTensorButton = document.getElementById(
      "context-menu-delete-tensor-button"
    );

    bindCommitOnBlurAndEnter(nameInput, () => {
      propertyCommands.renameTensor({
        invalidate: propertyInvalidation({ graph: true }),
        proposedName: nameInput.value,
        statusMessage: `Updated tensor ${nameInput.value.trim()}.`,
        tensor,
      });
    });

    if (
      addIndexButton &&
      propertyCommands &&
      typeof propertyCommands.addTensorIndex === "function"
    ) {
      addIndexButton.addEventListener("click", () => {
        propertyCommands.addTensorIndex({
          primaryId: tensor.id,
          selectionIds: [tensor.id],
          statusMessage: `Added one index to ${tensor.name}.`,
          tensor,
        });
        renderCanvasContextMenu();
      });
    }

    bindColorInput(colorInput, {
      statusMessage: `Updated tensor ${tensor.name}.`,
      target: tensor,
    });

    if (
      deleteTensorButton &&
      propertyCommands &&
      typeof propertyCommands.deleteTensor === "function"
    ) {
      deleteTensorButton.addEventListener("click", () => {
        propertyCommands.deleteTensor({
          selectionIds: [],
          statusMessage: `Deleted tensor ${tensor.name}.`,
          tensorId: tensor.id,
        });
        closeCanvasContextMenu();
      });
    }

    bindInlineMetadataEditor({
      annotationScope: "tensor",
      inputPrefix: "context-menu-tensor",
      invalidate: propertyInvalidation(),
      statusMessage: `Updated tensor ${tensor.name}.`,
      target: tensor,
    });
  }

  function bindIndexContextTarget(resolvedTarget) {
    const { index, indexPosition, tensor } = resolvedTarget;
    const nameInput = document.getElementById("context-menu-name-input");
    const dimensionInput = document.getElementById("context-menu-dimension-input");
    const colorInput = document.getElementById("context-menu-index-color-input");
    const moveUpButton = document.getElementById("context-menu-move-up-button");
    const moveDownButton = document.getElementById("context-menu-move-down-button");
    const deleteIndexButton = document.getElementById("context-menu-delete-index-button");

    bindCommitOnBlurAndEnter(nameInput, () => {
      propertyCommands.renameIndex({
        index,
        invalidate: propertyInvalidation({ graph: true }),
        proposedName: nameInput.value,
        statusMessage: `Updated index ${nameInput.value.trim()}.`,
        tensor,
      });
    });

    bindCommitOnBlurAndEnter(dimensionInput, () => {
      propertyCommands.updateIndexDimension({
        indexId: index.id,
        invalidate: propertyInvalidation({
          analysis: true,
          graph: true,
        }),
        rawValue: dimensionInput.value,
        statusMessage: `Updated index ${index.name}.`,
      });
    });

    bindColorInput(colorInput, {
      statusMessage: `Updated index ${index.name}.`,
      target: index,
    });

    if (
      moveUpButton &&
      propertyCommands &&
      typeof propertyCommands.moveTensorIndex === "function"
    ) {
      moveUpButton.addEventListener("click", () => {
        propertyCommands.moveTensorIndex({
          direction: -1,
          indexPosition,
          invalidate: propertyInvalidation({
            graph: true,
            lookups: true,
            properties: true,
          }),
          primaryId: index.id,
          selectionIds: [index.id],
          statusMessage: `Moved index ${index.name}.`,
          tensorId: tensor.id,
        });
        closeCanvasContextMenu();
      });
    }

    if (
      moveDownButton &&
      propertyCommands &&
      typeof propertyCommands.moveTensorIndex === "function"
    ) {
      moveDownButton.addEventListener("click", () => {
        propertyCommands.moveTensorIndex({
          direction: 1,
          indexPosition,
          invalidate: propertyInvalidation({
            graph: true,
            lookups: true,
            properties: true,
          }),
          primaryId: index.id,
          selectionIds: [index.id],
          statusMessage: `Moved index ${index.name}.`,
          tensorId: tensor.id,
        });
        closeCanvasContextMenu();
      });
    }

    if (
      deleteIndexButton &&
      propertyCommands &&
      typeof propertyCommands.deleteTensorIndex === "function"
    ) {
      deleteIndexButton.addEventListener("click", () => {
        propertyCommands.deleteTensorIndex({
          indexId: index.id,
          primaryId: tensor.id,
          selectionIds: [tensor.id],
          statusMessage: `Deleted index ${index.name}.`,
          tensorId: tensor.id,
        });
        closeCanvasContextMenu();
      });
    }

    bindInlineMetadataEditor({
      annotationScope: "index",
      inputPrefix: "context-menu-index",
      invalidate: propertyInvalidation(),
      statusMessage: `Updated index ${index.name}.`,
      target: index,
    });
  }

  function bindEdgeContextTarget(resolvedTarget) {
    const edge = resolvedTarget.target;
    const nameInput = document.getElementById("context-menu-name-input");
    const colorInput = document.getElementById("context-menu-edge-color-input");
    const deleteEdgeButton = document.getElementById("context-menu-delete-edge-button");

    bindCommitOnBlurAndEnter(nameInput, () => {
      propertyCommands.renameEdge({
        edge,
        invalidate: propertyInvalidation({ graph: true }),
        proposedName: nameInput.value,
        statusMessage: `Updated connection ${nameInput.value.trim()}.`,
      });
    });

    bindColorInput(colorInput, {
      statusMessage: `Updated connection ${edge.name}.`,
      target: edge,
    });

    if (
      deleteEdgeButton &&
      propertyCommands &&
      typeof propertyCommands.deleteEdge === "function"
    ) {
      deleteEdgeButton.addEventListener("click", () => {
        propertyCommands.deleteEdge({
          edgeId: edge.id,
          selectionIds: [],
          statusMessage: `Deleted connection ${edge.name}.`,
        });
        closeCanvasContextMenu();
      });
    }

    bindInlineMetadataEditor({
      annotationScope: "edge",
      inputPrefix: "context-menu-edge",
      invalidate: propertyInvalidation({ graph: false, minimap: false }),
      statusMessage: `Updated connection ${edge.name}.`,
      target: edge,
    });
  }

  function bindGroupContextTarget(resolvedTarget) {
    const group = resolvedTarget.target;
    const nameInput = document.getElementById("context-menu-name-input");
    const colorInput = document.getElementById("context-menu-group-color-input");
    const addIndexToGroupButton = document.getElementById(
      "context-menu-add-index-to-group-button"
    );
    const extractGroupButton = document.getElementById("context-menu-extract-group-button");
    const saveGroupLibraryButton = document.getElementById(
      "context-menu-save-group-subnetwork-library-button"
    );
    const promoteGroupTemplateButton = document.getElementById(
      "context-menu-promote-group-template-button"
    );
    const toggleGroupButton = document.getElementById("context-menu-toggle-group-button");
    const deleteGroupButton = document.getElementById("context-menu-delete-group-button");

    bindCommitOnBlurAndEnter(nameInput, () => {
      propertyCommands.renameGroup({
        group,
        invalidate: propertyInvalidation({ overlays: true }),
        proposedName: nameInput.value,
        statusMessage: `Updated group ${nameInput.value.trim()}.`,
      });
    });

    bindColorInput(colorInput, {
      statusMessage: `Updated group ${group.name}.`,
      target: group,
    });

    if (
      addIndexToGroupButton &&
      propertyCommands &&
      typeof propertyCommands.addIndexToSelectedTensors === "function"
    ) {
      addIndexToGroupButton.addEventListener("click", () => {
        propertyCommands.addIndexToSelectedTensors({
          primaryId: group.id,
          selectionIds: [group.id],
          statusMessage: "Added one index to each group tensor.",
          tensorIds: [...group.tensor_ids],
        });
        closeCanvasContextMenu();
      });
    }

    if (extractGroupButton) {
      extractGroupButton.addEventListener("click", () => {
        if (typeof exportGroupSubnetwork === "function") {
          exportGroupSubnetwork(group.id);
        }
        closeCanvasContextMenu();
      });
    }

    if (saveGroupLibraryButton) {
      saveGroupLibraryButton.addEventListener("click", () => {
        if (typeof saveGroupToSubnetworkLibrary === "function") {
          saveGroupToSubnetworkLibrary(group.id);
        }
        closeCanvasContextMenu();
      });
    }

    if (promoteGroupTemplateButton) {
      promoteGroupTemplateButton.addEventListener("click", () => {
        if (typeof promoteGroupToTemplate === "function") {
          promoteGroupToTemplate(group.id);
        }
        closeCanvasContextMenu();
      });
    }

    if (toggleGroupButton) {
      toggleGroupButton.addEventListener("click", () => {
        if (typeof toggleGroupCollapse === "function") {
          toggleGroupCollapse(group.id);
        }
        closeCanvasContextMenu();
      });
    }

    if (
      deleteGroupButton &&
      propertyCommands &&
      typeof propertyCommands.deleteGroup === "function"
    ) {
      deleteGroupButton.addEventListener("click", () => {
        propertyCommands.deleteGroup({
          groupId: group.id,
          invalidate: propertyInvalidation({
            lookups: true,
            overlays: true,
            properties: true,
          }),
          selectionIds: [],
          statusMessage: `Deleted group ${group.name}.`,
        });
        closeCanvasContextMenu();
      });
    }

    bindInlineMetadataEditor({
      annotationScope: "group",
      inputPrefix: "context-menu-group",
      invalidate: propertyInvalidation({ overlays: false }),
      statusMessage: `Updated group ${group.name}.`,
      target: group,
    });
  }

  function bindResolvedTarget(resolvedTarget) {
    if (!resolvedTarget) {
      return;
    }
    if (resolvedTarget.kind === "selection") {
      bindSelectionContextTarget(resolvedTarget);
      return;
    }
    if (resolvedTarget.kind === "tensor") {
      bindTensorContextTarget(resolvedTarget);
      return;
    }
    if (resolvedTarget.kind === "index") {
      bindIndexContextTarget(resolvedTarget);
      return;
    }
    if (resolvedTarget.kind === "edge") {
      bindEdgeContextTarget(resolvedTarget);
      return;
    }
    if (resolvedTarget.kind === "group") {
      bindGroupContextTarget(resolvedTarget);
    }
  }

  function installCanvasContextMenuGlobalListeners() {
    if (document && typeof document.addEventListener === "function") {
      document.addEventListener("click", (event) => {
        if (!state.canvasContextMenu) {
          return;
        }
        if (
          event &&
          event.target &&
          typeof event.target.closest === "function" &&
          event.target.closest(".canvas-context-menu")
        ) {
          return;
        }
        closeCanvasContextMenu();
      });
    }
    if (window && typeof window.addEventListener === "function") {
      window.addEventListener("resize", () => {
        if (state.canvasContextMenu) {
          closeCanvasContextMenu();
        }
      });
    }
  }

  return {
    bindResolvedTarget,
    closeCanvasContextMenu,
    installCanvasContextMenuGlobalListeners,
  };
}
