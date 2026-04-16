export function createOverviewPropertiesBindings({
  documentRef,
  support,
  actions,
}) {
  const {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    commands,
    propertyInvalidation,
    selectionColorInvalidation,
  } = support;

  function bindClick(buttonId, handler) {
    const button = documentRef.getElementById(buttonId);
    if (!button) {
      return;
    }
    button.addEventListener("click", handler);
  }

  function bindNetworkProperties({ state }) {
    const networkNameInput = documentRef.getElementById("network-name-input");
    const networkTagsInput = documentRef.getElementById("network-tags-input");
    const networkCustomMetadataInput = documentRef.getElementById(
      "network-custom-metadata-input"
    );

    bindDebouncedAutosave(networkNameInput, "network:name", () => {
      commands.renameNetwork({
        spec: state.spec,
        proposedName: networkNameInput.value,
        invalidate: propertyInvalidation(),
        statusMessage: "Updated design name.",
      });
    });
    bindMetadataEditors({
      target: state.spec,
      tagsInput: networkTagsInput,
      tagsFieldKey: "network:tags",
      customMetadataInput: networkCustomMetadataInput,
      customMetadataFieldKey: "network:custom-metadata",
      statusMessage: "Updated design metadata.",
      invalidate: propertyInvalidation(),
    });
  }

  function bindMultiSelectionProperties({ state, selectedEntries, batchColor, tensorsOnly }) {
    const multiColorInput = documentRef.getElementById("multi-color-input");
    bindImmediateAutosave(
      multiColorInput,
      "selection:color",
      () => {
        if (multiColorInput.value === batchColor) {
          return;
        }
        commands.applySelectionColor({
          nextColor: multiColorInput.value,
          invalidate: selectionColorInvalidation(selectedEntries),
          statusMessage: "Updated the selection color.",
        });
      },
      "input"
    );

    const addIndexButton = documentRef.getElementById(
      "add-index-to-selection-button"
    );
    if (addIndexButton) {
      addIndexButton.addEventListener("click", () => {
        commands.addIndexToSelectedTensors({
          selectionIds: [...state.selectionIds],
          primaryId: state.primarySelectionId,
          statusMessage: "Added one index to each selected tensor.",
        });
      });
    }

    bindClick("delete-selection-button", () => {
      commands.deleteCurrentSelection();
    });

    if (!tensorsOnly) {
      return;
    }

    bindClick("align-selection-left-button", () => actions.alignSelectedTensors("left"));
    bindClick("align-selection-center-button", () =>
      actions.alignSelectedTensors("center")
    );
    bindClick("align-selection-right-button", () =>
      actions.alignSelectedTensors("right")
    );
    bindClick("align-selection-top-button", () => actions.alignSelectedTensors("top"));
    bindClick("align-selection-middle-button", () =>
      actions.alignSelectedTensors("middle")
    );
    bindClick("align-selection-bottom-button", () =>
      actions.alignSelectedTensors("bottom")
    );
    bindClick("arrange-selection-chain-button", () =>
      actions.arrangeSelectedTensors("chain")
    );
    bindClick("arrange-selection-tree-button", () =>
      actions.arrangeSelectedTensors("tree")
    );
    bindClick("arrange-selection-grid-button", () =>
      actions.arrangeSelectedTensors("grid")
    );
    bindClick("distribute-selection-horizontal-button", () =>
      actions.distributeSelectedTensors("horizontal")
    );
    bindClick("distribute-selection-vertical-button", () =>
      actions.distributeSelectedTensors("vertical")
    );
    bindClick("snap-selection-button", () => actions.snapSelectedTensorsToGrid());
    bindClick("extract-selection-button", () => actions.exportSelectedSubnetwork());
    bindClick("promote-selection-template-button", () =>
      actions.promoteSelectedSubnetworkToTemplate()
    );
  }

  return {
    bindNetworkProperties,
    bindMultiSelectionProperties,
  };
}
