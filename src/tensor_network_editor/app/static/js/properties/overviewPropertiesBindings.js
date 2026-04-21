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

  function bindMultiSelectionProperties({
    state,
    selectedEntries,
    batchColor,
    hasMultipleTensors,
  }) {
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

    if (!hasMultipleTensors) {
      return;
    }

    bindClick("extract-selection-button", () => actions.exportSelectedSubnetwork());
    bindClick("save-selection-subnetwork-library-button", () =>
      actions.saveSelectionToSubnetworkLibrary()
    );
    bindClick("promote-selection-template-button", () =>
      actions.promoteSelectedSubnetworkToTemplate()
    );
    bindClick("group-selection-button", () => actions.createGroupFromSelection());
  }

  return {
    bindNetworkProperties,
    bindMultiSelectionProperties,
  };
}
