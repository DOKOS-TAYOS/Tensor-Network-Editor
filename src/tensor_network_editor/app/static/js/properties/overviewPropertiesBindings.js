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
    editableTensorIds,
    selectedIndexIds,
    hasMultipleTensors,
  }) {
    const multiColorInput = documentRef.getElementById("multi-color-input");
    const multiIndexDimensionInput = documentRef.getElementById(
      "multi-index-dimension-input"
    );
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
    bindDebouncedAutosave(
      multiIndexDimensionInput,
      "selection:index-dimension",
      () => {
        const rawValue = multiIndexDimensionInput.value.trim();
        if (!rawValue) {
          return;
        }
        commands.updateIndexDimensions({
          indexIds: selectedIndexIds,
          rawValue,
          invalidate: propertyInvalidation({
            analysis: true,
            graph: true,
            properties: true,
          }),
          statusMessage: `Updated ${selectedIndexIds.length} indices.`,
        });
      }
    );

    const addIndexButton = documentRef.getElementById(
      "add-index-to-selection-button"
    );
    if (addIndexButton) {
      addIndexButton.addEventListener("click", () => {
        commands.addIndexToSelectedTensors({
          tensorIds: editableTensorIds,
          selectionIds: [...state.selectionIds],
          primaryId: state.primarySelectionId,
          statusMessage: "Added one index to each selected tensor.",
        });
      });
    }

    bindClick("delete-selection-button", () => {
      commands.deleteCurrentSelection();
    });

    bindClick("create-hyperedge-button", () => {
      actions.createHyperedgeFromSelection({
        invalidate: propertyInvalidation({
          analysis: true,
          graph: true,
          lookups: true,
          minimap: true,
          planner: true,
          toolbar: true,
        }),
        statusMessage: "Created a hyperedge.",
      });
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
