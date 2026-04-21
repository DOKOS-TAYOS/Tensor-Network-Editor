export function createEntityPropertiesBindings({
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
  } = support;

  function bindClick(buttonId, handler) {
    const button = documentRef.getElementById(buttonId);
    if (!button) {
      return;
    }
    button.addEventListener("click", handler);
  }

  function bindGroupProperties({ group, groupColor }) {
    const groupNameInput = documentRef.getElementById("group-name-input");
    const groupColorInput = documentRef.getElementById("group-color-input");
    const groupTagsInput = documentRef.getElementById("group-tags-input");
    const groupCustomMetadataInput = documentRef.getElementById(
      "group-custom-metadata-input"
    );

    bindDebouncedAutosave(groupNameInput, `group:${group.id}:name`, () => {
      commands.renameGroup({
        group,
        proposedName: groupNameInput.value,
        invalidate: propertyInvalidation({ overlays: true }),
        statusMessage: `Updated group ${groupNameInput.value.trim()}.`,
      });
    });
    bindImmediateAutosave(
      groupColorInput,
      `group:${group.id}:color`,
      () => {
        if (groupColorInput.value === groupColor) {
          return;
        }
        commands.updateTargetColor({
          target: group,
          nextColor: groupColorInput.value,
          invalidate: propertyInvalidation({ overlays: true }),
          statusMessage: `Updated group ${group.name}.`,
        });
      },
      "input"
    );
    bindClick("add-index-to-group-button", () => {
      commands.addIndexToSelectedTensors({
        tensorIds: [...group.tensor_ids],
        selectionIds: [group.id],
        primaryId: group.id,
        statusMessage: "Added one index to each group tensor.",
      });
    });
    bindClick("toggle-group-button", () => actions.toggleGroupCollapse(group.id));
    bindClick("extract-group-button", () => actions.exportGroupSubnetwork(group.id));
    bindClick("save-group-subnetwork-library-button", () =>
      actions.saveGroupToSubnetworkLibrary(group.id)
    );
    bindClick("promote-group-template-button", () =>
      actions.promoteGroupToTemplate(group.id)
    );
    bindMetadataEditors({
      target: group,
      tagsInput: groupTagsInput,
      tagsFieldKey: `group:${group.id}:tags`,
      customMetadataInput: groupCustomMetadataInput,
      customMetadataFieldKey: `group:${group.id}:custom-metadata`,
      statusMessage: `Updated group ${group.name}.`,
      invalidate: propertyInvalidation({ overlays: false }),
    });
    bindClick("delete-group-button", () => {
      commands.deleteGroup({
        groupId: group.id,
        invalidate: propertyInvalidation({
          lookups: true,
          overlays: true,
        }),
        selectionIds: [],
        statusMessage: `Deleted group ${group.name}.`,
      });
    });
  }

  function bindEdgeProperties({ edge, edgeColor }) {
    const edgeNameInput = documentRef.getElementById("edge-name-input");
    const edgeColorInput = documentRef.getElementById("edge-color-input");
    const edgeTagsInput = documentRef.getElementById("edge-tags-input");
    const edgeCustomMetadataInput = documentRef.getElementById(
      "edge-custom-metadata-input"
    );

    bindDebouncedAutosave(edgeNameInput, `edge:${edge.id}:name`, () => {
      commands.renameEdge({
        edge,
        proposedName: edgeNameInput.value,
        invalidate: propertyInvalidation({ graph: true }),
        statusMessage: `Updated connection ${edgeNameInput.value.trim()}.`,
      });
    });
    bindImmediateAutosave(
      edgeColorInput,
      `edge:${edge.id}:color`,
      () => {
        if (edgeColorInput.value === edgeColor) {
          return;
        }
        commands.updateTargetColor({
          target: edge,
          nextColor: edgeColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated connection ${edge.name}.`,
        });
      },
      "input"
    );
    bindMetadataEditors({
      target: edge,
      tagsInput: edgeTagsInput,
      tagsFieldKey: `edge:${edge.id}:tags`,
      customMetadataInput: edgeCustomMetadataInput,
      customMetadataFieldKey: `edge:${edge.id}:custom-metadata`,
      statusMessage: `Updated connection ${edge.name}.`,
      invalidate: propertyInvalidation({ graph: false, minimap: false }),
    });
    bindClick("delete-edge-button", () => {
      commands.deleteEdge({
        edgeId: edge.id,
        selectionIds: [],
        statusMessage: `Deleted connection ${edge.name}.`,
      });
    });
  }

  function bindHyperedgeProperties({ hyperedge, hyperedgeColor }) {
    const hyperedgeNameInput = documentRef.getElementById("hyperedge-name-input");
    const hyperedgeColorInput = documentRef.getElementById("hyperedge-color-input");
    const hyperedgeTagsInput = documentRef.getElementById("hyperedge-tags-input");
    const hyperedgeCustomMetadataInput = documentRef.getElementById(
      "hyperedge-custom-metadata-input"
    );

    bindDebouncedAutosave(
      hyperedgeNameInput,
      `hyperedge:${hyperedge.id}:name`,
      () => {
        commands.renameHyperedge({
          hyperedge,
          proposedName: hyperedgeNameInput.value,
          invalidate: propertyInvalidation({ graph: true }),
          statusMessage: `Updated hyperedge ${hyperedgeNameInput.value.trim()}.`,
        });
      }
    );
    bindImmediateAutosave(
      hyperedgeColorInput,
      `hyperedge:${hyperedge.id}:color`,
      () => {
        if (hyperedgeColorInput.value === hyperedgeColor) {
          return;
        }
        commands.updateTargetColor({
          target: hyperedge,
          nextColor: hyperedgeColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated hyperedge ${hyperedge.name}.`,
        });
      },
      "input"
    );
    bindMetadataEditors({
      target: hyperedge,
      tagsInput: hyperedgeTagsInput,
      tagsFieldKey: `hyperedge:${hyperedge.id}:tags`,
      customMetadataInput: hyperedgeCustomMetadataInput,
      customMetadataFieldKey: `hyperedge:${hyperedge.id}:custom-metadata`,
      statusMessage: `Updated hyperedge ${hyperedge.name}.`,
      invalidate: propertyInvalidation({ graph: false, minimap: false }),
      annotationScope: "edge",
    });
    bindClick("delete-hyperedge-button", () => {
      commands.deleteHyperedge({
        hyperedgeId: hyperedge.id,
        invalidate: propertyInvalidation({
          graph: true,
          lookups: true,
          analysis: true,
          planner: true,
          minimap: true,
        }),
        selectionIds: [],
        statusMessage: `Deleted hyperedge ${hyperedge.name}.`,
      });
    });
  }

  function bindNoteProperties({ note, noteColor }) {
    const noteTextInput = documentRef.getElementById("note-text-input");
    const noteColorInput = documentRef.getElementById("note-color-input");
    const noteTagsInput = documentRef.getElementById("note-tags-input");
    const noteCustomMetadataInput = documentRef.getElementById(
      "note-custom-metadata-input"
    );

    bindDebouncedAutosave(
      noteTextInput,
      `note:${note.id}:text`,
      () => {
        commands.updateNoteText({
          note,
          proposedText: noteTextInput.value,
          invalidate: propertyInvalidation({ overlays: true }),
          statusMessage: "Updated the note.",
        });
      },
      { commitOnEnter: false, scheduleOnInput: false }
    );
    bindImmediateAutosave(
      noteColorInput,
      `note:${note.id}:color`,
      () => {
        if (noteColorInput.value === noteColor) {
          return;
        }
        commands.updateTargetColor({
          target: note,
          nextColor: noteColorInput.value,
          invalidate: propertyInvalidation({ overlays: true }),
          statusMessage: "Updated the note.",
        });
      },
      "input"
    );
    bindMetadataEditors({
      target: note,
      tagsInput: noteTagsInput,
      tagsFieldKey: `note:${note.id}:tags`,
      customMetadataInput: noteCustomMetadataInput,
      customMetadataFieldKey: `note:${note.id}:custom-metadata`,
      statusMessage: "Updated the note.",
      invalidate: propertyInvalidation({ overlays: false }),
    });
    bindClick("delete-note-button", () => {
      commands.deleteNote({
        noteId: note.id,
        invalidate: propertyInvalidation({
          lookups: true,
          overlays: true,
        }),
        selectionIds: [],
        statusMessage: "Deleted the note.",
      });
    });
  }

  return {
    bindGroupProperties,
    bindEdgeProperties,
    bindHyperedgeProperties,
    bindNoteProperties,
  };
}
