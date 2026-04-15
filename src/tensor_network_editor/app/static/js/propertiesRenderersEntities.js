export function createEntityPropertiesRenderers({
  ctx,
  state,
  document,
  propertiesPanel,
  support,
}) {
  const {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    buildMetadataEditorMarkup,
    propertyInvalidation,
    renderTrashIcon,
    getTotalElementCountForTensorIds,
    formatTotalElementCount,
  } = support;

  function renderGroupProperties(groupId) {
    const group = ctx.findGroupById(groupId);
    if (!group) {
      ctx.clearSelection();
      return;
    }
    const groupColor = ctx.getMetadataColor(group.metadata, "#61a8ff");
    const isCollapsed = Boolean(group.metadata && group.metadata.collapsed);
    const linearPeriodicMode =
      typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode();
    const totalElementCount = getTotalElementCountForTensorIds(
      Array.isArray(group.tensor_ids) ? group.tensor_ids : []
    );
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="group-name-input">Group name</label>
        <input
          id="group-name-input"
          data-focus-key="group:${group.id}:name"
          value="${ctx.escapeHtml(group.name)}"
        />
      </div>
      <div class="properties-chip-wrap">
        <div class="properties-chip">
          <span>Member tensors</span>
          <strong>${Array.isArray(group.tensor_ids) ? group.tensor_ids.length : 0}</strong>
        </div>
        ${
          totalElementCount !== null
            ? `
              <div class="properties-chip">
                <span>Total elements</span>
                <strong>${formatTotalElementCount(totalElementCount)}</strong>
              </div>
            `
            : ""
        }
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="group-color-input">
          <input
            id="group-color-input"
            data-focus-key="group:${group.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(groupColor)}"
          />
        </label>
        <button
          id="extract-group-button"
          type="button"
          ${linearPeriodicMode ? "disabled" : ""}
        >
          Extract Group
        </button>
        <button
          id="promote-group-template-button"
          type="button"
          ${linearPeriodicMode ? "disabled" : ""}
        >
          Promote Group to Template
        </button>
        <button id="toggle-group-button" type="button">${isCollapsed ? "Expand Group" : "Collapse Group"}</button>
        <button
          id="delete-group-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete group"
          title="Delete group"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${
        linearPeriodicMode
          ? '<p class="property-meta">Subnetwork export and template promotion are not available in For mode yet.</p>'
          : ""
      }
      <p class="property-meta">Drag the group box on the canvas to move all tensors together.</p>
      ${buildMetadataEditorMarkup({
        tagsInputId: "group-tags-input",
        tagsFocusKey: `group:${group.id}:tags`,
        customMetadataInputId: "group-custom-metadata-input",
        customMetadataFocusKey: `group:${group.id}:custom-metadata`,
        target: group,
      })}
    `;

    const groupNameInput = document.getElementById("group-name-input");
    const groupColorInput = document.getElementById("group-color-input");
    const groupTagsInput = document.getElementById("group-tags-input");
    const groupCustomMetadataInput = document.getElementById(
      "group-custom-metadata-input"
    );

    bindDebouncedAutosave(groupNameInput, `group:${group.id}:name`, () => {
      const proposedName = groupNameInput.value.trim();
      if (!proposedName) {
        ctx.setStatus("Group name cannot be empty.", "error");
        return;
      }
      if (proposedName === group.name) {
        return;
      }
      ctx.applyDesignChange(
        () => {
          group.name = proposedName;
        },
        {
          invalidate: propertyInvalidation({ overlays: true }),
          statusMessage: `Updated group ${proposedName}.`,
        }
      );
    });
    bindImmediateAutosave(
      groupColorInput,
      `group:${group.id}:color`,
      () => {
        if (groupColorInput.value === groupColor) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            group.metadata.color = groupColorInput.value;
          },
          {
            invalidate: propertyInvalidation({ overlays: true }),
            statusMessage: `Updated group ${group.name}.`,
          }
        );
      },
      "input"
    );
    document.getElementById("toggle-group-button").addEventListener("click", () => {
      ctx.toggleGroupCollapse(group.id);
    });
    const extractGroupButton = document.getElementById("extract-group-button");
    if (extractGroupButton) {
      extractGroupButton.addEventListener("click", () => {
        ctx.exportGroupSubnetwork(group.id);
      });
    }
    const promoteGroupTemplateButton = document.getElementById(
      "promote-group-template-button"
    );
    if (promoteGroupTemplateButton) {
      promoteGroupTemplateButton.addEventListener("click", () => {
        ctx.promoteGroupToTemplate(group.id);
      });
    }
    bindMetadataEditors({
      target: group,
      tagsInput: groupTagsInput,
      tagsFieldKey: `group:${group.id}:tags`,
      customMetadataInput: groupCustomMetadataInput,
      customMetadataFieldKey: `group:${group.id}:custom-metadata`,
      statusMessage: `Updated group ${group.name}.`,
      invalidate: propertyInvalidation({ overlays: false }),
    });
    document.getElementById("delete-group-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          state.spec.groups = state.spec.groups.filter(
            (candidate) => candidate.id !== group.id
          );
        },
        {
          invalidate: propertyInvalidation({
            lookups: true,
            overlays: true,
          }),
          selectionIds: [],
          statusMessage: `Deleted group ${group.name}.`,
        }
      );
    });
  }

  function renderEdgeProperties(edgeId) {
    const edge = ctx.findEdgeById(edgeId);
    if (!edge) {
      ctx.clearSelection();
      return;
    }
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="edge-name-input">Edge name</label>
        <input
          id="edge-name-input"
          data-focus-key="edge:${edge.id}:name"
          value="${ctx.escapeHtml(edge.name)}"
        />
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="edge-color-input">
          <input
            id="edge-color-input"
            data-focus-key="edge:${edge.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(ctx.getMetadataColor(edge.metadata, "#8da1c3"))}"
          />
        </label>
        <button
          id="delete-edge-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete connection"
          title="Delete connection"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId: "edge-tags-input",
        tagsFocusKey: `edge:${edge.id}:tags`,
        customMetadataInputId: "edge-custom-metadata-input",
        customMetadataFocusKey: `edge:${edge.id}:custom-metadata`,
        target: edge,
      })}
    `;

    const edgeNameInput = document.getElementById("edge-name-input");
    const edgeColorInput = document.getElementById("edge-color-input");
    const edgeTagsInput = document.getElementById("edge-tags-input");
    const edgeCustomMetadataInput = document.getElementById(
      "edge-custom-metadata-input"
    );

    bindDebouncedAutosave(edgeNameInput, `edge:${edge.id}:name`, () => {
      const proposedName = edgeNameInput.value.trim();
      if (!proposedName) {
        ctx.setStatus("Connection name cannot be empty.", "error");
        return;
      }
      if (proposedName === edge.name) {
        return;
      }
      ctx.applyDesignChange(
        () => {
          edge.name = proposedName;
        },
        {
          invalidate: propertyInvalidation({ graph: true }),
          statusMessage: `Updated connection ${proposedName}.`,
        }
      );
    });
    bindImmediateAutosave(
      edgeColorInput,
      `edge:${edge.id}:color`,
      () => {
        if (
          edgeColorInput.value ===
          ctx.getMetadataColor(edge.metadata, "#8da1c3")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            edge.metadata.color = edgeColorInput.value;
          },
          {
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated connection ${edge.name}.`,
          }
        );
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

    document.getElementById("delete-edge-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.removeEdge(edge.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted connection ${edge.name}.`,
        }
      );
    });
  }

  function renderNoteProperties(noteId) {
    const note = ctx.findNoteById(noteId);
    if (!note) {
      ctx.clearSelection();
      return;
    }

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="note-text-input">Note text</label>
        <textarea
          id="note-text-input"
          data-focus-key="note:${note.id}:text"
          rows="6"
        >${ctx.escapeHtml(note.text)}</textarea>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="note-color-input">
          <input
            id="note-color-input"
            data-focus-key="note:${note.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(ctx.getMetadataColor(note.metadata, "#5f95ff"))}"
          />
        </label>
        <button
          id="delete-note-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete note"
          title="Delete note"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      <p class="property-meta">Move the note from its title bar directly on the canvas.</p>
      ${buildMetadataEditorMarkup({
        tagsInputId: "note-tags-input",
        tagsFocusKey: `note:${note.id}:tags`,
        customMetadataInputId: "note-custom-metadata-input",
        customMetadataFocusKey: `note:${note.id}:custom-metadata`,
        target: note,
      })}
    `;

    const noteTextInput = document.getElementById("note-text-input");
    const noteColorInput = document.getElementById("note-color-input");
    const noteTagsInput = document.getElementById("note-tags-input");
    const noteCustomMetadataInput = document.getElementById(
      "note-custom-metadata-input"
    );

    bindDebouncedAutosave(
      noteTextInput,
      `note:${note.id}:text`,
      () => {
        const proposedText = noteTextInput.value.trim();
        if (!proposedText) {
          ctx.setStatus("Notes cannot be empty.", "error");
          return;
        }
        if (proposedText === note.text) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.text = proposedText;
          },
          {
            invalidate: propertyInvalidation({ overlays: true }),
            statusMessage: "Updated the note.",
          }
        );
      },
      { commitOnEnter: false }
    );
    bindImmediateAutosave(
      noteColorInput,
      `note:${note.id}:color`,
      () => {
        if (
          noteColorInput.value ===
          ctx.getMetadataColor(note.metadata, "#5f95ff")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.metadata.color = noteColorInput.value;
          },
          {
            invalidate: propertyInvalidation({ overlays: true }),
            statusMessage: "Updated the note.",
          }
        );
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

    document.getElementById("delete-note-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.removeNote(note.id);
        },
        {
          invalidate: propertyInvalidation({
            lookups: true,
            overlays: true,
          }),
          selectionIds: [],
          statusMessage: "Deleted the note.",
        }
      );
    });
  }

  return {
    renderGroupProperties,
    renderEdgeProperties,
    renderNoteProperties,
  };
}
