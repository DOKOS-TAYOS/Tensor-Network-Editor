export function buildGroupPropertiesMarkup({
  group,
  groupColor,
  linearPeriodicMode,
  totalElementCount,
  formatTotalElementCount,
  renderTrashIcon,
  buildMetadataEditorMarkup,
  escapeHtml,
}) {
  const isCollapsed = Boolean(group.metadata && group.metadata.collapsed);
  return `
      <div class="field-group">
        <label for="group-name-input">Group name</label>
        <input
          id="group-name-input"
          data-focus-key="group:${group.id}:name"
          value="${escapeHtml(group.name)}"
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
            value="${escapeHtml(groupColor)}"
          />
        </label>
        <button id="add-index-to-group-button" type="button">
          Add index
        </button>
        <button
          id="extract-group-button"
          type="button"
          ${linearPeriodicMode ? "disabled" : ""}
        >
          Extract
        </button>
        <button
          id="promote-group-template-button"
          type="button"
          ${linearPeriodicMode ? "disabled" : ""}
        >
          To Template
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
        collapsible: true,
      })}
    `;
}

export function buildEdgePropertiesMarkup({
  edge,
  edgeColor,
  renderTrashIcon,
  buildMetadataEditorMarkup,
  escapeHtml,
}) {
  return `
      <div class="field-group">
        <label for="edge-name-input">Edge name</label>
        <input
          id="edge-name-input"
          data-focus-key="edge:${edge.id}:name"
          value="${escapeHtml(edge.name)}"
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
            value="${escapeHtml(edgeColor)}"
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
        collapsible: true,
      })}
    `;
}

export function buildNotePropertiesMarkup({
  note,
  noteColor,
  renderTrashIcon,
  buildMetadataEditorMarkup,
  escapeHtml,
}) {
  return `
      <div class="field-group">
        <label for="note-text-input">Note text</label>
        <textarea
          id="note-text-input"
          data-focus-key="note:${note.id}:text"
          rows="6"
        >${escapeHtml(note.text)}</textarea>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="note-color-input">
          <input
            id="note-color-input"
            data-focus-key="note:${note.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${escapeHtml(noteColor)}"
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
        collapsible: true,
      })}
    `;
}
