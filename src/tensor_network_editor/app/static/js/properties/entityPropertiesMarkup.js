function buildTooltipAttributes(label, description = "", shortcut = "") {
  const safeLabel = String(label || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
  const safeDescription = String(description || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
  const safeShortcut = String(shortcut || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
  return `data-tooltip-enabled="true" data-shortcut-label="${safeLabel}"${
    safeShortcut ? ` data-shortcut="${safeShortcut}"` : ""
  }${
    safeDescription ? ` data-shortcut-description="${safeDescription}"` : ""
  }`;
}

export function buildGroupPropertiesMarkup({
  group,
  groupColor,
  disableSubnetworkActions,
  subnetworkActionsMessage,
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
        <label
          class="control-inline-color"
          for="group-color-input"
          data-tooltip-enabled="true"
          data-shortcut-label="Choose color"
          data-shortcut-description="Set the display color for this item."
        >
          <input
            id="group-color-input"
            data-focus-key="group:${group.id}:color"
            type="color"
            aria-label="Choose color"
            value="${escapeHtml(groupColor)}"
          />
        </label>
        <button
          id="add-index-to-group-button"
          type="button"
          class="button-accent-insert"
          data-tooltip-enabled="true"
          data-shortcut-label="Add index"
          data-shortcut="I"
          data-shortcut-description="Add one new open index to each tensor inside this group."
        >
          Add index
        </button>
        <button
          id="extract-group-button"
          type="button"
          class="button-accent-positive"
          ${disableSubnetworkActions ? "disabled" : ""}
          ${buildTooltipAttributes(
            "Extract",
            "Extract the tensors inside this group as a reusable subnetwork.",
            "Shift+E"
          )}
        >
          Extract
        </button>
        <button
          id="save-group-subnetwork-library-button"
          type="button"
          class="button-accent-template"
          ${disableSubnetworkActions ? "disabled" : ""}
          ${buildTooltipAttributes(
            "To Library",
            "Save the tensors inside this group to the subnetwork library."
          )}
        >
          To Library
        </button>
        <button
          id="promote-group-template-button"
          type="button"
          class="button-accent-template"
          ${disableSubnetworkActions ? "disabled" : ""}
          ${buildTooltipAttributes(
            "To Template",
            "Promote the tensors inside this group to a reusable template."
          )}
        >
          To Template
        </button>
        <button id="toggle-group-button" type="button">${isCollapsed ? "Expand Group" : "Collapse Group"}</button>
        <button
          id="delete-group-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete group"
          data-tooltip-enabled="true"
          data-shortcut="Delete"
          data-shortcut-label="Delete group"
          data-shortcut-description="Remove this group from the network."
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${
        subnetworkActionsMessage
          ? `<p class="property-meta">${escapeHtml(subnetworkActionsMessage)}</p>`
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
  return buildConnectionPropertiesMarkup({
    annotationScope: "edge",
    colorInputId: "edge-color-input",
    colorValue: edgeColor,
    connection: edge,
    customMetadataFocusKey: `edge:${edge.id}:custom-metadata`,
    customMetadataInputId: "edge-custom-metadata-input",
    deleteButtonId: "delete-edge-button",
    deleteDescription: "Remove this connection from the network.",
    focusPrefix: "edge",
    label: "Edge name",
    renderTrashIcon,
    tagsFocusKey: `edge:${edge.id}:tags`,
    tagsInputId: "edge-tags-input",
    buildMetadataEditorMarkup,
    escapeHtml,
  });
}

export function buildHyperedgePropertiesMarkup({
  hyperedge,
  hyperedgeColor,
  endpointCount,
  renderTrashIcon,
  buildMetadataEditorMarkup,
  escapeHtml,
}) {
  return `
      <div class="properties-chip-wrap">
        <div class="properties-chip">
          <span>Endpoints</span>
          <strong>${endpointCount}</strong>
        </div>
      </div>
      ${buildConnectionPropertiesMarkup({
        annotationScope: "edge",
        colorInputId: "hyperedge-color-input",
        colorValue: hyperedgeColor,
        connection: hyperedge,
        customMetadataFocusKey: `hyperedge:${hyperedge.id}:custom-metadata`,
        customMetadataInputId: "hyperedge-custom-metadata-input",
        deleteButtonId: "delete-hyperedge-button",
        deleteDescription: "Remove this hyperedge from the network.",
        focusPrefix: "hyperedge",
        label: "Hyperedge name",
        renderTrashIcon,
        tagsFocusKey: `hyperedge:${hyperedge.id}:tags`,
        tagsInputId: "hyperedge-tags-input",
        buildMetadataEditorMarkup,
        escapeHtml,
      })}
    `;
}

function buildConnectionPropertiesMarkup({
  annotationScope,
  colorInputId,
  colorValue,
  connection,
  customMetadataFocusKey,
  customMetadataInputId,
  deleteButtonId,
  deleteDescription,
  focusPrefix,
  label,
  renderTrashIcon,
  tagsFocusKey,
  tagsInputId,
  buildMetadataEditorMarkup,
  escapeHtml,
}) {
  return `
      <div class="field-group">
        <label for="${escapeHtml(focusPrefix)}-name-input">${escapeHtml(label)}</label>
        <input
          id="${escapeHtml(focusPrefix)}-name-input"
          data-focus-key="${escapeHtml(focusPrefix)}:${escapeHtml(connection.id)}:name"
          value="${escapeHtml(connection.name)}"
        />
      </div>
      <div class="button-row">
        <label
          class="control-inline-color"
          for="${escapeHtml(colorInputId)}"
          data-tooltip-enabled="true"
          data-shortcut-label="Choose color"
          data-shortcut-description="Set the display color for this item."
        >
          <input
            id="${escapeHtml(colorInputId)}"
            data-focus-key="${escapeHtml(focusPrefix)}:${escapeHtml(connection.id)}:color"
            type="color"
            aria-label="Choose color"
            value="${escapeHtml(colorValue)}"
          />
        </label>
        <button
          id="${escapeHtml(deleteButtonId)}"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete connection"
          data-tooltip-enabled="true"
          data-shortcut="Delete"
          data-shortcut-label="Delete connection"
          data-shortcut-description="${escapeHtml(deleteDescription)}"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId,
        tagsFocusKey,
        customMetadataInputId,
        customMetadataFocusKey,
        target: connection,
        annotationScope,
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
        <label
          class="control-inline-color"
          for="note-color-input"
          data-tooltip-enabled="true"
          data-shortcut-label="Choose color"
          data-shortcut-description="Set the display color for this item."
        >
          <input
            id="note-color-input"
            data-focus-key="note:${note.id}:color"
            type="color"
            aria-label="Choose color"
            value="${escapeHtml(noteColor)}"
          />
        </label>
        <button
          id="delete-note-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete note"
          data-tooltip-enabled="true"
          data-shortcut="Delete"
          data-shortcut-label="Delete note"
          data-shortcut-description="Remove this note from the canvas."
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
