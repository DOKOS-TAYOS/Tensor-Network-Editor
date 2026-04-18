export function buildNetworkPropertiesMarkup({
  spec,
  escapeHtml,
  buildMetadataEditorMarkup,
}) {
  return `
      <div class="field-group">
        <label for="network-name-input">Design name</label>
        <input
          id="network-name-input"
          data-focus-key="network:name"
          value="${escapeHtml(spec.name)}"
        />
      </div>
      <div class="properties-chip">
        <span>Tensors</span>
        <strong>${spec.tensors.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Connections</span>
        <strong>${spec.edges.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Groups</span>
        <strong>${Array.isArray(spec.groups) ? spec.groups.length : 0}</strong>
      </div>
      <div class="properties-chip">
        <span>Notes</span>
        <strong>${Array.isArray(spec.notes) ? spec.notes.length : 0}</strong>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId: "network-tags-input",
        tagsFocusKey: "network:tags",
        customMetadataInputId: "network-custom-metadata-input",
        customMetadataFocusKey: "network:custom-metadata",
        target: spec,
        collapsible: true,
      })}
    `;
}

export function buildMultiSelectionPropertiesMarkup({
  selectedEntries,
  baseTensorCount,
  tensorCount,
  indexCount,
  edgeCount,
  groupCount,
  noteCount,
  hasMultipleTensors,
  linearPeriodicMode,
  batchColor,
  totalElementCount,
  formatTotalElementCount,
  renderTrashIcon,
  escapeHtml,
}) {
  return `
      <div class="properties-summary">
        <div class="properties-chip">
          <span>Selected</span>
          <strong>${selectedEntries.length}</strong>
        </div>
        <div class="properties-chip-wrap">
          <div class="properties-chip">
            <span>Tensors</span>
            <strong>${tensorCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Indices</span>
            <strong>${indexCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Connections</span>
            <strong>${edgeCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Groups</span>
            <strong>${groupCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Notes</span>
            <strong>${noteCount}</strong>
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
      </div>
      ${
        hasMultipleTensors
          ? `
            <div class="button-row selection-tensor-actions-row">
              <button id="add-index-to-selection-button" type="button" class="button-accent-insert">Add Index to Tensors</button>
              <button
                id="extract-selection-button"
                type="button"
                ${linearPeriodicMode ? "disabled" : ""}
              >
                Extract Selection
              </button>
              <button
                id="promote-selection-template-button"
                type="button"
                ${linearPeriodicMode ? "disabled" : ""}
              >
                Promote to Template
              </button>
              <button id="group-selection-button" type="button">Group</button>
            </div>
            ${
              linearPeriodicMode
                ? '<p class="property-meta">Subnetwork export and template promotion are not available in For mode yet.</p>'
                : ""
            }
          `
          : ""
      }
      <div class="button-row">
        <label class="control-inline-color" for="multi-color-input">
          <input
            id="multi-color-input"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${escapeHtml(batchColor)}"
          />
        </label>
        <button
          id="delete-selection-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete selection"
          title="Delete selection"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      <p class="property-meta">
        Drag any selected tensor to move the selected tensor group together.
      </p>
    `;
}
