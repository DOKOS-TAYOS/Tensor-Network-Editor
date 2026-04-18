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
  tensorsOnly,
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
      <div class="button-row">
        ${
          tensorsOnly
            ? '<button id="add-index-to-selection-button" type="button" class="button-accent-insert">Add Index to Tensors</button>'
            : ""
        }
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
      ${
        tensorsOnly
          ? `
            <div class="properties-section-heading">Layout</div>
            <div class="button-row layout-align-row">
              <button
                id="align-selection-left-button"
                type="button"
                aria-label="Align left"
                title="Align left"
              >
                &larr;
              </button>
              <button
                id="align-selection-right-button"
                type="button"
                aria-label="Align right"
                title="Align right"
              >
                &rarr;
              </button>
              <button
                id="align-selection-top-button"
                type="button"
                aria-label="Align top"
                title="Align top"
              >
                &uarr;
              </button>
              <button
                id="align-selection-middle-button"
                type="button"
                aria-label="Align middle"
                title="Align middle"
              >
                &#8857;
              </button>
              <button
                id="align-selection-bottom-button"
                type="button"
                aria-label="Align bottom"
                title="Align bottom"
              >
                &darr;
              </button>
            </div>
            <div class="button-row">
              <button id="arrange-selection-chain-button" type="button">Arrange Chain</button>
              <button id="arrange-selection-tree-button" type="button">Arrange Tree</button>
              <button id="arrange-selection-grid-button" type="button">Arrange Grid</button>
            </div>
            <div class="button-row">
              <button
                id="distribute-selection-horizontal-button"
                type="button"
                ${baseTensorCount < 3 ? "disabled" : ""}
              >
                Distribute Horizontally
              </button>
              <button
                id="distribute-selection-vertical-button"
                type="button"
                ${baseTensorCount < 3 ? "disabled" : ""}
              >
                Distribute Vertically
              </button>
              <button id="snap-selection-button" type="button">Snap to Grid</button>
            </div>
            <div class="properties-section-heading">Subnetwork</div>
            <div class="button-row">
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
                Promote Selection to Template
              </button>
            </div>
            ${
              linearPeriodicMode
                ? '<p class="property-meta">Subnetwork export and template promotion are not available in For mode yet.</p>'
                : ""
            }
          `
          : ""
      }
      <p class="property-meta">
        Drag any selected tensor to move the selected tensor group together.
      </p>
    `;
}
