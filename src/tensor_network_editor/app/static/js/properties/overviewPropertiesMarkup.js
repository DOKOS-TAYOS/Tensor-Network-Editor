function buildTooltipAttributes(label, description = "") {
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
  return `data-tooltip-enabled="true" data-shortcut-label="${safeLabel}"${
    safeDescription ? ` data-shortcut-description="${safeDescription}"` : ""
  }`;
}

export function buildNetworkPropertiesMarkup({
  spec,
  connectionCount,
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
        <strong>${connectionCount}</strong>
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
  connectionCount,
  groupCount,
  noteCount,
  hasMultipleTensors,
  showAddIndexAction,
  hyperedgeCreationCandidate,
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
            <strong>${connectionCount}</strong>
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
              ${
                showAddIndexAction
                  ? `
                    <button
                      id="add-index-to-selection-button"
                      type="button"
                      class="button-accent-insert"
                      data-tooltip-enabled="true"
                      data-shortcut-label="Add index"
                      data-shortcut-description="Add one new open index to each selected tensor."
                    >Add index</button>
                  `
                  : ""
              }
              <button
                id="extract-selection-button"
                type="button"
                class="button-accent-positive"
                ${linearPeriodicMode ? "disabled" : ""}
                ${buildTooltipAttributes(
                  "Extract",
                  "Extract the selected tensors as a reusable subnetwork."
                )}
              >
                Extract
              </button>
              <button
                id="save-selection-subnetwork-library-button"
                type="button"
                class="button-accent-template"
                ${linearPeriodicMode ? "disabled" : ""}
                ${buildTooltipAttributes(
                  "To Library",
                  "Save the selected tensors to the subnetwork library."
                )}
              >
                To Library
              </button>
              <button
                id="promote-selection-template-button"
                type="button"
                class="button-accent-template"
                ${linearPeriodicMode ? "disabled" : ""}
                ${buildTooltipAttributes(
                  "To Template",
                  "Promote the selected tensors to a reusable template."
                )}
              >
                To Template
              </button>
              <button
                id="group-selection-button"
                type="button"
                class="button-accent-insert"
                ${buildTooltipAttributes(
                  "Group",
                  "Create a visual group from the selected tensors."
                )}
              >
                Group
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
      <div class="button-row">
        ${
          hyperedgeCreationCandidate?.selectionContainsOnlyIndices &&
          Array.isArray(hyperedgeCreationCandidate.selectedIndexIds) &&
          hyperedgeCreationCandidate.selectedIndexIds.length
            ? `
              <button
                id="create-hyperedge-button"
                type="button"
                ${hyperedgeCreationCandidate.canCreate ? "" : "disabled"}
                ${buildTooltipAttributes(
                  "Create hyperedge",
                  hyperedgeCreationCandidate.canCreate
                    ? "Create a hyperedge from the selected open indices."
                    : hyperedgeCreationCandidate.message || "This selection cannot form a hyperedge."
                )}
              >
                Create hyperedge
              </button>
            `
            : ""
        }
        <label
          class="control-inline-color"
          for="multi-color-input"
          data-tooltip-enabled="true"
          data-shortcut-label="Choose color"
          data-shortcut-description="Set the display color for this item."
        >
          <input
            id="multi-color-input"
            type="color"
            aria-label="Choose color"
            value="${escapeHtml(batchColor)}"
          />
        </label>
        <button
          id="delete-selection-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete selection"
          data-tooltip-enabled="true"
          data-shortcut-label="Delete selection"
          data-shortcut-description="Remove the current selection from the network."
        >
          ${renderTrashIcon()}
        </button>
      </div>
      <p class="property-meta">
        Drag any selected tensor to move the selected tensor group together.
      </p>
    `;
}
