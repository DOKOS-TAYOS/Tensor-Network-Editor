export function createOverviewPropertiesRenderers({
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
    selectionColorInvalidation,
    renderTrashIcon,
    getSelectionTotalElementCount,
    formatTotalElementCount,
  } = support;

  function renderNetworkProperties() {
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="network-name-input">Design name</label>
        <input
          id="network-name-input"
          data-focus-key="network:name"
          value="${ctx.escapeHtml(state.spec.name)}"
        />
      </div>
      <div class="properties-chip">
        <span>Tensors</span>
        <strong>${state.spec.tensors.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Connections</span>
        <strong>${state.spec.edges.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Groups</span>
        <strong>${Array.isArray(state.spec.groups) ? state.spec.groups.length : 0}</strong>
      </div>
      <div class="properties-chip">
        <span>Notes</span>
        <strong>${Array.isArray(state.spec.notes) ? state.spec.notes.length : 0}</strong>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId: "network-tags-input",
        tagsFocusKey: "network:tags",
        customMetadataInputId: "network-custom-metadata-input",
        customMetadataFocusKey: "network:custom-metadata",
        target: state.spec,
      })}
    `;

    const networkNameInput = document.getElementById("network-name-input");
    const networkTagsInput = document.getElementById("network-tags-input");
    const networkCustomMetadataInput = document.getElementById(
      "network-custom-metadata-input"
    );
    bindDebouncedAutosave(networkNameInput, "network:name", () => {
      const proposedName = networkNameInput.value.trim();
      if (!proposedName) {
        ctx.setStatus("Design name cannot be empty.", "error");
        return;
      }
      if (proposedName === state.spec.name) {
        return;
      }
      ctx.applyDesignChange(
        () => {
          state.spec.name = proposedName;
        },
        {
          invalidate: propertyInvalidation(),
          statusMessage: "Updated design name.",
        }
      );
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

  function renderMultiSelectionProperties() {
    const selectedEntries = ctx.getSelectedEntries();
    const baseTensorCount = selectedEntries.filter(
      (entry) => entry.kind === "tensor"
    ).length;
    const tensorCount = selectedEntries.filter(
      (entry) => entry.kind === "tensor" || entry.kind === "contraction-tensor"
    ).length;
    const indexCount = selectedEntries.filter(
      (entry) => entry.kind === "index"
    ).length;
    const edgeCount = selectedEntries.filter(
      (entry) => entry.kind === "edge"
    ).length;
    const groupCount = selectedEntries.filter(
      (entry) => entry.kind === "group"
    ).length;
    const noteCount = selectedEntries.filter(
      (entry) => entry.kind === "note"
    ).length;
    const tensorsOnly =
      baseTensorCount > 0 && baseTensorCount === selectedEntries.length;
    const batchColor = ctx.getBatchColorValue(selectedEntries);
    const totalElementCount = getSelectionTotalElementCount(selectedEntries);

    propertiesPanel.innerHTML = `
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
            value="${ctx.escapeHtml(batchColor)}"
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

    const multiColorInput = document.getElementById("multi-color-input");
    bindImmediateAutosave(
      multiColorInput,
      "selection:color",
      () => {
        const colorValue = multiColorInput.value;
        if (colorValue === batchColor) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            ctx.applyColorToSelection(colorValue);
          },
          {
            invalidate: selectionColorInvalidation(selectedEntries),
            statusMessage: "Updated the selection color.",
          }
        );
      },
      "input"
    );

    const addIndexButton = document.getElementById(
      "add-index-to-selection-button"
    );
    if (addIndexButton) {
      addIndexButton.addEventListener("click", () => {
        ctx.applyDesignChange(
          () => {
            ctx.getSelectedIdsByKind("tensor").forEach((tensorId) => {
              const tensor = ctx.findTensorById(tensorId);
              if (tensor) {
                tensor.indices.push(ctx.createIndex(tensor, tensor.indices.length));
              }
            });
          },
          {
            statusMessage: "Added one index to each selected tensor.",
          }
        );
      });
    }

    document
      .getElementById("delete-selection-button")
      .addEventListener("click", ctx.deleteSelection);
  }

  return {
    renderNetworkProperties,
    renderMultiSelectionProperties,
  };
}
