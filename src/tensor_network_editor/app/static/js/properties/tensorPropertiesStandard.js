import { GRAPH_THEME } from "../theme.js";

export function createStandardTensorPropertiesRenderer({
  bindDebouncedAutosave,
  bindImmediateAutosave,
  bindMetadataEditors,
  buildMetadataEditorMarkup,
  commands,
  ctx,
  document,
  formatTotalElementCount,
  getTensorTotalElementCount,
  isTensorIndexDisclosureOpen,
  propertiesPanel,
  propertyInvalidation,
  renderTrashIcon,
  toggleTensorIndexDisclosure,
}) {
  function buildTooltipAttributes(label, description = "") {
    const safeLabel = ctx.escapeHtml(label);
    const safeDescription = ctx.escapeHtml(description);
    return `data-tooltip-enabled="true" data-shortcut-label="${safeLabel}"${
      safeDescription ? ` data-shortcut-description="${safeDescription}"` : ""
    }`;
  }

  function createPaleIndexColor(color) {
    const matchedHex = /^#?([0-9a-f]{6})$/iu.exec(String(color || "").trim());
    if (!matchedHex) {
      return "rgba(216, 226, 245, 0.92)";
    }
    const hexColor = matchedHex[1];
    const red = Number.parseInt(hexColor.slice(0, 2), 16);
    const green = Number.parseInt(hexColor.slice(2, 4), 16);
    const blue = Number.parseInt(hexColor.slice(4, 6), 16);
    const blendToWhite = (component) =>
      Math.round(component + (255 - component) * 0.58);
    return `rgba(${blendToWhite(red)}, ${blendToWhite(green)}, ${blendToWhite(blue)}, 0.94)`;
  }

  function renderTensorProperties(tensor, options = {}) {
    const focusedIndexId = options.focusedIndexId || null;
    const tensorIndexCount = Array.isArray(tensor.indices) ? tensor.indices.length : 0;
    const totalElementCount = getTensorTotalElementCount(tensor);
    const indexEditors = tensor.indices
      .map((index, indexPosition) => {
        const isOpen = isTensorIndexDisclosureOpen(tensor.id, index.id);
        const isConnected = Boolean(ctx.findEdgeByIndexId(index.id));
        const indexColor = ctx.getIndexColor(index, isConnected);

        return `
          <section class="planner-section planner-disclosure index-disclosure${isOpen ? " is-open" : ""}">
            <button
              type="button"
              class="planner-disclosure-toggle index-disclosure-toggle${
                isOpen ? " is-open" : ""
              }${focusedIndexId === index.id ? " is-focused" : ""}"
              data-index-toggle="${index.id}"
              aria-expanded="${isOpen}"
              style="--index-border-color: ${ctx.escapeHtml(
                indexColor
              )}; --index-state-color: ${ctx.escapeHtml(
                createPaleIndexColor(indexColor)
              )};"
            >
              <span class="index-disclosure-title">
                <strong>${indexPosition + 1}. ${ctx.escapeHtml(index.name)}</strong>
                <span>${isConnected ? "Connected" : "Open"} &middot; dim ${index.dimension}</span>
              </span>
              <strong class="planner-disclosure-state index-disclosure-state">${
                isOpen ? "Hide" : "Show"
              }</strong>
            </button>
            ${
              isOpen
                ? `
                  <div class="planner-disclosure-body index-disclosure-body">
                    <div class="field-row index-disclosure-fields">
                      <div class="field-group index-name-field">
                        <label for="index-name-input-${index.id}">Index name</label>
                        <input
                          id="index-name-input-${index.id}"
                          data-focus-key="index:${index.id}:name"
                          value="${ctx.escapeHtml(index.name)}"
                        />
                      </div>
                      <div class="field-group compact-number-field index-dimension-field">
                        <label for="index-dimension-input-${index.id}">Dimension</label>
                        <input
                          id="index-dimension-input-${index.id}"
                          data-focus-key="index:${index.id}:dimension"
                          ${buildTooltipAttributes(
                            "Index dimension",
                            "Set the size of this index. Connected indices should share the same dimension."
                          )}
                          type="number"
                          min="1"
                          step="1"
                          value="${index.dimension}"
                        />
                      </div>
                    </div>
                    <div class="button-row">
                      <label
                        class="control-inline-color"
                        for="index-color-input-${index.id}"
                        ${buildTooltipAttributes(
                          "Choose color",
                          "Set the display color for this item."
                        )}
                      >
                        <input
                          id="index-color-input-${index.id}"
                          data-focus-key="index:${index.id}:color"
                          type="color"
                          aria-label="Choose color"
                          value="${ctx.escapeHtml(
                            ctx.getMetadataColor(
                              index.metadata,
                              ctx.getIndexColor(index, isConnected)
                            )
                          )}"
                        />
                      </label>
                      <button
                        id="move-index-up-button-${index.id}"
                        type="button"
                        class="icon-button index-action-button"
                        aria-label="Move index up"
                        ${buildTooltipAttributes(
                          "Move index up",
                          "Move this index one position earlier in the tensor index order."
                        )}
                        ${indexPosition === 0 ? "disabled" : ""}
                      >
                        <span aria-hidden="true">&#8593;</span>
                      </button>
                      <button
                        id="move-index-down-button-${index.id}"
                        type="button"
                        class="icon-button index-action-button"
                        aria-label="Move index down"
                        ${buildTooltipAttributes(
                          "Move index down",
                          "Move this index one position later in the tensor index order."
                        )}
                        ${
                          indexPosition === tensor.indices.length - 1 ? "disabled" : ""
                        }
                      >
                        <span aria-hidden="true">&#8595;</span>
                      </button>
                      <button
                        id="delete-index-button-${index.id}"
                        type="button"
                        class="icon-button index-action-button danger"
                        aria-label="Delete index"
                        ${buildTooltipAttributes(
                          "Delete index",
                          "Remove this index from the tensor."
                        )}
                      >
                        ${renderTrashIcon()}
                      </button>
                    </div>
                    ${buildMetadataEditorMarkup({
                      tagsInputId: `index-tags-input-${index.id}`,
                      tagsFocusKey: `index:${index.id}:tags`,
                      customMetadataInputId: `index-custom-metadata-input-${index.id}`,
                      customMetadataFocusKey: `index:${index.id}:custom-metadata`,
                      target: index,
                      annotationScope: "index",
                      collapsible: true,
                    })}
                  </div>
                `
                : ""
            }
          </section>
        `;
      })
      .join("");

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="tensor-name-input">Tensor name</label>
        <input
          id="tensor-name-input"
          data-focus-key="tensor:${tensor.id}:name"
          value="${ctx.escapeHtml(tensor.name)}"
        />
      </div>
      <div class="properties-chip-wrap">
        <div class="properties-chip">
          <span>Indices</span>
          <strong>${tensorIndexCount}</strong>
        </div>
        <div class="properties-chip">
          <span>Total elements</span>
          <strong>${formatTotalElementCount(totalElementCount)}</strong>
        </div>
      </div>
      <div class="button-row">
        <button
          id="add-index-button"
          type="button"
          class="icon-button button-accent-insert"
          aria-label="Add index"
          ${buildTooltipAttributes(
            "Add index",
            "Create a new open index on this tensor."
          )}
        >
          +
        </button>
        <label
          class="control-inline-color"
          for="tensor-color-input"
          ${buildTooltipAttributes(
            "Choose color",
            "Set the display color for this item."
          )}
        >
          <input
            id="tensor-color-input"
            data-focus-key="tensor:${tensor.id}:color"
            type="color"
            aria-label="Choose color"
            value="${ctx.escapeHtml(ctx.getMetadataColor(tensor.metadata, GRAPH_THEME.tensorFallback))}"
          />
        </label>
        <button
          id="delete-tensor-button"
          type="button"
          class="icon-button danger"
          aria-label="Delete tensor"
          ${buildTooltipAttributes(
            "Delete tensor",
            "Remove this tensor from the network."
          )}
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId: "tensor-tags-input",
        tagsFocusKey: `tensor:${tensor.id}:tags`,
        customMetadataInputId: "tensor-custom-metadata-input",
        customMetadataFocusKey: `tensor:${tensor.id}:custom-metadata`,
        target: tensor,
        annotationScope: "tensor",
        collapsible: true,
      })}
      <div class="properties-list">
        ${indexEditors || "<p class='property-meta'>This tensor has no indices yet.</p>"}
      </div>
    `;

    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");
    const tensorTagsInput = document.getElementById("tensor-tags-input");
    const tensorCustomMetadataInput = document.getElementById(
      "tensor-custom-metadata-input"
    );

    bindDebouncedAutosave(
      tensorNameInput,
      `tensor:${tensor.id}:name`,
      () => {
        commands.renameTensor({
          tensor,
          proposedName: tensorNameInput.value,
          invalidate: propertyInvalidation({ graph: true, planner: true }),
          statusMessage: `Updated tensor ${tensorNameInput.value.trim()}.`,
        });
      }
    );
    bindImmediateAutosave(
      tensorColorInput,
      `tensor:${tensor.id}:color`,
      () => {
        commands.updateTargetColor({
          target: tensor,
          nextColor: tensorColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated tensor ${tensor.name}.`,
        });
      },
      "input"
    );
    bindMetadataEditors({
      target: tensor,
      tagsInput: tensorTagsInput,
      tagsFieldKey: `tensor:${tensor.id}:tags`,
      customMetadataInput: tensorCustomMetadataInput,
      customMetadataFieldKey: `tensor:${tensor.id}:custom-metadata`,
      statusMessage: `Updated tensor ${tensor.name}.`,
      invalidate: propertyInvalidation(),
      annotationScope: "tensor",
    });

    document.getElementById("add-index-button").addEventListener("click", () => {
      commands.addTensorIndex({
        tensor,
        selectionIds: [tensor.id],
        primaryId: tensor.id,
        statusMessage: `Added one index to ${tensor.name}.`,
      });
    });
    document
      .getElementById("delete-tensor-button")
      .addEventListener("click", () => {
        commands.deleteTensor({
          tensorId: tensor.id,
          selectionIds: [],
          statusMessage: `Deleted tensor ${tensor.name}.`,
        });
      });

    propertiesPanel.querySelectorAll("[data-index-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        toggleTensorIndexDisclosure(tensor.id, button.dataset.indexToggle);
      });
    });

    tensor.indices.forEach((index, indexPosition) => {
      const indexColorInput = document.getElementById(
        `index-color-input-${index.id}`
      );
      const indexNameInput = document.getElementById(
        `index-name-input-${index.id}`
      );
      const indexDimensionInput = document.getElementById(
        `index-dimension-input-${index.id}`
      );
      const indexTagsInput = document.getElementById(
        `index-tags-input-${index.id}`
      );
      const indexCustomMetadataInput = document.getElementById(
        `index-custom-metadata-input-${index.id}`
      );
      const moveIndexUpButton = document.getElementById(
        `move-index-up-button-${index.id}`
      );
      const moveIndexDownButton = document.getElementById(
        `move-index-down-button-${index.id}`
      );
      const deleteIndexButton = document.getElementById(
        `delete-index-button-${index.id}`
      );

      bindDebouncedAutosave(
        indexNameInput,
        `index:${index.id}:name`,
        () => {
          commands.renameIndex({
            tensor,
            index,
            proposedName: indexNameInput.value,
            invalidate: propertyInvalidation({ graph: true }),
            statusMessage: `Updated index ${indexNameInput.value.trim()}.`,
          });
        }
      );
      bindDebouncedAutosave(
        indexDimensionInput,
        `index:${index.id}:dimension`,
        () => {
          commands.updateIndexDimension({
            indexId: index.id,
            rawValue: indexDimensionInput.value,
            invalidate: propertyInvalidation({
              graph: true,
              analysis: true,
            }),
            statusMessage: `Updated index ${index.name}.`,
          });
        }
      );
      bindImmediateAutosave(
        indexColorInput,
        `index:${index.id}:color`,
        () => {
          commands.updateTargetColor({
            target: index,
            nextColor: indexColorInput.value,
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated index ${index.name}.`,
          });
        },
        "input"
      );
      bindMetadataEditors({
        target: index,
        tagsInput: indexTagsInput,
        tagsFieldKey: `index:${index.id}:tags`,
        customMetadataInput: indexCustomMetadataInput,
        customMetadataFieldKey: `index:${index.id}:custom-metadata`,
        statusMessage: `Updated index ${index.name}.`,
        invalidate: propertyInvalidation(),
        annotationScope: "index",
      });

      if (moveIndexUpButton) {
        moveIndexUpButton.addEventListener("click", () => {
          commands.moveTensorIndex({
            tensorId: tensor.id,
            indexPosition,
            direction: -1,
            invalidate: propertyInvalidation({
              graph: true,
              lookups: true,
              properties: true,
            }),
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Moved index ${index.name}.`,
          });
        });
      }
      if (moveIndexDownButton) {
        moveIndexDownButton.addEventListener("click", () => {
          commands.moveTensorIndex({
            tensorId: tensor.id,
            indexPosition,
            direction: 1,
            invalidate: propertyInvalidation({
              graph: true,
              lookups: true,
              properties: true,
            }),
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Moved index ${index.name}.`,
          });
        });
      }
      if (deleteIndexButton) {
        deleteIndexButton.addEventListener("click", () => {
          commands.deleteTensorIndex({
            tensorId: tensor.id,
            indexId: index.id,
            selectionIds: [tensor.id],
            primaryId: tensor.id,
            statusMessage: `Deleted index ${index.name}.`,
          });
        });
      }
    });
  }

  return {
    renderTensorProperties,
  };
}
