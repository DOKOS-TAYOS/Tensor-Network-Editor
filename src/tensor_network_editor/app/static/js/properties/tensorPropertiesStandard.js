import { GRAPH_THEME } from "../core/theme.js";

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

  function getTensorShape(tensor) {
    return Array.isArray(tensor?.indices)
      ? tensor.indices.map((index) => index.dimension)
      : [];
  }

  function formatTensorShape(shape) {
    return Array.isArray(shape) && shape.length ? `[${shape.join(", ")}]` : "scalar";
  }

  function tensorShapesMatch(leftShape, rightShape) {
    if (!Array.isArray(leftShape) || !Array.isArray(rightShape)) {
      return false;
    }
    if (leftShape.length !== rightShape.length) {
      return false;
    }
    return leftShape.every((dimension, position) => dimension === rightShape[position]);
  }

  function normalizeTensorLiteralNode(value) {
    if (typeof value === "boolean") {
      return null;
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        return null;
      }
      return {
        normalized: value,
        shape: [],
      };
    }
    if (!Array.isArray(value) || !value.length) {
      return null;
    }

    const normalizedChildren = [];
    let childShape = null;
    for (const item of value) {
      const normalizedItem = normalizeTensorLiteralNode(item);
      if (!normalizedItem) {
        return null;
      }
      if (!childShape) {
        childShape = normalizedItem.shape;
      } else if (!tensorShapesMatch(childShape, normalizedItem.shape)) {
        return null;
      }
      normalizedChildren.push(normalizedItem.normalized);
    }

    return {
      normalized: normalizedChildren,
      shape: [normalizedChildren.length, ...(childShape || [])],
    };
  }

  function getTensorDataMode(tensor) {
    const mode = String(tensor?.tensor_data?.mode || "").trim();
    if (mode === "ones" || mode === "fill" || mode === "literal") {
      return mode;
    }
    return "zeros";
  }

  function cloneTensorLiteral(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function buildFilledTensorLiteral(shape, fillValue) {
    if (!Array.isArray(shape) || !shape.length) {
      return fillValue;
    }
    return Array.from({ length: shape[0] }, () =>
      buildFilledTensorLiteral(shape.slice(1), fillValue)
    );
  }

  function buildDefaultTensorLiteralValues(tensor, tensorShape) {
    if (tensor?.tensor_data?.mode === "literal" && tensor.tensor_data.values !== undefined) {
      return cloneTensorLiteral(tensor.tensor_data.values);
    }
    if (tensor?.tensor_data?.mode === "ones") {
      return buildFilledTensorLiteral(tensorShape, 1);
    }
    if (
      tensor?.tensor_data?.mode === "fill" &&
      typeof tensor.tensor_data.fill_value === "number" &&
      Number.isFinite(tensor.tensor_data.fill_value)
    ) {
      return buildFilledTensorLiteral(tensorShape, tensor.tensor_data.fill_value);
    }
    return buildFilledTensorLiteral(tensorShape, 0);
  }

  function analyzeTensorDataFillInput(rawValue) {
    const trimmedValue = String(rawValue || "").trim();
    if (!trimmedValue) {
      return {
        ok: false,
        message: "Fill value must be a finite number.",
      };
    }
    const parsedValue = Number(trimmedValue);
    if (!Number.isFinite(parsedValue)) {
      return {
        ok: false,
        message: "Fill value must be a finite number.",
      };
    }
    return {
      ok: true,
      tensorData: {
        mode: "fill",
        fill_value: parsedValue,
      },
    };
  }

  function analyzeTensorLiteralInput(rawValue, expectedShape) {
    let parsedValue = null;
    try {
      parsedValue = JSON.parse(String(rawValue || ""));
    } catch (error) {
      return {
        ok: false,
        message: `Explicit values must be valid JSON: ${error.message}`,
      };
    }

    const normalizedLiteral = normalizeTensorLiteralNode(parsedValue);
    if (!normalizedLiteral) {
      return {
        ok: false,
        message: "Explicit values must be finite numbers arranged as a non-ragged tensor.",
      };
    }
    if (!tensorShapesMatch(normalizedLiteral.shape, expectedShape)) {
      return {
        ok: false,
        message: `Explicit values must match the tensor shape ${formatTensorShape(
          expectedShape
        )}.`,
      };
    }
    return {
      ok: true,
      tensorData: {
        mode: "literal",
        values: normalizedLiteral.normalized,
      },
    };
  }

  function describeTensorData(tensor, tensorShape) {
    const tensorDataMode = getTensorDataMode(tensor);
    if (tensorDataMode === "ones") {
      return `Current initializer: ones for ${formatTensorShape(tensorShape)}.`;
    }
    if (tensorDataMode === "fill") {
      return `Current initializer: fill with ${tensor.tensor_data.fill_value}.`;
    }
    if (tensorDataMode === "literal") {
      return `Current initializer: explicit JSON values for ${formatTensorShape(
        tensorShape
      )}.`;
    }
    return `Current initializer: generated zeros for ${formatTensorShape(tensorShape)}.`;
  }

  function renderTensorProperties(tensor, options = {}) {
    const focusedIndexId = options.focusedIndexId || null;
    const tensorIndexCount = Array.isArray(tensor.indices) ? tensor.indices.length : 0;
    const totalElementCount = getTensorTotalElementCount(tensor);
    const tensorShape = getTensorShape(tensor);
    const tensorDataMode = getTensorDataMode(tensor);
    const tensorFillValue =
      tensorDataMode === "fill" &&
      typeof tensor.tensor_data?.fill_value === "number" &&
      Number.isFinite(tensor.tensor_data.fill_value)
        ? tensor.tensor_data.fill_value
        : 0;
    const tensorLiteralText =
      tensorDataMode === "literal"
        ? JSON.stringify(buildDefaultTensorLiteralValues(tensor, tensorShape), null, 2)
        : "";
    const indexEditors = tensor.indices
      .map((index, indexPosition) => {
        const isOpen = isTensorIndexDisclosureOpen(tensor.id, index.id);
        const isConnected = Boolean(
          typeof ctx.findConnectionByIndexId === "function"
            ? ctx.findConnectionByIndexId(index.id)
            : ctx.findEdgeByIndexId(index.id)
        );
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
      <section class="planner-section">
        <h3>Tensor values</h3>
        <div class="field-group">
          <label for="tensor-data-mode-select">Initialization</label>
          <select
            id="tensor-data-mode-select"
            data-focus-key="tensor:${tensor.id}:tensor-data-mode"
          >
            <option value="zeros"${tensorDataMode === "zeros" ? " selected" : ""}>
              Generated zeros
            </option>
            <option value="ones"${tensorDataMode === "ones" ? " selected" : ""}>
              Ones
            </option>
            <option value="fill"${tensorDataMode === "fill" ? " selected" : ""}>
              Fill value
            </option>
            <option value="literal"${tensorDataMode === "literal" ? " selected" : ""}>
              Explicit values
            </option>
          </select>
        </div>
        <p class="property-meta">Expected shape: ${ctx.escapeHtml(formatTensorShape(tensorShape))}</p>
        ${
          tensorDataMode === "fill"
            ? `
              <div class="field-group compact-number-field">
                <label for="tensor-data-fill-input">Fill value</label>
                <input
                  id="tensor-data-fill-input"
                  data-focus-key="tensor:${tensor.id}:tensor-data-fill"
                  type="number"
                  step="any"
                  value="${ctx.escapeHtml(String(tensorFillValue))}"
                />
              </div>
            `
            : ""
        }
        ${
          tensorDataMode === "literal"
            ? `
              <div class="field-group">
                <label for="tensor-data-values-input">Explicit values (JSON)</label>
                <textarea
                  id="tensor-data-values-input"
                  data-focus-key="tensor:${tensor.id}:tensor-data-values"
                  rows="6"
                  spellcheck="false"
                >${ctx.escapeHtml(tensorLiteralText)}</textarea>
              </div>
              <p class="property-meta">Use JSON numbers that match the tensor shape exactly.</p>
            `
            : ""
        }
        <p id="tensor-data-validation-message" class="property-meta" hidden></p>
        <p class="property-meta">${ctx.escapeHtml(describeTensorData(tensor, tensorShape))}</p>
      </section>
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
    const tensorDataModeSelect = document.getElementById("tensor-data-mode-select");
    const tensorDataFillInput = document.getElementById("tensor-data-fill-input");
    const tensorDataValuesInput = document.getElementById("tensor-data-values-input");
    const tensorDataValidationMessage = document.getElementById(
      "tensor-data-validation-message"
    );

    function setTensorDataValidationMessage(message = "") {
      if (!tensorDataValidationMessage) {
        return;
      }
      tensorDataValidationMessage.textContent = message;
      tensorDataValidationMessage.hidden = !message;
    }

    function reportTensorDataValidationError(message) {
      setTensorDataValidationMessage(message);
      if (typeof ctx.setStatus === "function") {
        ctx.setStatus(message, "error");
      }
    }

    function buildTensorDataForMode(nextMode) {
      if (nextMode === "ones") {
        return { mode: "ones" };
      }
      if (nextMode === "fill") {
        return {
          mode: "fill",
          fill_value: tensorFillValue,
        };
      }
      if (nextMode === "literal") {
        return {
          mode: "literal",
          values: buildDefaultTensorLiteralValues(tensor, tensorShape),
        };
      }
      return null;
    }

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
    bindImmediateAutosave(
      tensorDataModeSelect,
      `tensor:${tensor.id}:tensor-data-mode`,
      () => {
        const nextMode = String(tensorDataModeSelect.value || "zeros");
        setTensorDataValidationMessage("");
        commands.updateTensorData({
          tensorId: tensor.id,
          nextTensorData: buildTensorDataForMode(nextMode),
          invalidate: propertyInvalidation({ properties: true }),
          statusMessage: `Updated tensor ${tensor.name}.`,
        });
      }
    );

    if (tensorDataFillInput) {
      tensorDataFillInput.addEventListener("input", () => {
        const validation = analyzeTensorDataFillInput(tensorDataFillInput.value);
        setTensorDataValidationMessage(validation.ok ? "" : validation.message);
      });
      bindDebouncedAutosave(
        tensorDataFillInput,
        `tensor:${tensor.id}:tensor-data-fill`,
        () => {
          const validation = analyzeTensorDataFillInput(tensorDataFillInput.value);
          if (!validation.ok) {
            reportTensorDataValidationError(validation.message);
            return;
          }
          setTensorDataValidationMessage("");
          commands.updateTensorData({
            tensorId: tensor.id,
            nextTensorData: validation.tensorData,
            invalidate: propertyInvalidation({ properties: true }),
            statusMessage: `Updated tensor ${tensor.name}.`,
          });
        },
        {
          scheduleOnInput: false,
        }
      );
    }

    if (tensorDataValuesInput) {
      tensorDataValuesInput.addEventListener("input", () => {
        const validation = analyzeTensorLiteralInput(
          tensorDataValuesInput.value,
          tensorShape
        );
        setTensorDataValidationMessage(validation.ok ? "" : validation.message);
      });
      bindDebouncedAutosave(
        tensorDataValuesInput,
        `tensor:${tensor.id}:tensor-data-values`,
        () => {
          const validation = analyzeTensorLiteralInput(
            tensorDataValuesInput.value,
            tensorShape
          );
          if (!validation.ok) {
            reportTensorDataValidationError(validation.message);
            return;
          }
          setTensorDataValidationMessage("");
          commands.updateTensorData({
            tensorId: tensor.id,
            nextTensorData: validation.tensorData,
            invalidate: propertyInvalidation({ properties: true }),
            statusMessage: `Updated tensor ${tensor.name}.`,
          });
        },
        {
          scheduleOnInput: false,
          commitOnEnter: false,
        }
      );
    }

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
