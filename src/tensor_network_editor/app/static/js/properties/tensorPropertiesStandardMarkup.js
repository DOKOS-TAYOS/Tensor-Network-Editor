import { GRAPH_THEME } from "../core/theme.js";

export function createStandardTensorPropertiesMarkupSupport({
  ctx,
  buildMetadataEditorMarkup,
  formatTotalElementCount,
  getTensorTotalElementCount,
  isTensorIndexDisclosureOpen,
  renderTrashIcon,
  dataSupport,
}) {
  const {
    getTensorShape,
    formatTensorShape,
    getTensorDataMode,
    buildDefaultTensorLiteralValues,
  } = dataSupport;

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

  function buildIndexEditorsMarkup(tensor, focusedIndexId = null) {
    return tensor.indices
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
  }

  function renderTensorPropertiesMarkup(tensor, options = {}) {
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
    const indexEditors = buildIndexEditorsMarkup(tensor, focusedIndexId);

    return `
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
      <div class="field-group">
        <label for="tensor-data-mode-select">Initialization</label>
        <div
          id="tensor-data-mode-field"
          class="select-chevron-field tensor-data-mode-field"
          data-expanded="false"
        >
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
      </div>
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
            <p class="property-meta">Expected shape: ${ctx.escapeHtml(
              formatTensorShape(tensorShape)
            )}</p>
          `
          : ""
      }
      <p id="tensor-data-validation-message" class="property-meta" hidden></p>
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
  }

  return {
    buildTooltipAttributes,
    createPaleIndexColor,
    renderTensorPropertiesMarkup,
  };
}
