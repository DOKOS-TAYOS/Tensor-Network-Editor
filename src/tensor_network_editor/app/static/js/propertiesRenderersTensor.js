export function createTensorPropertiesRenderers({
  ctx,
  document,
  propertiesPanel,
  support,
}) {
  const {
    bindDebouncedAutosave,
    bindImmediateAutosave,
    bindMetadataEditors,
    bindSuggestedAnnotationEditors,
    buildMetadataEditorMarkup,
    buildSuggestedAnnotationsMarkup,
    propertyInvalidation,
    renderTrashIcon,
    getTensorTotalElementCount,
    getContractionTensorTotalElementCount,
    formatTotalElementCount,
    isTensorIndexDisclosureOpen,
    toggleTensorIndexDisclosure,
  } = support;

  function tensorAnnotationInputId(key) {
    return `tensor-annotation-${key}-input`;
  }

  function tensorAnnotationFocusKey(tensorId, key) {
    return `tensor:${tensorId}:annotation:${key}`;
  }

  function tensorAnnotationSuggestionButtonId(key, suggestion) {
    return `tensor-annotation-${key}-suggestion-${ctx.sanitizeFilename(
      suggestion
    )}`;
  }

  function indexAnnotationInputId(indexId, key) {
    return `index-annotation-${key}-input-${indexId}`;
  }

  function indexAnnotationFocusKey(indexId, key) {
    return `index:${indexId}:annotation:${key}`;
  }

  function indexAnnotationSuggestionButtonId(indexId, key, suggestion) {
    return `index-annotation-${key}-suggestion-${ctx.sanitizeFilename(
      suggestion
    )}-${indexId}`;
  }

  function renderContractionTensorProperties(tensor) {
    const sourceTensorLabels = Array.isArray(tensor.sourceTensorIds)
      ? tensor.sourceTensorIds
          .map((sourceTensorId) => {
            const sourceTensor = ctx.findTensorById(sourceTensorId);
            return sourceTensor ? sourceTensor.name : sourceTensorId;
          })
          .join(", ")
      : "";
    const totalElementCount = getContractionTensorTotalElementCount(tensor);

    propertiesPanel.innerHTML = `
      <div class="properties-summary">
        <div class="properties-chip">
          <span>Result tensor</span>
          <strong>${ctx.escapeHtml(tensor.name)}</strong>
        </div>
        <div class="properties-chip-wrap">
          <div class="properties-chip">
            <span>Contains</span>
            <strong>${Number(tensor.resultCount || 0)}</strong>
          </div>
          <div class="properties-chip">
            <span>Open indices</span>
            <strong>${Array.isArray(tensor.indices) ? tensor.indices.length : 0}</strong>
          </div>
          <div class="properties-chip">
            <span>Total elements</span>
            <strong>${formatTotalElementCount(totalElementCount)}</strong>
          </div>
        </div>
      </div>
      <p class="property-meta">
        This tensor is a contracted result shown only in the planner scene, so its structure is read-only here.
      </p>
      <div class="button-row">
        <button
          id="delete-contraction-tensor-button"
          type="button"
          class="icon-button index-action-button danger"
          aria-label="Delete result"
          title="Delete result"
        >
          ${renderTrashIcon()}
        </button>
      </div>
      ${
        sourceTensorLabels
          ? `
            <div class="field-group">
              <label>Base tensors inside</label>
              <div class="property-readonly">${ctx.escapeHtml(sourceTensorLabels)}</div>
            </div>
          `
          : ""
      }
    `;

    document
      .getElementById("delete-contraction-tensor-button")
      .addEventListener("click", () => {
        ctx.deleteSelection();
      });
  }

  function renderContractionIndexProperties(located) {
    const ownerLabel = located && located.tensor ? located.tensor.name : "Result tensor";
    const isConnected = Boolean(ctx.findEdgeByIndexId(located.index.id));

    propertiesPanel.innerHTML = `
      <div class="properties-summary">
        <div class="properties-chip">
          <span>Port</span>
          <strong>${ctx.escapeHtml(located.index.name)}</strong>
        </div>
        <div class="properties-chip-wrap">
          <div class="properties-chip">
            <span>Owner</span>
            <strong>${ctx.escapeHtml(ownerLabel)}</strong>
          </div>
          <div class="properties-chip">
            <span>Dimension</span>
            <strong>${located.index.dimension}</strong>
          </div>
          <div class="properties-chip">
            <span>Status</span>
            <strong>${isConnected ? "Connected" : "Open"}</strong>
          </div>
        </div>
      </div>
      <p class="property-meta">
        This port is shown in the contraction scene. You can use Connect here, but tensor structure edits still belong to the base graph.
      </p>
    `;
  }

  function renderLinearPeriodicBoundaryTensorProperties(tensor) {
    const roleLabel =
      tensor.linear_periodic_role === "previous"
        ? "Previous cell"
        : "Next cell";
    const indexEditors = tensor.indices
      .map(
        (index) => `
          <section class="planner-section planner-disclosure index-disclosure is-open">
            <div class="planner-disclosure-body index-disclosure-body">
              <div class="properties-chip-wrap">
                <div class="properties-chip">
                  <span>Port</span>
                  <strong>${ctx.escapeHtml(index.name)}</strong>
                </div>
                <div class="properties-chip">
                  <span>Dimension</span>
                  <strong>${index.dimension}</strong>
                </div>
              </div>
              <div class="field-row">
                <label class="control-inline-color" for="index-color-input-${index.id}">
                  <input
                    id="index-color-input-${index.id}"
                    data-focus-key="index:${index.id}:color"
                    type="color"
                    title="Choose tint"
                    aria-label="Choose tint"
                    value="${ctx.escapeHtml(
                      ctx.getMetadataColor(
                        index.metadata,
                        ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(index.id)))
                      )
                    )}"
                  />
                </label>
              </div>
              ${buildMetadataEditorMarkup({
                tagsInputId: `index-tags-input-${index.id}`,
                tagsFocusKey: `index:${index.id}:tags`,
                customMetadataInputId: `index-custom-metadata-input-${index.id}`,
                customMetadataFocusKey: `index:${index.id}:custom-metadata`,
                target: index,
                annotationScope: "index",
                collapsible: true,
                suggestedAnnotationsMarkup: buildSuggestedAnnotationsMarkup({
                  annotationScope: "index",
                  target: index,
                  inputIdForKey: (key) => indexAnnotationInputId(index.id, key),
                  focusKeyForKey: (key) =>
                    indexAnnotationFocusKey(index.id, key),
                  suggestionButtonIdForValue: (key, suggestion) =>
                    indexAnnotationSuggestionButtonId(
                      index.id,
                      key,
                      suggestion
                    ),
                }),
              })}
            </div>
          </section>
        `
      )
      .join("");

    propertiesPanel.innerHTML = `
      <div class="properties-summary">
        <div class="properties-chip">
          <span>Virtual tensor</span>
          <strong>${ctx.escapeHtml(roleLabel)}</strong>
        </div>
        <div class="properties-chip-wrap">
          <div class="properties-chip">
            <span>Ports</span>
            <strong>${tensor.indices.length}</strong>
          </div>
          <div class="properties-chip">
            <span>Role</span>
            <strong>${ctx.escapeHtml(tensor.linear_periodic_role || "")}</strong>
          </div>
        </div>
      </div>
      <div class="button-row">
        <button id="center-tensor-button" type="button">Center</button>
        <label class="control-inline-color" for="tensor-color-input">
          <input
            id="tensor-color-input"
            data-focus-key="tensor:${tensor.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(ctx.getMetadataColor(tensor.metadata, "#456cbf"))}"
          />
        </label>
      </div>
      ${buildMetadataEditorMarkup({
        tagsInputId: "tensor-tags-input",
        tagsFocusKey: `tensor:${tensor.id}:tags`,
        customMetadataInputId: "tensor-custom-metadata-input",
        customMetadataFocusKey: `tensor:${tensor.id}:custom-metadata`,
        target: tensor,
        annotationScope: "tensor",
        collapsible: true,
        suggestedAnnotationsMarkup: buildSuggestedAnnotationsMarkup({
          annotationScope: "tensor",
          target: tensor,
          inputIdForKey: (key) => tensorAnnotationInputId(key),
          focusKeyForKey: (key) => tensorAnnotationFocusKey(tensor.id, key),
          suggestionButtonIdForValue: (key, suggestion) =>
            tensorAnnotationSuggestionButtonId(key, suggestion),
        }),
      })}
      <div class="properties-list">
        ${indexEditors || "<p class='property-meta'>Ports will appear automatically when this cell exposes free non-virtual indices.</p>"}
      </div>
    `;

    const tensorColorInput = document.getElementById("tensor-color-input");
    const tensorTagsInput = document.getElementById("tensor-tags-input");
    const tensorCustomMetadataInput = document.getElementById(
      "tensor-custom-metadata-input"
    );
    bindImmediateAutosave(
      tensorColorInput,
      `tensor:${tensor.id}:color`,
      () => {
        const currentColor = ctx.getMetadataColor(tensor.metadata, "#456cbf");
        if (tensorColorInput.value === currentColor) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            tensor.metadata.color = tensorColorInput.value;
          },
          {
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
          }
        );
      },
      "input"
    );
    document
      .getElementById("center-tensor-button")
      .addEventListener("click", () => {
        ctx.applyDesignChange(
          () => {
            ctx.centerTensor(tensor.id);
          },
          {
            invalidate: propertyInvalidation({
              graph: true,
              overlays: true,
              minimap: true,
            }),
            statusMessage: `Centered ${roleLabel.toLowerCase()}.`,
          }
        );
      });
    bindMetadataEditors({
      target: tensor,
      tagsInput: tensorTagsInput,
      tagsFieldKey: `tensor:${tensor.id}:tags`,
      customMetadataInput: tensorCustomMetadataInput,
      customMetadataFieldKey: `tensor:${tensor.id}:custom-metadata`,
      statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
      invalidate: propertyInvalidation(),
      annotationScope: "tensor",
    });
    bindSuggestedAnnotationEditors({
      target: tensor,
      annotationScope: "tensor",
      inputForKey: (key) => document.getElementById(tensorAnnotationInputId(key)),
      fieldKeyForKey: (key) => tensorAnnotationFocusKey(tensor.id, key),
      suggestionButtonForValue: (key, suggestion) =>
        document.getElementById(
          tensorAnnotationSuggestionButtonId(key, suggestion)
        ),
      customMetadataInput: tensorCustomMetadataInput,
      statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
      invalidate: propertyInvalidation(),
    });
    tensor.indices.forEach((index) => {
      const indexColorInput = document.getElementById(
        `index-color-input-${index.id}`
      );
      const indexTagsInput = document.getElementById(
        `index-tags-input-${index.id}`
      );
      const indexCustomMetadataInput = document.getElementById(
        `index-custom-metadata-input-${index.id}`
      );
      bindImmediateAutosave(
        indexColorInput,
        `index:${index.id}:color`,
        () => {
          const currentColor = ctx.getMetadataColor(
            index.metadata,
            ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(index.id)))
          );
          if (indexColorInput.value === currentColor) {
            return;
          }
          ctx.applyDesignChange(
            () => {
              index.metadata.color = indexColorInput.value;
            },
            {
              invalidate: propertyInvalidation({ graph: true, minimap: true }),
              statusMessage: `Updated ${index.name}.`,
            }
          );
        },
        "input"
      );
      bindMetadataEditors({
        target: index,
        tagsInput: indexTagsInput,
        tagsFieldKey: `index:${index.id}:tags`,
        customMetadataInput: indexCustomMetadataInput,
        customMetadataFieldKey: `index:${index.id}:custom-metadata`,
        statusMessage: `Updated ${index.name}.`,
        invalidate: propertyInvalidation(),
        annotationScope: "index",
      });
      bindSuggestedAnnotationEditors({
        target: index,
        annotationScope: "index",
        inputForKey: (key) =>
          document.getElementById(indexAnnotationInputId(index.id, key)),
        fieldKeyForKey: (key) => indexAnnotationFocusKey(index.id, key),
        suggestionButtonForValue: (key, suggestion) =>
          document.getElementById(
            indexAnnotationSuggestionButtonId(index.id, key, suggestion)
          ),
        customMetadataInput: indexCustomMetadataInput,
        statusMessage: `Updated ${index.name}.`,
        invalidate: propertyInvalidation(),
      });
    });
  }

  function renderTensorProperties(tensorId, options = {}) {
    const tensor = ctx.findTensorById(tensorId);
    if (!tensor) {
      ctx.clearSelection();
      return;
    }
    if (ctx.isLinearPeriodicBoundaryTensor(tensor)) {
      renderLinearPeriodicBoundaryTensorProperties(tensor);
      return;
    }

    const focusedIndexId = options.focusedIndexId || null;
    const tensorIndexCount = Array.isArray(tensor.indices) ? tensor.indices.length : 0;
    const totalElementCount = getTensorTotalElementCount(tensor);
    const indexEditors = tensor.indices
      .map((index, indexPosition) => {
        const isOpen = isTensorIndexDisclosureOpen(tensor.id, index.id);
        const isConnected = Boolean(ctx.findEdgeByIndexId(index.id));

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
                ctx.getIndexColor(index, isConnected)
              )};"
            >
              <span class="index-disclosure-title">
                <strong>${indexPosition + 1}. ${ctx.escapeHtml(index.name)}</strong>
                <span>${isConnected ? "Connected" : "Open"} &middot; dim ${index.dimension}</span>
              </span>
              <strong>${isOpen ? "Hide" : "Show"}</strong>
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
                          type="number"
                          min="1"
                          step="1"
                          value="${index.dimension}"
                        />
                      </div>
                    </div>
                    <div class="button-row">
                      <label class="control-inline-color" for="index-color-input-${index.id}">
                        <input
                          id="index-color-input-${index.id}"
                          data-focus-key="index:${index.id}:color"
                          type="color"
                          title="Choose tint"
                          aria-label="Choose tint"
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
                        title="Move index up"
                        ${indexPosition === 0 ? "disabled" : ""}
                      >
                        <span aria-hidden="true">&#8593;</span>
                      </button>
                      <button
                        id="move-index-down-button-${index.id}"
                        type="button"
                        class="icon-button index-action-button"
                        aria-label="Move index down"
                        title="Move index down"
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
                        title="Delete index"
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
                      suggestedAnnotationsMarkup: buildSuggestedAnnotationsMarkup({
                        annotationScope: "index",
                        target: index,
                        inputIdForKey: (key) =>
                          indexAnnotationInputId(index.id, key),
                        focusKeyForKey: (key) =>
                          indexAnnotationFocusKey(index.id, key),
                        suggestionButtonIdForValue: (key, suggestion) =>
                          indexAnnotationSuggestionButtonId(
                            index.id,
                            key,
                            suggestion
                          ),
                      }),
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
          title="Add index"
        >
          +
        </button>
        <button id="center-tensor-button" type="button">Center</button>
        <label class="control-inline-color" for="tensor-color-input">
          <input
            id="tensor-color-input"
            data-focus-key="tensor:${tensor.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(ctx.getMetadataColor(tensor.metadata, "#18212c"))}"
          />
        </label>
        <button
          id="delete-tensor-button"
          type="button"
          class="icon-button danger"
          aria-label="Delete tensor"
          title="Delete tensor"
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
        suggestedAnnotationsMarkup: buildSuggestedAnnotationsMarkup({
          annotationScope: "tensor",
          target: tensor,
          inputIdForKey: (key) => tensorAnnotationInputId(key),
          focusKeyForKey: (key) => tensorAnnotationFocusKey(tensor.id, key),
          suggestionButtonIdForValue: (key, suggestion) =>
            tensorAnnotationSuggestionButtonId(key, suggestion),
        }),
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
        const proposedName = tensorNameInput.value.trim();
        if (!proposedName) {
          ctx.setStatus("Tensor name cannot be empty.", "error");
          return;
        }
        if (proposedName === tensor.name) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            tensor.name = proposedName;
          },
          {
            invalidate: propertyInvalidation({ graph: true, planner: true }),
            statusMessage: `Updated tensor ${proposedName}.`,
          }
        );
      }
    );
    bindImmediateAutosave(
      tensorColorInput,
      `tensor:${tensor.id}:color`,
      () => {
        if (
          tensorColorInput.value ===
          ctx.getMetadataColor(tensor.metadata, "#18212c")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            tensor.metadata.color = tensorColorInput.value;
          },
          {
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated tensor ${tensor.name}.`,
          }
        );
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
    bindSuggestedAnnotationEditors({
      target: tensor,
      annotationScope: "tensor",
      inputForKey: (key) => document.getElementById(tensorAnnotationInputId(key)),
      fieldKeyForKey: (key) => tensorAnnotationFocusKey(tensor.id, key),
      suggestionButtonForValue: (key, suggestion) =>
        document.getElementById(
          tensorAnnotationSuggestionButtonId(key, suggestion)
        ),
      customMetadataInput: tensorCustomMetadataInput,
      statusMessage: `Updated tensor ${tensor.name}.`,
      invalidate: propertyInvalidation(),
    });

    document.getElementById("add-index-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          tensor.indices.push(ctx.createIndex(tensor, tensor.indices.length));
        },
        {
          selectionIds: [tensor.id],
          primaryId: tensor.id,
          statusMessage: `Added one index to ${tensor.name}.`,
        }
      );
    });
    document
      .getElementById("center-tensor-button")
      .addEventListener("click", () => {
        ctx.applyDesignChange(
          () => {
            ctx.centerTensor(tensor.id);
          },
          {
            invalidate: propertyInvalidation({
              graph: true,
              overlays: true,
              minimap: true,
            }),
            statusMessage: `Centered tensor ${tensor.name} in the current view.`,
          }
        );
      });
    document
      .getElementById("delete-tensor-button")
      .addEventListener("click", () => {
        ctx.applyDesignChange(
          () => {
            ctx.removeTensor(tensor.id);
          },
          {
            selectionIds: [],
            statusMessage: `Deleted tensor ${tensor.name}.`,
          }
        );
      });

    propertiesPanel.querySelectorAll("[data-index-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        toggleTensorIndexDisclosure(tensor.id, button.dataset.indexToggle);
      });
    });

    tensor.indices.forEach((index, indexPosition) => {
      const isConnected = Boolean(ctx.findEdgeByIndexId(index.id));
      const indexNameInput = document.getElementById(
        `index-name-input-${index.id}`
      );
      const indexDimensionInput = document.getElementById(
        `index-dimension-input-${index.id}`
      );
      const indexColorInput = document.getElementById(
        `index-color-input-${index.id}`
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
          const proposedName = indexNameInput.value.trim();
          if (!proposedName) {
            ctx.setStatus("Index name cannot be empty.", "error");
            return;
          }
          if (ctx.tensorIndexNameExists(tensor, proposedName, index.id)) {
            ctx.setStatus(
              `Tensor ${tensor.name} already has an index named ${proposedName}.`,
              "error"
            );
            return;
          }
          if (proposedName === index.name) {
            return;
          }
          ctx.applyDesignChange(
            () => {
              index.name = proposedName;
            },
            {
              invalidate: propertyInvalidation({ graph: true }),
              statusMessage: `Updated index ${proposedName}.`,
            }
          );
        }
      );
      bindDebouncedAutosave(
        indexDimensionInput,
        `index:${index.id}:dimension`,
        () => {
          const currentOwner = ctx.findIndexOwner(index.id);
          const currentIndex = currentOwner ? currentOwner.index : null;
          if (!currentIndex) {
            return;
          }
          const parsed = Number.parseInt(indexDimensionInput.value, 10);
          if (!Number.isFinite(parsed) || parsed <= 0) {
            ctx.setStatus("Index dimension must be a positive integer.", "error");
            return;
          }
          if (parsed === currentIndex.dimension) {
            return;
          }
          ctx.applyDesignChange(
            () => {
              const nextOwner = ctx.findIndexOwner(index.id);
              if (!nextOwner || !nextOwner.index) {
                return;
              }
              nextOwner.index.dimension = parsed;
              if (typeof ctx.syncConnectedIndexDimension === "function") {
                ctx.syncConnectedIndexDimension(index.id, parsed);
              }
            },
            {
              invalidate: propertyInvalidation({
                graph: true,
                analysis: true,
              }),
              statusMessage: `Updated index ${index.name}.`,
            }
          );
        }
      );
      bindImmediateAutosave(
        indexColorInput,
        `index:${index.id}:color`,
        () => {
          const currentColor = ctx.getMetadataColor(
            index.metadata,
            ctx.getIndexColor(index, isConnected)
          );
          if (indexColorInput.value === currentColor) {
            return;
          }
          ctx.applyDesignChange(
            () => {
              index.metadata.color = indexColorInput.value;
            },
            {
              invalidate: propertyInvalidation({ graph: true, minimap: true }),
              statusMessage: `Updated index ${index.name}.`,
            }
          );
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
      bindSuggestedAnnotationEditors({
        target: index,
        annotationScope: "index",
        inputForKey: (key) =>
          document.getElementById(indexAnnotationInputId(index.id, key)),
        fieldKeyForKey: (key) => indexAnnotationFocusKey(index.id, key),
        suggestionButtonForValue: (key, suggestion) =>
          document.getElementById(
            indexAnnotationSuggestionButtonId(index.id, key, suggestion)
          ),
        customMetadataInput: indexCustomMetadataInput,
        statusMessage: `Updated index ${index.name}.`,
        invalidate: propertyInvalidation(),
      });

      if (moveIndexUpButton) {
        moveIndexUpButton.addEventListener("click", () => {
          ctx.applyDesignChange(
            () => {
              ctx.moveIndex(tensor.id, indexPosition, -1);
            },
            {
              invalidate: propertyInvalidation({
                graph: true,
                lookups: true,
                properties: true,
              }),
              selectionIds: [index.id],
              primaryId: index.id,
              statusMessage: `Moved index ${index.name}.`,
            }
          );
        });
      }
      if (moveIndexDownButton) {
        moveIndexDownButton.addEventListener("click", () => {
          ctx.applyDesignChange(
            () => {
              ctx.moveIndex(tensor.id, indexPosition, 1);
            },
            {
              invalidate: propertyInvalidation({
                graph: true,
                lookups: true,
                properties: true,
              }),
              selectionIds: [index.id],
              primaryId: index.id,
              statusMessage: `Moved index ${index.name}.`,
            }
          );
        });
      }
      if (deleteIndexButton) {
        deleteIndexButton.addEventListener("click", () => {
          ctx.applyDesignChange(
            () => {
              ctx.removeIndex(tensor.id, index.id);
            },
            {
              selectionIds: [tensor.id],
              primaryId: tensor.id,
              statusMessage: `Deleted index ${index.name}.`,
            }
          );
        });
      }
    });
  }

  return {
    renderContractionTensorProperties,
    renderContractionIndexProperties,
    renderLinearPeriodicBoundaryTensorProperties,
    renderTensorProperties,
  };
}
