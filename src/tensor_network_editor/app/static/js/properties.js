export function registerProperties(ctx) {
  const state = ctx.state;
  const { propertiesPanel } = ctx.dom;
  const { document, window } = ctx;
  const AUTOSAVE_DELAY_MS = 300;
  const autosaveTimers = new Map();

  function clearAutosaveTimer(fieldKey) {
    const timerId = autosaveTimers.get(fieldKey);
    if (typeof timerId === "number") {
      window.clearTimeout(timerId);
    }
    autosaveTimers.delete(fieldKey);
  }

  function commitAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    commit();
  }

  function scheduleAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    autosaveTimers.set(
      fieldKey,
      window.setTimeout(() => {
        autosaveTimers.delete(fieldKey);
        commit();
      }, AUTOSAVE_DELAY_MS)
    );
  }

  function bindDebouncedAutosave(element, fieldKey, commit, options = {}) {
    if (!element) {
      return;
    }
    element.dataset.focusKey = fieldKey;
    element.addEventListener("input", () => {
      scheduleAutosave(fieldKey, commit);
    });
    element.addEventListener("blur", () => {
      commitAutosave(fieldKey, commit);
    });
    if (options.commitOnEnter !== false) {
      element.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.shiftKey) {
          return;
        }
        event.preventDefault();
        commitAutosave(fieldKey, commit);
      });
    }
  }

  function bindImmediateAutosave(
    element,
    fieldKey,
    commit,
    eventName = "change"
  ) {
    if (!element) {
      return;
    }
    if (fieldKey) {
      element.dataset.focusKey = fieldKey;
    }
    element.addEventListener(eventName, () => {
      commit();
    });
  }

  function renderTrashIcon() {
    return `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
      </svg>
    `;
  }

  function tensorDisclosureState(tensorId) {
    if (!state.tensorIndexDisclosureState[tensorId]) {
      state.tensorIndexDisclosureState[tensorId] = {};
    }
    return state.tensorIndexDisclosureState[tensorId];
  }

  function isTensorIndexDisclosureOpen(tensorId, indexId) {
    return Boolean(tensorDisclosureState(tensorId)[indexId]);
  }

  function setTensorIndexDisclosureOpen(tensorId, indexId, isOpen) {
    const disclosureState = tensorDisclosureState(tensorId);
    if (isOpen) {
      disclosureState[indexId] = true;
      return;
    }
    delete disclosureState[indexId];
  }

  function syncPendingTensorIndexDisclosure() {
    const pendingIndexId = state.pendingPropertiesIndexFocusId;
    if (!pendingIndexId) {
      return;
    }

    const located = ctx.findIndexOwner(pendingIndexId);
    state.pendingPropertiesIndexFocusId = null;
    if (!located) {
      return;
    }

    const wasOpen = isTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId);
    setTensorIndexDisclosureOpen(located.tensor.id, pendingIndexId, true);
    state.autoExpandedTensorIndex = {
      tensorId: located.tensor.id,
      indexId: pendingIndexId,
      wasOpen,
    };
  }

  function toggleTensorIndexDisclosure(tensorId, indexId) {
    const nextOpen = !isTensorIndexDisclosureOpen(tensorId, indexId);
    setTensorIndexDisclosureOpen(tensorId, indexId, nextOpen);
    if (
      state.autoExpandedTensorIndex &&
      state.autoExpandedTensorIndex.tensorId === tensorId &&
      state.autoExpandedTensorIndex.indexId === indexId
    ) {
      state.autoExpandedTensorIndex = null;
    }
    ctx.renderProperties();
  }

  function renderProperties() {
    ctx.pruneSelectionToExisting();
    syncPendingTensorIndexDisclosure();
    if (!state.selectionIds.length) {
      renderNetworkProperties();
      return;
    }
    if (state.selectionIds.length > 1) {
      renderMultiSelectionProperties();
      return;
    }
    const singleSelection = ctx.getSelectionEntry(state.selectionIds[0]);
    if (!singleSelection) {
      renderNetworkProperties();
      return;
    }
    if (singleSelection.kind === "tensor") {
      renderTensorProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "contraction-tensor") {
      renderContractionTensorProperties(singleSelection.tensor);
      return;
    }
    if (singleSelection.kind === "index") {
      renderTensorProperties(singleSelection.located.tensor.id, {
        focusedIndexId: singleSelection.id,
      });
      return;
    }
    if (singleSelection.kind === "edge") {
      renderEdgeProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "group") {
      renderGroupProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "note") {
      renderNoteProperties(singleSelection.id);
      return;
    }
    renderNetworkProperties();
  }

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
    `;

    const networkNameInput = document.getElementById("network-name-input");
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
          statusMessage: "Updated design name.",
        }
      );
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
        </div>
      </div>
      <div class="field-group">
        <label for="multi-color-input">Selection color</label>
        <label class="control-inline-color" for="multi-color-input">
          <input
            id="multi-color-input"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(batchColor)}"
          />
        </label>
      </div>
      <div class="button-row">
        ${
          tensorsOnly
            ? '<button id="add-index-to-selection-button" type="button" class="button-accent-insert">Add Index to Tensors</button>'
            : ""
        }
        <button id="delete-selection-button" type="button" class="danger">Delete Selected</button>
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

  function renderContractionTensorProperties(tensor) {
    const sourceTensorLabels = Array.isArray(tensor.sourceTensorIds)
      ? tensor.sourceTensorIds
          .map((sourceTensorId) => {
            const sourceTensor = ctx.findTensorById(sourceTensorId);
            return sourceTensor ? sourceTensor.name : sourceTensorId;
          })
          .join(", ")
      : "";

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
        </div>
      </div>
      <p class="property-meta">
        This tensor is a contracted result shown only in the planner scene, so its structure is read-only here.
      </p>
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
  }

  function renderTensorProperties(tensorId, options = {}) {
    const tensor = ctx.findTensorById(tensorId);
    if (!tensor) {
      ctx.clearSelection();
      return;
    }

    const focusedIndexId = options.focusedIndexId || null;
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
            >
              <span class="index-disclosure-title">
                <strong>${indexPosition + 1}. ${ctx.escapeHtml(index.name)}</strong>
                <span>${isConnected ? "Connected" : "Open"} · dim ${index.dimension}</span>
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
      <div class="properties-list">
        ${indexEditors || "<p class='property-meta'>This tensor has no indices yet.</p>"}
      </div>
    `;

    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");

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
            statusMessage: `Updated tensor ${tensor.name}.`,
          }
        );
      },
      "input"
    );

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
              statusMessage: `Updated index ${proposedName}.`,
            }
          );
        }
      );
      bindDebouncedAutosave(
        indexDimensionInput,
        `index:${index.id}:dimension`,
        () => {
          const parsed = Number.parseInt(indexDimensionInput.value, 10);
          if (!Number.isFinite(parsed) || parsed <= 0) {
            ctx.setStatus("Index dimension must be a positive integer.", "error");
            return;
          }
          if (parsed === index.dimension) {
            return;
          }
          ctx.applyDesignChange(
            () => {
              index.dimension = parsed;
            },
            {
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
              statusMessage: `Updated index ${index.name}.`,
            }
          );
        },
        "input"
      );

      if (moveIndexUpButton) {
        moveIndexUpButton.addEventListener("click", () => {
          ctx.applyDesignChange(
            () => {
              ctx.moveIndex(tensor.id, indexPosition, -1);
            },
            {
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

  function renderGroupProperties(groupId) {
    const group = ctx.findGroupById(groupId);
    if (!group) {
      ctx.clearSelection();
      return;
    }
    const groupColor = ctx.getMetadataColor(group.metadata, "#61a8ff");
    const isCollapsed = Boolean(group.metadata && group.metadata.collapsed);
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="group-name-input">Group name</label>
        <input
          id="group-name-input"
          data-focus-key="group:${group.id}:name"
          value="${ctx.escapeHtml(group.name)}"
        />
      </div>
      <div class="properties-chip">
        <span>Member tensors</span>
        <strong>${Array.isArray(group.tensor_ids) ? group.tensor_ids.length : 0}</strong>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="group-color-input">
          <input
            id="group-color-input"
            data-focus-key="group:${group.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(groupColor)}"
          />
        </label>
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
      <p class="property-meta">Drag the group box on the canvas to move all tensors together.</p>
    `;

    const groupNameInput = document.getElementById("group-name-input");
    const groupColorInput = document.getElementById("group-color-input");

    bindDebouncedAutosave(groupNameInput, `group:${group.id}:name`, () => {
      const proposedName = groupNameInput.value.trim();
      if (!proposedName) {
        ctx.setStatus("Group name cannot be empty.", "error");
        return;
      }
      if (proposedName === group.name) {
        return;
      }
      ctx.applyDesignChange(
        () => {
          group.name = proposedName;
        },
        {
          statusMessage: `Updated group ${proposedName}.`,
        }
      );
    });
    bindImmediateAutosave(
      groupColorInput,
      `group:${group.id}:color`,
      () => {
        if (groupColorInput.value === groupColor) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            group.metadata.color = groupColorInput.value;
          },
          {
            statusMessage: `Updated group ${group.name}.`,
          }
        );
      },
      "input"
    );
    document.getElementById("toggle-group-button").addEventListener("click", () => {
      ctx.toggleGroupCollapse(group.id);
    });
    document.getElementById("delete-group-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          state.spec.groups = state.spec.groups.filter(
            (candidate) => candidate.id !== group.id
          );
        },
        {
          selectionIds: [],
          statusMessage: `Deleted group ${group.name}.`,
        }
      );
    });
  }

  function renderEdgeProperties(edgeId) {
    const edge = ctx.findEdgeById(edgeId);
    if (!edge) {
      ctx.clearSelection();
      return;
    }
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="edge-name-input">Edge name</label>
        <input
          id="edge-name-input"
          data-focus-key="edge:${edge.id}:name"
          value="${ctx.escapeHtml(edge.name)}"
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
            value="${ctx.escapeHtml(ctx.getMetadataColor(edge.metadata, "#8da1c3"))}"
          />
        </label>
        <button id="delete-edge-button" type="button" class="danger">Delete Connection</button>
      </div>
    `;

    const edgeNameInput = document.getElementById("edge-name-input");
    const edgeColorInput = document.getElementById("edge-color-input");

    bindDebouncedAutosave(edgeNameInput, `edge:${edge.id}:name`, () => {
      const proposedName = edgeNameInput.value.trim();
      if (!proposedName) {
        ctx.setStatus("Connection name cannot be empty.", "error");
        return;
      }
      if (proposedName === edge.name) {
        return;
      }
      ctx.applyDesignChange(
        () => {
          edge.name = proposedName;
        },
        {
          statusMessage: `Updated connection ${proposedName}.`,
        }
      );
    });
    bindImmediateAutosave(
      edgeColorInput,
      `edge:${edge.id}:color`,
      () => {
        if (
          edgeColorInput.value ===
          ctx.getMetadataColor(edge.metadata, "#8da1c3")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            edge.metadata.color = edgeColorInput.value;
          },
          {
            statusMessage: `Updated connection ${edge.name}.`,
          }
        );
      },
      "input"
    );

    document.getElementById("delete-edge-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.removeEdge(edge.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted connection ${edge.name}.`,
        }
      );
    });
  }

  function renderNoteProperties(noteId) {
    const note = ctx.findNoteById(noteId);
    if (!note) {
      ctx.clearSelection();
      return;
    }

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="note-text-input">Note text</label>
        <textarea
          id="note-text-input"
          data-focus-key="note:${note.id}:text"
          rows="6"
        >${ctx.escapeHtml(note.text)}</textarea>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="note-color-input">
          <input
            id="note-color-input"
            data-focus-key="note:${note.id}:color"
            type="color"
            title="Choose tint"
            aria-label="Choose tint"
            value="${ctx.escapeHtml(ctx.getMetadataColor(note.metadata, "#5f95ff"))}"
          />
        </label>
        <button id="delete-note-button" type="button" class="danger">Delete Note</button>
      </div>
      <p class="property-meta">Move the note from its title bar directly on the canvas.</p>
    `;

    const noteTextInput = document.getElementById("note-text-input");
    const noteColorInput = document.getElementById("note-color-input");

    bindDebouncedAutosave(
      noteTextInput,
      `note:${note.id}:text`,
      () => {
        const proposedText = noteTextInput.value.trim();
        if (!proposedText) {
          ctx.setStatus("Notes cannot be empty.", "error");
          return;
        }
        if (proposedText === note.text) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.text = proposedText;
          },
          {
            statusMessage: "Updated the note.",
          }
        );
      },
      { commitOnEnter: false }
    );
    bindImmediateAutosave(
      noteColorInput,
      `note:${note.id}:color`,
      () => {
        if (
          noteColorInput.value ===
          ctx.getMetadataColor(note.metadata, "#5f95ff")
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.metadata.color = noteColorInput.value;
          },
          {
            statusMessage: "Updated the note.",
          }
        );
      },
      "input"
    );

    document.getElementById("delete-note-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.removeNote(note.id);
        },
        {
          selectionIds: [],
          statusMessage: "Deleted the note.",
        }
      );
    });
  }

  Object.assign(ctx, {
    renderProperties,
    renderNetworkProperties,
    renderMultiSelectionProperties,
    renderTensorProperties,
    renderGroupProperties,
    renderEdgeProperties,
    renderNoteProperties,
  });
}
