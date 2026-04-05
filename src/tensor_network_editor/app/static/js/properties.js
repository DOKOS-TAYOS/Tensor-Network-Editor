export function registerProperties(ctx) {
  const state = ctx.state;
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  } = ctx.constants;
  const {
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportPyButton,
    exportPngButton,
    exportSvgButton,
    templateSelect,
    insertTemplateButton,
    createGroupButton,
    helpButton,
    helpModal,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    groupLayer,
    resizeLayer,
    selectionBox,
    minimapCanvas,
  } = ctx.dom;
  const { apiGet, apiPost, window, document, cytoscape } = ctx;

  function renderProperties() {
    ctx.pruneSelectionToExisting();
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
    if (singleSelection.kind === "index") {
      renderIndexProperties(singleSelection.id);
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
        <input id="network-name-input" value="${ctx.escapeHtml(state.spec.name)}" />
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
    networkNameInput.addEventListener("change", () => {
      const proposedName = networkNameInput.value.trim() || "Untitled Network";
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
    const tensorCount = selectedEntries.filter((entry) => entry.kind === "tensor").length;
    const indexCount = selectedEntries.filter((entry) => entry.kind === "index").length;
    const edgeCount = selectedEntries.filter((entry) => entry.kind === "edge").length;
    const groupCount = selectedEntries.filter((entry) => entry.kind === "group").length;
    const noteCount = selectedEntries.filter((entry) => entry.kind === "note").length;
    const tensorsOnly = tensorCount > 0 && tensorCount === selectedEntries.length;
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
      <div class="button-row">
        <label class="control-inline-color" for="multi-color-input">
          <input id="multi-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(batchColor)}" />
        </label>
        <button id="apply-multi-color-button" type="button">Apply Color</button>
        ${tensorsOnly ? '<button id="add-index-to-selection-button" type="button">Add Index to Tensors</button>' : ""}
        <button id="delete-selection-button" type="button" class="danger">Delete Selected</button>
      </div>
      <p class="property-meta">
        Drag any selected tensor to move the selected tensor group together.
      </p>
    `;

    document.getElementById("apply-multi-color-button").addEventListener("click", () => {
      const colorValue = document.getElementById("multi-color-input").value;
      ctx.applyDesignChange(
        () => {
          ctx.applyColorToSelection(colorValue);
        },
        {
          preserveSelection: true,
          statusMessage: "Updated the selection color.",
        }
      );
    });

    const addIndexButton = document.getElementById("add-index-to-selection-button");
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
            preserveSelection: true,
            statusMessage: "Added one index to each selected tensor.",
          }
        );
      });
    }

    document.getElementById("delete-selection-button").addEventListener("click", ctx.deleteSelection);
  }

  function renderTensorProperties(tensorId) {
    const tensor = ctx.findTensorById(tensorId);
    if (!tensor) {
      ctx.clearSelection();
      return;
    }

    const indexList = tensor.indices
      .map(
        (index, indexPosition) => `
          <button type="button" class="properties-chip index-select-button" data-index-id="${index.id}">
            <span>${indexPosition + 1}. ${ctx.escapeHtml(index.name)}</span>
            <strong>${index.dimension}</strong>
          </button>
        `
      )
      .join("");

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="tensor-name-input">Tensor name</label>
        <input id="tensor-name-input" value="${ctx.escapeHtml(tensor.name)}" />
      </div>
      <div class="button-row">
        <button id="add-index-button" type="button">Add Index</button>
        <button id="center-tensor-button" type="button">Center in View</button>
        <label class="control-inline-color" for="tensor-color-input">
          <input id="tensor-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(ctx.getMetadataColor(tensor.metadata, "#18212c"))}" />
        </label>
      </div>
      <div class="button-row">
        <button id="delete-tensor-button" type="button" class="danger">Delete Tensor</button>
        <button id="apply-tensor-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
      <div class="properties-list">${indexList || "<p class='property-meta'>This tensor has no indices yet.</p>"}</div>
      <p class="property-meta">Current size: ${Math.round(ctx.tensorWidth(tensor))} × ${Math.round(ctx.tensorHeight(tensor))}. Resize from the corner handles on the canvas.</p>
    `;

    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-tensor-button"),
      inputElements: [tensorNameInput, tensorColorInput],
      isDirty: () =>
        tensorNameInput.value !== tensor.name ||
        tensorColorInput.value !== ctx.getMetadataColor(tensor.metadata, "#18212c"),
      onApply: () => {
        ctx.applyDesignChange(
          () => {
            tensor.name = tensorNameInput.value.trim() || tensor.name;
            tensor.metadata.color = tensorColorInput.value;
          },
          {
            selectionIds: [tensor.id],
            primaryId: tensor.id,
            statusMessage: `Updated tensor ${tensorNameInput.value.trim() || tensor.name}.`,
          }
        );
      },
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
    document.getElementById("center-tensor-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.centerTensor(tensor.id);
        },
        {
          selectionIds: [tensor.id],
          primaryId: tensor.id,
          statusMessage: `Centered tensor ${tensor.name} in the current view.`,
        }
      );
    });
    document.getElementById("delete-tensor-button").addEventListener("click", () => {
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
    document.querySelectorAll(".index-select-button").forEach((button) => {
      button.addEventListener("click", () => {
        ctx.setSelection([button.dataset.indexId], { primaryId: button.dataset.indexId });
      });
    });
  }

  function renderIndexProperties(indexId) {
    const located = ctx.findIndexOwner(indexId);
    if (!located) {
      ctx.clearSelection();
      return;
    }

    const { tensor, index, indexPosition } = located;
    propertiesPanel.innerHTML = `
      <div class="field-row">
        <div class="field-group">
          <label for="index-name-input">Index name</label>
          <input id="index-name-input" value="${ctx.escapeHtml(index.name)}" />
        </div>
        <div class="field-group compact-number-field">
          <label for="index-dimension-input">Dimension</label>
          <input id="index-dimension-input" type="number" min="1" step="1" value="${index.dimension}" />
        </div>
      </div>
      <div class="button-row">
        <button id="move-index-up-button" type="button">Move Earlier</button>
        <button id="move-index-down-button" type="button">Move Later</button>
        <label class="control-inline-color" for="index-color-input">
          <input id="index-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(ctx.getMetadataColor(index.metadata, ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(indexId)))))}" />
        </label>
        <button id="delete-index-button" type="button" class="danger">Delete Index</button>
        <button id="apply-index-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const indexNameInput = document.getElementById("index-name-input");
    const indexDimensionInput = document.getElementById("index-dimension-input");
    const indexColorInput = document.getElementById("index-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-index-button"),
      inputElements: [indexNameInput, indexDimensionInput, indexColorInput],
      isDirty: () =>
        indexNameInput.value !== index.name ||
        indexDimensionInput.value !== String(index.dimension) ||
        indexColorInput.value !== ctx.getMetadataColor(index.metadata, ctx.getIndexColor(index, Boolean(ctx.findEdgeByIndexId(indexId)))),
      onApply: () => {
        const proposedName = indexNameInput.value.trim();
        const parsed = Number.parseInt(indexDimensionInput.value, 10);
        if (!proposedName) {
          ctx.setStatus("Index name cannot be empty.", "error");
          return;
        }
        if (!Number.isFinite(parsed) || parsed <= 0) {
          ctx.setStatus("Index dimension must be a positive integer.", "error");
          return;
        }
        if (ctx.tensorIndexNameExists(tensor, proposedName, index.id)) {
          ctx.setStatus(`Tensor ${tensor.name} already has an index named ${proposedName}.`, "error");
          return;
        }
        ctx.applyDesignChange(
          () => {
            index.name = proposedName;
            index.dimension = parsed;
            index.metadata.color = indexColorInput.value;
          },
          {
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Updated index ${proposedName}.`,
          }
        );
      },
    });

    document.getElementById("move-index-up-button").addEventListener("click", () => {
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
    document.getElementById("move-index-down-button").addEventListener("click", () => {
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
    document.getElementById("delete-index-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          ctx.removeIndex(tensor.id, index.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted index ${index.name}.`,
        }
      );
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
        <input id="group-name-input" value="${ctx.escapeHtml(group.name)}" />
      </div>
      <div class="properties-chip">
        <span>Member tensors</span>
        <strong>${Array.isArray(group.tensor_ids) ? group.tensor_ids.length : 0}</strong>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="group-color-input">
          <input id="group-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(groupColor)}" />
        </label>
        <button id="toggle-group-button" type="button">${isCollapsed ? "Expand Group" : "Collapse Group"}</button>
        <button id="delete-group-button" type="button" class="danger">Delete Group</button>
      </div>
      <p class="property-meta">Drag the group box on the canvas to move all tensors together.</p>
    `;

    document.getElementById("group-name-input").addEventListener("change", (event) => {
      const proposedName = event.target.value.trim() || "Group";
      ctx.applyDesignChange(
        () => {
          group.name = proposedName;
        },
        {
          selectionIds: [group.id],
          primaryId: group.id,
          statusMessage: `Updated group ${proposedName}.`,
        }
      );
    });
    document.getElementById("group-color-input").addEventListener("change", (event) => {
      ctx.applyDesignChange(
        () => {
          group.metadata.color = event.target.value;
        },
        {
          selectionIds: [group.id],
          primaryId: group.id,
          statusMessage: `Updated group ${group.name}.`,
        }
      );
    });
    document.getElementById("toggle-group-button").addEventListener("click", () => {
      ctx.toggleGroupCollapse(group.id);
    });
    document.getElementById("delete-group-button").addEventListener("click", () => {
      ctx.applyDesignChange(
        () => {
          state.spec.groups = state.spec.groups.filter((candidate) => candidate.id !== group.id);
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
        <input id="edge-name-input" value="${ctx.escapeHtml(edge.name)}" />
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="edge-color-input">
          <input id="edge-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(ctx.getMetadataColor(edge.metadata, "#8da1c3"))}" />
        </label>
        <button id="delete-edge-button" type="button" class="danger">Delete Connection</button>
        <button id="apply-edge-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const edgeNameInput = document.getElementById("edge-name-input");
    const edgeColorInput = document.getElementById("edge-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-edge-button"),
      inputElements: [edgeNameInput, edgeColorInput],
      isDirty: () =>
        edgeNameInput.value !== edge.name ||
        edgeColorInput.value !== ctx.getMetadataColor(edge.metadata, "#8da1c3"),
      onApply: () => {
        ctx.applyDesignChange(
          () => {
            edge.name = edgeNameInput.value.trim() || edge.name;
            edge.metadata.color = edgeColorInput.value;
          },
          {
            selectionIds: [edge.id],
            primaryId: edge.id,
            statusMessage: `Updated connection ${edgeNameInput.value.trim() || edge.name}.`,
          }
        );
      },
    });

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
        <textarea id="note-text-input" rows="6">${ctx.escapeHtml(note.text)}</textarea>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="note-color-input">
          <input id="note-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${ctx.escapeHtml(ctx.getMetadataColor(note.metadata, "#5f95ff"))}" />
        </label>
        <button id="delete-note-button" type="button" class="danger">Delete Note</button>
        <button id="apply-note-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
      <p class="property-meta">Move the note from its title bar directly on the canvas.</p>
    `;

    const noteTextInput = document.getElementById("note-text-input");
    const noteColorInput = document.getElementById("note-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-note-button"),
      inputElements: [noteTextInput, noteColorInput],
      isDirty: () =>
        noteTextInput.value !== note.text ||
        noteColorInput.value !== ctx.getMetadataColor(note.metadata, "#5f95ff"),
      onApply: () => {
        const proposedText = noteTextInput.value.trim();
        if (!proposedText) {
          ctx.setStatus("Notes cannot be empty.", "error");
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.text = proposedText;
            note.metadata.color = noteColorInput.value;
          },
          {
            selectionIds: [note.id],
            primaryId: note.id,
            statusMessage: "Updated the note.",
          }
        );
      },
    });

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

  function installDirtyApply({ buttonElement, inputElements, isDirty, onApply }) {
    const refreshVisibility = () => {
      buttonElement.classList.toggle("is-hidden", !isDirty());
    };
    inputElements.forEach((element) => {
      element.addEventListener("input", refreshVisibility);
    });
    buttonElement.addEventListener("click", onApply);
    refreshVisibility();
  }

  Object.assign(ctx, {
    renderProperties,
    renderNetworkProperties,
    renderMultiSelectionProperties,
    renderTensorProperties,
    renderIndexProperties,
    renderGroupProperties,
    renderEdgeProperties,
    renderNoteProperties,
    installDirtyApply
  });
}
