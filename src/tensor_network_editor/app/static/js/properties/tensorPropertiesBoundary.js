export function createBoundaryTensorPropertiesRenderer({
  bindImmediateAutosave,
  bindMetadataEditors,
  buildMetadataEditorMarkup,
  commands,
  ctx,
  document,
  propertiesPanel,
  propertyInvalidation,
}) {
  function getBoundaryRoleDetails(tensor) {
    if (tensor.grid_periodic_role === "up") {
      return { roleKey: "up", roleLabel: "Upper cell", fallbackColor: "#456cbf" };
    }
    if (tensor.grid_periodic_role === "right") {
      return { roleKey: "right", roleLabel: "Right cell", fallbackColor: "#2f9b8f" };
    }
    if (tensor.grid_periodic_role === "down") {
      return { roleKey: "down", roleLabel: "Lower cell", fallbackColor: "#d38a37" };
    }
    if (tensor.grid_periodic_role === "left") {
      return { roleKey: "left", roleLabel: "Left cell", fallbackColor: "#8e5bcc" };
    }
    return {
      roleKey: tensor.linear_periodic_role === "previous" ? "previous" : "next",
      roleLabel:
        tensor.linear_periodic_role === "previous" ? "Previous cell" : "Next cell",
      fallbackColor: tensor.linear_periodic_role === "previous" ? "#456cbf" : "#2f9b8f",
    };
  }

  function renderLinearPeriodicBoundaryTensorProperties(tensor) {
    const { roleKey, roleLabel, fallbackColor } = getBoundaryRoleDetails(tensor);
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
                <label
                  class="control-inline-color"
                  for="index-color-input-${index.id}"
                  data-tooltip-enabled="true"
                  data-shortcut-label="Choose color"
                  data-shortcut-description="Set the display color for this item."
                >
                  <input
                    id="index-color-input-${index.id}"
                    data-focus-key="index:${index.id}:color"
                    type="color"
                    aria-label="Choose color"
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
            <strong>${ctx.escapeHtml(roleKey)}</strong>
          </div>
        </div>
      </div>
      <div class="button-row">
        <label
          class="control-inline-color"
          for="tensor-color-input"
          data-tooltip-enabled="true"
          data-shortcut-label="Choose color"
          data-shortcut-description="Set the display color for this item."
        >
          <input
            id="tensor-color-input"
            data-focus-key="tensor:${tensor.id}:color"
            type="color"
            aria-label="Choose color"
            value="${ctx.escapeHtml(ctx.getMetadataColor(tensor.metadata, fallbackColor))}"
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
        commands.updateTargetColor({
          target: tensor,
          nextColor: tensorColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
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
      statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
      invalidate: propertyInvalidation(),
      annotationScope: "tensor",
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
          commands.updateTargetColor({
            target: index,
            nextColor: indexColorInput.value,
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated ${index.name}.`,
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
        statusMessage: `Updated ${index.name}.`,
        invalidate: propertyInvalidation(),
        annotationScope: "index",
      });
    });
  }

  return {
    renderLinearPeriodicBoundaryTensorProperties,
  };
}
