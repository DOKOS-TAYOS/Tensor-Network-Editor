export function createBoundaryTensorPropertiesRenderer({
  bindImmediateAutosave,
  bindMetadataEditors,
  bindSuggestedAnnotationEditors,
  buildMetadataEditorMarkup,
  buildSuggestedAnnotationsMarkup,
  commands,
  ctx,
  document,
  indexAnnotationFocusKey,
  indexAnnotationInputId,
  indexAnnotationSuggestionButtonId,
  propertiesPanel,
  propertyInvalidation,
  tensorAnnotationFocusKey,
  tensorAnnotationInputId,
  tensorAnnotationSuggestionButtonId,
}) {
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
                    indexAnnotationSuggestionButtonId(index.id, key, suggestion),
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
        commands.updateTargetColor({
          target: tensor,
          nextColor: tensorColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated ${roleLabel.toLowerCase()}.`,
        });
      },
      "input"
    );
    document
      .getElementById("center-tensor-button")
      .addEventListener("click", () => {
        commands.centerTensorInView({
          tensorId: tensor.id,
          invalidate: propertyInvalidation({
            graph: true,
            overlays: true,
            minimap: true,
          }),
          statusMessage: `Centered ${roleLabel.toLowerCase()}.`,
        });
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
      inputForKey: (key) =>
        document.getElementById(tensorAnnotationInputId(key)),
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

  return {
    renderLinearPeriodicBoundaryTensorProperties,
  };
}
