export function createContractionTensorPropertiesRenderer({
  commands,
  ctx,
  document,
  formatTotalElementCount,
  getContractionTensorTotalElementCount,
  propertiesPanel,
  renderTrashIcon,
}) {
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
        commands.deleteCurrentSelection();
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

  return {
    renderContractionIndexProperties,
    renderContractionTensorProperties,
  };
}
