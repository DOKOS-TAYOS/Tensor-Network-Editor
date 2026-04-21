export function createUtilityLayoutIndexSupport({ ctx, constants }) {
  const INDEX_RADIUS =
    Number.isFinite(constants && constants.INDEX_RADIUS) &&
    constants.INDEX_RADIUS >= 0
      ? constants.INDEX_RADIUS
      : 15;
  const INDEX_PADDING =
    Number.isFinite(constants && constants.INDEX_PADDING) &&
    constants.INDEX_PADDING >= 0
      ? constants.INDEX_PADDING
      : 8;
  const INDEX_REFLOW_GAP = INDEX_RADIUS * 2 + Math.max(INDEX_PADDING, 8);

  function buildReflowIndexOffsets(tensors, mode) {
    if (
      mode !== "left" &&
      mode !== "right" &&
      mode !== "top" &&
      mode !== "bottom" &&
      mode !== "reset"
    ) {
      return null;
    }
    return Object.fromEntries(
      tensors.map((tensor) => [tensor.id, buildTensorIndexOffsets(tensor, mode)])
    );
  }

  function buildTensorIndexOffsets(tensor, mode) {
    if (mode === "reset") {
      return tensor.indices.map((index, indexPosition) =>
        ctx.defaultIndexOffsetForOrder(indexPosition, tensor)
      );
    }

    const leftOffset = -ctx.tensorWidth(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING;
    const rightOffset = ctx.tensorWidth(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;
    const topOffset = -ctx.tensorHeight(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING;
    const bottomOffset = ctx.tensorHeight(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;

    if (mode === "left" || mode === "right") {
      return buildPackedBoundaryIndexOffsets(
        tensor,
        tensor.indices.length,
        topOffset,
        bottomOffset,
        (offsetAlongEdge, bandIndex) => ({
          x:
            (mode === "left" ? leftOffset : rightOffset) +
            (mode === "left" ? 1 : -1) * bandIndex * INDEX_REFLOW_GAP,
          y: offsetAlongEdge,
        })
      );
    }

    return buildPackedBoundaryIndexOffsets(
      tensor,
      tensor.indices.length,
      leftOffset,
      rightOffset,
      (offsetAlongEdge, bandIndex) => ({
        x: offsetAlongEdge,
        y:
          (mode === "top" ? topOffset : bottomOffset) +
          (mode === "top" ? 1 : -1) * bandIndex * INDEX_REFLOW_GAP,
      })
    );
  }

  function buildPackedBoundaryIndexOffsets(
    tensor,
    count,
    start,
    end,
    offsetBuilder
  ) {
    return buildIndexBands(count, start, end).flatMap((bandOffsets, bandIndex) =>
      bandOffsets.map((offsetAlongEdge) =>
        ctx.clampIndexOffset(offsetBuilder(offsetAlongEdge, bandIndex), tensor)
      )
    );
  }

  function buildIndexBands(count, start, end) {
    if (count <= 0) {
      return [];
    }
    const span = Math.abs(end - start);
    const maxPerBand = Math.max(1, Math.floor(span / INDEX_REFLOW_GAP) + 1);
    const bands = [];
    for (let index = 0; index < count; index += maxPerBand) {
      const bandCount = Math.min(maxPerBand, count - index);
      bands.push(buildDistributedIndexAxisOffsets(bandCount, start, end));
    }
    return bands;
  }

  function buildDistributedIndexAxisOffsets(count, start, end) {
    if (count <= 1) {
      return [(start + end) / 2];
    }
    const step = (end - start) / (count - 1);
    return Array.from({ length: count }, (_, index) => start + step * index);
  }

  return {
    buildReflowIndexOffsets,
  };
}
