export function formatShape(shape) {
  if (!Array.isArray(shape) || !shape.length) {
    return "scalar";
  }
  return shape.join(" × ");
}

export function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function normalizeShapeDimension(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }
  return BigInt(Math.max(1, Math.round(numericValue)));
}

export function getShapeElementCount(shape) {
  if (!Array.isArray(shape)) {
    return null;
  }
  return shape.reduce((product, dimension) => {
    const normalizedDimension = normalizeShapeDimension(dimension);
    if (normalizedDimension === null) {
      return product;
    }
    return product * normalizedDimension;
  }, 1n);
}

export function formatShapeElementCount(shape) {
  const elementCount = getShapeElementCount(shape);
  return elementCount === null ? "" : elementCount.toString();
}

export function renderShapeElementDetail(shape) {
  const formattedElementCount = formatShapeElementCount(shape);
  return formattedElementCount ? `Total elements ${formattedElementCount}` : "";
}

export function getAnalysisMemoryDtype(payload) {
  if (payload && typeof payload.memory_dtype === "string" && payload.memory_dtype) {
    return payload.memory_dtype;
  }
  return "float64";
}

function getMemoryBytesPerElement(memoryDtype) {
  switch (memoryDtype) {
    case "float16":
      return 2;
    case "float32":
      return 4;
    case "complex64":
      return 8;
    case "complex128":
      return 16;
    case "float64":
    default:
      return 8;
  }
}

export function getPeakMemoryBytes(summary, memoryDtype) {
  if (!summary || typeof summary !== "object") {
    return 0;
  }
  if (Number.isFinite(Number(summary.peak_intermediate_bytes))) {
    return Number(summary.peak_intermediate_bytes);
  }
  return (
    Number(summary.peak_intermediate_size || 0) *
    getMemoryBytesPerElement(memoryDtype)
  );
}

export function formatBytes(value) {
  return `${formatNumber(value)} bytes`;
}

export function formatSignedDelta(value, unit = "") {
  const numericValue = Number(value || 0);
  if (!Number.isFinite(numericValue)) {
    return unit ? `0 ${unit}` : "0";
  }
  const prefix = numericValue > 0 ? "+" : numericValue < 0 ? "-" : "";
  const suffix = unit ? ` ${unit}` : "";
  return `${prefix}${formatNumber(Math.abs(numericValue))}${suffix}`;
}
