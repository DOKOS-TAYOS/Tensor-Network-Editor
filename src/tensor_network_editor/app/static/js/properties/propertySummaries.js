export function normalizeElementDimension(value, asFiniteNumber) {
  return BigInt(Math.max(1, Math.round(asFiniteNumber(value, 1))));
}

export function getTensorTotalElementCount(tensor, asFiniteNumber) {
  const indices = Array.isArray(tensor && tensor.indices) ? tensor.indices : [];
  return indices.reduce(
    (product, index) =>
      product * normalizeElementDimension(index.dimension, asFiniteNumber),
    1n
  );
}

export function getTotalElementCountForTensorIds(
  tensorIds,
  findTensorById,
  asFiniteNumber
) {
  const uniqueTensorIds = [...new Set(Array.isArray(tensorIds) ? tensorIds : [])];
  let resolvedTensorCount = 0;
  const totalElementCount = uniqueTensorIds.reduce((sum, tensorId) => {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return sum;
    }
    resolvedTensorCount += 1;
    return sum + getTensorTotalElementCount(tensor, asFiniteNumber);
  }, 0n);
  return resolvedTensorCount ? totalElementCount : null;
}

export function getSelectionEntryTensorIds(entry) {
  if (!entry) {
    return [];
  }
  if (entry.kind === "group") {
    return Array.isArray(entry.group && entry.group.tensor_ids)
      ? [...entry.group.tensor_ids]
      : [];
  }
  if (entry.kind === "contraction-tensor") {
    return Array.isArray(entry.tensor && entry.tensor.sourceTensorIds)
      ? [...entry.tensor.sourceTensorIds]
      : [];
  }
  if (entry.kind === "tensor" && entry.id) {
    return [entry.id];
  }
  return [];
}

export function getSelectionTotalElementCount(
  selectedEntries,
  findTensorById,
  asFiniteNumber
) {
  const tensorIds = selectedEntries.flatMap((entry) =>
    getSelectionEntryTensorIds(entry)
  );
  return getTotalElementCountForTensorIds(
    tensorIds,
    findTensorById,
    asFiniteNumber
  );
}

export function getContractionTensorTotalElementCount(
  tensor,
  findTensorById,
  asFiniteNumber
) {
  const sourceTensorIds = Array.isArray(tensor && tensor.sourceTensorIds)
    ? tensor.sourceTensorIds
    : [];
  const totalElementCount = getTotalElementCountForTensorIds(
    sourceTensorIds,
    findTensorById,
    asFiniteNumber
  );
  return totalElementCount === null
    ? getTensorTotalElementCount(tensor, asFiniteNumber)
    : totalElementCount;
}

export function formatTotalElementCount(totalElementCount) {
  return totalElementCount === null ? "" : totalElementCount.toString();
}
