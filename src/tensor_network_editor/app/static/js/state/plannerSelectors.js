function getPlannerStepId(step) {
  if (!step || typeof step !== "object") {
    return null;
  }
  if (typeof step.id === "string" && step.id) {
    return step.id;
  }
  if (typeof step.step_id === "string" && step.step_id) {
    return step.step_id;
  }
  return null;
}

export function buildPlannerSeedOperands({
  tensors,
  specTensors,
  isLinearPeriodicMode,
  isLinearPeriodicBoundaryTensor,
  getLinearPeriodicReservedOperandIdForTensor,
  isGridPeriodicMode,
  isGridPeriodicBoundaryTensor,
  getGridPeriodicReservedOperandIdForTensor,
  isTreePeriodicMode,
  isTreePeriodicBoundaryTensor,
  getTreePeriodicReservedOperandIdForTensor,
  syntheticOperands,
}) {
  const baseOperands = (Array.isArray(tensors) ? tensors : []).map((tensor) => ({
    id: tensor.id,
    sourceTensorIds: [tensor.id],
    selectionIds: [tensor.id],
  }));
  const syntheticSeedOperands = (Array.isArray(syntheticOperands) ? syntheticOperands : [])
    .filter((operand) => operand && typeof operand.operand_id === "string")
    .map((operand) => ({
      id: operand.operand_id,
      sourceTensorIds: Array.isArray(operand.source_tensor_ids)
        ? operand.source_tensor_ids
        : [],
      selectionIds: [operand.operand_id],
      visibleRepresentative: false,
    }));

  const boundaryOperandConfigs = [
    {
      enabled: isLinearPeriodicMode,
      isBoundaryTensor: isLinearPeriodicBoundaryTensor,
      getReservedOperandIdForTensor: getLinearPeriodicReservedOperandIdForTensor,
    },
    {
      enabled: isGridPeriodicMode,
      isBoundaryTensor: isGridPeriodicBoundaryTensor,
      getReservedOperandIdForTensor: getGridPeriodicReservedOperandIdForTensor,
    },
    {
      enabled: isTreePeriodicMode,
      isBoundaryTensor: isTreePeriodicBoundaryTensor,
      getReservedOperandIdForTensor: getTreePeriodicReservedOperandIdForTensor,
    },
  ];
  const activeBoundaryConfig = boundaryOperandConfigs.find(
    (config) =>
      config.enabled &&
      typeof config.isBoundaryTensor === "function" &&
      typeof config.getReservedOperandIdForTensor === "function"
  );
  if (!activeBoundaryConfig) {
    return [...baseOperands, ...syntheticSeedOperands];
  }

  const boundaryOperands = (Array.isArray(specTensors) ? specTensors : [])
    .filter((tensor) => activeBoundaryConfig.isBoundaryTensor(tensor))
    .map((tensor) => {
      const reservedOperandId =
        activeBoundaryConfig.getReservedOperandIdForTensor(tensor);
      return reservedOperandId
        ? {
            id: reservedOperandId,
            sourceTensorIds: [tensor.id],
            selectionIds: [tensor.id, reservedOperandId],
          }
        : null;
    })
    .filter(Boolean);
  return [...baseOperands, ...boundaryOperands, ...syntheticSeedOperands];
}

function buildPlannerResultSelectionIds(stepId, leftOperand, rightOperand, carrySourceOperand) {
  const mergedSelectionIds = carrySourceOperand
    ? carrySourceOperand.selectionIds
    : [...leftOperand.selectionIds, ...rightOperand.selectionIds];
  return [...new Set([stepId, ...(Array.isArray(mergedSelectionIds) ? mergedSelectionIds : [])])];
}

export function buildPlannerOperandState({
  tensors,
  steps,
  seedOperands,
  previousOperandId,
  nextOperandId,
}) {
  const activeOperands = new Map();
  const representativeByTensorId = {};
  const representativeByOperandId = {};
  const sourceTensorIdsByOperandId = {};
  const validSteps = [];
  const reservedOperandIds = new Set();
  const stepOrdersByTensorId = {};

  (Array.isArray(seedOperands) ? seedOperands : []).forEach((operand) => {
    const sourceTensorIds =
      Array.isArray(operand.sourceTensorIds) && operand.sourceTensorIds.length
        ? [...operand.sourceTensorIds]
        : [operand.id];
    const selectionIds = [
      ...new Set(
        Array.isArray(operand.selectionIds) && operand.selectionIds.length
          ? [operand.id, ...operand.selectionIds]
          : [operand.id]
      ),
    ];
    activeOperands.set(operand.id, { sourceTensorIds, selectionIds });
    representativeByOperandId[operand.id] = operand.id;
    selectionIds.forEach((selectionId) => {
      representativeByOperandId[selectionId] = operand.id;
    });
    sourceTensorIdsByOperandId[operand.id] = sourceTensorIds;
    if (operand.visibleRepresentative !== false) {
      sourceTensorIds.forEach((tensorId) => {
        representativeByTensorId[tensorId] = operand.id;
      });
    }
    reservedOperandIds.add(operand.id);
  });

  for (const step of Array.isArray(steps) ? steps : []) {
    const stepId = getPlannerStepId(step);
    const usesPreviousOperand =
      step.left_operand_id === previousOperandId ||
      step.right_operand_id === previousOperandId;
    const usesNextOperand =
      step.left_operand_id === nextOperandId ||
      step.right_operand_id === nextOperandId;
    if (
      !step ||
      !stepId ||
      step.left_operand_id === step.right_operand_id ||
      !activeOperands.has(step.left_operand_id) ||
      !activeOperands.has(step.right_operand_id) ||
      reservedOperandIds.has(stepId) ||
      (usesPreviousOperand && usesNextOperand)
    ) {
      break;
    }
    const leftOperand = activeOperands.get(step.left_operand_id);
    const rightOperand = activeOperands.get(step.right_operand_id);
    if (!leftOperand || !rightOperand) {
      break;
    }
    const carrySourceOperand = usesNextOperand
      ? step.left_operand_id === nextOperandId
        ? rightOperand
        : leftOperand
      : null;
    const selectionIds = buildPlannerResultSelectionIds(
      stepId,
      leftOperand,
      rightOperand,
      carrySourceOperand
    );
    const sourceTensorIds = carrySourceOperand
      ? [...carrySourceOperand.sourceTensorIds]
      : [...new Set([...leftOperand.sourceTensorIds, ...rightOperand.sourceTensorIds])];

    activeOperands.delete(step.left_operand_id);
    activeOperands.delete(step.right_operand_id);
    activeOperands.set(stepId, { sourceTensorIds, selectionIds });
    reservedOperandIds.add(stepId);
    validSteps.push(step);
    sourceTensorIdsByOperandId[stepId] = sourceTensorIds;

    sourceTensorIds.forEach((tensorId) => {
      representativeByTensorId[tensorId] = stepId;
      representativeByOperandId[tensorId] = stepId;
      if (!Array.isArray(stepOrdersByTensorId[tensorId])) {
        stepOrdersByTensorId[tensorId] = [];
      }
      stepOrdersByTensorId[tensorId].push(validSteps.length);
    });
    selectionIds.forEach((selectionId) => {
      representativeByOperandId[selectionId] = stepId;
    });
    if (usesNextOperand) {
      break;
    }
  }

  return {
    activeOperandIds: [...activeOperands.keys()],
    representativeByTensorId,
    representativeByOperandId,
    sourceTensorIdsByOperandId,
    validSteps,
    stepOrdersByTensorId,
    tensors,
  };
}

export function buildPreviewOrderByVisibleTensorId(visibleTensors, steps) {
  const previewOrderByTensorId = Object.fromEntries(
    visibleTensors.map((tensor) => [tensor.id, []])
  );
  const sourceTensorIdsByOperandId = {};
  const visibleTensorIdsBySourceTensorId = {};

  visibleTensors.forEach((tensor) => {
    const sourceTensorIds =
      Array.isArray(tensor.sourceTensorIds) && tensor.sourceTensorIds.length
        ? [...tensor.sourceTensorIds]
        : [tensor.id];
    sourceTensorIdsByOperandId[tensor.id] = sourceTensorIds;
    sourceTensorIds.forEach((sourceTensorId) => {
      if (!Array.isArray(visibleTensorIdsBySourceTensorId[sourceTensorId])) {
        visibleTensorIdsBySourceTensorId[sourceTensorId] = [];
      }
      visibleTensorIdsBySourceTensorId[sourceTensorId].push(tensor.id);
    });
  });

  (Array.isArray(steps) ? steps : []).forEach((step, index) => {
    const leftSourceTensorIds = sourceTensorIdsByOperandId[step.left_operand_id] || [
      step.left_operand_id,
    ];
    const rightSourceTensorIds = sourceTensorIdsByOperandId[step.right_operand_id] || [
      step.right_operand_id,
    ];
    const resultSourceTensorIds = [...new Set([...leftSourceTensorIds, ...rightSourceTensorIds])];
    sourceTensorIdsByOperandId[step.result_operand_id] = resultSourceTensorIds;

    const affectedVisibleTensorIds = new Set();
    resultSourceTensorIds.forEach((sourceTensorId) => {
      (visibleTensorIdsBySourceTensorId[sourceTensorId] || []).forEach((visibleTensorId) => {
        affectedVisibleTensorIds.add(visibleTensorId);
      });
    });
    affectedVisibleTensorIds.forEach((visibleTensorId) => {
      previewOrderByTensorId[visibleTensorId].push(index + 1);
    });
  });

  return previewOrderByTensorId;
}

export function getAutomaticAnalysisByMode(payload, mode) {
  if (!payload) {
    return null;
  }
  if (mode === "automaticFuture") {
    return payload.automatic_future || null;
  }
  if (mode === "automaticPast") {
    return payload.automatic_past || null;
  }
  return null;
}
