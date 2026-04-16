export function cloneOperand(operand) {
  return {
    id: operand.id,
    name: operand.name,
    isDerived: Boolean(operand.isDerived),
    linearPeriodicRole: operand.linearPeriodicRole || null,
    sourceTensorIds: [...operand.sourceTensorIds],
    tokens: operand.tokens.map((token) => ({ ...token })),
  };
}

export function analyzeOperandPair(leftOperand, rightOperand) {
  if (!leftOperand || !rightOperand) {
    return null;
  }
  const rightTokenKeys = new Set(rightOperand.tokens.map((token) => token.key));
  const sharedTokenKeys = [
    ...new Set(
      leftOperand.tokens
        .filter((token) => rightTokenKeys.has(token.key))
        .map((token) => token.key)
    ),
  ];
  return {
    sharedTokenKeys,
    sharedTokenCount: sharedTokenKeys.length,
    isOuterProduct: sharedTokenKeys.length === 0,
  };
}

export function cloneSourceTensorIdsByOperandId(progression, stepCount) {
  const sourceTensorIdsByOperandId = {};
  progression.seedOperandIds.forEach((operandId) => {
    sourceTensorIdsByOperandId[operandId] = [
      ...(progression.sourceTensorIdsByOperandId[operandId] || []),
    ];
  });
  progression.validSteps.slice(0, stepCount).forEach((step) => {
    sourceTensorIdsByOperandId[step.id] = [
      ...(progression.sourceTensorIdsByOperandId[step.id] || []),
    ];
  });
  return sourceTensorIdsByOperandId;
}

export function cloneStepAnalysis(stepAnalysis) {
  return {
    ...stepAnalysis,
    sourceTensorIds: [...stepAnalysis.sourceTensorIds],
    sharedTokenKeys: [...stepAnalysis.sharedTokenKeys],
  };
}

export function getActiveOperandsFromProgression(
  progression,
  stepCount,
  shouldClone = true
) {
  const normalizedStepCount = Math.max(
    0,
    Math.min(progression.validSteps.length, stepCount)
  );
  const stage = progression.stages[normalizedStepCount] || progression.stages[0];
  const activeOperands = Array.isArray(stage.activeOperands)
    ? stage.activeOperands
    : stage.activeOperandIds
        .map((operandId) => progression.operandById.get(operandId))
        .filter(Boolean);
  return shouldClone ? activeOperands.map((operand) => cloneOperand(operand)) : activeOperands;
}

export function buildContractionStateFromProgression(progression, stepCount = null) {
  const normalizedStepCount =
    stepCount === null
      ? progression.validSteps.length
      : Math.max(0, Math.min(progression.validSteps.length, stepCount));
  return {
    activeOperands: getActiveOperandsFromProgression(
      progression,
      normalizedStepCount
    ),
    validSteps: progression.validSteps.slice(0, normalizedStepCount),
    sourceTensorIdsByOperandId: cloneSourceTensorIdsByOperandId(
      progression,
      normalizedStepCount
    ),
    stepAnalyses: progression.stepAnalyses
      .slice(0, normalizedStepCount)
      .map((stepAnalysis) => cloneStepAnalysis(stepAnalysis)),
  };
}

export function buildContractionOperandProgression({
  initialOperands,
  planSteps,
  previousOperandId,
  nextOperandId,
}) {
  const seedOperands = (Array.isArray(initialOperands) ? initialOperands : []).map((operand) =>
    cloneOperand(operand)
  );
  const operandById = new Map(seedOperands.map((operand) => [operand.id, operand]));
  let activeOperandIds = seedOperands.map((operand) => operand.id);
  const activeOperandIdSet = new Set(activeOperandIds);
  const seedOperandIds = [...activeOperandIds];
  const sourceTensorIdsByOperandId = Object.fromEntries(
    seedOperands.map((operand) => [operand.id, [...operand.sourceTensorIds]])
  );
  const validSteps = [];
  const stepAnalyses = [];
  const stepOrdersByTensorId = {};
  const stages = [
    {
      activeOperandIds: [...activeOperandIds],
      activeOperands: [...seedOperands],
    },
  ];

  const isPreviousOperandId = (operandId) => operandId === previousOperandId;
  const isNextOperandId = (operandId) => operandId === nextOperandId;

  for (const step of Array.isArray(planSteps) ? planSteps : []) {
    const usesPreviousOperand =
      isPreviousOperandId(step.left_operand_id) ||
      isPreviousOperandId(step.right_operand_id);
    const usesNextOperand =
      isNextOperandId(step.left_operand_id) || isNextOperandId(step.right_operand_id);
    if (
      !activeOperandIdSet.has(step.left_operand_id) ||
      !activeOperandIdSet.has(step.right_operand_id) ||
      step.left_operand_id === step.right_operand_id ||
      activeOperandIdSet.has(step.id) ||
      (usesPreviousOperand && usesNextOperand)
    ) {
      break;
    }

    const leftOperand = operandById.get(step.left_operand_id);
    const rightOperand = operandById.get(step.right_operand_id);
    if (!leftOperand || !rightOperand) {
      break;
    }
    const pairAnalysis = analyzeOperandPair(leftOperand, rightOperand);
    const contractedTokenKeys = new Set(pairAnalysis ? pairAnalysis.sharedTokenKeys : []);
    const carrySourceOperand = usesNextOperand
      ? isNextOperandId(step.left_operand_id)
        ? rightOperand
        : leftOperand
      : null;
    const resultOperand = {
      id: step.id,
      name: `Result ${validSteps.length + 1}`,
      isDerived: true,
      sourceTensorIds: carrySourceOperand
        ? [...carrySourceOperand.sourceTensorIds]
        : [...new Set([...leftOperand.sourceTensorIds, ...rightOperand.sourceTensorIds])],
      tokens: carrySourceOperand
        ? carrySourceOperand.tokens.map((token) => ({ ...token }))
        : [
            ...leftOperand.tokens.filter((token) => !contractedTokenKeys.has(token.key)),
            ...rightOperand.tokens.filter((token) => !contractedTokenKeys.has(token.key)),
          ].map((token) => ({ ...token })),
    };

    stepAnalyses.push({
      stepId: step.id,
      stepNumber: validSteps.length + 1,
      leftOperandId: step.left_operand_id,
      rightOperandId: step.right_operand_id,
      sourceTensorIds: [...resultOperand.sourceTensorIds],
      sharedTokenKeys: pairAnalysis ? [...pairAnalysis.sharedTokenKeys] : [],
      sharedTokenCount: pairAnalysis ? pairAnalysis.sharedTokenCount : 0,
      isOuterProduct: pairAnalysis ? pairAnalysis.isOuterProduct : false,
    });
    validSteps.push(step);
    resultOperand.sourceTensorIds.forEach((tensorId) => {
      if (!Array.isArray(stepOrdersByTensorId[tensorId])) {
        stepOrdersByTensorId[tensorId] = [];
      }
      stepOrdersByTensorId[tensorId].push(validSteps.length);
    });
    sourceTensorIdsByOperandId[step.id] = [...resultOperand.sourceTensorIds];
    operandById.set(step.id, resultOperand);
    activeOperandIdSet.delete(step.left_operand_id);
    activeOperandIdSet.delete(step.right_operand_id);
    activeOperandIdSet.add(step.id);
    activeOperandIds = activeOperandIds.filter(
      (operandId) =>
        operandId !== step.left_operand_id && operandId !== step.right_operand_id
    );
    activeOperandIds.push(step.id);
    stages.push({
      activeOperandIds: [...activeOperandIds],
      activeOperands: activeOperandIds
        .map((operandId) => operandById.get(operandId))
        .filter(Boolean),
    });
    if (usesNextOperand) {
      break;
    }
  }

  return {
    validSteps,
    stepAnalyses,
    sourceTensorIdsByOperandId,
    seedOperandIds,
    operandById,
    stepOrdersByTensorId,
    stages,
  };
}
