export function cloneOperandLayout(layout) {
  return {
    position: {
      x: layout.position.x,
      y: layout.position.y,
    },
    size: {
      width: layout.size.width,
      height: layout.size.height,
    },
  };
}

export function buildSnapshotLayoutMap(
  snapshot,
  {
    asFiniteNumber = (value, fallbackValue) => {
      const numericValue = Number(value);
      return Number.isFinite(numericValue) ? numericValue : fallbackValue;
    },
    constants = {
      MIN_TENSOR_HEIGHT: 72,
      MIN_TENSOR_WIDTH: 120,
      TENSOR_HEIGHT: 84,
      TENSOR_WIDTH: 140,
    },
  } = {}
) {
  if (!snapshot || !Array.isArray(snapshot.operand_layouts)) {
    return {};
  }
  return Object.fromEntries(
    snapshot.operand_layouts.map((layout) => [
      layout.operand_id,
      {
        position: {
          x: asFiniteNumber(layout.position && layout.position.x, 120),
          y: asFiniteNumber(layout.position && layout.position.y, 120),
        },
        size: {
          width: Math.max(
            constants.MIN_TENSOR_WIDTH,
            asFiniteNumber(layout.size && layout.size.width, constants.TENSOR_WIDTH)
          ),
          height: Math.max(
            constants.MIN_TENSOR_HEIGHT,
            asFiniteNumber(layout.size && layout.size.height, constants.TENSOR_HEIGHT)
          ),
        },
      },
    ])
  );
}

export function buildExistingSnapshotLayoutsByStepCount(currentSnapshots, options = {}) {
  return new Map(
    (Array.isArray(currentSnapshots) ? currentSnapshots : [])
      .filter((snapshot) => snapshot && Number.isInteger(snapshot.applied_step_count))
      .map((snapshot) => [snapshot.applied_step_count, buildSnapshotLayoutMap(snapshot, options)])
  );
}

export function buildExistingSnapshotsByStepCount(currentSnapshots) {
  return new Map(
    (Array.isArray(currentSnapshots) ? currentSnapshots : [])
      .filter((snapshot) => snapshot && Number.isInteger(snapshot.applied_step_count))
      .map((snapshot) => [snapshot.applied_step_count, snapshot])
  );
}

export function buildSnapshotAndLayoutMapFromOperands(
  activeOperands,
  existingSnapshot,
  {
    defaultsByOperandId = {},
    existingLayouts = null,
    fallbackLayoutForOperand,
    snapshotOptions = {},
  } = {}
) {
  const resolvedExistingLayouts =
    existingLayouts || buildSnapshotLayoutMap(existingSnapshot, snapshotOptions);
  const operandLayouts = [];
  const layoutMap = {};

  (Array.isArray(activeOperands) ? activeOperands : []).forEach((operand) => {
    const fallbackLayout =
      defaultsByOperandId[operand.id] || fallbackLayoutForOperand(operand);
    const chosenLayout = cloneOperandLayout(
      resolvedExistingLayouts[operand.id] || fallbackLayout
    );
    layoutMap[operand.id] = chosenLayout;
    operandLayouts.push({
      operand_id: operand.id,
      position: {
        x: chosenLayout.position.x,
        y: chosenLayout.position.y,
      },
      size: {
        width: chosenLayout.size.width,
        height: chosenLayout.size.height,
      },
    });
  });

  return {
    layoutMap,
    snapshot: {
      applied_step_count: Number(existingSnapshot && existingSnapshot.applied_step_count) || 0,
      operand_layouts: operandLayouts,
    },
  };
}

export function getPreferredStepAnchorOperandId(
  step,
  { isNextOperandId = () => false, isPreviousOperandId = () => false } = {}
) {
  if (isPreviousOperandId(step.left_operand_id) || isNextOperandId(step.left_operand_id)) {
    return step.right_operand_id;
  }
  if (isPreviousOperandId(step.right_operand_id) || isNextOperandId(step.right_operand_id)) {
    return step.left_operand_id;
  }
  return step.left_operand_id;
}
