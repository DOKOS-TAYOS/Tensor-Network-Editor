import { getPreferredStepAnchorOperandId } from "../state/contractionSceneSnapshots.js";

export function createContractionSceneEditingSupport({
  state,
  constants,
  makeId,
  findTensorById,
  ensureContractionPlan,
  buildContractionScene,
  ensureContractionViewSnapshots,
  getLatestAppliedStepCount,
  getViewedAppliedStepCount,
  getSnapshotForStepCount,
  captureVisibleOperandLayoutMap,
  buildLayoutMapFromSnapshot,
  touchContractionViewRevision,
  isNextOperandId,
  isPreviousOperandId,
}) {
  function isInspectingPastStage() {
    return getViewedAppliedStepCount() !== getLatestAppliedStepCount();
  }

  function isContractionSceneVisible() {
    return Boolean(buildContractionScene());
  }

  function canEditCurrentContractionStage() {
    return Boolean(buildContractionScene()) && !isInspectingPastStage();
  }

  function updateCurrentStageOperandLayout(operandId, updates) {
    const plan =
      state.spec && state.spec.contraction_plan ? state.spec.contraction_plan : null;
    if (!plan) {
      return false;
    }
    if (!canEditCurrentContractionStage()) {
      return false;
    }
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const snapshot = getSnapshotForStepCount(latestAppliedStepCount);
    if (!snapshot) {
      return false;
    }
    const operandLayout = snapshot.operand_layouts.find((layout) => layout.operand_id === operandId);
    if (!operandLayout) {
      return false;
    }
    if (updates.position) {
      operandLayout.position = {
        x: Math.round(updates.position.x),
        y: Math.round(updates.position.y),
      };
    }
    if (updates.size) {
      operandLayout.size = {
        height: Math.max(constants.MIN_TENSOR_HEIGHT, Math.round(updates.size.height)),
        width: Math.max(constants.MIN_TENSOR_WIDTH, Math.round(updates.size.width)),
      };
    }
    touchContractionViewRevision();
    return true;
  }

  function applySnapshotLayoutMap(stepCount, layoutMap) {
    const snapshot = getSnapshotForStepCount(stepCount);
    if (!snapshot || !layoutMap || typeof layoutMap !== "object") {
      return false;
    }
    snapshot.operand_layouts.forEach((layout) => {
      const nextLayout = layoutMap[layout.operand_id];
      if (!nextLayout) {
        return;
      }
      layout.position = {
        x: Math.round(nextLayout.position.x),
        y: Math.round(nextLayout.position.y),
      };
      layout.size = {
        height: Math.max(constants.MIN_TENSOR_HEIGHT, Math.round(nextLayout.size.height)),
        width: Math.max(constants.MIN_TENSOR_WIDTH, Math.round(nextLayout.size.width)),
      };
    });
    touchContractionViewRevision();
    return true;
  }

  function clearPastInspection() {
    state.plannerInspectionStepCount = null;
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
  }

  function beginPastInspection(stepIndex) {
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const inspectedStepCount = Math.max(0, Math.min(stepIndex, latestAppliedStepCount));
    state.plannerInspectionStepCount =
      inspectedStepCount === latestAppliedStepCount ? null : inspectedStepCount;
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
  }

  function togglePastInspection(stepIndex) {
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const inspectedStepCount = Math.max(0, Math.min(stepIndex, latestAppliedStepCount));
    if (state.plannerInspectionStepCount === inspectedStepCount) {
      state.plannerInspectionStepCount = null;
    } else if (inspectedStepCount === latestAppliedStepCount) {
      state.plannerInspectionStepCount = null;
    } else {
      state.plannerInspectionStepCount = inspectedStepCount;
    }
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
  }

  function toggleFutureBadgeDisclosure(operandId) {
    state.plannerFutureBadgeDisclosure[operandId] = !Boolean(
      state.plannerFutureBadgeDisclosure[operandId]
    );
  }

  function applyManualContractionStep(leftOperandId, rightOperandId) {
    const plan = ensureContractionPlan();
    ensureContractionViewSnapshots();
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const latestSnapshot = getSnapshotForStepCount(latestAppliedStepCount);
    const latestLayoutMap = latestSnapshot
      ? buildLayoutMapFromSnapshot(latestSnapshot)
      : captureVisibleOperandLayoutMap(latestAppliedStepCount);
    const latestScene = buildContractionScene(latestAppliedStepCount);
    const leftVisibleOperand = latestScene
      ? latestScene.operandMap[leftOperandId]
      : findTensorById(leftOperandId);
    const rightVisibleOperand = latestScene
      ? latestScene.operandMap[rightOperandId]
      : findTensorById(rightOperandId);
    const nextStepId = makeId("step");
    const preferredAnchorOperandId = getPreferredStepAnchorOperandId(
      {
        left_operand_id: leftOperandId,
        right_operand_id: rightOperandId,
      },
      {
        isNextOperandId,
        isPreviousOperandId,
      }
    );
    const clickedAnchorOperandId =
      isPreviousOperandId(rightOperandId) || isNextOperandId(rightOperandId)
        ? leftOperandId
        : rightOperandId;

    plan.steps.push({
      id: nextStepId,
      left_operand_id: leftOperandId,
      metadata: {},
      right_operand_id: rightOperandId,
    });
    ensureContractionViewSnapshots();

    const nextSnapshot = getSnapshotForStepCount(latestAppliedStepCount + 1);
    const nextLayout = nextSnapshot
      ? nextSnapshot.operand_layouts.find((layout) => layout.operand_id === nextStepId)
      : null;
    const preferredLayout =
      latestLayoutMap[clickedAnchorOperandId] ||
      latestLayoutMap[preferredAnchorOperandId] ||
      latestLayoutMap[leftOperandId] ||
      latestLayoutMap[rightOperandId] ||
      (clickedAnchorOperandId === rightOperandId ? rightVisibleOperand : leftVisibleOperand) ||
      (isPreviousOperandId(leftOperandId) || isNextOperandId(leftOperandId)
        ? rightVisibleOperand
        : leftVisibleOperand) ||
      (isPreviousOperandId(rightOperandId) || isNextOperandId(rightOperandId)
        ? leftVisibleOperand
        : rightVisibleOperand) ||
      leftVisibleOperand ||
      rightVisibleOperand ||
      null;
    if (nextLayout && preferredLayout) {
      nextLayout.position = {
        x: Math.round(preferredLayout.position.x),
        y: Math.round(preferredLayout.position.y),
      };
      nextLayout.size = {
        height: Math.round(preferredLayout.size.height),
        width: Math.round(preferredLayout.size.width),
      };
      touchContractionViewRevision();
    } else if (nextLayout && latestSnapshot) {
      const fallbackLayout =
        latestLayoutMap[clickedAnchorOperandId] ||
        latestLayoutMap[preferredAnchorOperandId] ||
        latestSnapshot.operand_layouts.find(
          (layout) => layout.operand_id === clickedAnchorOperandId
        ) ||
        latestSnapshot.operand_layouts.find(
          (layout) => layout.operand_id === preferredAnchorOperandId
        );
      if (fallbackLayout) {
        nextLayout.position = { ...fallbackLayout.position };
        nextLayout.size = { ...fallbackLayout.size };
        touchContractionViewRevision();
      }
    }
  }

  return {
    applyManualContractionStep,
    applySnapshotLayoutMap,
    beginPastInspection,
    canEditCurrentContractionStage,
    clearPastInspection,
    isContractionSceneVisible,
    isInspectingPastStage,
    toggleFutureBadgeDisclosure,
    togglePastInspection,
    updateCurrentStageOperandLayout,
  };
}
