import {
  buildContractionOperandProgression as buildContractionOperandProgressionFromState,
  buildContractionStateFromProgression,
} from "../state/contractionSceneProgression.js";

export function createContractionSceneOperandsSupport({
  state,
  getLinearPeriodicReservedOperandId,
  isLinearPeriodicMode,
  getLinearPeriodicReservedOperandIdForTensor,
  isLinearPeriodicBoundaryTensor,
  ensureSpecLookups,
  getContractibleEdges,
  getContractibleTensors,
}) {
  const TENSORKROWCH_MANUAL_PLAN_BASE_MESSAGE =
    "TensorKrowch manual plans cannot include outer product steps.";
  const LINEAR_PERIODIC_PREVIOUS_OPERAND_ID =
    typeof getLinearPeriodicReservedOperandId === "function"
      ? getLinearPeriodicReservedOperandId("previous")
      : "__linear_previous__";
  const LINEAR_PERIODIC_NEXT_OPERAND_ID =
    typeof getLinearPeriodicReservedOperandId === "function"
      ? getLinearPeriodicReservedOperandId("next")
      : "__linear_next__";

  function getContractionPlan() {
    return state.spec && state.spec.contraction_plan ? state.spec.contraction_plan : null;
  }

  function getPlanSteps() {
    const plan = getContractionPlan();
    return plan && Array.isArray(plan.steps) ? plan.steps : [];
  }

  function hasFreshContractibleCollections() {
    const tensors = Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [];
    const edges = Array.isArray(state.spec && state.spec.edges) ? state.spec.edges : [];
    return (
      state.contractibleCacheRevision === state.specRevision &&
      state.contractibleCacheTensorRef === tensors &&
      state.contractibleCacheTensorCount === tensors.length &&
      state.contractibleCacheEdgeRef === edges &&
      state.contractibleCacheEdgeCount === edges.length
    );
  }

  function isPreviousOperandId(operandId) {
    return operandId === LINEAR_PERIODIC_PREVIOUS_OPERAND_ID;
  }

  function isNextOperandId(operandId) {
    return operandId === LINEAR_PERIODIC_NEXT_OPERAND_ID;
  }

  function buildBoundaryOperands() {
    if (
      typeof isLinearPeriodicMode !== "function" ||
      typeof getLinearPeriodicReservedOperandIdForTensor !== "function" ||
      typeof isLinearPeriodicBoundaryTensor !== "function" ||
      !isLinearPeriodicMode()
    ) {
      return [];
    }
    if (typeof ensureSpecLookups === "function") {
      ensureSpecLookups();
    }
    return (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [])
      .filter((tensor) => isLinearPeriodicBoundaryTensor(tensor))
      .map((boundaryTensor) => {
        const operandId = getLinearPeriodicReservedOperandIdForTensor(boundaryTensor);
        if (!operandId) {
          return null;
        }
        const tokens = (Array.isArray(boundaryTensor.indices) ? boundaryTensor.indices : []).map(
          (boundaryIndex) => {
            const boundaryEdge = state.edgeByIndexId[boundaryIndex.id] || null;
            const otherIndexId =
              boundaryEdge && boundaryEdge.left && boundaryEdge.left.index_id === boundaryIndex.id
                ? boundaryEdge.right && boundaryEdge.right.index_id
                : boundaryEdge && boundaryEdge.left && boundaryEdge.left.index_id;
            const otherOwner = otherIndexId ? state.indexOwnerById[otherIndexId] || null : null;
            const connectedToRealTensor =
              otherOwner &&
              otherOwner.tensor &&
              !isLinearPeriodicBoundaryTensor(otherOwner.tensor);
            return {
              key: connectedToRealTensor
                ? `open:${otherOwner.index.id}`
                : `boundary:${boundaryTensor.linear_periodic_role}:${boundaryIndex.id}`,
              name: connectedToRealTensor ? otherOwner.index.name : boundaryIndex.name,
              dimension: Number(boundaryIndex.dimension) || 1,
              textColorSeed: boundaryIndex.id,
              sourceEdgeId: null,
              sourceIndexId: boundaryIndex.id,
            };
          }
        );
        return {
          id: operandId,
          isDerived: false,
          linearPeriodicRole: boundaryTensor.linear_periodic_role,
          name: boundaryTensor.name,
          sourceTensorIds: [boundaryTensor.id],
          tokens,
        };
      })
      .filter(Boolean);
  }

  function buildInitialOperands() {
    const edgeByIndexId = {};
    getContractibleEdges().forEach((edge) => {
      edgeByIndexId[edge.left.index_id] = edge;
      edgeByIndexId[edge.right.index_id] = edge;
    });
    const realOperands = getContractibleTensors().map((tensor) => ({
      id: tensor.id,
      isDerived: false,
      name: tensor.name,
      sourceTensorIds: [tensor.id],
      tokens: tensor.indices.map((index) => {
        const edge = edgeByIndexId[index.id];
        return {
          key: edge ? `edge:${edge.id}` : `open:${index.id}`,
          name: edge ? edge.name : index.name,
          dimension: Number(index.dimension) || 1,
          textColorSeed: edge ? edge.id : index.id,
          sourceEdgeId: edge ? edge.id : null,
          sourceIndexId: edge ? null : index.id,
        };
      }),
    }));
    return [...realOperands, ...buildBoundaryOperands()];
  }

  function buildContractionOperandProgressionForSteps(planSteps) {
    return buildContractionOperandProgressionFromState({
      initialOperands: buildInitialOperands(),
      nextOperandId: LINEAR_PERIODIC_NEXT_OPERAND_ID,
      planSteps,
      previousOperandId: LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    });
  }

  function buildContractionOperandProgression(planSteps = getPlanSteps()) {
    const currentPlanSteps = getPlanSteps();
    if (planSteps !== currentPlanSteps) {
      return buildContractionOperandProgressionForSteps(planSteps);
    }

    const cacheIsFresh =
      hasFreshContractibleCollections() &&
      state.contractionProgressionCacheRevision === state.specRevision &&
      state.contractionProgressionCacheStepsRef === currentPlanSteps &&
      state.contractionProgressionCacheStepCount === currentPlanSteps.length &&
      state.contractionProgressionCacheContractibleToken === state.contractibleCacheToken &&
      state.contractionProgressionCache;
    if (cacheIsFresh) {
      return state.contractionProgressionCache;
    }

    getContractibleTensors();
    const contractibleToken = state.contractibleCacheToken;
    const progression = buildContractionOperandProgressionForSteps(currentPlanSteps);
    state.contractionProgressionCacheRevision = state.specRevision;
    state.contractionProgressionCacheStepsRef = currentPlanSteps;
    state.contractionProgressionCacheStepCount = currentPlanSteps.length;
    state.contractionProgressionCacheContractibleToken = contractibleToken;
    state.contractionProgressionCache = progression;
    state.contractionProgressionCacheToken += 1;
    return progression;
  }

  function buildContractionOperandState(stepLimit = null, planSteps = getPlanSteps()) {
    const progression = buildContractionOperandProgression(planSteps);
    return buildContractionStateFromProgression(progression, stepLimit);
  }

  function getTensorKrowchManualPlanIssue() {
    if (state.selectedEngine !== "tensorkrowch") {
      return null;
    }
    const plan = getContractionPlan();
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      return null;
    }
    const progression = buildContractionOperandProgression();
    const incompatibleStep = progression.stepAnalyses.find(
      (stepAnalysis) => stepAnalysis.isOuterProduct
    );
    if (!incompatibleStep) {
      return null;
    }
    return {
      ...incompatibleStep,
      message: `${TENSORKROWCH_MANUAL_PLAN_BASE_MESSAGE} Step ${incompatibleStep.stepNumber} has no shared index.`,
    };
  }

  function getTensorKrowchManualPlanIssueMessage() {
    const issue = getTensorKrowchManualPlanIssue();
    return issue ? issue.message : "";
  }

  return {
    buildContractionOperandProgression,
    buildContractionOperandProgressionForSteps,
    buildContractionOperandState,
    getContractionPlan,
    getPlanSteps,
    getTensorKrowchManualPlanIssue,
    getTensorKrowchManualPlanIssueMessage,
    hasFreshContractibleCollections,
    isNextOperandId,
    isPreviousOperandId,
  };
}
