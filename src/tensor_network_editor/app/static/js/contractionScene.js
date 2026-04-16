import {
  buildContractionOperandProgression as buildContractionOperandProgressionFromState,
  buildContractionStateFromProgression,
  getActiveOperandsFromProgression,
} from "./state/contractionSceneProgression.js";
import {
  buildExistingSnapshotLayoutsByStepCount,
  buildExistingSnapshotsByStepCount,
  buildSnapshotAndLayoutMapFromOperands,
  buildSnapshotLayoutMap,
  cloneOperandLayout,
  getPreferredStepAnchorOperandId,
} from "./state/contractionSceneSnapshots.js";

export function registerContractionScene(ctx) {
  const state = ctx.state;
  const TENSORKROWCH_MANUAL_PLAN_BASE_MESSAGE =
    "TensorKrowch manual plans cannot include outer product steps.";
  const LINEAR_PERIODIC_PREVIOUS_OPERAND_ID =
    typeof ctx.getLinearPeriodicReservedOperandId === "function"
      ? ctx.getLinearPeriodicReservedOperandId("previous")
      : "__linear_previous__";
  const LINEAR_PERIODIC_NEXT_OPERAND_ID =
    typeof ctx.getLinearPeriodicReservedOperandId === "function"
      ? ctx.getLinearPeriodicReservedOperandId("next")
      : "__linear_next__";

  function getContractionPlan() {
    return state.spec && state.spec.contraction_plan ? state.spec.contraction_plan : null;
  }

  function getPlanSteps() {
    const plan = getContractionPlan();
    return plan && Array.isArray(plan.steps) ? plan.steps : [];
  }

  function invalidateContractionSceneCache() {
    state.contractionSceneCacheRevision = -1;
    state.contractionSceneCacheViewRevision = -1;
    state.contractionSceneCacheProgressionToken = -1;
    state.contractionSceneCacheByAppliedStepCount = {};
  }

  function touchContractionViewRevision() {
    state.contractionSceneViewRevision += 1;
    invalidateContractionSceneCache();
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
      typeof ctx.isLinearPeriodicMode !== "function" ||
      typeof ctx.getLinearPeriodicReservedOperandIdForTensor !== "function" ||
      !ctx.isLinearPeriodicMode()
    ) {
      return [];
    }
    if (typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    return (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [])
      .filter((tensor) => ctx.isLinearPeriodicBoundaryTensor(tensor))
      .map((boundaryTensor) => {
        const operandId = ctx.getLinearPeriodicReservedOperandIdForTensor(boundaryTensor);
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
              !ctx.isLinearPeriodicBoundaryTensor(otherOwner.tensor);
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
          name: boundaryTensor.name,
          isDerived: false,
          linearPeriodicRole: boundaryTensor.linear_periodic_role,
          sourceTensorIds: [boundaryTensor.id],
          tokens,
        };
      })
      .filter(Boolean);
  }

  function buildInitialOperands() {
    const edgeByIndexId = {};
    ctx.getContractibleEdges().forEach((edge) => {
      edgeByIndexId[edge.left.index_id] = edge;
      edgeByIndexId[edge.right.index_id] = edge;
    });
    const realOperands = ctx.getContractibleTensors().map((tensor) => ({
      id: tensor.id,
      name: tensor.name,
      isDerived: false,
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
      planSteps,
      previousOperandId: LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
      nextOperandId: LINEAR_PERIODIC_NEXT_OPERAND_ID,
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

    ctx.getContractibleTensors();
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

  function buildFallbackLayoutForOperand(operand) {
    const anchorTensorId = operand.sourceTensorIds[0];
    if (typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    const anchorTensor = state.tensorById[anchorTensorId] || null;
    if (anchorTensor) {
      return {
        position: { x: anchorTensor.position.x, y: anchorTensor.position.y },
        size: {
          width: ctx.tensorWidth(anchorTensor),
          height: ctx.tensorHeight(anchorTensor),
        },
      };
    }
    return {
      position: { x: 120, y: 120 },
      size: { width: ctx.constants.TENSOR_WIDTH, height: ctx.constants.TENSOR_HEIGHT },
    };
  }

  function getSnapshotOptions() {
    return {
      asFiniteNumber: (value, fallbackValue) =>
        ctx.asFiniteNumber(value, fallbackValue),
      constants: ctx.constants,
    };
  }

  function getFallbackLayoutForOperand(operand, fallbackLayoutsByOperandId = null) {
    if (!fallbackLayoutsByOperandId) {
      return buildFallbackLayoutForOperand(operand);
    }
    if (!fallbackLayoutsByOperandId[operand.id]) {
      fallbackLayoutsByOperandId[operand.id] = buildFallbackLayoutForOperand(operand);
    }
    return fallbackLayoutsByOperandId[operand.id];
  }

  function buildSnapshotDefaultsByOperandId(
    activeOperands,
    step,
    previousLayouts,
    fallbackLayoutsByOperandId
  ) {
    const defaultsByOperandId = {};
    activeOperands.forEach((operand) => {
      const fallbackLayout =
        previousLayouts[operand.id] || getFallbackLayoutForOperand(operand, fallbackLayoutsByOperandId);
      if (step && operand.id === step.id) {
        const anchorOperandId = getPreferredStepAnchorOperandId(step, {
          isNextOperandId,
          isPreviousOperandId,
        });
        defaultsByOperandId[operand.id] = cloneOperandLayout(
          previousLayouts[anchorOperandId] || fallbackLayout
        );
        return;
      }
      defaultsByOperandId[operand.id] = cloneOperandLayout(fallbackLayout);
    });
    return defaultsByOperandId;
  }

  function snapshotsMatchProgression(snapshots, progression) {
    if (
      !Array.isArray(snapshots) ||
      snapshots.length !== progression.validSteps.length + 1
    ) {
      return false;
    }

    for (let stepCount = 0; stepCount < snapshots.length; stepCount += 1) {
      const snapshot = snapshots[stepCount];
      if (
        !snapshot ||
        snapshot.applied_step_count !== stepCount ||
        !Array.isArray(snapshot.operand_layouts)
      ) {
        return false;
      }
      const activeOperands = getActiveOperandsFromProgression(
        progression,
        stepCount,
        false
      );
      if (snapshot.operand_layouts.length !== activeOperands.length) {
        return false;
      }
      for (let index = 0; index < activeOperands.length; index += 1) {
        if (snapshot.operand_layouts[index].operand_id !== activeOperands[index].id) {
          return false;
        }
      }
    }

    return true;
  }

  function ensureContractionViewSnapshots(
    progression = buildContractionOperandProgression()
  ) {
    const plan = getContractionPlan();
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      if (plan) {
        const hadSnapshots =
          Array.isArray(plan.view_snapshots) && plan.view_snapshots.length > 0;
        plan.view_snapshots = [];
        if (hadSnapshots) {
          touchContractionViewRevision();
        }
      }
      return [];
    }

    const validSteps = progression.validSteps;
    const currentSnapshots = Array.isArray(plan.view_snapshots) ? plan.view_snapshots : [];
    if (snapshotsMatchProgression(currentSnapshots, progression)) {
      return currentSnapshots;
    }
    const fallbackLayoutsByOperandId = {};
    const snapshotOptions = getSnapshotOptions();
    const existingSnapshots = buildExistingSnapshotsByStepCount(currentSnapshots);
    const existingLayoutsByStepCount = buildExistingSnapshotLayoutsByStepCount(
      currentSnapshots,
      snapshotOptions
    );
    const initialSnapshotRecord = buildSnapshotAndLayoutMapFromOperands(
      getActiveOperandsFromProgression(progression, 0, false),
      existingSnapshots.get(0),
      {
        existingLayouts: existingLayoutsByStepCount.get(0) || null,
        fallbackLayoutForOperand: (operand) =>
          getFallbackLayoutForOperand(operand, fallbackLayoutsByOperandId),
        snapshotOptions,
      }
    );
    const nextSnapshots = [
      {
        ...initialSnapshotRecord.snapshot,
        applied_step_count: 0,
      },
    ];
    let previousLayouts = initialSnapshotRecord.layoutMap;

    for (let stepCount = 1; stepCount <= validSteps.length; stepCount += 1) {
      const activeOperands = getActiveOperandsFromProgression(
        progression,
        stepCount,
        false
      );
      const step = validSteps[stepCount - 1];
      const defaultsByOperandId = buildSnapshotDefaultsByOperandId(
        activeOperands,
        step,
        previousLayouts,
        fallbackLayoutsByOperandId
      );
      const nextSnapshotRecord = buildSnapshotAndLayoutMapFromOperands(
        activeOperands,
        existingSnapshots.get(stepCount),
        {
          defaultsByOperandId,
          existingLayouts: existingLayoutsByStepCount.get(stepCount) || null,
          fallbackLayoutForOperand: (operand) =>
            getFallbackLayoutForOperand(operand, fallbackLayoutsByOperandId),
          snapshotOptions,
        }
      );

      nextSnapshots.push({
        ...nextSnapshotRecord.snapshot,
        applied_step_count: stepCount,
      });
      previousLayouts = nextSnapshotRecord.layoutMap;
    }

    plan.view_snapshots = nextSnapshots;
    touchContractionViewRevision();
    return nextSnapshots;
  }

  function getLatestAppliedStepCount(progression = null) {
    return (progression || buildContractionOperandProgression()).validSteps.length;
  }

  function getViewedAppliedStepCount(progression = null) {
    const latestAppliedStepCount = getLatestAppliedStepCount(progression);
    if (!Number.isInteger(state.plannerInspectionStepCount)) {
      return latestAppliedStepCount;
    }
    return Math.max(0, Math.min(latestAppliedStepCount, state.plannerInspectionStepCount));
  }

  function isInspectingPastStage() {
    return getViewedAppliedStepCount() !== getLatestAppliedStepCount();
  }

  function getSnapshotForStepCount(stepCount) {
    const plan = getContractionPlan();
    if (!plan) {
      return null;
    }
    const snapshots = ensureContractionViewSnapshots();
    const snapshot = snapshots[stepCount];
    return snapshot && snapshot.applied_step_count === stepCount ? snapshot : null;
  }

  function buildFutureOrdersByOperandId(appliedStepCount, progression, activeOperands) {
    if (appliedStepCount >= progression.validSteps.length) {
      return {};
    }
    const stepOrdersByTensorId = progression.stepOrdersByTensorId || {};
    const futureOrdersByOperandId = {};
    activeOperands.forEach((operand) => {
      const futureOrderSet = new Set();
      operand.sourceTensorIds.forEach((tensorId) => {
        (stepOrdersByTensorId[tensorId] || []).forEach((stepOrder) => {
          if (stepOrder > appliedStepCount) {
            futureOrderSet.add(stepOrder);
          }
        });
      });
      futureOrdersByOperandId[operand.id] = [...futureOrderSet].sort((left, right) => left - right);
    });
    return futureOrdersByOperandId;
  }

  function buildContractionScene(appliedStepCount = null) {
    const plan = getContractionPlan();
    const progression = buildContractionOperandProgression();
    const latestAppliedStepCount = getLatestAppliedStepCount(progression);
    if (!plan || latestAppliedStepCount <= 0) {
      return null;
    }

    const requestedAppliedStepCount =
      appliedStepCount === null ? getViewedAppliedStepCount(progression) : appliedStepCount;
    const normalizedAppliedStepCount = Math.max(
      0,
      Math.min(latestAppliedStepCount, requestedAppliedStepCount)
    );
    const snapshots = ensureContractionViewSnapshots(progression);
    const progressionToken = state.contractionProgressionCacheToken;
    const shouldResetSceneCache =
      state.contractionSceneCacheRevision !== state.specRevision ||
      state.contractionSceneCacheViewRevision !== state.contractionSceneViewRevision ||
      state.contractionSceneCacheProgressionToken !== progressionToken;
    if (shouldResetSceneCache) {
      state.contractionSceneCacheRevision = state.specRevision;
      state.contractionSceneCacheViewRevision = state.contractionSceneViewRevision;
      state.contractionSceneCacheProgressionToken = progressionToken;
      state.contractionSceneCacheByAppliedStepCount = {};
    }
    if (
      Object.prototype.hasOwnProperty.call(
        state.contractionSceneCacheByAppliedStepCount,
        normalizedAppliedStepCount
      )
    ) {
      return state.contractionSceneCacheByAppliedStepCount[normalizedAppliedStepCount];
    }

    const activeOperands = getActiveOperandsFromProgression(
      progression,
      normalizedAppliedStepCount
    );
    const snapshot = snapshots[normalizedAppliedStepCount] || null;
    const layoutMap = buildSnapshotLayoutMap(snapshot, getSnapshotOptions());
    const operandMap = {};
    const tokenOccurrencesByKey = {};
    const edgeMap = {};
    const edgeIdByBaseEdgeId = {};
    if (typeof ctx.ensureSpecLookups === "function") {
      ctx.ensureSpecLookups();
    }
    const tensors = activeOperands.map((operand) => {
      const layout = layoutMap[operand.id] || buildFallbackLayoutForOperand(operand);
      const visibleTensor = {
        id: operand.id,
        name: operand.name,
        isDerived: Boolean(operand.isDerived),
        linear_periodic_role: operand.linearPeriodicRole || null,
        sourceTensorIds: [...operand.sourceTensorIds],
        resultCount: operand.sourceTensorIds.length,
        position: {
          x: layout.position.x,
          y: layout.position.y,
        },
        size: {
          width: layout.size.width,
          height: layout.size.height,
        },
        indices: operand.tokens.map((token, indexPosition) => {
          const indexId = `scene-index:${operand.id}:${indexPosition}:${token.key}`;
          const proxyTensor = { size: layout.size };
          const offset = ctx.defaultIndexOffsetForOrder(indexPosition, proxyTensor);
          const visibleIndex = {
            id: indexId,
            key: token.key,
            name: token.name,
            dimension: token.dimension,
            offset,
            sourceIndexId: token.sourceIndexId || null,
          };
          if (!Array.isArray(tokenOccurrencesByKey[token.key])) {
            tokenOccurrencesByKey[token.key] = [];
          }
          tokenOccurrencesByKey[token.key].push({
            tensorId: operand.id,
            indexId,
            token,
          });
          return visibleIndex;
        }),
      };
      operandMap[visibleTensor.id] = visibleTensor;
      return visibleTensor;
    });

    const edges = Object.entries(tokenOccurrencesByKey)
      .filter(([, occurrences]) => Array.isArray(occurrences) && occurrences.length === 2)
      .map(([tokenKey, occurrences]) => {
        const baseEdgeId =
          occurrences.find(
            (occurrence) =>
              occurrence &&
              occurrence.token &&
              typeof occurrence.token.sourceEdgeId === "string" &&
              occurrence.token.sourceEdgeId
          )?.token.sourceEdgeId || null;
        const baseEdge = baseEdgeId ? state.edgeById[baseEdgeId] || null : null;
        const visibleEdge = {
          id: `scene-edge:${tokenKey}`,
          key: tokenKey,
          name: baseEdge ? baseEdge.name : occurrences[0].token.name,
          label: baseEdge ? baseEdge.name : occurrences[0].token.name,
          metadata: baseEdge && baseEdge.metadata ? baseEdge.metadata : {},
          baseEdgeId,
          leftIndexId: occurrences[0].indexId,
          rightIndexId: occurrences[1].indexId,
        };
        edgeMap[visibleEdge.id] = visibleEdge;
        if (baseEdgeId) {
          edgeIdByBaseEdgeId[baseEdgeId] = visibleEdge.id;
        }
        return visibleEdge;
      });

    const scene = {
      appliedStepCount: normalizedAppliedStepCount,
      latestAppliedStepCount,
      totalStepCount: progression.validSteps.length,
      validSteps: progression.validSteps,
      operandMap,
      edgeMap,
      edgeIdByBaseEdgeId,
      tensors,
      edges,
      futureOrdersByOperandId: buildFutureOrdersByOperandId(
        normalizedAppliedStepCount,
        progression,
        activeOperands
      ),
    };
    state.contractionSceneCacheByAppliedStepCount[normalizedAppliedStepCount] = scene;
    return scene;
  }

  function isContractionSceneVisible() {
    return Boolean(buildContractionScene());
  }

  function captureVisibleOperandLayoutMap(appliedStepCount = getViewedAppliedStepCount()) {
    const scene = buildContractionScene(appliedStepCount);
    if (!scene) {
      return {};
    }
    return Object.fromEntries(
      scene.tensors.map((tensor) => [
        tensor.id,
        {
          position: {
            x: Math.round(tensor.position.x),
            y: Math.round(tensor.position.y),
          },
          size: {
            width: Math.round(tensor.size.width),
            height: Math.round(tensor.size.height),
          },
        },
      ])
    );
  }

  function findVisibleTensorById(tensorId) {
    const scene = buildContractionScene();
    if (!scene) {
      return ctx.findTensorById(tensorId);
    }
    return scene.operandMap[tensorId] || null;
  }

  function getVisibleTensors() {
    const scene = buildContractionScene();
    return scene ? scene.tensors : state.spec.tensors;
  }

  function getVisibleEdges() {
    const scene = buildContractionScene();
    return scene ? scene.edges : state.spec.edges;
  }

  function findVisibleEdgeById(edgeId) {
    const scene = buildContractionScene();
    if (!scene) {
      if (typeof ctx.ensureSpecLookups === "function") {
        ctx.ensureSpecLookups();
      }
      return state.edgeById[edgeId] || null;
    }
    return scene.edgeMap[edgeId] || null;
  }

  function findVisibleEdgeSelectionIdByBaseEdgeId(baseEdgeId) {
    if (!baseEdgeId) {
      return null;
    }
    const scene = buildContractionScene();
    if (!scene) {
      return baseEdgeId;
    }
    return scene.edgeIdByBaseEdgeId[baseEdgeId] || baseEdgeId;
  }

  function canEditCurrentContractionStage() {
    return Boolean(buildContractionScene()) && !isInspectingPastStage();
  }

  function updateCurrentStageOperandLayout(operandId, updates) {
    const plan = getContractionPlan();
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
        width: Math.max(ctx.constants.MIN_TENSOR_WIDTH, Math.round(updates.size.width)),
        height: Math.max(ctx.constants.MIN_TENSOR_HEIGHT, Math.round(updates.size.height)),
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
        width: Math.max(ctx.constants.MIN_TENSOR_WIDTH, Math.round(nextLayout.size.width)),
        height: Math.max(ctx.constants.MIN_TENSOR_HEIGHT, Math.round(nextLayout.size.height)),
      };
    });
    touchContractionViewRevision();
    return true;
  }

  function clearPastInspection() {
    state.plannerInspectionStepCount = null;
    state.plannerFutureBadgeDisclosure = {};
  }

  function beginPastInspection(stepIndex) {
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const inspectedStepCount = Math.max(0, Math.min(stepIndex, latestAppliedStepCount));
    state.plannerInspectionStepCount =
      inspectedStepCount === latestAppliedStepCount ? null : inspectedStepCount;
    state.plannerFutureBadgeDisclosure = {};
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
  }

  function toggleFutureBadgeDisclosure(operandId) {
    state.plannerFutureBadgeDisclosure[operandId] = !Boolean(
      state.plannerFutureBadgeDisclosure[operandId]
    );
  }

  function applyManualContractionStep(leftOperandId, rightOperandId) {
    const plan = ctx.ensureContractionPlan();
    ensureContractionViewSnapshots();
    const latestAppliedStepCount = getLatestAppliedStepCount();
    const latestSnapshot = getSnapshotForStepCount(latestAppliedStepCount);
    const latestScene = buildContractionScene(latestAppliedStepCount);
    const leftVisibleOperand = latestScene ? latestScene.operandMap[leftOperandId] : ctx.findTensorById(leftOperandId);
    const rightVisibleOperand = latestScene ? latestScene.operandMap[rightOperandId] : ctx.findTensorById(rightOperandId);
    const nextStepId = ctx.makeId("step");

    plan.steps.push({
      id: nextStepId,
      left_operand_id: leftOperandId,
      right_operand_id: rightOperandId,
      metadata: {},
    });
    ensureContractionViewSnapshots();

    const nextSnapshot = getSnapshotForStepCount(latestAppliedStepCount + 1);
    const nextLayout = nextSnapshot
      ? nextSnapshot.operand_layouts.find((layout) => layout.operand_id === nextStepId)
      : null;
    const preferredLayout =
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
        width: Math.round(preferredLayout.size.width),
        height: Math.round(preferredLayout.size.height),
      };
    } else if (nextLayout && latestSnapshot) {
      const fallbackLayout = latestSnapshot.operand_layouts.find(
        (layout) => layout.operand_id === getPreferredStepAnchorOperandId({
          left_operand_id: leftOperandId,
          right_operand_id: rightOperandId,
        }, {
          isNextOperandId,
          isPreviousOperandId,
        })
      );
      if (fallbackLayout) {
        nextLayout.position = { ...fallbackLayout.position };
        nextLayout.size = { ...fallbackLayout.size };
      }
    }
  }

  Object.assign(ctx, {
    getContractionPlan,
    getPlanSteps,
    buildContractionOperandState,
    getTensorKrowchManualPlanIssue,
    getTensorKrowchManualPlanIssueMessage,
    ensureContractionViewSnapshots,
    getLatestAppliedStepCount,
    getViewedAppliedStepCount,
    isInspectingPastStage,
    getSnapshotForStepCount,
    buildContractionScene,
    isContractionSceneVisible,
    captureVisibleOperandLayoutMap,
    findVisibleTensorById,
    getVisibleTensors,
    getVisibleEdges,
    findVisibleEdgeById,
    findVisibleEdgeSelectionIdByBaseEdgeId,
    canEditCurrentContractionStage,
    updateCurrentStageOperandLayout,
    applySnapshotLayoutMap,
    clearPastInspection,
    beginPastInspection,
    togglePastInspection,
    toggleFutureBadgeDisclosure,
    applyManualContractionStep,
  });
}
