import { getActiveOperandsFromProgression } from "../state/contractionSceneProgression.js";
import {
  buildExistingSnapshotLayoutsByStepCount,
  buildExistingSnapshotsByStepCount,
  buildSnapshotAndLayoutMapFromOperands,
  buildSnapshotLayoutMap,
  cloneOperandLayout,
  getPreferredStepAnchorOperandId,
} from "../state/contractionSceneSnapshots.js";

export function createContractionSceneCacheSupport({
  state,
  constants,
  asFiniteNumber,
  defaultIndexOffsetForOrder,
  ensureSpecLookups,
  tensorWidth,
  tensorHeight,
  findTensorById,
  buildContractionOperandProgression,
  getContractionPlan,
  isNextOperandId,
  isPreviousOperandId,
}) {
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

  function buildFallbackLayoutForOperand(operand) {
    const anchorTensorId = operand.sourceTensorIds[0];
    if (typeof ensureSpecLookups === "function") {
      ensureSpecLookups();
    }
    const anchorTensor = state.tensorById[anchorTensorId] || null;
    if (anchorTensor) {
      return {
        position: { x: anchorTensor.position.x, y: anchorTensor.position.y },
        size: {
          height: tensorHeight(anchorTensor),
          width: tensorWidth(anchorTensor),
        },
      };
    }
    return {
      position: { x: 120, y: 120 },
      size: { height: constants.TENSOR_HEIGHT, width: constants.TENSOR_WIDTH },
    };
  }

  function getSnapshotOptions() {
    return {
      asFiniteNumber: (value, fallbackValue) => asFiniteNumber(value, fallbackValue),
      constants,
    };
  }

  function buildLayoutMapFromSnapshot(snapshot) {
    return buildSnapshotLayoutMap(snapshot, getSnapshotOptions());
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
        previousLayouts[operand.id] ||
        getFallbackLayoutForOperand(operand, fallbackLayoutsByOperandId);
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
      futureOrdersByOperandId[operand.id] = [...futureOrderSet].sort(
        (left, right) => left - right
      );
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
    const layoutMap = buildLayoutMapFromSnapshot(snapshot);
    const operandMap = {};
    const tokenOccurrencesByKey = {};
    const edgeMap = {};
    const edgeIdByBaseEdgeId = {};
    if (typeof ensureSpecLookups === "function") {
      ensureSpecLookups();
    }
    const tensors = activeOperands.map((operand) => {
      const layout = layoutMap[operand.id] || buildFallbackLayoutForOperand(operand);
      const visibleTensor = {
        id: operand.id,
        indices: operand.tokens.map((token, indexPosition) => {
          const indexId = `scene-index:${operand.id}:${indexPosition}:${token.key}`;
          const proxyTensor = { size: layout.size };
          const offset =
            typeof defaultIndexOffsetForOrder === "function"
              ? defaultIndexOffsetForOrder(indexPosition, proxyTensor)
              : null;
          const visibleIndex = {
            dimension: token.dimension,
            id: indexId,
            key: token.key,
            name: token.name,
            offset,
            sourceIndexId: token.sourceIndexId || null,
          };
          if (!Array.isArray(tokenOccurrencesByKey[token.key])) {
            tokenOccurrencesByKey[token.key] = [];
          }
          tokenOccurrencesByKey[token.key].push({
            indexId,
            tensorId: operand.id,
            token,
          });
          return visibleIndex;
        }),
        isDerived: Boolean(operand.isDerived),
        linear_periodic_role: operand.linearPeriodicRole || null,
        name: operand.name,
        position: {
          x: layout.position.x,
          y: layout.position.y,
        },
        resultCount: operand.sourceTensorIds.length,
        size: {
          height: layout.size.height,
          width: layout.size.width,
        },
        sourceTensorIds: [...operand.sourceTensorIds],
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
          baseEdgeId,
          id: `scene-edge:${tokenKey}`,
          key: tokenKey,
          label: baseEdge ? baseEdge.name : occurrences[0].token.name,
          leftIndexId: occurrences[0].indexId,
          metadata: baseEdge && baseEdge.metadata ? baseEdge.metadata : {},
          name: baseEdge ? baseEdge.name : occurrences[0].token.name,
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
      edgeIdByBaseEdgeId,
      edgeMap,
      edges,
      futureOrdersByOperandId: buildFutureOrdersByOperandId(
        normalizedAppliedStepCount,
        progression,
        activeOperands
      ),
      latestAppliedStepCount,
      operandMap,
      tensors,
      totalStepCount: progression.validSteps.length,
      validSteps: progression.validSteps,
    };
    state.contractionSceneCacheByAppliedStepCount[normalizedAppliedStepCount] = scene;
    return scene;
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
            height: Math.round(tensor.size.height),
            width: Math.round(tensor.size.width),
          },
        },
      ])
    );
  }

  function findVisibleTensorById(tensorId) {
    const scene = buildContractionScene();
    if (!scene) {
      return findTensorById(tensorId);
    }
    return scene.operandMap[tensorId] || null;
  }

  function getVisibleTensors() {
    const scene = buildContractionScene();
    return scene
      ? scene.tensors
      : Array.isArray(state.spec && state.spec.tensors)
        ? state.spec.tensors
        : [];
  }

  function getVisibleEdges() {
    const scene = buildContractionScene();
    return scene
      ? scene.edges
      : Array.isArray(state.spec && state.spec.edges)
        ? state.spec.edges
        : [];
  }

  function findVisibleEdgeById(edgeId) {
    const scene = buildContractionScene();
    if (!scene) {
      if (typeof ensureSpecLookups === "function") {
        ensureSpecLookups();
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

  return {
    buildContractionScene,
    buildLayoutMapFromSnapshot,
    buildSnapshotOptions: getSnapshotOptions,
    captureVisibleOperandLayoutMap,
    ensureContractionViewSnapshots,
    findVisibleEdgeById,
    findVisibleEdgeSelectionIdByBaseEdgeId,
    findVisibleTensorById,
    getLatestAppliedStepCount,
    getSnapshotForStepCount,
    getVisibleEdges,
    getVisibleTensors,
    getViewedAppliedStepCount,
    touchContractionViewRevision,
  };
}
