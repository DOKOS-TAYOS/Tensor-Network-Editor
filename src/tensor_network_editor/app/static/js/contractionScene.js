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
    const tensorById = Object.fromEntries(
      (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : []).map((tensor) => [
        tensor.id,
        tensor,
      ])
    );
    const indexOwnerById = {};
    (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : []).forEach((tensor) => {
      (Array.isArray(tensor.indices) ? tensor.indices : []).forEach((index) => {
        indexOwnerById[index.id] = { tensor, index };
      });
    });
    return (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [])
      .filter((tensor) => ctx.isLinearPeriodicBoundaryTensor(tensor))
      .map((boundaryTensor) => {
        const operandId = ctx.getLinearPeriodicReservedOperandIdForTensor(boundaryTensor);
        if (!operandId) {
          return null;
        }
        const tokens = (Array.isArray(boundaryTensor.indices) ? boundaryTensor.indices : []).map(
          (boundaryIndex) => {
            const boundaryEdge = (Array.isArray(state.spec && state.spec.edges) ? state.spec.edges : [])
              .find(
                (edge) =>
                  (edge.left && edge.left.index_id === boundaryIndex.id) ||
                  (edge.right && edge.right.index_id === boundaryIndex.id)
              ) || null;
            const otherIndexId =
              boundaryEdge && boundaryEdge.left && boundaryEdge.left.index_id === boundaryIndex.id
                ? boundaryEdge.right && boundaryEdge.right.index_id
                : boundaryEdge && boundaryEdge.left && boundaryEdge.left.index_id;
            const otherOwner = otherIndexId ? indexOwnerById[otherIndexId] || null : null;
            const otherTensor = otherOwner ? tensorById[otherOwner.tensor.id] || null : null;
            const connectedToRealTensor =
              otherOwner &&
              otherTensor &&
              !ctx.isLinearPeriodicBoundaryTensor(otherTensor);
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

  function cloneOperand(operand) {
    return {
      id: operand.id,
      name: operand.name,
      isDerived: Boolean(operand.isDerived),
      linearPeriodicRole: operand.linearPeriodicRole || null,
      sourceTensorIds: [...operand.sourceTensorIds],
      tokens: operand.tokens.map((token) => ({ ...token })),
    };
  }

  function analyzeOperandPair(leftOperand, rightOperand) {
    if (!leftOperand || !rightOperand) {
      return null;
    }
    const rightTokenKeys = new Set(rightOperand.tokens.map((token) => token.key));
    const sharedTokenKeys = [...new Set(
      leftOperand.tokens
        .filter((token) => rightTokenKeys.has(token.key))
        .map((token) => token.key)
    )];
    return {
      sharedTokenKeys,
      sharedTokenCount: sharedTokenKeys.length,
      isOuterProduct: sharedTokenKeys.length === 0,
    };
  }

  function buildContractionOperandState(stepLimit = null, planSteps = getPlanSteps()) {
    const activeOperands = buildInitialOperands();
    const sourceTensorIdsByOperandId = Object.fromEntries(
      activeOperands.map((operand) => [operand.id, [...operand.sourceTensorIds]])
    );
    const validSteps = [];
    const stepAnalyses = [];
    const totalSteps = stepLimit === null ? planSteps.length : Math.max(0, stepLimit);

    for (const step of planSteps.slice(0, totalSteps)) {
      const usesPreviousOperand =
        isPreviousOperandId(step.left_operand_id) ||
        isPreviousOperandId(step.right_operand_id);
      const usesNextOperand =
        isNextOperandId(step.left_operand_id) ||
        isNextOperandId(step.right_operand_id);
      const leftIndex = activeOperands.findIndex((operand) => operand.id === step.left_operand_id);
      const rightIndex = activeOperands.findIndex((operand) => operand.id === step.right_operand_id);
      if (
        leftIndex < 0 ||
        rightIndex < 0 ||
        leftIndex === rightIndex ||
        activeOperands.some((operand) => operand.id === step.id) ||
        (usesPreviousOperand && usesNextOperand)
      ) {
        break;
      }

      const leftOperand = cloneOperand(activeOperands[leftIndex]);
      const rightOperand = cloneOperand(activeOperands[rightIndex]);
      const pairAnalysis = analyzeOperandPair(leftOperand, rightOperand);
      const contractedTokenKeys = new Set(
        pairAnalysis ? pairAnalysis.sharedTokenKeys : []
      );
      const resultOperand = {
        id: step.id,
        name: `Result ${validSteps.length + 1}`,
        isDerived: true,
        sourceTensorIds: [...new Set([...leftOperand.sourceTensorIds, ...rightOperand.sourceTensorIds])],
        tokens: [
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
      sourceTensorIdsByOperandId[step.id] = [...resultOperand.sourceTensorIds];
      const indicesToRemove = [leftIndex, rightIndex].sort((left, right) => right - left);
      indicesToRemove.forEach((index) => {
        activeOperands.splice(index, 1);
      });
      activeOperands.push(resultOperand);
      if (usesNextOperand) {
        break;
      }
    }

    return {
      activeOperands,
      validSteps,
      sourceTensorIdsByOperandId,
      stepAnalyses,
    };
  }

  function getTensorKrowchManualPlanIssue() {
    if (state.selectedEngine !== "tensorkrowch") {
      return null;
    }
    const plan = getContractionPlan();
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      return null;
    }
    const fullState = buildContractionOperandState();
    const incompatibleStep = fullState.stepAnalyses.find(
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
    const anchorTensor =
      state.spec && Array.isArray(state.spec.tensors)
        ? state.spec.tensors.find((tensor) => tensor.id === anchorTensorId) || null
        : null;
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

  function buildSnapshotLayoutMap(snapshot) {
    if (!snapshot || !Array.isArray(snapshot.operand_layouts)) {
      return {};
    }
    return Object.fromEntries(
      snapshot.operand_layouts.map((layout) => [
        layout.operand_id,
        {
          position: {
            x: ctx.asFiniteNumber(layout.position && layout.position.x, 120),
            y: ctx.asFiniteNumber(layout.position && layout.position.y, 120),
          },
          size: {
            width: Math.max(
              ctx.constants.MIN_TENSOR_WIDTH,
              ctx.asFiniteNumber(layout.size && layout.size.width, ctx.constants.TENSOR_WIDTH)
            ),
            height: Math.max(
              ctx.constants.MIN_TENSOR_HEIGHT,
              ctx.asFiniteNumber(layout.size && layout.size.height, ctx.constants.TENSOR_HEIGHT)
            ),
          },
        },
      ])
    );
  }

  function buildSnapshotFromOperands(activeOperands, existingSnapshot, defaultsByOperandId = {}) {
    const existingLayouts = buildSnapshotLayoutMap(existingSnapshot);
    return {
      applied_step_count: Number(existingSnapshot && existingSnapshot.applied_step_count) || 0,
      operand_layouts: activeOperands.map((operand) => {
        const fallbackLayout = defaultsByOperandId[operand.id] || buildFallbackLayoutForOperand(operand);
        const chosenLayout = existingLayouts[operand.id] || fallbackLayout;
        return {
          operand_id: operand.id,
          position: {
            x: chosenLayout.position.x,
            y: chosenLayout.position.y,
          },
          size: {
            width: chosenLayout.size.width,
            height: chosenLayout.size.height,
          },
        };
      }),
    };
  }

  function getPreferredStepAnchorOperandId(step) {
    if (isPreviousOperandId(step.left_operand_id) || isNextOperandId(step.left_operand_id)) {
      return step.right_operand_id;
    }
    if (isPreviousOperandId(step.right_operand_id) || isNextOperandId(step.right_operand_id)) {
      return step.left_operand_id;
    }
    return step.left_operand_id;
  }

  function ensureContractionViewSnapshots() {
    const plan = getContractionPlan();
    if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
      if (plan) {
        plan.view_snapshots = [];
      }
      return [];
    }

    const fullState = buildContractionOperandState();
    const validSteps = fullState.validSteps;
    const existingSnapshots = new Map(
      (Array.isArray(plan.view_snapshots) ? plan.view_snapshots : [])
        .filter((snapshot) => snapshot && Number.isInteger(snapshot.applied_step_count))
        .map((snapshot) => [snapshot.applied_step_count, snapshot])
    );

    const stageZeroState = buildContractionOperandState(0);
    const nextSnapshots = [
      {
        ...buildSnapshotFromOperands(stageZeroState.activeOperands, existingSnapshots.get(0)),
        applied_step_count: 0,
      },
    ];

    for (let stepCount = 1; stepCount <= validSteps.length; stepCount += 1) {
      const previousState = buildContractionOperandState(stepCount - 1);
      const currentState = buildContractionOperandState(stepCount);
      const step = validSteps[stepCount - 1];
      const previousLayouts = buildSnapshotLayoutMap(nextSnapshots[stepCount - 1]);
      const defaultsByOperandId = {};

      currentState.activeOperands.forEach((operand) => {
        if (operand.id === step.id) {
          const anchorOperandId = getPreferredStepAnchorOperandId(step);
          defaultsByOperandId[operand.id] = ctx.deepClone(
            previousLayouts[anchorOperandId] || buildFallbackLayoutForOperand(operand)
          );
          return;
        }
        defaultsByOperandId[operand.id] = ctx.deepClone(
          previousLayouts[operand.id] || buildFallbackLayoutForOperand(operand)
        );
      });

      nextSnapshots.push({
        ...buildSnapshotFromOperands(
          currentState.activeOperands,
          existingSnapshots.get(stepCount),
          defaultsByOperandId
        ),
        applied_step_count: stepCount,
      });
    }

    plan.view_snapshots = nextSnapshots;
    return nextSnapshots;
  }

  function getLatestAppliedStepCount() {
    return buildContractionOperandState().validSteps.length;
  }

  function getViewedAppliedStepCount() {
    const latestAppliedStepCount = getLatestAppliedStepCount();
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
    return snapshots.find((snapshot) => snapshot.applied_step_count === stepCount) || null;
  }

  function buildFutureOrdersByOperandId(appliedStepCount, fullState, activeOperands) {
    if (appliedStepCount >= fullState.validSteps.length) {
      return {};
    }
    const futureOrdersByOperandId = {};
    activeOperands.forEach((operand) => {
      const sourceTensorIds = new Set(operand.sourceTensorIds);
      const futureOrders = [];
      fullState.validSteps.slice(appliedStepCount).forEach((step, futureOffset) => {
        const stepSourceTensorIds = new Set(fullState.sourceTensorIdsByOperandId[step.id] || []);
        if ([...stepSourceTensorIds].some((tensorId) => sourceTensorIds.has(tensorId))) {
          futureOrders.push(appliedStepCount + futureOffset + 1);
        }
      });
      futureOrdersByOperandId[operand.id] = futureOrders;
    });
    return futureOrdersByOperandId;
  }

  function buildContractionScene(appliedStepCount = getViewedAppliedStepCount()) {
    const plan = getContractionPlan();
    const latestAppliedStepCount = getLatestAppliedStepCount();
    if (!plan || latestAppliedStepCount <= 0) {
      return null;
    }

    const normalizedAppliedStepCount = Math.max(0, Math.min(latestAppliedStepCount, appliedStepCount));
    const fullState = buildContractionOperandState();
    const stageState = buildContractionOperandState(normalizedAppliedStepCount);
    const snapshot = getSnapshotForStepCount(normalizedAppliedStepCount);
    const layoutMap = buildSnapshotLayoutMap(snapshot);
    const operandMap = {};
    const tokenOccurrencesByKey = {};
    const tensors = stageState.activeOperands.map((operand) => {
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
        const baseEdge = baseEdgeId
          ? state.spec.edges.find((edge) => edge.id === baseEdgeId) || null
          : null;
        return {
          id: `scene-edge:${tokenKey}`,
          key: tokenKey,
          name: baseEdge ? baseEdge.name : occurrences[0].token.name,
          label: baseEdge ? baseEdge.name : occurrences[0].token.name,
          metadata: baseEdge && baseEdge.metadata ? baseEdge.metadata : {},
          baseEdgeId,
          leftIndexId: occurrences[0].indexId,
          rightIndexId: occurrences[1].indexId,
        };
      });

    return {
      appliedStepCount: normalizedAppliedStepCount,
      latestAppliedStepCount,
      totalStepCount: fullState.validSteps.length,
      validSteps: fullState.validSteps,
      operandMap,
      tensors,
      edges,
      futureOrdersByOperandId: buildFutureOrdersByOperandId(
        normalizedAppliedStepCount,
        fullState,
        stageState.activeOperands
      ),
    };
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
    return getVisibleEdges().find((edge) => edge.id === edgeId) || null;
  }

  function findVisibleEdgeSelectionIdByBaseEdgeId(baseEdgeId) {
    if (!baseEdgeId) {
      return null;
    }
    const visibleEdge = getVisibleEdges().find(
      (edge) => edge.baseEdgeId === baseEdgeId
    );
    return visibleEdge ? visibleEdge.id : baseEdgeId;
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
