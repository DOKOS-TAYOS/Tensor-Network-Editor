export function createUtilitySpecBindings({ ctx, state, constants, runtime }) {
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    NOTE_WIDTH,
    NOTE_HEIGHT,
    NOTE_MIN_WIDTH,
    NOTE_MIN_HEIGHT,
  } = constants;
  const { document } = ctx;

  function buildEmptyGraphSection() {
    return {
      tensors: [],
      groups: [],
      edges: [],
      notes: [],
      contraction_plan: null,
      metadata: {},
    };
  }

  function clearGraphSectionOnSpec(spec) {
    spec.tensors = [];
    spec.groups = [];
    spec.edges = [];
    spec.notes = [];
    spec.contraction_plan = null;
  }

  function buildGraphSectionFromSpec(spec, existingCell = null) {
    return {
      tensors: runtime.deepClone(Array.isArray(spec && spec.tensors) ? spec.tensors : []),
      groups: runtime.deepClone(Array.isArray(spec && spec.groups) ? spec.groups : []),
      edges: runtime.deepClone(Array.isArray(spec && spec.edges) ? spec.edges : []),
      notes: runtime.deepClone(Array.isArray(spec && spec.notes) ? spec.notes : []),
      contraction_plan: runtime.deepClone(
        spec && runtime.isObject(spec.contraction_plan) ? spec.contraction_plan : null
      ),
      metadata: runtime.deepClone(
        existingCell && runtime.isObject(existingCell.metadata) ? existingCell.metadata : {}
      ),
    };
  }

  function replaceGraphSectionOnSpec(spec, graphSection) {
    const nextSection = normalizeGraphSectionInPlace(
      runtime.deepClone(graphSection || buildEmptyGraphSection())
    );
    spec.tensors = nextSection.tensors;
    spec.groups = nextSection.groups;
    spec.edges = nextSection.edges;
    spec.notes = nextSection.notes;
    spec.contraction_plan = nextSection.contraction_plan;
  }

  function normalizeGraphSectionInPlace(graphSection) {
    graphSection.metadata = runtime.isObject(graphSection.metadata)
      ? graphSection.metadata
      : {};
    graphSection.tensors = Array.isArray(graphSection.tensors)
      ? graphSection.tensors
      : [];
    graphSection.groups = Array.isArray(graphSection.groups) ? graphSection.groups : [];
    graphSection.edges = Array.isArray(graphSection.edges) ? graphSection.edges : [];
    graphSection.notes = Array.isArray(graphSection.notes) ? graphSection.notes : [];
    graphSection.contraction_plan = runtime.isObject(graphSection.contraction_plan)
      ? graphSection.contraction_plan
      : null;

    graphSection.tensors.forEach((tensor) => {
      tensor.metadata = runtime.isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.position = {
        x: runtime.asFiniteNumber(tensor.position && tensor.position.x, 120),
        y: runtime.asFiniteNumber(tensor.position && tensor.position.y, 120),
      };
      tensor.size = {
        width: Math.max(
          MIN_TENSOR_WIDTH,
          runtime.asFiniteNumber(tensor.size && tensor.size.width, TENSOR_WIDTH)
        ),
        height: Math.max(
          MIN_TENSOR_HEIGHT,
          runtime.asFiniteNumber(tensor.size && tensor.size.height, TENSOR_HEIGHT)
        ),
      };
      tensor.linear_periodic_role =
        tensor.linear_periodic_role === "previous" ||
        tensor.linear_periodic_role === "next"
          ? tensor.linear_periodic_role
          : null;
      tensor.indices = Array.isArray(tensor.indices) ? tensor.indices : [];
      tensor.indices.forEach((index, indexPosition) => {
        index.metadata = runtime.isObject(index.metadata) ? index.metadata : {};
        index.dimension = Math.max(
          1,
          Math.round(runtime.asFiniteNumber(index.dimension, 2))
        );
        index.offset = {
          x: runtime.asFiniteNumber(index.offset && index.offset.x, 0),
          y: runtime.asFiniteNumber(index.offset && index.offset.y, 0),
        };
        if (!index.id) {
          index.id = runtime.makeId("index");
        }
        if (!index.name) {
          index.name = runtime.nextName(
            "i",
            tensor.indices
              .slice(0, indexPosition)
              .map((candidate) => candidate.name)
          );
        }
      });
      if (!tensor.id) {
        tensor.id = runtime.makeId("tensor");
      }
      if (!tensor.name) {
        tensor.name = runtime.nextName(
          "T",
          graphSection.tensors
            .slice(0, graphSection.tensors.indexOf(tensor))
            .map((candidate) => candidate.name)
        );
      }
      runtime.ensureTensorIndexOffsets(tensor);
    });

    graphSection.groups.forEach((group, groupPosition) => {
      group.metadata = runtime.isObject(group.metadata) ? group.metadata : {};
      group.tensor_ids = Array.isArray(group.tensor_ids)
        ? group.tensor_ids.map((tensorId) => String(tensorId))
        : [];
      if (!group.id) {
        group.id = runtime.makeId("group");
      }
      if (!group.name) {
        group.name = `Group ${groupPosition + 1}`;
      }
    });

    graphSection.edges.forEach((edge, edgePosition) => {
      edge.metadata = runtime.isObject(edge.metadata) ? edge.metadata : {};
      if (!edge.id) {
        edge.id = runtime.makeId("edge");
      }
      if (!edge.name) {
        edge.name = `bond_${edgePosition + 1}`;
      }
      edge.left = runtime.isObject(edge.left) ? edge.left : {};
      edge.right = runtime.isObject(edge.right) ? edge.right : {};
      edge.left.tensor_id = String(edge.left.tensor_id || "");
      edge.left.index_id = String(edge.left.index_id || "");
      edge.right.tensor_id = String(edge.right.tensor_id || "");
      edge.right.index_id = String(edge.right.index_id || "");
    });

    graphSection.notes.forEach((note) => {
      note.metadata = runtime.isObject(note.metadata) ? note.metadata : {};
      note.position = {
        x: runtime.asFiniteNumber(note.position && note.position.x, 120),
        y: runtime.asFiniteNumber(note.position && note.position.y, 120),
      };
      note.size = {
        width: Math.max(
          NOTE_MIN_WIDTH,
          runtime.asFiniteNumber(note.size && note.size.width, NOTE_WIDTH)
        ),
        height: Math.max(
          NOTE_MIN_HEIGHT,
          runtime.asFiniteNumber(note.size && note.size.height, NOTE_HEIGHT)
        ),
      };
      note.text =
        typeof note.text === "string" && note.text.trim() ? note.text : "Note";
      if (!note.id) {
        note.id = runtime.makeId("note");
      }
    });

    if (graphSection.contraction_plan) {
      graphSection.contraction_plan.metadata = runtime.isObject(
        graphSection.contraction_plan.metadata
      )
        ? graphSection.contraction_plan.metadata
        : {};
      graphSection.contraction_plan.steps = Array.isArray(
        graphSection.contraction_plan.steps
      )
        ? graphSection.contraction_plan.steps
        : [];
      graphSection.contraction_plan.view_snapshots = Array.isArray(
        graphSection.contraction_plan.view_snapshots
      )
        ? graphSection.contraction_plan.view_snapshots
        : [];
      if (!graphSection.contraction_plan.id) {
        graphSection.contraction_plan.id = runtime.makeId("plan");
      }
      if (!graphSection.contraction_plan.name) {
        graphSection.contraction_plan.name = "Manual path";
      }
      graphSection.contraction_plan.steps.forEach((step) => {
        step.metadata = runtime.isObject(step.metadata) ? step.metadata : {};
        if (!step.id) {
          step.id = runtime.makeId("step");
        }
        step.left_operand_id = String(step.left_operand_id || "");
        step.right_operand_id = String(step.right_operand_id || "");
      });
      graphSection.contraction_plan.view_snapshots.forEach(
        (snapshot, snapshotIndex) => {
          snapshot.applied_step_count = Math.max(
            0,
            Math.round(
              runtime.asFiniteNumber(snapshot.applied_step_count, snapshotIndex)
            )
          );
          snapshot.operand_layouts = Array.isArray(snapshot.operand_layouts)
            ? snapshot.operand_layouts
            : [];
          snapshot.operand_layouts.forEach((layout) => {
            layout.operand_id = String(layout.operand_id || "");
            layout.position = {
              x: runtime.asFiniteNumber(layout.position && layout.position.x, 120),
              y: runtime.asFiniteNumber(layout.position && layout.position.y, 120),
            };
            layout.size = {
              width: Math.max(
                MIN_TENSOR_WIDTH,
                runtime.asFiniteNumber(layout.size && layout.size.width, TENSOR_WIDTH)
              ),
              height: Math.max(
                MIN_TENSOR_HEIGHT,
                runtime.asFiniteNumber(layout.size && layout.size.height, TENSOR_HEIGHT)
              ),
            };
          });
        }
      );
    }

    return graphSection;
  }

  function buildHistorySnapshotSpec(spec = state.spec) {
    const snapshotSpec = runtime.deepClone(spec || {});
    if (runtime.isLinearPeriodicMode(snapshotSpec)) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain(snapshotSpec);
    }
    return snapshotSpec;
  }

  function buildSerializedSpec(spec = state.spec) {
    const serializedSpec = runtime.deepClone(spec || {});
    if (runtime.isLinearPeriodicMode(serializedSpec)) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain(serializedSpec);
      clearGraphSectionOnSpec(serializedSpec);
    }
    return serializedSpec;
  }

  function bumpSpecRevision() {
    state.specRevision += 1;
    state.lookupRevision = -1;
  }

  function ensureSpecLookups() {
    if (!state.spec) {
      state.lookupRevision = state.specRevision;
      state.tensorById = {};
      state.edgeById = {};
      state.edgeByIndexId = {};
      state.groupById = {};
      state.indexOwnerById = {};
      state.groupsByTensorId = {};
      state.noteById = {};
      return;
    }
    if (state.lookupRevision === state.specRevision) {
      return;
    }

    const tensorById = {};
    const edgeById = {};
    const edgeByIndexId = {};
    const groupById = {};
    const indexOwnerById = {};
    const groupsByTensorId = {};
    const noteById = {};

    state.spec.tensors.forEach((tensor) => {
      tensorById[tensor.id] = tensor;
      tensor.indices.forEach((index, indexPosition) => {
        indexOwnerById[index.id] = { tensor, index, indexPosition };
      });
    });
    state.spec.edges.forEach((edge) => {
      edgeById[edge.id] = edge;
      edgeByIndexId[edge.left.index_id] = edge;
      edgeByIndexId[edge.right.index_id] = edge;
    });
    state.spec.groups.forEach((group) => {
      groupById[group.id] = group;
      group.tensor_ids.forEach((tensorId) => {
        if (!Array.isArray(groupsByTensorId[tensorId])) {
          groupsByTensorId[tensorId] = [];
        }
        groupsByTensorId[tensorId].push(group);
      });
    });
    state.spec.notes.forEach((note) => {
      noteById[note.id] = note;
    });

    state.tensorById = tensorById;
    state.edgeById = edgeById;
    state.edgeByIndexId = edgeByIndexId;
    state.groupById = groupById;
    state.indexOwnerById = indexOwnerById;
    state.groupsByTensorId = groupsByTensorId;
    state.noteById = noteById;
    state.lookupRevision = state.specRevision;
  }

  function serializeCurrentSpec(options = {}) {
    const { persistViewSnapshots = false } = options;
    if (
      persistViewSnapshots &&
      state.spec &&
      state.spec.contraction_plan &&
      typeof ctx.ensureContractionViewSnapshots === "function"
    ) {
      ctx.ensureContractionViewSnapshots();
    }
    return {
      schema_version: state.schemaVersion,
      network: buildSerializedSpec(),
    };
  }

  function captureEditableFocus() {
    const activeElement = document.activeElement;
    if (!activeElement || !(activeElement instanceof HTMLElement)) {
      return null;
    }
    const focusKey = activeElement.dataset ? activeElement.dataset.focusKey : "";
    if (!focusKey) {
      return null;
    }
    const focusState = {
      key: focusKey,
      selectionStart: null,
      selectionEnd: null,
    };
    if (
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement
    ) {
      focusState.selectionStart = activeElement.selectionStart;
      focusState.selectionEnd = activeElement.selectionEnd;
    }
    return focusState;
  }

  function restoreEditableFocus(focusState) {
    if (!focusState) {
      return;
    }
    const target = Array.from(document.querySelectorAll("[data-focus-key]")).find(
      (element) => element.dataset.focusKey === focusState.key
    );
    if (!(target instanceof HTMLElement)) {
      return;
    }
    target.focus({ preventScroll: true });
    if (
      typeof focusState.selectionStart === "number" &&
      typeof focusState.selectionEnd === "number" &&
      (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)
    ) {
      target.setSelectionRange(
        focusState.selectionStart,
        focusState.selectionEnd
      );
    }
  }

  function stripImportLines(code) {
    const keptLines = code
      .split(/\r?\n/)
      .filter((line) => !/^\s*(import|from)\s+/.test(line));
    while (keptLines.length && keptLines[0].trim() === "") {
      keptLines.shift();
    }
    while (keptLines.length && keptLines[keptLines.length - 1].trim() === "") {
      keptLines.pop();
    }
    return keptLines.join("\n");
  }

  function moveIndex(tensorId, indexPosition, direction) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const targetPosition = indexPosition + direction;
    if (targetPosition < 0 || targetPosition >= tensor.indices.length) {
      return;
    }
    const [index] = tensor.indices.splice(indexPosition, 1);
    tensor.indices.splice(targetPosition, 0, index);
  }

  function removeTensor(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor || runtime.isLinearPeriodicBoundaryTensor(tensor)) {
      return;
    }
    const tensorIndexIds = new Set(tensor.indices.map((index) => index.id));
    state.spec.edges = state.spec.edges.filter(
      (edge) =>
        !tensorIndexIds.has(edge.left.index_id) && !tensorIndexIds.has(edge.right.index_id)
    );
    state.spec.tensors = state.spec.tensors.filter((candidate) => candidate.id !== tensorId);
    state.spec.groups = state.spec.groups
      .map((group) => ({
        ...group,
        tensor_ids: group.tensor_ids.filter((candidateId) => candidateId !== tensorId),
      }))
      .filter((group) => group.tensor_ids.length > 0);
    state.tensorOrder = state.tensorOrder.filter((candidateId) => candidateId !== tensorId);
  }

  function findBaseIndexOwner(indexId) {
    ensureSpecLookups();
    return state.indexOwnerById[indexId] || null;
  }

  function removeIndex(tensorId, indexId) {
    const tensor = findTensorById(tensorId);
    if (!tensor || runtime.isLinearPeriodicBoundaryTensor(tensor)) {
      return;
    }
    state.spec.edges = state.spec.edges.filter(
      (edge) => edge.left.index_id !== indexId && edge.right.index_id !== indexId
    );
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function resolveBaseEdgeId(edgeId) {
    if (!edgeId) {
      return null;
    }
    ensureSpecLookups();
    const baseEdge = state.edgeById[edgeId];
    if (baseEdge) {
      return baseEdge.id;
    }
    const visibleEdge =
      typeof ctx.findVisibleEdgeById === "function" ? ctx.findVisibleEdgeById(edgeId) : null;
    if (
      visibleEdge &&
      typeof visibleEdge.baseEdgeId === "string" &&
      visibleEdge.baseEdgeId
    ) {
      return visibleEdge.baseEdgeId;
    }
    return null;
  }

  function removeEdge(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId) || edgeId;
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== resolvedEdgeId);
  }

  function findTensorById(tensorId) {
    ensureSpecLookups();
    return state.tensorById[tensorId] || null;
  }

  function findGroupById(groupId) {
    ensureSpecLookups();
    return state.groupById[groupId] || null;
  }

  function findGroupsByTensorId(tensorId) {
    ensureSpecLookups();
    return state.groupsByTensorId[tensorId] || [];
  }

  function findEdgeById(edgeId) {
    const resolvedEdgeId = resolveBaseEdgeId(edgeId);
    if (!resolvedEdgeId) {
      return null;
    }
    ensureSpecLookups();
    return state.edgeById[resolvedEdgeId] || null;
  }

  function findVisibleIndexOwner(indexId) {
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : [];
    for (const tensor of visibleTensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return null;
  }

  function findIndexOwner(indexId) {
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    return findVisibleIndexOwner(indexId);
  }

  function resolveConnectableIndexOwner(indexId) {
    const baseOwner = findBaseIndexOwner(indexId);
    if (baseOwner) {
      return baseOwner;
    }
    const visibleOwner = findVisibleIndexOwner(indexId);
    if (
      !visibleOwner ||
      typeof visibleOwner.index.sourceIndexId !== "string" ||
      !visibleOwner.index.sourceIndexId
    ) {
      return null;
    }
    return findBaseIndexOwner(visibleOwner.index.sourceIndexId);
  }

  function findEdgeByIndexId(indexId) {
    ensureSpecLookups();
    const baseEdge = state.edgeByIndexId[indexId];
    if (baseEdge) {
      return baseEdge;
    }
    const visibleEdges =
      typeof ctx.getVisibleEdges === "function" ? ctx.getVisibleEdges() : [];
    return (
      visibleEdges.find(
        (edge) => edge.leftIndexId === indexId || edge.rightIndexId === indexId
      ) || null
    );
  }

  function syncConnectedIndexDimension(indexId, nextDimension) {
    const connectedEdge = findEdgeByIndexId(indexId);
    if (!connectedEdge) {
      return;
    }
    const connectedIndexId =
      connectedEdge.left && connectedEdge.left.index_id === indexId
        ? connectedEdge.right && connectedEdge.right.index_id
        : connectedEdge.left && connectedEdge.left.index_id;
    if (!connectedIndexId) {
      return;
    }
    const connectedOwner = findIndexOwner(connectedIndexId);
    if (!connectedOwner || !connectedOwner.index) {
      return;
    }
    connectedOwner.index.dimension = nextDimension;
  }

  function createTensor(x, y) {
    const tensor = {
      id: runtime.makeId("tensor"),
      name: runtime.nextName("T", state.spec.tensors.map((tensor) => tensor.name)),
      position: { x, y },
      size: { width: TENSOR_WIDTH, height: TENSOR_HEIGHT },
      indices: [],
      metadata: {},
    };
    tensor.indices.push(createIndex(tensor, 0));
    tensor.indices.push(createIndex(tensor, 1));
    return tensor;
  }

  function createIndex(tensor, indexPosition) {
    return {
      id: runtime.makeId("index"),
      name: runtime.nextName("i", tensor.indices.map((index) => index.name)),
      dimension: 2,
      offset: runtime.defaultIndexOffsetForOrder(indexPosition, tensor),
      metadata: {},
    };
  }

  function normalizeSpec(spec) {
    const normalized = runtime.deepClone(spec || {});
    normalized.metadata = runtime.isObject(normalized.metadata) ? normalized.metadata : {};
    normalizeGraphSectionInPlace(normalized);
    normalized.linear_periodic_chain = runtime.isObject(normalized.linear_periodic_chain)
      ? runtime.normalizeLinearPeriodicChainInPlace(normalized.linear_periodic_chain)
      : null;
    if (normalized.linear_periodic_chain) {
      runtime.hydrateActiveLinearPeriodicCell(normalized);
    }
    return normalized;
  }

  function applyColorToSelection(colorValue) {
    ctx.getSelectedEntries().forEach((entry) => {
      if (entry.kind === "tensor") {
        entry.tensor.metadata.color = colorValue;
      } else if (entry.kind === "index") {
        entry.located.index.metadata.color = colorValue;
      } else if (entry.kind === "edge") {
        entry.edge.metadata.color = colorValue;
      } else if (entry.kind === "group") {
        entry.group.metadata.color = colorValue;
      } else if (entry.kind === "note") {
        entry.note.metadata.color = colorValue;
      }
    });
  }

  function getEntryColor(entry) {
    if (entry.kind === "tensor") {
      return runtime.getMetadataColor(entry.tensor.metadata, "#18212c");
    }
    if (entry.kind === "index") {
      return runtime.getMetadataColor(
        entry.located.index.metadata,
        runtime.getIndexColor(entry.located.index, Boolean(findEdgeByIndexId(entry.id)))
      );
    }
    if (entry.kind === "group") {
      return runtime.getMetadataColor(entry.group.metadata, "#61a8ff");
    }
    if (entry.kind === "note") {
      return runtime.getMetadataColor(entry.note.metadata, "#5f95ff");
    }
    return runtime.getMetadataColor(entry.edge.metadata, "#8da1c3");
  }

  function getBatchColorValue(selectedEntries) {
    if (!selectedEntries.length) {
      return "#61a8ff";
    }
    return getEntryColor(selectedEntries[0]);
  }

  return {
    buildEmptyGraphSection,
    clearGraphSectionOnSpec,
    buildGraphSectionFromSpec,
    replaceGraphSectionOnSpec,
    normalizeGraphSectionInPlace,
    buildHistorySnapshotSpec,
    buildSerializedSpec,
    bumpSpecRevision,
    ensureSpecLookups,
    serializeCurrentSpec,
    captureEditableFocus,
    restoreEditableFocus,
    stripImportLines,
    moveIndex,
    removeTensor,
    removeIndex,
    resolveBaseEdgeId,
    removeEdge,
    findTensorById,
    findGroupById,
    findGroupsByTensorId,
    findEdgeById,
    findVisibleIndexOwner,
    findIndexOwner,
    resolveConnectableIndexOwner,
    findEdgeByIndexId,
    syncConnectedIndexDimension,
    createTensor,
    createIndex,
    normalizeSpec,
    applyColorToSelection,
    getBatchColorValue,
    getEntryColor,
  };
}
