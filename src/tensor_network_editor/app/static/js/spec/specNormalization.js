export function createSpecNormalizationBindings({ state, constants, runtime }) {
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
    spec.hyperedges = [];
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
        existingCell && runtime.isObject(existingCell.metadata)
          ? existingCell.metadata
          : {}
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

  function tensorShapesMatch(leftShape, rightShape) {
    if (!Array.isArray(leftShape) || !Array.isArray(rightShape)) {
      return false;
    }
    if (leftShape.length !== rightShape.length) {
      return false;
    }
    return leftShape.every((dimension, position) => dimension === rightShape[position]);
  }

  function normalizeTensorDataDType(tensorData) {
    const dtype = String(tensorData?.dtype || "").trim();
    return ["float32", "float64", "complex64", "complex128"].includes(dtype)
      ? dtype
      : null;
  }

  function withOptionalTensorDataDType(payload, tensorData) {
    const dtype = normalizeTensorDataDType(tensorData);
    if (dtype) {
      payload.dtype = dtype;
    }
    return payload;
  }

  function normalizeTensorScalarNode(value) {
    if (typeof value === "boolean") {
      return null;
    }
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    if (
      runtime.isObject(value) &&
      typeof value.real === "number" &&
      typeof value.imag === "number" &&
      Number.isFinite(value.real) &&
      Number.isFinite(value.imag)
    ) {
      return {
        real: value.real,
        imag: value.imag,
      };
    }
    return null;
  }

  function normalizeTensorLiteralNode(value) {
    const scalarValue = normalizeTensorScalarNode(value);
    if (scalarValue !== null) {
      return {
        normalized: scalarValue,
        shape: [],
      };
    }
    if (!Array.isArray(value)) {
      return null;
    }
    if (!value.length) {
      return null;
    }

    const normalizedChildren = [];
    let childShape = null;
    for (const item of value) {
      const normalizedItem = normalizeTensorLiteralNode(item);
      if (!normalizedItem) {
        return null;
      }
      if (!childShape) {
        childShape = normalizedItem.shape;
      } else if (!tensorShapesMatch(childShape, normalizedItem.shape)) {
        return null;
      }
      normalizedChildren.push(normalizedItem.normalized);
    }

    return {
      normalized: normalizedChildren,
      shape: [normalizedChildren.length, ...(childShape || [])],
    };
  }

  function normalizeTensorDataForShape(tensorData, expectedShape) {
    if (!runtime.isObject(tensorData)) {
      return null;
    }
    const mode = String(tensorData.mode || "").trim();
    if (mode === "zeros") {
      return withOptionalTensorDataDType({ mode: "zeros" }, tensorData);
    }
    if (mode === "ones") {
      return withOptionalTensorDataDType({ mode: "ones" }, tensorData);
    }
    if (mode === "fill") {
      const fillValue = normalizeTensorScalarNode(tensorData.fill_value);
      if (fillValue === null) {
        return null;
      }
      return withOptionalTensorDataDType({
        mode: "fill",
        fill_value: fillValue,
      }, tensorData);
    }
    if (mode === "literal") {
      const normalizedLiteral = normalizeTensorLiteralNode(tensorData.values);
      if (!normalizedLiteral || !tensorShapesMatch(normalizedLiteral.shape, expectedShape)) {
        return null;
      }
      return withOptionalTensorDataDType({
        mode: "literal",
        values: runtime.deepClone(normalizedLiteral.normalized),
      }, tensorData);
    }
    if (mode === "identity") {
      if (
        !Array.isArray(expectedShape) ||
        expectedShape.length !== 2 ||
        expectedShape[0] !== expectedShape[1]
      ) {
        return null;
      }
      return withOptionalTensorDataDType({ mode: "identity" }, tensorData);
    }
    if (mode === "copy") {
      if (
        !Array.isArray(expectedShape) ||
        !expectedShape.length ||
        expectedShape.some((dimension) => dimension !== expectedShape[0])
      ) {
        return null;
      }
      return withOptionalTensorDataDType({ mode: "copy" }, tensorData);
    }
    if (mode === "random") {
      const seed = Number.isInteger(tensorData.seed) && tensorData.seed >= 0
        ? tensorData.seed
        : 0;
      const distribution = String(tensorData.distribution || "normal").trim();
      return withOptionalTensorDataDType({
        mode: "random",
        seed,
        distribution: distribution === "uniform" ? "uniform" : "normal",
      }, tensorData);
    }
    if (mode === "external") {
      const normalizedExternal = {
        mode: "external",
        file_path: String(tensorData.file_path || "").trim(),
      };
      const arrayKey = String(tensorData.array_key || "").trim();
      if (arrayKey) {
        normalizedExternal.array_key = arrayKey;
      }
      return withOptionalTensorDataDType(normalizedExternal, tensorData);
    }
    return null;
  }

  function normalizeContractionPlanInPlace(contractionPlan) {
    if (!runtime.isObject(contractionPlan)) {
      return null;
    }
    contractionPlan.metadata = runtime.isObject(contractionPlan.metadata)
      ? contractionPlan.metadata
      : {};
    contractionPlan.steps = Array.isArray(contractionPlan.steps)
      ? contractionPlan.steps
      : [];
    contractionPlan.view_snapshots = Array.isArray(contractionPlan.view_snapshots)
      ? contractionPlan.view_snapshots
      : [];
    if (!contractionPlan.id) {
      contractionPlan.id = runtime.makeId("plan");
    }
    if (!contractionPlan.name) {
      contractionPlan.name = "Manual path";
    }
    contractionPlan.steps.forEach((step) => {
      step.metadata = runtime.isObject(step.metadata) ? step.metadata : {};
      if (!step.id) {
        step.id = runtime.makeId("step");
      }
      step.left_operand_id = String(step.left_operand_id || "");
      step.right_operand_id = String(step.right_operand_id || "");
    });
    contractionPlan.view_snapshots.forEach((snapshot, snapshotIndex) => {
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
    });
    return contractionPlan;
  }

  function normalizeHyperedgeInPlace(hyperedge, hyperedgePosition) {
    hyperedge.metadata = runtime.isObject(hyperedge.metadata)
      ? hyperedge.metadata
      : {};
    hyperedge.hub_offset = {
      x: runtime.asFiniteNumber(hyperedge.hub_offset && hyperedge.hub_offset.x, 0),
      y: runtime.asFiniteNumber(hyperedge.hub_offset && hyperedge.hub_offset.y, 0),
    };
    hyperedge.endpoints = (Array.isArray(hyperedge.endpoints)
      ? hyperedge.endpoints
      : []
    ).map((endpoint) => {
      const normalizedEndpoint = runtime.isObject(endpoint) ? endpoint : {};
      return {
        tensor_id: String(normalizedEndpoint.tensor_id || ""),
        index_id: String(normalizedEndpoint.index_id || ""),
      };
    });
    if (!hyperedge.id) {
      hyperedge.id = runtime.makeId("hyperedge");
    }
    if (!hyperedge.name) {
      hyperedge.name = `hyperedge_${hyperedgePosition + 1}`;
    }
    return hyperedge;
  }

  function normalizeGraphSectionInPlace(graphSection) {
    graphSection.metadata = runtime.isObject(graphSection.metadata)
      ? graphSection.metadata
      : {};
    graphSection.tensors = Array.isArray(graphSection.tensors)
      ? graphSection.tensors
      : [];
    graphSection.groups = Array.isArray(graphSection.groups)
      ? graphSection.groups
      : [];
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
      tensor.grid_periodic_role =
        tensor.grid_periodic_role === "up" ||
        tensor.grid_periodic_role === "right" ||
        tensor.grid_periodic_role === "down" ||
        tensor.grid_periodic_role === "left"
          ? tensor.grid_periodic_role
          : null;
      tensor.tree_periodic_role =
        tensor.tree_periodic_role === "parent" ||
        tensor.tree_periodic_role === "child"
          ? tensor.tree_periodic_role
          : null;
      tensor.tree_periodic_child_index =
        tensor.tree_periodic_role === "child"
        && Number.isInteger(tensor.tree_periodic_child_index)
        && tensor.tree_periodic_child_index >= 0
          ? tensor.tree_periodic_child_index
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
      tensor.tensor_data = normalizeTensorDataForShape(
        tensor.tensor_data,
        tensor.indices.map((index) => index.dimension)
      );
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
      if (typeof runtime.ensureTensorIndexOffsets === "function") {
        runtime.ensureTensorIndexOffsets(tensor);
      }
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
      normalizeContractionPlanInPlace(graphSection.contraction_plan);
    }

    return graphSection;
  }

  function buildHistorySnapshotSpec(spec = state.spec) {
    const snapshotSpec = runtime.deepClone(spec || {});
    if (runtime.isLinearPeriodicMode(snapshotSpec)) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain(snapshotSpec);
    }
    if (typeof runtime.isGridPeriodicMode === "function" && runtime.isGridPeriodicMode(snapshotSpec)) {
      runtime.syncCurrentGraphIntoGridPeriodicGrid(snapshotSpec);
    }
    if (
      typeof runtime.isTreePeriodicMode === "function" &&
      runtime.isTreePeriodicMode(snapshotSpec)
    ) {
      runtime.syncCurrentGraphIntoTreePeriodicTree(snapshotSpec);
    }
    return snapshotSpec;
  }

  function buildSerializedSpec(spec = state.spec) {
    const serializedSpec = runtime.deepClone(spec || {});
    if (runtime.isLinearPeriodicMode(serializedSpec)) {
      runtime.syncCurrentGraphIntoLinearPeriodicChain(serializedSpec);
      clearGraphSectionOnSpec(serializedSpec);
    }
    if (
      typeof runtime.isGridPeriodicMode === "function" &&
      runtime.isGridPeriodicMode(serializedSpec)
    ) {
      runtime.syncCurrentGraphIntoGridPeriodicGrid(serializedSpec);
      clearGraphSectionOnSpec(serializedSpec);
    }
    if (
      typeof runtime.isTreePeriodicMode === "function" &&
      runtime.isTreePeriodicMode(serializedSpec)
    ) {
      runtime.syncCurrentGraphIntoTreePeriodicTree(serializedSpec);
      clearGraphSectionOnSpec(serializedSpec);
    }
    return serializedSpec;
  }

  function normalizeSpec(spec) {
    const normalized = runtime.deepClone(spec || {});
    normalized.metadata = runtime.isObject(normalized.metadata)
      ? normalized.metadata
      : {};
    normalizeGraphSectionInPlace(normalized);
    normalized.hyperedges = Array.isArray(normalized.hyperedges)
      ? normalized.hyperedges
      : [];
    normalized.hyperedges.forEach((hyperedge, hyperedgePosition) => {
      normalizeHyperedgeInPlace(hyperedge, hyperedgePosition);
    });
    normalized.linear_periodic_chain = runtime.isObject(normalized.linear_periodic_chain)
      ? runtime.normalizeLinearPeriodicChainInPlace(normalized.linear_periodic_chain)
      : null;
    normalized.grid_periodic_grid =
      typeof runtime.normalizeGridPeriodicGridInPlace === "function" &&
      runtime.isObject(normalized.grid_periodic_grid)
        ? runtime.normalizeGridPeriodicGridInPlace(normalized.grid_periodic_grid)
        : null;
    normalized.tree_periodic_tree =
      typeof runtime.normalizeTreePeriodicTreeInPlace === "function" &&
      runtime.isObject(normalized.tree_periodic_tree)
        ? runtime.normalizeTreePeriodicTreeInPlace(normalized.tree_periodic_tree)
        : null;
    if (normalized.tree_periodic_tree) {
      normalized.linear_periodic_chain = null;
      normalized.grid_periodic_grid = null;
      runtime.hydrateActiveTreePeriodicCell(normalized);
      return normalized;
    }
    if (normalized.grid_periodic_grid) {
      normalized.linear_periodic_chain = null;
      normalized.tree_periodic_tree = null;
      runtime.hydrateActiveGridPeriodicCell(normalized);
      return normalized;
    }
    if (normalized.linear_periodic_chain) {
      normalized.tree_periodic_tree = null;
      normalized.grid_periodic_grid = null;
      runtime.hydrateActiveLinearPeriodicCell(normalized);
    }
    return normalized;
  }

  return {
    buildEmptyGraphSection,
    clearGraphSectionOnSpec,
    buildGraphSectionFromSpec,
    replaceGraphSectionOnSpec,
    normalizeContractionPlanInPlace,
    normalizeGraphSectionInPlace,
    buildHistorySnapshotSpec,
    buildSerializedSpec,
    normalizeSpec,
  };
}
