const BENCHMARK_BASE_LABEL = "Tensor network";
const BENCHMARK_STATUS_HINT = "Move right to edit or create a contraction scheme.";
const BENCHMARK_METRICS = [
  {
    key: "flop",
    label: "FLOP",
    summaryKey: "total_estimated_flops",
    formatDisplay: (value) => formatBenchmarkNumber(value),
  },
  {
    key: "mac",
    label: "MAC",
    summaryKey: "total_estimated_macs",
    formatDisplay: (value) => formatBenchmarkNumber(value),
  },
  {
    key: "peak",
    label: "Peak",
    summaryKey: "peak_intermediate_size",
    formatDisplay: (value) => formatBenchmarkNumber(value),
  },
  {
    key: "memory",
    label: "Peak Memory",
    summaryKey: "peak_intermediate_bytes",
    formatDisplay: (value) => formatBenchmarkBytes(value),
  },
];

const BENCHMARK_COMPARE_COLUMNS = [
  {
    key: "name",
    label: "Name",
    getDisplay: (row) => row?.cells?.name?.display || row?.schemeName || "-",
  },
  ...BENCHMARK_METRICS.map((metric) => ({
    key: metric.key,
    label: metric.label,
    getDisplay: (row) => row?.cells?.[metric.key]?.display || "-",
  })),
];

function createEmptyBenchmarkCompareState() {
  return {
    open: false,
    loading: false,
    errorMessage: "",
    tableModel: null,
    rows: [],
    activeRequestId: 0,
  };
}

function createEmptyBenchmarkSession() {
  return {
    enabled: false,
    activePosition: 0,
    originalPlan: null,
    schemes: [],
    compareModal: createEmptyBenchmarkCompareState(),
  };
}

function formatBenchmarkNumber(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toLocaleString() : "-";
}

function formatBenchmarkBytes(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? `${numericValue.toLocaleString()} bytes`
    : "-";
}

function isComparableBenchmarkAnalysis(analysis) {
  return Boolean(
    analysis &&
      typeof analysis === "object" &&
      analysis.status === "complete" &&
      analysis.summary &&
      typeof analysis.summary === "object"
  );
}

function buildBenchmarkMetricCell(metric, analysis) {
  const comparable = isComparableBenchmarkAnalysis(analysis);
  const summary = comparable ? analysis.summary : null;
  const value = comparable ? Number(summary[metric.summaryKey]) : NaN;
  const isComparable = comparable && Number.isFinite(value);
  return {
    value: isComparable ? value : null,
    display: isComparable ? metric.formatDisplay(value) : "-",
    isComparable,
    isBest: false,
    isWorst: false,
  };
}

export function buildBenchmarkCompareTableModel(entries = []) {
  const rows = (Array.isArray(entries) ? entries : []).map((entry, index) => {
    const schemeId =
      typeof entry?.scheme_id === "string" && entry.scheme_id
        ? entry.scheme_id
        : `scheme_${index + 1}`;
    const schemeName =
      typeof entry?.scheme_name === "string" && entry.scheme_name.trim()
        ? entry.scheme_name.trim()
        : `Scheme ${index + 1}`;
    const analysis =
      entry?.analysis && typeof entry.analysis === "object" ? entry.analysis : null;
    const cells = {
      name: {
        value: schemeName,
        display: schemeName,
        isComparable: false,
        isBest: false,
        isWorst: false,
      },
    };

    BENCHMARK_METRICS.forEach((metric) => {
      cells[metric.key] = buildBenchmarkMetricCell(metric, analysis);
    });

    return {
      schemeId,
      schemeName,
      analysisStatus:
        typeof analysis?.status === "string" && analysis.status
          ? analysis.status
          : "unknown",
      analysisMessage:
        typeof analysis?.message === "string" ? analysis.message : "",
      cells,
    };
  });

  BENCHMARK_METRICS.forEach((metric) => {
    const comparableCells = rows
      .map((row) => row.cells[metric.key])
      .filter((cell) => cell && cell.isComparable);
    if (!comparableCells.length) {
      return;
    }
    const values = comparableCells.map((cell) => cell.value);
    const bestValue = Math.min(...values);
    const worstValue = Math.max(...values);
    comparableCells.forEach((cell) => {
      cell.isBest = cell.value === bestValue;
      cell.isWorst = bestValue === worstValue ? false : cell.value === worstValue;
    });
  });

  return {
    rows,
  };
}

function getBenchmarkCompareExportRows(tableModel) {
  return Array.isArray(tableModel?.rows) ? tableModel.rows : [];
}

function buildBenchmarkCompareExportMatrix(tableModel) {
  const headers = BENCHMARK_COMPARE_COLUMNS.map((column) => column.label);
  const rows = getBenchmarkCompareExportRows(tableModel).map((row) =>
    BENCHMARK_COMPARE_COLUMNS.map((column) => column.getDisplay(row))
  );
  return [headers, ...rows];
}

function escapeBenchmarkCompareCsvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text)
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

function escapeBenchmarkCompareLatex(value) {
  return String(value ?? "")
    .replaceAll("\\", "\\textbackslash{}")
    .replaceAll("&", "\\&")
    .replaceAll("%", "\\%")
    .replaceAll("$", "\\$")
    .replaceAll("#", "\\#")
    .replaceAll("_", "\\_")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}")
    .replaceAll("~", "\\textasciitilde{}")
    .replaceAll("^", "\\textasciicircum{}");
}

export function serializeBenchmarkCompareTableCsv(tableModel) {
  return buildBenchmarkCompareExportMatrix(tableModel)
    .map((row) => row.map((value) => escapeBenchmarkCompareCsvCell(value)).join(","))
    .join("\n");
}

export function serializeBenchmarkCompareTableText(tableModel) {
  const matrix = buildBenchmarkCompareExportMatrix(tableModel);
  if (!matrix.length) {
    return "";
  }
  const columnWidths = matrix[0].map((_, columnIndex) =>
    Math.max(
      ...matrix.map((row) => String(row[columnIndex] ?? "").length)
    )
  );
  const formatRow = (row) =>
    row
      .map((value, columnIndex) =>
        String(value ?? "").padEnd(columnWidths[columnIndex], " ")
      )
      .join("  ");
  const separator = columnWidths.map((width) => "-".repeat(width)).join("  ");
  return [formatRow(matrix[0]), separator, ...matrix.slice(1).map(formatRow)].join(
    "\n"
  );
}

export function serializeBenchmarkCompareTableLatex(tableModel) {
  const [headerRow, ...rows] = buildBenchmarkCompareExportMatrix(tableModel);
  const formatRow = (row) =>
    `${row.map((value) => escapeBenchmarkCompareLatex(value)).join(" & ")} \\\\`;
  return [
    "\\begin{tabular}{lrrrr}",
    "\\hline",
    formatRow(headerRow),
    "\\hline",
    ...rows.map(formatRow),
    "\\hline",
    "\\end{tabular}",
  ].join("\n");
}

export function createBenchmarkSessionSupport({
  actions,
  state,
  dom,
  runtime,
}) {
  const ctx = actions;

  function normalizeCompareModalInPlace(compareModal) {
    const nextCompareModal =
      runtime.isObject(compareModal) ? compareModal : createEmptyBenchmarkCompareState();
    const defaults = createEmptyBenchmarkCompareState();
    nextCompareModal.open = Boolean(nextCompareModal.open);
    nextCompareModal.loading = Boolean(nextCompareModal.loading);
    nextCompareModal.errorMessage =
      typeof nextCompareModal.errorMessage === "string"
        ? nextCompareModal.errorMessage
        : defaults.errorMessage;
    nextCompareModal.tableModel =
      nextCompareModal.tableModel && runtime.isObject(nextCompareModal.tableModel)
        ? nextCompareModal.tableModel
        : null;
    nextCompareModal.rows = Array.isArray(nextCompareModal.rows)
      ? nextCompareModal.rows
      : nextCompareModal.tableModel &&
          Array.isArray(nextCompareModal.tableModel.rows)
        ? nextCompareModal.tableModel.rows
        : [];
    nextCompareModal.activeRequestId = Number.isInteger(nextCompareModal.activeRequestId)
      ? nextCompareModal.activeRequestId
      : defaults.activeRequestId;
    return nextCompareModal;
  }

  function buildNormalizedPlanLike(
    plan,
    {
      fallbackId = runtime.makeId("plan"),
      fallbackName = "Manual path",
      fallbackMetadata = {},
    } = {}
  ) {
    const nextPlan = runtime.isObject(plan) ? runtime.deepClone(plan) : {};
    const providedId =
      typeof nextPlan.id === "string" && nextPlan.id ? nextPlan.id : "";
    const providedName =
      typeof nextPlan.name === "string" && nextPlan.name.trim()
        ? nextPlan.name.trim()
        : "";
    if (typeof runtime.normalizeContractionPlanInPlace === "function") {
      runtime.normalizeContractionPlanInPlace(nextPlan);
    } else {
      nextPlan.steps = Array.isArray(nextPlan.steps) ? nextPlan.steps : [];
      nextPlan.view_snapshots = Array.isArray(nextPlan.view_snapshots)
        ? nextPlan.view_snapshots
        : [];
      nextPlan.metadata = runtime.isObject(nextPlan.metadata) ? nextPlan.metadata : {};
    }
    nextPlan.id = providedId || nextPlan.id || fallbackId;
    nextPlan.name = providedName || fallbackName;
    nextPlan.metadata = runtime.isObject(nextPlan.metadata)
      ? nextPlan.metadata
      : runtime.deepClone(fallbackMetadata);
    return nextPlan;
  }

  function buildDefaultBenchmarkSchemeName(index) {
    return `Scheme ${index + 1}`;
  }

  function buildNormalizedBenchmarkScheme(plan, index, seed = null) {
    const fallbackName =
      typeof seed?.name === "string" && seed.name.trim()
        ? seed.name.trim()
        : typeof plan?.name === "string" && plan.name.trim()
          ? plan.name.trim()
          : buildDefaultBenchmarkSchemeName(index);
    const fallbackId =
      typeof seed?.id === "string" && seed.id
        ? seed.id
        : typeof plan?.id === "string" && plan.id
          ? plan.id
          : runtime.makeId("plan");
    const fallbackMetadata = runtime.isObject(seed?.metadata)
      ? seed.metadata
      : runtime.isObject(plan?.metadata)
        ? plan.metadata
        : {};
    return buildNormalizedPlanLike(plan, {
      fallbackId,
      fallbackName,
      fallbackMetadata,
    });
  }

  function buildNormalizedOriginalPlan(plan) {
    const fallbackName =
      typeof plan?.name === "string" && plan.name.trim() ? plan.name.trim() : "Manual path";
    const fallbackId =
      typeof plan?.id === "string" && plan.id ? plan.id : runtime.makeId("plan");
    const fallbackMetadata = runtime.isObject(plan?.metadata) ? plan.metadata : {};
    return buildNormalizedPlanLike(plan, {
      fallbackId,
      fallbackName,
      fallbackMetadata,
    });
  }

  function normalizeBenchmarkSessionInPlace(session) {
    const nextSession =
      runtime.isObject(session) ? session : createEmptyBenchmarkSession();
    nextSession.enabled = Boolean(nextSession.enabled);
    nextSession.activePosition = Number.isInteger(nextSession.activePosition)
      ? Math.max(0, nextSession.activePosition)
      : 0;
    nextSession.originalPlan = nextSession.originalPlan
      ? buildNormalizedOriginalPlan(nextSession.originalPlan)
      : null;
    nextSession.schemes = Array.isArray(nextSession.schemes)
      ? nextSession.schemes.map((scheme, index) =>
          buildNormalizedBenchmarkScheme(scheme, index)
        )
      : [];
    nextSession.compareModal = normalizeCompareModalInPlace(nextSession.compareModal);
    if (!nextSession.enabled) {
      nextSession.activePosition = 0;
    } else if (nextSession.activePosition > nextSession.schemes.length) {
      nextSession.activePosition = nextSession.schemes.length;
    }
    return nextSession;
  }

  function getBenchmarkSession() {
    state.benchmarkSession = normalizeBenchmarkSessionInPlace(state.benchmarkSession);
    return state.benchmarkSession;
  }

  function setBenchmarkSession(session) {
    state.benchmarkSession = normalizeBenchmarkSessionInPlace(session);
    return state.benchmarkSession;
  }

  function resetBenchmarkCompareState(preserveOpen = false) {
    const session = getBenchmarkSession();
    session.compareModal = createEmptyBenchmarkCompareState();
    session.compareModal.open = preserveOpen;
    return session.compareModal;
  }

  function isBenchmarkMode() {
    return Boolean(getBenchmarkSession().enabled);
  }

  function isBenchmarkBasePosition() {
    const session = getBenchmarkSession();
    return Boolean(session.enabled && session.activePosition === 0);
  }

  function getBenchmarkSchemeCount() {
    return getBenchmarkSession().schemes.length;
  }

  function getBenchmarkActiveSchemeIndex() {
    const session = getBenchmarkSession();
    return session.enabled && session.activePosition > 0
      ? session.activePosition - 1
      : -1;
  }

  function getActiveBenchmarkScheme() {
    const activeSchemeIndex = getBenchmarkActiveSchemeIndex();
    return activeSchemeIndex >= 0
      ? getBenchmarkSession().schemes[activeSchemeIndex] || null
      : null;
  }

  function getBenchmarkBaseLabel() {
    return BENCHMARK_BASE_LABEL;
  }

  function getBenchmarkSchemeName(index) {
    return buildDefaultBenchmarkSchemeName(index);
  }

  function hasHyperedges() {
    return Boolean(Array.isArray(state.spec?.hyperedges) && state.spec.hyperedges.length);
  }

  function canOpenBenchmarkCompare() {
    const session = getBenchmarkSession();
    return Boolean(session.enabled && session.schemes.length && !hasHyperedges());
  }

  function getBenchmarkCompareTableModel(compareModal = getBenchmarkSession().compareModal) {
    return compareModal?.tableModel && runtime.isObject(compareModal.tableModel)
      ? compareModal.tableModel
      : { rows: [] };
  }

  function canExportBenchmarkCompare(compareModal = getBenchmarkSession().compareModal) {
    return Boolean(
      compareModal?.open &&
        !compareModal.loading &&
        !compareModal.errorMessage &&
        getBenchmarkCompareExportRows(getBenchmarkCompareTableModel(compareModal)).length
    );
  }

  function getBenchmarkCompareFilename(extension) {
    const baseName = runtime.sanitizeFilename(state.spec?.name || "tensor-network");
    return `${baseName}-benchmark-compare.${extension}`;
  }

  function resolveBenchmarkCompareTextDownloader() {
    if (typeof ctx.downloadText === "function") {
      return ctx.downloadText.bind(ctx);
    }
    const blobCtor =
      typeof ctx.window?.Blob === "function"
        ? ctx.window.Blob
        : typeof Blob === "function"
          ? Blob
          : null;
    if (typeof ctx.downloadBlob === "function" && blobCtor) {
      return (filename, text, contentType = "text/plain;charset=utf-8") =>
        ctx.downloadBlob(filename, new blobCtor([text], { type: contentType }));
    }
    return null;
  }

  function syncBenchmarkCompareExportButtons(compareModal = getBenchmarkSession().compareModal) {
    const disabled = !canExportBenchmarkCompare(compareModal);
    if (dom.benchmarkCompareExportCsvButton) {
      dom.benchmarkCompareExportCsvButton.disabled = disabled;
    }
    if (dom.benchmarkCompareExportTextButton) {
      dom.benchmarkCompareExportTextButton.disabled = disabled;
    }
    if (dom.benchmarkCompareCopyLatexButton) {
      dom.benchmarkCompareCopyLatexButton.disabled = disabled;
    }
  }

  function getBenchmarkNextButtonLabel() {
    const session = getBenchmarkSession();
    if (!session.enabled) {
      return ">";
    }
    return session.activePosition >= session.schemes.length ? "+" : ">";
  }

  function clearBenchmarkTransientEditorState() {
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerInspectionStepCount = null;
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
    state.activeTensorDrag = null;
    state.activeIndexDrag = null;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.boxSelection = null;
    state.connectMode = false;
  }

  function syncActiveBenchmarkScheme() {
    const session = getBenchmarkSession();
    if (!session.enabled || !state.spec) {
      return null;
    }
    const activeSchemeIndex = getBenchmarkActiveSchemeIndex();
    if (activeSchemeIndex < 0) {
      state.spec.contraction_plan = null;
      return null;
    }
    const existingScheme =
      session.schemes[activeSchemeIndex] || buildNormalizedBenchmarkScheme(null, activeSchemeIndex);
    const livePlan = state.spec.contraction_plan || existingScheme;
    const nextScheme = buildNormalizedBenchmarkScheme(
      livePlan,
      activeSchemeIndex,
      existingScheme
    );
    session.schemes[activeSchemeIndex] = nextScheme;
    state.spec.contraction_plan = nextScheme;
    return nextScheme;
  }

  function repairBenchmarkSchemeAt(index) {
    const session = getBenchmarkSession();
    const seedScheme =
      session.schemes[index] || buildNormalizedBenchmarkScheme(null, index);
    if (!state.spec || typeof ctx.repairContractionPlan !== "function") {
      const normalizedScheme = buildNormalizedBenchmarkScheme(seedScheme, index, seedScheme);
      session.schemes[index] = normalizedScheme;
      if (session.activePosition === index + 1) {
        state.spec.contraction_plan = normalizedScheme;
      }
      return normalizedScheme;
    }

    const previousPlan = state.spec.contraction_plan;
    const previousPlannerState = {
      pendingPlannerOperandId: state.pendingPlannerOperandId,
      pendingPlannerSelectionId: state.pendingPlannerSelectionId,
      plannerInspectionStepCount: state.plannerInspectionStepCount,
      plannerPreviewMode: state.plannerPreviewMode,
      plannerPreviewBadgeDisclosure: runtime.deepClone(
        state.plannerPreviewBadgeDisclosure || {}
      ),
      plannerFutureBadgeDisclosure: runtime.deepClone(
        state.plannerFutureBadgeDisclosure || {}
      ),
    };
    let repairedPlan = null;

    try {
      state.spec.contraction_plan = buildNormalizedBenchmarkScheme(
        seedScheme,
        index,
        seedScheme
      );
      ctx.repairContractionPlan();
      repairedPlan = state.spec.contraction_plan
        ? buildNormalizedBenchmarkScheme(
            state.spec.contraction_plan,
            index,
            seedScheme
          )
        : buildNormalizedPlanLike(
            {
              id: seedScheme.id,
              name: seedScheme.name,
              steps: [],
              view_snapshots: [],
              metadata: runtime.deepClone(seedScheme.metadata || {}),
            },
            {
              fallbackId: seedScheme.id,
              fallbackName: seedScheme.name,
              fallbackMetadata: seedScheme.metadata || {},
            }
          );
      session.schemes[index] = repairedPlan;
    } finally {
      state.spec.contraction_plan = previousPlan;
      state.pendingPlannerOperandId = previousPlannerState.pendingPlannerOperandId;
      state.pendingPlannerSelectionId = previousPlannerState.pendingPlannerSelectionId;
      state.plannerInspectionStepCount =
        previousPlannerState.plannerInspectionStepCount;
      state.plannerPreviewMode = previousPlannerState.plannerPreviewMode;
      state.plannerPreviewBadgeDisclosure =
        previousPlannerState.plannerPreviewBadgeDisclosure;
      state.plannerFutureBadgeDisclosure =
        previousPlannerState.plannerFutureBadgeDisclosure;
    }
    if (session.activePosition === index + 1) {
      state.spec.contraction_plan = repairedPlan;
    }
    return repairedPlan;
  }

  function projectBenchmarkPosition(position) {
    const session = getBenchmarkSession();
    session.activePosition = runtime.clamp(position, 0, session.schemes.length);
    if (!state.spec) {
      return null;
    }
    if (session.activePosition === 0) {
      state.spec.contraction_plan = null;
      return null;
    }
    const repairedScheme = repairBenchmarkSchemeAt(session.activePosition - 1);
    state.spec.contraction_plan = repairedScheme;
    session.schemes[session.activePosition - 1] = repairedScheme;
    return repairedScheme;
  }

  function syncBenchmarkProjection(statusMessage = "", statusKind = "success") {
    clearBenchmarkTransientEditorState();
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
    if (typeof runtime.reconcileTensorOrder === "function") {
      runtime.reconcileTensorOrder();
    }
    if (typeof ctx.render === "function") {
      ctx.render();
    }
    if (typeof ctx.renderPlanner === "function") {
      ctx.renderPlanner();
    }
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis({ immediate: true });
    }
    if (typeof runtime.updateToolbarState === "function") {
      runtime.updateToolbarState();
    }
    if (statusMessage && typeof ctx.setStatus === "function") {
      ctx.setStatus(statusMessage, statusKind);
    }
  }

  function setBenchmarkMode(enabled) {
    const shouldEnable = Boolean(enabled);
    const session = getBenchmarkSession();
    if (shouldEnable === Boolean(session.enabled)) {
      return session.enabled;
    }
    if (!state.spec) {
      return false;
    }
    if (
      shouldEnable &&
      ((typeof runtime.isForMode === "function" && runtime.isForMode()) ||
        (typeof runtime.isLinearPeriodicMode === "function" &&
          runtime.isLinearPeriodicMode()) ||
        (typeof runtime.isGridPeriodicMode === "function" &&
          runtime.isGridPeriodicMode()))
    ) {
      if (typeof ctx.setStatus === "function") {
        ctx.setStatus(
          "Benchmark mode is unavailable while a For mode is active.",
          "error"
        );
      }
      return false;
    }
    if (shouldEnable && hasHyperedges()) {
      if (typeof ctx.setStatus === "function") {
        ctx.setStatus(
          "Benchmark mode is unavailable while the design contains hyperedges.",
          "error"
        );
      }
      return false;
    }

    if (shouldEnable) {
      const importedPlan = state.spec.contraction_plan
        ? buildNormalizedOriginalPlan(state.spec.contraction_plan)
        : null;
      const nextSession = createEmptyBenchmarkSession();
      nextSession.enabled = true;
      nextSession.originalPlan = importedPlan
        ? runtime.deepClone(importedPlan)
        : null;
      nextSession.schemes = importedPlan
        ? [buildNormalizedBenchmarkScheme(importedPlan, 0, importedPlan)]
        : [];
      state.spec.contraction_plan = null;
      setBenchmarkSession(nextSession);
      syncBenchmarkProjection(
        "Benchmark mode enabled. Edit the tensor network here, then move right to manage schemes."
      );
      return true;
    }

    syncActiveBenchmarkScheme();
    const activeScheme = getActiveBenchmarkScheme();
    const restoredPlan =
      session.activePosition > 0 && activeScheme
        ? buildNormalizedBenchmarkScheme(activeScheme, session.activePosition - 1, activeScheme)
        : session.originalPlan
          ? buildNormalizedOriginalPlan(session.originalPlan)
          : null;
    const exitMessage =
      session.activePosition > 0 && activeScheme
        ? "Benchmark mode disabled. Kept the active contraction scheme."
        : restoredPlan
          ? "Benchmark mode disabled. Restored the original contraction scheme."
          : "Benchmark mode disabled. Restored the tensor network without a manual scheme.";
    state.spec.contraction_plan = restoredPlan;
    setBenchmarkSession(createEmptyBenchmarkSession());
    syncBenchmarkProjection(exitMessage);
    return false;
  }

  function toggleBenchmarkMode() {
    return setBenchmarkMode(!isBenchmarkMode());
  }

  function switchBenchmarkPosition(direction) {
    const session = getBenchmarkSession();
    if (!session.enabled || !Number.isInteger(direction) || direction === 0) {
      return session.activePosition;
    }

    syncActiveBenchmarkScheme();

    let nextPosition = session.activePosition;
    if (direction > 0) {
      if (session.activePosition < session.schemes.length) {
        nextPosition = session.activePosition + 1;
      } else {
        const nextSchemeIndex = session.schemes.length;
        session.schemes.push(buildNormalizedBenchmarkScheme(null, nextSchemeIndex));
        nextPosition = session.schemes.length;
      }
    } else {
      nextPosition = Math.max(0, session.activePosition - 1);
    }

    if (nextPosition === session.activePosition) {
      if (typeof runtime.updateToolbarState === "function") {
        runtime.updateToolbarState();
      }
      return session.activePosition;
    }

    projectBenchmarkPosition(nextPosition);
    const activePositionLabel =
      nextPosition === 0
        ? BENCHMARK_BASE_LABEL
        : getBenchmarkSession().schemes[nextPosition - 1].name;
    syncBenchmarkProjection(`Editing ${activePositionLabel}.`);
    return nextPosition;
  }

  function renameActiveBenchmarkScheme(name) {
    if (!isBenchmarkMode() || isBenchmarkBasePosition() || !state.spec) {
      return null;
    }
    const activeSchemeIndex = getBenchmarkActiveSchemeIndex();
    const activeScheme = syncActiveBenchmarkScheme();
    if (!activeScheme) {
      return null;
    }
    const nextName =
      typeof name === "string" && name.trim()
        ? name.trim()
        : buildDefaultBenchmarkSchemeName(activeSchemeIndex);
    activeScheme.name = nextName;
    getBenchmarkSession().schemes[activeSchemeIndex] = activeScheme;
    if (typeof runtime.updateToolbarState === "function") {
      runtime.updateToolbarState();
    }
    return nextName;
  }

  function buildBenchmarkCompareBodyHtml(compareModal) {
    if (!dom.benchmarkCompareTableBody) {
      return "";
    }
    if (!compareModal.open) {
      return "";
    }
    if (compareModal.loading) {
      return `
        <tr>
          <td colspan="5" class="benchmark-compare-status-cell">
            Calculating metrics...
          </td>
        </tr>
      `;
    }
    if (compareModal.errorMessage) {
      return `
        <tr>
          <td colspan="5" class="benchmark-compare-status-cell benchmark-compare-status-error">
            ${runtime.escapeHtml(compareModal.errorMessage)}
          </td>
        </tr>
      `;
    }
    const tableModel =
      compareModal.tableModel && runtime.isObject(compareModal.tableModel)
        ? compareModal.tableModel
        : { rows: [] };
    if (!Array.isArray(tableModel.rows) || !tableModel.rows.length) {
      return `
        <tr>
          <td colspan="5" class="benchmark-compare-status-cell">
            No saved schemes yet.
          </td>
        </tr>
      `;
    }
    return tableModel.rows
      .map((row) => {
        const schemeStatusClass =
          row.analysisStatus === "complete"
            ? ""
            : " benchmark-compare-name-cell-muted";
        const renderMetricCell = (metricKey) => {
          const cell = row.cells[metricKey];
          const metricClass = [
            "benchmark-compare-cell",
            cell.isBest ? "is-best" : "",
            cell.isWorst ? "is-worst" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return `
            <td class="${metricClass}">
              ${runtime.escapeHtml(cell.display)}
            </td>
          `;
        };
        return `
          <tr data-benchmark-scheme-id="${runtime.escapeHtml(row.schemeId)}">
            <th scope="row" class="benchmark-compare-name-cell${schemeStatusClass}">
              ${runtime.escapeHtml(row.schemeName)}
            </th>
            ${renderMetricCell("flop")}
            ${renderMetricCell("mac")}
            ${renderMetricCell("peak")}
            ${renderMetricCell("memory")}
          </tr>
        `;
      })
      .join("");
  }

  function syncBenchmarkCompareModalState() {
    const session = getBenchmarkSession();
    const compareModal = session.compareModal;
    const isOpen = Boolean(session.enabled && compareModal.open);
    if (dom.benchmarkCompareModal?.classList) {
      dom.benchmarkCompareModal.classList.toggle("is-hidden", !isOpen);
    }
    if (dom.benchmarkCompareModal) {
      dom.benchmarkCompareModal.hidden = !isOpen;
    }
    if (dom.benchmarkCompareTableBody) {
      dom.benchmarkCompareTableBody.innerHTML = buildBenchmarkCompareBodyHtml(compareModal);
    }
    syncBenchmarkCompareExportButtons(compareModal);
  }

  function exportBenchmarkCompareTable(format) {
    const compareModal = getBenchmarkSession().compareModal;
    if (!canExportBenchmarkCompare(compareModal)) {
      ctx.setStatus(
        "Open a benchmark comparison with analyzed schemes before exporting.",
        "error"
      );
      return false;
    }
    const downloadText = resolveBenchmarkCompareTextDownloader();
    if (typeof downloadText !== "function") {
      ctx.setStatus("This browser cannot export comparison tables yet.", "error");
      return false;
    }
    const tableModel = getBenchmarkCompareTableModel(compareModal);
    if (format === "csv") {
      downloadText(
        getBenchmarkCompareFilename("csv"),
        serializeBenchmarkCompareTableCsv(tableModel),
        "text/csv;charset=utf-8"
      );
      ctx.setStatus("Downloaded the benchmark comparison as CSV.", "success");
      return true;
    }
    if (format === "txt") {
      downloadText(
        getBenchmarkCompareFilename("txt"),
        serializeBenchmarkCompareTableText(tableModel),
        "text/plain;charset=utf-8"
      );
      ctx.setStatus("Downloaded the benchmark comparison as text.", "success");
      return true;
    }
    ctx.setStatus("Unknown benchmark export format.", "error");
    return false;
  }

  async function copyBenchmarkCompareAsLatex() {
    const compareModal = getBenchmarkSession().compareModal;
    if (!canExportBenchmarkCompare(compareModal)) {
      ctx.setStatus(
        "Open a benchmark comparison with analyzed schemes before copying it.",
        "error"
      );
      return false;
    }
    const copyText =
      typeof ctx.copyText === "function"
        ? ctx.copyText.bind(ctx)
        : async (text) => {
            const clipboard = ctx.window?.navigator?.clipboard || null;
            if (!clipboard || typeof clipboard.writeText !== "function") {
              throw new Error("Clipboard access is not available in this browser.");
            }
            await clipboard.writeText(text);
          };
    try {
      await copyText(
        serializeBenchmarkCompareTableLatex(getBenchmarkCompareTableModel(compareModal))
      );
      ctx.setStatus("Copied the benchmark comparison as LaTeX.", "success");
      return true;
    } catch (error) {
      ctx.setStatus(
        `Could not copy the LaTeX table: ${error.message}`,
        "error"
      );
      return false;
    }
  }

  function exportBenchmarkCompareAsCsv() {
    return exportBenchmarkCompareTable("csv");
  }

  function exportBenchmarkCompareAsText() {
    return exportBenchmarkCompareTable("txt");
  }

  function closeBenchmarkCompareModal() {
    const compareModal = getBenchmarkSession().compareModal;
    compareModal.open = false;
    compareModal.loading = false;
    compareModal.activeRequestId += 1;
    syncBenchmarkCompareModalState();
    if (typeof runtime.updateToolbarState === "function") {
      runtime.updateToolbarState();
    }
    return false;
  }

  function isBenchmarkCompareModalOpen() {
    return Boolean(getBenchmarkSession().compareModal.open);
  }

  async function analyzeBenchmarkScheme(baseSpec, scheme, schemeIndex) {
    const repairedScheme = repairBenchmarkSchemeAt(schemeIndex);
    if (
      !ctx.apiPost ||
      typeof ctx.apiPost !== "function" ||
      state.schemaVersion == null
    ) {
      return {
        scheme_id: repairedScheme.id,
        scheme_name: repairedScheme.name,
        analysis: {
          status: "incomplete",
          message: "Benchmark comparison is unavailable in this session.",
        },
      };
    }
    const networkSpec = runtime.deepClone(baseSpec || state.spec || {});
    networkSpec.contraction_plan = runtime.deepClone(repairedScheme);
    try {
      const payload = await ctx.apiPost("/api/analyze-contraction", {
        spec: {
          schema_version: state.schemaVersion,
          network: networkSpec,
        },
      });
      if (!payload || payload.ok !== true) {
        return {
          scheme_id: repairedScheme.id,
          scheme_name: repairedScheme.name,
          analysis: {
            status: "failed",
            message:
              typeof payload?.message === "string" && payload.message
                ? payload.message
                : "The scheme could not be analyzed.",
          },
        };
      }
      return {
        scheme_id: repairedScheme.id,
        scheme_name: repairedScheme.name,
        analysis:
          payload.manual && typeof payload.manual === "object"
            ? payload.manual
            : {
                status: "incomplete",
                message: "No manual analysis was returned.",
              },
      };
    } catch (error) {
      return {
        scheme_id: repairedScheme.id,
        scheme_name: repairedScheme.name,
        analysis: {
          status: "failed",
          message:
            error && typeof error.message === "string"
              ? error.message
              : "The scheme could not be analyzed.",
        },
      };
    }
  }

  async function openBenchmarkCompareModal() {
    const session = getBenchmarkSession();
    if (hasHyperedges()) {
      ctx.setStatus(
        "Benchmark comparison is unavailable while the design contains hyperedges.",
        "error"
      );
      return null;
    }
    if (!session.enabled || !session.schemes.length) {
      return null;
    }
    syncActiveBenchmarkScheme();
    const previousRequestId = session.compareModal.activeRequestId;
    const compareModal = resetBenchmarkCompareState(true);
    compareModal.loading = true;
    compareModal.activeRequestId = previousRequestId + 1;
    const requestId = compareModal.activeRequestId;
    syncBenchmarkCompareModalState();

    const baseSpec = runtime.buildSerializedSpec
      ? runtime.buildSerializedSpec()
      : runtime.deepClone(state.spec || {});
    const benchmarkEntries = await Promise.all(
      session.schemes.map((scheme, index) =>
        analyzeBenchmarkScheme(baseSpec, scheme, index)
      )
    );

    const latestSession = getBenchmarkSession();
    const latestCompareModal = latestSession.compareModal;
    if (
      !latestSession.enabled ||
      latestCompareModal.activeRequestId !== requestId
    ) {
      return null;
    }

    latestCompareModal.loading = false;
    latestCompareModal.errorMessage = "";
    latestCompareModal.tableModel = buildBenchmarkCompareTableModel(benchmarkEntries);
    latestCompareModal.rows = latestCompareModal.tableModel.rows;
    syncBenchmarkCompareModalState();
    return latestCompareModal.tableModel;
  }

  return {
    getBenchmarkSession,
    getBenchmarkSchemeCount,
    getActiveBenchmarkScheme,
    getBenchmarkBaseLabel,
    getBenchmarkSchemeName,
    getBenchmarkNextButtonLabel,
    canOpenBenchmarkCompare,
    isBenchmarkMode,
    isBenchmarkBasePosition,
    isBenchmarkCompareModalOpen,
    syncActiveBenchmarkScheme,
    repairBenchmarkSchemeAt,
    setBenchmarkMode,
    toggleBenchmarkMode,
    switchBenchmarkPosition,
    renameActiveBenchmarkScheme,
    openBenchmarkCompareModal,
    closeBenchmarkCompareModal,
    syncBenchmarkCompareModalState,
    exportBenchmarkCompareAsCsv,
    exportBenchmarkCompareAsText,
    copyBenchmarkCompareAsLatex,
    benchmarkBaseStatusHint: BENCHMARK_STATUS_HINT,
  };
}
