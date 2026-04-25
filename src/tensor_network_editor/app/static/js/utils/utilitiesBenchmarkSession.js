import {
  buildDefaultBenchmarkSchemeName,
  createBenchmarkSessionStateSupport,
} from "./utilitiesBenchmarkSessionState.js";
import { createEmptyBenchmarkSession } from "../state/benchmarkState.js";
import { buildBenchmarkCompareTableModel } from "./utilitiesBenchmarkTable.js";
import {
  serializeBenchmarkCompareTableCsv,
  serializeBenchmarkCompareTableLatex,
  serializeBenchmarkCompareTableText,
} from "./utilitiesBenchmarkExports.js";

export { buildBenchmarkCompareTableModel } from "./utilitiesBenchmarkTable.js";
export {
  serializeBenchmarkCompareTableCsv,
  serializeBenchmarkCompareTableLatex,
  serializeBenchmarkCompareTableText,
} from "./utilitiesBenchmarkExports.js";

const BENCHMARK_BASE_LABEL = "Tensor network";
const BENCHMARK_STATUS_HINT = "Move right to edit or create a contraction scheme.";

export function createBenchmarkSessionSupport({
  actions,
  state,
  dom,
  runtime,
}) {
  const ctx = actions;

  const benchmarkState = createBenchmarkSessionStateSupport({
    state,
    runtime,
  });
  const {
    buildOriginalPlan: buildNormalizedOriginalPlan,
    buildPlanLike: buildNormalizedPlanLike,
    buildScheme: buildNormalizedBenchmarkScheme,
    getBenchmarkSession,
    resetBenchmarkCompareState,
    setBenchmarkSession,
  } = benchmarkState;

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
    const tableModel = getBenchmarkCompareTableModel(compareModal);
    return Boolean(
      compareModal?.open &&
        !compareModal.loading &&
        !compareModal.errorMessage &&
        Array.isArray(tableModel.rows) &&
        tableModel.rows.length
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
        ? name
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
