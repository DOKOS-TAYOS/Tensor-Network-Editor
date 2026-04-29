import { createPlannerAnalysisService } from "../services/plannerAnalysisService.js";

export function createPlannerAnalysisSupport({
  ctx,
  state,
  analysisRefreshDelayMs,
  setTimer,
  clearTimer,
  renderPlanner,
  guards,
}) {
  const {
    benchmarkBaseStatusMessage,
    isBenchmarkBasePosition,
  } = guards;
  let pendingContractionAnalysisOptions = null;
  const logger = ctx.logger || null;

  function getBenchmarkPosition() {
    if (typeof ctx.getBenchmarkSession !== "function") {
      return null;
    }
    const benchmarkSession = ctx.getBenchmarkSession();
    return Number.isInteger(benchmarkSession?.activePosition)
      ? benchmarkSession.activePosition
      : null;
  }

  function logPlannerAnalysisEvent(message, context = {}) {
    if (!logger || typeof logger.debug !== "function") {
      return;
    }
    logger.debug(message, {
      operation: "planner.analysis",
      analysis_source: "manual",
      benchmark_position: getBenchmarkPosition(),
      planner_mode: Boolean(state.plannerMode),
      ...context,
    });
  }

  function getCachedContractionAnalysisPayload() {
    return state.contractionAnalysisCacheRevision === state.specRevision
      ? state.contractionAnalysisCachePayload
      : null;
  }

  function markContractionAnalysisDirty() {
    state.contractionAnalysisDirty = true;
  }

  const analysisService = createPlannerAnalysisService({
    analysisRefreshDelayMs,
    analyze: (payload, requestOptions = {}) =>
      ctx.apiPost("/api/analyze-contraction", payload, {
        operation:
          requestOptions.analysisSource === "compare_batch"
            ? "benchmark.compare.analyze"
            : "planner.analysis",
        context: {
          analysis_source: requestOptions.analysisSource || "manual",
          refresh_reason: requestOptions.refreshReason || "explicit",
          cache_state: requestOptions.cacheState || null,
          benchmark_position: requestOptions.benchmarkPosition,
          planner_mode: requestOptions.plannerMode,
        },
      }),
    cancel: (timerId) => clearTimer(timerId),
    onAnalysisError: (error) => {
      state.contractionAnalysisCacheRevision = -1;
      state.contractionAnalysisCachePayload = null;
      state.contractionAnalysis = {
        status: "error",
        message: error.message,
      };
    },
    onAnalysisResult: (payload) => {
      if (!payload.ok) {
        state.contractionAnalysis = {
          status: "issues",
          issues: payload.issues || [],
        };
        return;
      }
      state.contractionAnalysisCacheRevision = state.specRevision;
      state.contractionAnalysisCachePayload = payload;
      state.contractionAnalysis = {
        status: "ready",
        payload,
      };
    },
    onRequestStarted: (options) => {
      if (options.focusTab && typeof ctx.setActiveSidebarTab === "function") {
        ctx.setActiveSidebarTab("planner");
      }
      state.contractionAnalysisRequestId += 1;
      state.contractionAnalysis = { status: "loading" };
      renderPlanner();
    },
    onRenderRequested: () => {
      renderPlanner();
      ctx.renderOverlayDecorations();
    },
    schedule: (callback, delay) => setTimer(callback, delay),
    serializeCurrentSpec: (options) => ctx.serializeCurrentSpec(options),
    logger,
  });

  function shouldRefreshContractionAnalysisImmediately(options = {}) {
    return (
      Boolean(options.immediate) ||
      state.activeSidebarTab === "planner" ||
      state.plannerMode ||
      Boolean(state.plannerPreviewMode) ||
      (typeof ctx.isInspectingPastStage === "function" && ctx.isInspectingPastStage()) ||
      (typeof ctx.isContractionSceneVisible === "function" &&
        ctx.isContractionSceneVisible())
    );
  }

  function refreshContractionAnalysis(options = {}) {
    const refreshReason =
      typeof options.refreshReason === "string" && options.refreshReason
        ? options.refreshReason
        : "explicit";
    if (isBenchmarkBasePosition()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      state.contractionAnalysis = { status: "benchmarkBase" };
      logPlannerAnalysisEvent("Skipped contraction analysis at benchmark base position", {
        analysis_status: "benchmark_base",
        cache_state: "bypass",
        refresh_reason: refreshReason,
      });
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    state.contractionAnalysisDirty = false;
    const cachedPayload = getCachedContractionAnalysisPayload();
    if (cachedPayload) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysis = {
        status: "ready",
        payload: cachedPayload,
      };
      logPlannerAnalysisEvent("Contraction analysis cache hit", {
        analysis_status: "ready",
        cache_state: "hit",
        refresh_reason: refreshReason,
      });
      renderPlanner();
      ctx.renderOverlayDecorations();
      return Promise.resolve(cachedPayload);
    }
    const cacheState =
      state.contractionAnalysisCachePayload &&
      state.contractionAnalysisCacheRevision !== state.specRevision
        ? "stale"
        : "miss";
    pendingContractionAnalysisOptions = {
      focusTab:
        Boolean(options.focusTab) ||
        Boolean(
          pendingContractionAnalysisOptions &&
            pendingContractionAnalysisOptions.focusTab
        ),
      refreshReason,
      cacheState,
      benchmarkPosition: getBenchmarkPosition(),
      plannerMode: Boolean(state.plannerMode),
    };
    return analysisService.requestRefresh(pendingContractionAnalysisOptions, {
      immediate: shouldRefreshContractionAnalysisImmediately(options),
    });
  }

  function togglePlannerDisclosure(disclosureKey) {
    state.plannerDisclosureState[disclosureKey] = !state.plannerDisclosureState[disclosureKey];
    renderPlanner();
  }

  return {
    analysisRefreshDelayMs,
    benchmarkBaseStatusMessage,
    markContractionAnalysisDirty,
    refreshContractionAnalysis,
    togglePlannerDisclosure,
  };
}
