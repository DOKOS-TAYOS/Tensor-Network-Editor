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
    gridPeriodicStatusMessage,
    treePeriodicStatusMessage,
    hyperedgeStatusMessage,
    hasHyperedges,
    isBenchmarkBasePosition,
    isGridPeriodicMode,
    isTreePeriodicMode,
  } = guards;
  let pendingContractionAnalysisOptions = null;

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
    analyze: (payload) => ctx.apiPost("/api/analyze-contraction", payload),
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
    if (hasHyperedges()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "hyperedgesDisabled",
        message: hyperedgeStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isTreePeriodicMode()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "treePeriodicDisabled",
        message: treePeriodicStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isGridPeriodicMode()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      if (state.spec) {
        state.spec.contraction_plan = null;
      }
      state.contractionAnalysis = {
        status: "gridPeriodicDisabled",
        message: gridPeriodicStatusMessage,
      };
      renderPlanner();
      ctx.renderOverlayDecorations();
      return;
    }
    if (isBenchmarkBasePosition()) {
      pendingContractionAnalysisOptions = null;
      state.contractionAnalysisDirty = false;
      state.contractionAnalysis = { status: "benchmarkBase" };
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
      renderPlanner();
      ctx.renderOverlayDecorations();
      return Promise.resolve(cachedPayload);
    }
    pendingContractionAnalysisOptions = {
      focusTab:
        Boolean(options.focusTab) ||
        Boolean(
          pendingContractionAnalysisOptions &&
            pendingContractionAnalysisOptions.focusTab
        ),
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
