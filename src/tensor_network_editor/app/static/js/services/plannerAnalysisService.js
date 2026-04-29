function emitLoggerEvent(logger, level, message, context = {}) {
  if (!logger || typeof logger[level] !== "function") {
    return;
  }
  logger[level](message, context);
}

function summarizeAnalysisPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return {
      analysis_status: "error",
    };
  }
  if (payload.ok === false) {
    return {
      analysis_status: "issues",
      issue_count: Array.isArray(payload.issues) ? payload.issues.length : 0,
    };
  }
  return {
    analysis_status: "ready",
    warning_count: Array.isArray(payload.warnings) ? payload.warnings.length : 0,
    manual_status:
      payload.manual && typeof payload.manual.status === "string"
        ? payload.manual.status
        : null,
    automatic_full_status:
      payload.automatic_full && typeof payload.automatic_full.status === "string"
        ? payload.automatic_full.status
        : null,
    automatic_future_status:
      payload.automatic_future && typeof payload.automatic_future.status === "string"
        ? payload.automatic_future.status
        : null,
    automatic_past_status:
      payload.automatic_past && typeof payload.automatic_past.status === "string"
        ? payload.automatic_past.status
        : null,
  };
}

function buildAnalysisLogContext(requestOptions = {}, extraContext = {}) {
  return {
    operation: "planner.analysis",
    analysis_source: requestOptions.analysisSource || "manual",
    refresh_reason: requestOptions.refreshReason || "explicit",
    cache_state: requestOptions.cacheState || null,
    benchmark_position:
      Number.isInteger(requestOptions.benchmarkPosition)
        ? requestOptions.benchmarkPosition
        : null,
    planner_mode:
      typeof requestOptions.plannerMode === "boolean"
        ? requestOptions.plannerMode
        : null,
    ...extraContext,
  };
}

export function createPlannerAnalysisService({
  analysisRefreshDelayMs,
  analyze,
  cancel,
  onAnalysisError = null,
  onAnalysisResult,
  onRequestStarted = null,
  onRenderRequested,
  schedule,
  serializeCurrentSpec,
  logger = null,
}) {
  let debounceId = null;
  let requestPending = false;
  let pendingOptions = null;

  function clearScheduledFlush() {
    if (debounceId === null) {
      return;
    }
    cancel(debounceId);
    debounceId = null;
  }

  async function flushQueue() {
    debounceId = null;
    if (requestPending) {
      return;
    }
    requestPending = true;
    const requestOptions = pendingOptions || {};
    pendingOptions = null;
    emitLoggerEvent(
      logger,
      "debug",
      "Contraction analysis request started",
      buildAnalysisLogContext(requestOptions, {
        analysis_status: "loading",
      })
    );
    try {
      if (typeof onRequestStarted === "function") {
        onRequestStarted(requestOptions);
      }
      const payload = await analyze(
        {
          ...requestOptions,
          spec: serializeCurrentSpec({ persistViewSnapshots: false }),
        },
        requestOptions
      );
      onAnalysisResult(payload, requestOptions);
      emitLoggerEvent(
        logger,
        "debug",
        "Contraction analysis request resolved",
        buildAnalysisLogContext(requestOptions, summarizeAnalysisPayload(payload))
      );
    } catch (error) {
      emitLoggerEvent(
        logger,
        "warn",
        "Contraction analysis request failed",
        buildAnalysisLogContext(requestOptions, {
          analysis_status: "error",
          outcome: "error",
        })
      );
      if (typeof onAnalysisError === "function") {
        onAnalysisError(error, requestOptions);
      }
    } finally {
      requestPending = false;
      if (pendingOptions) {
        emitLoggerEvent(
          logger,
          "debug",
          "Contraction analysis refresh queued",
          buildAnalysisLogContext(pendingOptions)
        );
        debounceId = schedule(flushQueue, analysisRefreshDelayMs);
      }
      onRenderRequested();
    }
  }

  function requestRefresh(options = {}, controls = {}) {
    pendingOptions = options;
    if (requestPending) {
      emitLoggerEvent(
        logger,
        "debug",
        "Contraction analysis refresh queued",
        buildAnalysisLogContext(options)
      );
      return null;
    }
    clearScheduledFlush();
    if (controls.immediate) {
      return flushQueue();
    }
    emitLoggerEvent(
      logger,
      "debug",
      "Contraction analysis refresh queued",
      buildAnalysisLogContext(options)
    );
    debounceId = schedule(flushQueue, analysisRefreshDelayMs);
  }

  return {
    flushQueue,
    requestRefresh,
  };
}
