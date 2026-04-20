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
    try {
      if (typeof onRequestStarted === "function") {
        onRequestStarted(requestOptions);
      }
      const payload = await analyze({
        ...requestOptions,
        spec: serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      onAnalysisResult(payload, requestOptions);
    } catch (error) {
      if (typeof onAnalysisError === "function") {
        onAnalysisError(error, requestOptions);
      }
    } finally {
      requestPending = false;
      if (pendingOptions) {
        debounceId = schedule(flushQueue, analysisRefreshDelayMs);
      }
      onRenderRequested();
    }
  }

  function requestRefresh(options = {}, controls = {}) {
    pendingOptions = options;
    clearScheduledFlush();
    if (controls.immediate) {
      return flushQueue();
    }
    debounceId = schedule(flushQueue, analysisRefreshDelayMs);
  }

  return {
    flushQueue,
    requestRefresh,
  };
}
