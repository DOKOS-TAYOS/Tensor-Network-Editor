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

  async function flushQueue() {
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

  function requestRefresh(options = {}) {
    pendingOptions = options;
    if (debounceId !== null) {
      cancel(debounceId);
    }
    debounceId = schedule(flushQueue, analysisRefreshDelayMs);
  }

  return {
    flushQueue,
    requestRefresh,
  };
}
