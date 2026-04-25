export function createEmptyBenchmarkCompareState() {
  return {
    open: false,
    loading: false,
    errorMessage: "",
    tableModel: null,
    rows: [],
    activeRequestId: 0,
  };
}

export function createEmptyBenchmarkSession() {
  return {
    enabled: false,
    activePosition: 0,
    originalPlan: null,
    schemes: [],
    compareModal: createEmptyBenchmarkCompareState(),
  };
}
