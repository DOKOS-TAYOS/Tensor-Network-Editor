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

function readNonEmptyBenchmarkName(value) {
  return typeof value === "string" && value.trim() ? value : "";
}

function buildNormalizedPlanLike(
  runtime,
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
  const providedName = readNonEmptyBenchmarkName(nextPlan.name);
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

export function buildDefaultBenchmarkSchemeName(index) {
  return `Scheme ${index + 1}`;
}

function buildNormalizedBenchmarkScheme(runtime, plan, index, seed = null) {
  const fallbackName =
    readNonEmptyBenchmarkName(seed?.name)
      ? readNonEmptyBenchmarkName(seed?.name)
      : readNonEmptyBenchmarkName(plan?.name)
        ? readNonEmptyBenchmarkName(plan?.name)
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
  return buildNormalizedPlanLike(runtime, plan, {
    fallbackId,
    fallbackName,
    fallbackMetadata,
  });
}

function buildNormalizedOriginalPlan(runtime, plan) {
  const fallbackName =
    readNonEmptyBenchmarkName(plan?.name) || "Manual path";
  const fallbackId =
    typeof plan?.id === "string" && plan.id ? plan.id : runtime.makeId("plan");
  const fallbackMetadata = runtime.isObject(plan?.metadata) ? plan.metadata : {};
  return buildNormalizedPlanLike(runtime, plan, {
    fallbackId,
    fallbackName,
    fallbackMetadata,
  });
}

function normalizeCompareModalInPlace(runtime, compareModal) {
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

export function normalizeBenchmarkSessionInPlace(runtime, session) {
  const nextSession =
    runtime.isObject(session) ? session : createEmptyBenchmarkSession();
  nextSession.enabled = Boolean(nextSession.enabled);
  nextSession.activePosition = Number.isInteger(nextSession.activePosition)
    ? Math.max(0, nextSession.activePosition)
    : 0;
  nextSession.originalPlan = nextSession.originalPlan
    ? buildNormalizedOriginalPlan(runtime, nextSession.originalPlan)
    : null;
  nextSession.schemes = Array.isArray(nextSession.schemes)
    ? nextSession.schemes.map((scheme, index) =>
        buildNormalizedBenchmarkScheme(runtime, scheme, index)
      )
    : [];
  nextSession.compareModal = normalizeCompareModalInPlace(
    runtime,
    nextSession.compareModal
  );
  if (!nextSession.enabled) {
    nextSession.activePosition = 0;
  } else if (nextSession.activePosition > nextSession.schemes.length) {
    nextSession.activePosition = nextSession.schemes.length;
  }
  return nextSession;
}

export function createBenchmarkSessionStateSupport({ state, runtime }) {
  function getBenchmarkSession() {
    state.benchmarkSession = normalizeBenchmarkSessionInPlace(
      runtime,
      state.benchmarkSession
    );
    return state.benchmarkSession;
  }

  function setBenchmarkSession(session) {
    state.benchmarkSession = normalizeBenchmarkSessionInPlace(runtime, session);
    return state.benchmarkSession;
  }

  function resetBenchmarkCompareState(preserveOpen = false) {
    const session = getBenchmarkSession();
    session.compareModal = createEmptyBenchmarkCompareState();
    session.compareModal.open = preserveOpen;
    return session.compareModal;
  }

  function buildScheme(plan, index, seed = null) {
    return buildNormalizedBenchmarkScheme(runtime, plan, index, seed);
  }

  function buildOriginalPlan(plan) {
    return buildNormalizedOriginalPlan(runtime, plan);
  }

  function buildPlanLike(plan, options = {}) {
    return buildNormalizedPlanLike(runtime, plan, options);
  }

  return {
    buildOriginalPlan,
    buildPlanLike,
    buildScheme,
    getBenchmarkSession,
    resetBenchmarkCompareState,
    setBenchmarkSession,
  };
}
