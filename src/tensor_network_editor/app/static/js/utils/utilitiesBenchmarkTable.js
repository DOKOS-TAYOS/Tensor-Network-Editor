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

export const BENCHMARK_COMPARE_COLUMNS = [
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

export function formatBenchmarkNumber(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toLocaleString() : "-";
}

export function formatBenchmarkBytes(value) {
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
