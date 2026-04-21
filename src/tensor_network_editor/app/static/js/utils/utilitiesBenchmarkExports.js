import { BENCHMARK_COMPARE_COLUMNS } from "./utilitiesBenchmarkTable.js";

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
    Math.max(...matrix.map((row) => String(row[columnIndex] ?? "").length))
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
