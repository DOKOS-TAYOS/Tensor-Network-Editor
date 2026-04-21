import { createBenchmarkSessionSupport } from "./utilitiesBenchmarkSession.js";
import { buildBenchmarkCompareTableModel } from "./utilitiesBenchmarkTable.js";
import {
  serializeBenchmarkCompareTableCsv,
  serializeBenchmarkCompareTableLatex,
  serializeBenchmarkCompareTableText,
} from "./utilitiesBenchmarkExports.js";

export {
  buildBenchmarkCompareTableModel,
  serializeBenchmarkCompareTableCsv,
  serializeBenchmarkCompareTableLatex,
  serializeBenchmarkCompareTableText,
};

export function createUtilityBenchmarkBindings({ ctx, state, dom, runtime }) {
  return createBenchmarkSessionSupport({
    actions: ctx,
    state,
    dom,
    runtime,
  });
}
