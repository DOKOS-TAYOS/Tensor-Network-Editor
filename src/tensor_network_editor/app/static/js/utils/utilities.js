import { createUtilityBaseBindings } from "./utilitiesBase.js";
import { createUtilityBenchmarkBindings } from "./utilitiesBenchmark.js";
import { createUtilityGeometryBindings } from "./utilitiesGeometry.js";
import { createUtilityGridPeriodicBindings } from "./utilitiesGridPeriodic.js";
import { createUtilityLayoutBindings } from "./utilitiesLayout.js";
import { createUtilityLinearPeriodicBindings } from "./utilitiesLinearPeriodic.js";
import { createUtilitySpecBindings } from "./utilitiesSpec.js";
import { createUtilityTreePeriodicBindings } from "./utilitiesTreePeriodic.js";
import { createTemplateOptionHelpers } from "./utilitiesTemplates.js";
import { createUtilityUiBindings } from "./utilitiesUi.js";

export function registerUtilities(ctx) {
  const runtime = {};
  const env = {
    ctx,
    state: ctx.state,
    constants: ctx.constants,
    dom: ctx.dom,
    runtime,
  };

  Object.assign(runtime, createUtilityBaseBindings(env));
  Object.assign(runtime, createUtilityGeometryBindings(env));
  Object.assign(runtime, createUtilityLayoutBindings(env));
  Object.assign(runtime, createUtilitySpecBindings(env));
  Object.assign(runtime, createUtilityLinearPeriodicBindings(env));
  Object.assign(runtime, createUtilityGridPeriodicBindings(env));
  Object.assign(runtime, createUtilityTreePeriodicBindings(env));
  Object.assign(runtime, createUtilityBenchmarkBindings(env));
  Object.assign(runtime, createUtilityUiBindings(env));

  const templateOptionHelpers = createTemplateOptionHelpers({
    state: ctx.state,
    document: ctx.document,
    engineSelect: ctx.dom.engineSelect,
    collectionFormatSelect: ctx.dom.collectionFormatSelect,
    templateSelect: ctx.dom.templateSelect,
    templateParameterPanel: ctx.dom.templateParameterPanel,
    templateGraphSizeField: ctx.dom.templateGraphSizeField,
    templateGraphSizeLabel: ctx.dom.templateGraphSizeLabel,
    templateGraphSizeInput: ctx.dom.templateGraphSizeInput,
    templateBondDimensionField: ctx.dom.templateBondDimensionField,
    templateBondDimensionInput: ctx.dom.templateBondDimensionInput,
    templatePhysicalDimensionField: ctx.dom.templatePhysicalDimensionField,
    templatePhysicalDimensionInput: ctx.dom.templatePhysicalDimensionInput,
    templateBoundaryConditionField: ctx.dom.templateBoundaryConditionField,
    templateBoundaryConditionSelect: ctx.dom.templateBoundaryConditionSelect,
    templateSymmetryField: ctx.dom.templateSymmetryField,
    templateSymmetrySelect: ctx.dom.templateSymmetrySelect,
    templateInitialStateField: ctx.dom.templateInitialStateField,
    templateInitialStateSelect: ctx.dom.templateInitialStateSelect,
    enforceLinearPeriodicEngineSupport: runtime.enforceLinearPeriodicEngineSupport,
    updateToolbarState: runtime.updateToolbarState,
  });

  Object.assign(ctx, templateOptionHelpers, runtime);
}
