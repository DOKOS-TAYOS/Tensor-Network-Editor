import { createUtilityBaseBindings } from "./utilitiesBase.js";
import { createUtilityGeometryBindings } from "./utilitiesGeometry.js";
import { createUtilityLinearPeriodicBindings } from "./utilitiesLinearPeriodic.js";
import { createUtilitySpecBindings } from "./utilitiesSpec.js";
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
  Object.assign(runtime, createUtilitySpecBindings(env));
  Object.assign(runtime, createUtilityLinearPeriodicBindings(env));
  Object.assign(runtime, createUtilityUiBindings(env));

  const templateOptionHelpers = createTemplateOptionHelpers({
    state: ctx.state,
    document: ctx.document,
    engineSelect: ctx.dom.engineSelect,
    collectionFormatSelect: ctx.dom.collectionFormatSelect,
    templateSelect: ctx.dom.templateSelect,
    templateParameterPanel: ctx.dom.templateParameterPanel,
    templateGraphSizeLabel: ctx.dom.templateGraphSizeLabel,
    templateGraphSizeInput: ctx.dom.templateGraphSizeInput,
    templateBondDimensionInput: ctx.dom.templateBondDimensionInput,
    templatePhysicalDimensionInput: ctx.dom.templatePhysicalDimensionInput,
    enforceLinearPeriodicEngineSupport: runtime.enforceLinearPeriodicEngineSupport,
    updateToolbarState: runtime.updateToolbarState,
  });

  Object.assign(ctx, templateOptionHelpers, runtime);
}
