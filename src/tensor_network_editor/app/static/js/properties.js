import { createPropertiesRenderers } from "./propertiesRenderers.js";
import { createPropertiesSupport } from "./propertiesSupport.js";

export function registerProperties(ctx) {
  const support = createPropertiesSupport({
    ctx,
    state: ctx.state,
    window: ctx.window,
  });
  const renderers = createPropertiesRenderers({
    ctx,
    state: ctx.state,
    document: ctx.document,
    propertiesPanel: ctx.dom.propertiesPanel,
    support,
  });

  Object.assign(ctx, {
    bindDebouncedAutosave: support.bindDebouncedAutosave,
    bindImmediateAutosave: support.bindImmediateAutosave,
    ...renderers,
  });
}
