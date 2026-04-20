import { createSessionCommands } from "../actions/sessionCommands.js";
import { createSessionEditorFlows } from "../session/sessionEditorFlows.js";
import { createSessionTemplateFlows } from "../session/sessionTemplateFlows.js";

export function createInteractionSessionBindings({
  ctx,
  state,
  dom,
  store,
  selectors,
  services,
  sessionUi,
  sessionActions,
}) {
  const commands = createSessionCommands({
    dom,
    state,
    store,
    document: ctx.document,
    window: ctx.window,
    setStatus: sessionActions.setStatus,
    applyTemplateCatalogPayload: sessionActions.applyTemplateCatalogPayload,
    normalizeSpec: sessionActions.normalizeSpec,
    applyDesignChange: sessionActions.applyDesignChange,
    bringTensorToFront: sessionActions.bringTensorToFront,
  });
  const editorFlows = createSessionEditorFlows({
    dom,
    state,
    store,
    selectors,
    services,
    commands,
    sessionUi,
    actions: sessionActions,
  });
  const templateFlows = createSessionTemplateFlows({
    dom,
    state,
    store,
    selectors,
    services,
    commands,
    sessionUi,
    actions: sessionActions,
  });

  return {
    ...editorFlows,
    ...templateFlows,
    insertPreparedSubnetwork: commands.insertPreparedSubnetwork,
  };
}
