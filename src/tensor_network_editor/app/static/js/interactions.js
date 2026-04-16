import { createInteractionCanvasBindings } from "./interactionsCanvas.js";
import { createInteractionEditorBindings } from "./interactionsEditor.js";
import { createInteractionSessionBindings } from "./interactionsSession.js";
import { createInteractionShortcutBindings } from "./interactionsShortcuts.js";
import { createEditorActionGroups } from "./interactions/editorActionGroups.js";
import { createSessionUiAdapters } from "./session/sessionUiAdapters.js";

export function registerInteractions(ctx) {
  const runtime = {};
  const actionGroups = createEditorActionGroups(ctx);
  const sessionUi = createSessionUiAdapters({
    windowRef: ctx.window,
    documentRef: ctx.document,
  });
  const env = {
    ctx,
    state: ctx.state,
    constants: ctx.constants,
    dom: ctx.dom,
    runtime,
  };

  Object.assign(runtime, createInteractionCanvasBindings(env));
  Object.assign(
    runtime,
    createInteractionShortcutBindings({
      ...env,
      shortcutActions: actionGroups.shortcuts,
    })
  );
  Object.assign(
    runtime,
    createInteractionEditorBindings({
      ...env,
      editorActions: actionGroups.editor,
    })
  );
  Object.assign(
    runtime,
    createInteractionSessionBindings({
      ...env,
      store: ctx.store,
      selectors: ctx.selectors,
      services: ctx.services,
      sessionUi,
      sessionActions: actionGroups.session,
    })
  );

  Object.assign(ctx, runtime);
}
