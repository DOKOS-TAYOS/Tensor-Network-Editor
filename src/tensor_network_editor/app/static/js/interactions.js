import { createInteractionCanvasBindings } from "./interactionsCanvas.js";
import { createInteractionEditorBindings } from "./interactionsEditor.js";
import { createInteractionSessionBindings } from "./interactionsSession.js";
import { createInteractionShortcutBindings } from "./interactionsShortcuts.js";

export function registerInteractions(ctx) {
  const runtime = {};
  const env = {
    ctx,
    state: ctx.state,
    constants: ctx.constants,
    dom: ctx.dom,
    runtime,
  };

  Object.assign(runtime, createInteractionCanvasBindings(env));
  Object.assign(runtime, createInteractionShortcutBindings(env));
  Object.assign(runtime, createInteractionEditorBindings(env));
  Object.assign(runtime, createInteractionSessionBindings(env));

  Object.assign(ctx, runtime);
}
