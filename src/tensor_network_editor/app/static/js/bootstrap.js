import { createEditorBootstrapFlow } from "./shell/editorBootstrapFlow.js";
import { createShellActions } from "./shell/shellActions.js";
import { createEditorShellBindings } from "./shell/editorShellBindings.js";
import { createShortcutTooltip } from "./shell/shortcutTooltip.js";

export function startEditor(ctx) {
  const state = ctx.state;
  const store = ctx.store;
  const { window, document } = ctx;
  const sessionService = ctx.services.session;
  const actions = createShellActions(ctx);
  const shortcutTooltip = createShortcutTooltip({
    documentRef: document,
    windowRef: window,
  });
  ctx.shortcutTooltip = shortcutTooltip;
  const bootstrapFlow = createEditorBootstrapFlow({
    state,
    store,
    sessionService,
    actions,
    documentRef: document,
    confirmAction(message) {
      return window.confirm(message);
    },
  });
  const shellBindings = createEditorShellBindings({
    state,
    store,
    dom: ctx.dom,
    documentRef: document,
    windowRef: window,
    actions,
    shortcutTooltip,
    redoShortcutLabel: ctx.constants.REDO_SHORTCUT_LABEL,
  });

  document.addEventListener("DOMContentLoaded", () => {
    shellBindings.attachToolbarHandlers();
    bootstrapFlow.bootstrap().catch((error) => {
      actions.setStatus(`Failed to load the editor: ${error.message}`, "error");
    });
  });
}
