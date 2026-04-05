import { startEditor } from "./bootstrap.js";
import { createEditorContext } from "./editorContext.js";
import { registerExportMinimap } from "./exportMinimap.js";
import { registerGraphRender } from "./graphRender.js";
import { registerHistorySelection } from "./historySelection.js";
import { registerInteractions } from "./interactions.js";
import { registerOverlaysLayoutTemplates } from "./overlaysLayoutTemplates.js";
import { registerProperties } from "./properties.js";
import { registerUtilities } from "./utilities.js";

const context = createEditorContext({
  window,
  document,
  cytoscape: window.cytoscape,
});

registerUtilities(context);
registerHistorySelection(context);
registerGraphRender(context);
registerProperties(context);
registerExportMinimap(context);
registerOverlaysLayoutTemplates(context);
registerInteractions(context);
startEditor(context);
