import { startEditor } from "./bootstrap.js";
import { registerContractionScene } from "./contractionScene.js";
import { createEditorContext } from "./editorContext.js";
import { registerExportMinimap } from "./exportMinimap.js";
import { registerGraphRender } from "./graphRender.js";
import { registerHistorySelection } from "./historySelection.js";
import { registerInteractions } from "./interactions.js";
import { registerNotesPlanner } from "./notesPlanner.js";
import { registerOverlaysLayoutTemplates } from "./overlaysLayoutTemplates.js";
import { registerProperties } from "./properties.js";
import { registerSidebarTabs } from "./sidebarTabs.js";
import { registerUtilities } from "./utilities.js";

const context = createEditorContext({
  window,
  document,
  cytoscape: window.cytoscape,
});

registerUtilities(context);
registerContractionScene(context);
registerHistorySelection(context);
registerGraphRender(context);
registerSidebarTabs(context);
registerProperties(context);
registerExportMinimap(context);
registerOverlaysLayoutTemplates(context);
registerNotesPlanner(context);
registerInteractions(context);
startEditor(context);
