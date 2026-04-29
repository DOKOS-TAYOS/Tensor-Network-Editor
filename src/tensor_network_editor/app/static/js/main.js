import { startEditor } from "./bootstrap.js";
import { registerCanvasContextMenu } from "./graph/canvasContextMenu.js";
import { registerContractionScene } from "./graph/contractionScene.js";
import { createEditorContext } from "./core/editorContext.js";
import {
  createFrontendLogger,
  readFrontendRuntimeConfig,
} from "./core/frontendLogger.js";
import { registerExportMinimap } from "./graph/exportMinimap.js";
import { registerGraphRender } from "./graph/graphRender.js";
import { registerHistorySelection } from "./graph/historySelection.js";
import { registerInteractions } from "./interactions/interactions.js";
import { registerMetadataFilters } from "./graph/metadataFilters.js";
import { registerNotesPlanner } from "./planner/notesPlanner.js";
import { registerOverlaysLayoutTemplates } from "./graph/overlaysLayoutTemplates.js";
import { registerProperties } from "./properties/properties.js";
import { registerSidebarTabs } from "./core/sidebarTabs.js";
import { registerUtilities } from "./utils/utilities.js";

const runtimeConfig = readFrontendRuntimeConfig({ documentRef: document });
const logger = createFrontendLogger(runtimeConfig);
const context = createEditorContext({
  window,
  document,
  cytoscape: window.cytoscape,
  runtimeConfig,
  logger,
});

registerUtilities(context);
registerContractionScene(context);
registerHistorySelection(context);
registerGraphRender(context);
registerSidebarTabs(context);
registerProperties(context);
registerCanvasContextMenu(context);
registerMetadataFilters(context);
registerExportMinimap(context);
registerOverlaysLayoutTemplates(context);
registerNotesPlanner(context);
registerInteractions(context);
startEditor(context);
