import { apiGet, apiPost } from "../services/api.js";
import { constants } from "./constants.js";
import { getDomRefs } from "./dom.js";
import { createEditorSessionService } from "../services/editorSessionService.js";
import { createSubnetworkService } from "../services/subnetworkService.js";
import { createTemplateCatalogService } from "../services/templateCatalogService.js";
import { createInitialState } from "../state/state.js";
import { createEditorSelectors } from "../state/editorSelectors.js";
import { createEditorStore } from "../state/editorStore.js";

export function createEditorContext({ window, document, cytoscape }) {
  const state = createInitialState();
  const store = createEditorStore(state);
  const services = {
    session: createEditorSessionService({ apiGet, apiPost }),
    templateCatalog: createTemplateCatalogService({ apiPost }),
    subnetwork: createSubnetworkService({ apiPost }),
  };
  return {
    apiGet,
    apiPost,
    constants,
    cytoscape,
    document,
    dom: getDomRefs(document),
    selectors: createEditorSelectors({ store }),
    services,
    state,
    store,
    window,
  };
}
