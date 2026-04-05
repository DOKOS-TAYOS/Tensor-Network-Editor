import { apiGet, apiPost } from "./api.js";
import { constants } from "./constants.js";
import { getDomRefs } from "./dom.js";
import { createInitialState } from "./state.js";

export function createEditorContext({ window, document, cytoscape }) {
  return {
    apiGet,
    apiPost,
    constants,
    cytoscape,
    document,
    dom: getDomRefs(document),
    state: createInitialState(),
    window,
  };
}
