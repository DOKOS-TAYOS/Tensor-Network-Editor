import { apiGet, apiPost } from "../services/api.js";
import { constants } from "./constants.js";
import { getDomRefs } from "./dom.js";
import { createEditorSessionService } from "../services/editorSessionService.js";
import { createSubnetworkService } from "../services/subnetworkService.js";
import { createTemplateCatalogService } from "../services/templateCatalogService.js";
import { createInitialState } from "../state/state.js";
import { createEditorSelectors } from "../state/editorSelectors.js";
import { createEditorStore } from "../state/editorStore.js";

function resolveRuntimeApiToken(runtimeConfig) {
  const rawToken =
    runtimeConfig && typeof runtimeConfig.apiToken === "string"
      ? runtimeConfig.apiToken
      : runtimeConfig && typeof runtimeConfig.api_token === "string"
        ? runtimeConfig.api_token
        : null;
  return typeof rawToken === "string" && rawToken.trim() ? rawToken.trim() : null;
}

export function createEditorContext({
  window,
  document,
  cytoscape,
  runtimeConfig = {},
  logger = null,
}) {
  const state = createInitialState();
  const store = createEditorStore(state);
  const apiToken = resolveRuntimeApiToken(runtimeConfig);
  const requestApiGet = (path, options = {}) =>
    apiGet(path, {
      apiToken,
      logger,
      ...options,
    });
  const requestApiPost = (path, payload, options = {}) =>
    apiPost(path, payload, {
      apiToken,
      logger,
      ...options,
    });
  const services = {
    session: createEditorSessionService({
      apiGet: requestApiGet,
      apiPost: requestApiPost,
    }),
    templateCatalog: createTemplateCatalogService({ apiPost: requestApiPost }),
    subnetwork: createSubnetworkService({ apiPost: requestApiPost }),
  };
  return {
    apiGet: requestApiGet,
    apiPost: requestApiPost,
    constants,
    cytoscape,
    document,
    dom: getDomRefs(document),
    logger,
    selectors: createEditorSelectors({ store }),
    services,
    state,
    store,
    runtimeConfig,
    window,
  };
}
