async function readResponseBody(response) {
  const text = await response.text();
  if (!text) {
    return { text: "", json: null };
  }
  try {
    return { text, json: JSON.parse(text) };
  } catch {
    return { text, json: null };
  }
}

function buildErrorMessage({ text, json }) {
  if (json && typeof json === "object" && !Array.isArray(json)) {
    const messageParts = [];
    if (typeof json.message === "string" && json.message.trim()) {
      messageParts.push(json.message.trim());
    }
    if (typeof json.guidance === "string" && json.guidance.trim()) {
      messageParts.push(json.guidance.trim());
    }
    if (typeof json.reference === "string" && json.reference.trim()) {
      messageParts.push(`Reference: ${json.reference.trim()}`);
    }
    if (messageParts.length) {
      return messageParts.join(" ");
    }
  }
  return typeof text === "string" && text.trim() ? text.trim() : "Request failed.";
}

function resolveApiToken(options = {}) {
  const rawToken =
    typeof options.apiToken === "string"
      ? options.apiToken
      : typeof options.api_token === "string"
        ? options.api_token
        : typeof options.sessionToken === "string"
          ? options.sessionToken
          : null;
  return typeof rawToken === "string" && rawToken.trim() ? rawToken.trim() : null;
}

function buildRequestHeaders(options = {}, baseHeaders = {}) {
  const headers = { ...baseHeaders };
  const apiToken = resolveApiToken(options);
  if (apiToken) {
    headers["X-TNE-Session-Token"] = apiToken;
  }
  return headers;
}

function requireJsonBody({ json }) {
  if (json === null) {
    throw new Error("Expected a JSON response.");
  }
  return json;
}

function summarizeSpec(serializedSpec) {
  const network =
    serializedSpec && typeof serializedSpec === "object" && serializedSpec.network
      ? serializedSpec.network
      : null;
  if (!network || typeof network !== "object") {
    return {};
  }
  return {
    tensor_count: Array.isArray(network.tensors) ? network.tensors.length : 0,
    edge_count: Array.isArray(network.edges) ? network.edges.length : 0,
    group_count: Array.isArray(network.groups) ? network.groups.length : 0,
    note_count: Array.isArray(network.notes) ? network.notes.length : 0,
    mode:
      network.linear_periodic_chain
        ? "linear_periodic"
        : network.grid_periodic_grid
          ? "grid_periodic"
          : network.tree_periodic_tree
            ? "tree_periodic"
            : "normal",
  };
}

function summarizeResponsePayload(json) {
  if (!json || typeof json !== "object" || Array.isArray(json)) {
    return {};
  }
  const summary = {};
  if (typeof json.ok === "boolean") {
    summary.response_state = json.ok ? "ok" : "not_ok";
  }
  if (Array.isArray(json.warnings) && json.warnings.length) {
    summary.warning_count = json.warnings.length;
  }
  if (Array.isArray(json.issues) && json.issues.length) {
    summary.issue_count = json.issues.length;
    summary.analysis_status = "issues";
  }
  if (json.spec && typeof json.spec === "object") {
    Object.assign(summary, summarizeSpec(json.spec));
  }
  if (json.manual && typeof json.manual === "object") {
    summary.analysis_status =
      json.ok === false ? "issues" : summary.analysis_status || "ready";
    if (typeof json.manual.status === "string") {
      summary.manual_status = json.manual.status;
    }
    if (Array.isArray(json.manual.steps)) {
      summary.manual_step_count = json.manual.steps.length;
    }
  }
  if (
    json.automatic_full
    && typeof json.automatic_full === "object"
    && typeof json.automatic_full.status === "string"
  ) {
    summary.automatic_full_status = json.automatic_full.status;
  }
  if (
    json.automatic_future
    && typeof json.automatic_future === "object"
    && typeof json.automatic_future.status === "string"
  ) {
    summary.automatic_future_status = json.automatic_future.status;
  }
  if (
    json.automatic_past
    && typeof json.automatic_past === "object"
    && typeof json.automatic_past.status === "string"
  ) {
    summary.automatic_past_status = json.automatic_past.status;
  }
  if (
    json.frontend_logging
    && typeof json.frontend_logging === "object"
    && typeof json.frontend_logging.level === "string"
  ) {
    summary.frontend_logging_level = json.frontend_logging.level;
  }
  return summary;
}

function buildTracker(logger, method, path, options = {}) {
  if (!logger || typeof logger.startRequest !== "function") {
    return null;
  }
  return logger.startRequest(method, path, {
    operation: options.operation || null,
    context:
      options.context && typeof options.context === "object" ? options.context : {},
  });
}

async function performJsonRequest(method, path, init = {}, options = {}) {
  const tracker = buildTracker(options.logger || null, method, path, options);
  let trackedFailure = false;
  try {
    const response = await fetch(path, init);
    const body = await readResponseBody(response);
    if (!response.ok) {
      const error = new Error(buildErrorMessage(body));
      tracker?.fail(error, {
        status: response.status,
        ...summarizeResponsePayload(body.json),
      });
      trackedFailure = true;
      throw error;
    }
    const jsonBody = requireJsonBody(body);
    tracker?.finish({
      status: response.status,
      ...summarizeResponsePayload(jsonBody),
    });
    return jsonBody;
  } catch (error) {
    if (!trackedFailure) {
      tracker?.fail(error, options.context);
    }
    throw error;
  }
}

export async function apiGet(path, options = {}) {
  return performJsonRequest(
    "GET",
    path,
    {
      headers: buildRequestHeaders(options),
    },
    options
  );
}

export async function apiPost(path, payload, options = {}) {
  return performJsonRequest(
    "POST",
    path,
    {
      method: "POST",
      headers: buildRequestHeaders(options, { "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    },
    options
  );
}
