const RUNTIME_CONFIG_ELEMENT_ID = "tne-runtime-config";
const LOG_CONTEXT_FIELD_ORDER = [
  "session",
  "operation",
  "method",
  "route",
  "request_id",
  "engine",
  "format",
  "export_format",
  "collection_format",
  "python_import_mode",
  "source_profile",
  "python_reconstruction_level",
  "theme",
  "refresh_reason",
  "cache_state",
  "analysis_status",
  "analysis_source",
  "planner_mode",
  "benchmark_position",
  "scheme_count",
  "template_name",
  "subnetwork_name",
  "selected_template",
  "selected_subnetwork",
  "tensor_id_count",
  "tag_count",
  "warning_count",
  "issue_count",
  "client_ts_ms",
  "status",
  "outcome",
  "elapsed_ms",
];
const LOG_LEVEL_PRIORITY = {
  debug: 10,
  info: 20,
  warning: 30,
  error: 40,
  off: Number.POSITIVE_INFINITY,
};
const LOG_LEVEL_CONSOLE_METHOD = {
  debug: "debug",
  info: "info",
  warning: "warn",
  error: "error",
};
const MAX_QUEUED_EVENTS = 200;
const MAX_FLUSH_BATCH_SIZE = 20;
const FLUSH_DELAY_MS = 500;
const NOOP_OPERATION = {
  branch() {},
  finish() {},
  fail() {},
};

function normalizeLevel(rawLevel, enabled) {
  const level =
    typeof rawLevel === "string" && rawLevel.trim()
      ? rawLevel.trim().toLowerCase()
      : enabled
        ? "debug"
        : "off";
  if (Object.prototype.hasOwnProperty.call(LOG_LEVEL_PRIORITY, level)) {
    return level;
  }
  return enabled ? "debug" : "off";
}

function normalizeRuntimeConfig(runtimeConfig = {}) {
  const candidate =
    runtimeConfig && typeof runtimeConfig === "object" ? runtimeConfig : {};
  const frontendLogging =
    candidate.frontend_logging && typeof candidate.frontend_logging === "object"
      ? candidate.frontend_logging
      : candidate;
  const rawSessionId =
    typeof candidate.sessionId === "string"
      ? candidate.sessionId
      : typeof candidate.session_id === "string"
        ? candidate.session_id
        : null;
  const sessionId =
    typeof rawSessionId === "string" && rawSessionId.trim()
      ? rawSessionId.trim()
      : null;
  const rawApiToken =
    typeof candidate.apiToken === "string"
      ? candidate.apiToken
      : typeof candidate.api_token === "string"
        ? candidate.api_token
        : null;
  const apiToken =
    typeof rawApiToken === "string" && rawApiToken.trim()
      ? rawApiToken.trim()
      : null;
  const rawEnabled = frontendLogging.enabled === true;
  const level = normalizeLevel(frontendLogging.level, rawEnabled);
  const enabled = rawEnabled || level !== "off";
  const rawTransportEndpoint =
    typeof frontendLogging.transportEndpoint === "string"
      ? frontendLogging.transportEndpoint
      : typeof frontendLogging.transport_endpoint === "string"
        ? frontendLogging.transport_endpoint
        : null;
  const transportEndpoint =
    typeof rawTransportEndpoint === "string" && rawTransportEndpoint.trim()
      ? rawTransportEndpoint.trim()
      : null;
  const persist = frontendLogging.persist === true && transportEndpoint !== null;
  return {
    apiToken,
    enabled,
    level,
    persist,
    sessionId,
    transportEndpoint,
  };
}

function elapsedMsText(startTime, endTime) {
  return String(Math.max(0, Math.round(endTime - startTime)));
}

function formatLogMessage(message, context = {}) {
  const mergedContext = {};
  Object.entries(context || {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      mergedContext[key] = typeof value === "boolean" ? String(value) : String(value);
    }
  });
  const orderedEntries = [];
  LOG_CONTEXT_FIELD_ORDER.forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(mergedContext, key)) {
      orderedEntries.push([key, mergedContext[key]]);
      delete mergedContext[key];
    }
  });
  Object.keys(mergedContext)
    .sort()
    .forEach((key) => {
      orderedEntries.push([key, mergedContext[key]]);
    });
  if (!orderedEntries.length) {
    return message;
  }
  return `${message} ${orderedEntries.map(([key, value]) => `${key}=${value}`).join(" ")}`;
}

function createConsoleDispatcher(consoleRef, level) {
  if (!consoleRef || typeof consoleRef !== "object") {
    return null;
  }
  const methodName = LOG_LEVEL_CONSOLE_METHOD[level] || "log";
  const method =
    typeof consoleRef[methodName] === "function"
      ? consoleRef[methodName].bind(consoleRef)
      : typeof consoleRef.log === "function"
        ? consoleRef.log.bind(consoleRef)
        : null;
  return method;
}

function normalizeContext(context = {}) {
  const normalizedContext = {};
  Object.entries(context || {}).forEach(([key, value]) => {
    if (
      typeof key === "string"
      && key.trim()
      && value !== null
      && value !== undefined
    ) {
      normalizedContext[key.trim()] =
        typeof value === "boolean" ? String(value) : String(value);
    }
  });
  return normalizedContext;
}

function createQueuedEvent(level, message, context, nowValue) {
  return {
    level,
    message,
    context: {
      ...context,
      client_ts_ms: String(Math.max(0, Math.round(nowValue))),
    },
  };
}

export function readFrontendRuntimeConfig({ documentRef = globalThis.document ?? null } = {}) {
  const configElement =
    documentRef && typeof documentRef.getElementById === "function"
      ? documentRef.getElementById(RUNTIME_CONFIG_ELEMENT_ID)
      : null;
  if (!configElement || typeof configElement.textContent !== "string") {
    return normalizeRuntimeConfig();
  }
  try {
    return normalizeRuntimeConfig(JSON.parse(configElement.textContent || "{}"));
  } catch {
    return normalizeRuntimeConfig();
  }
}

export function createFrontendLogger(
  runtimeConfig = {},
  {
    consoleRef = globalThis.console ?? null,
    performanceRef = globalThis.performance ?? null,
    navigatorRef = globalThis.navigator ?? null,
    windowRef = globalThis.window ?? null,
    fetchRef = globalThis.fetch ?? null,
  } = {}
) {
  let resolvedRuntimeConfig = normalizeRuntimeConfig(runtimeConfig);
  let nextRequestNumber = 1;
  let flushTimerId = null;
  let droppedEventCount = 0;
  let queuedEvents = [];
  let pagehideHandler = null;
  const dispatchers = {
    debug: createConsoleDispatcher(consoleRef, "debug"),
    info: createConsoleDispatcher(consoleRef, "info"),
    warning: createConsoleDispatcher(consoleRef, "warning"),
    error: createConsoleDispatcher(consoleRef, "error"),
  };

  function now() {
    return performanceRef && typeof performanceRef.now === "function"
      ? performanceRef.now()
      : Date.now();
  }

  function shouldEmit(level) {
    if (!resolvedRuntimeConfig.enabled) {
      return false;
    }
    return (
      (LOG_LEVEL_PRIORITY[level] || LOG_LEVEL_PRIORITY.off)
      >= (LOG_LEVEL_PRIORITY[resolvedRuntimeConfig.level] || LOG_LEVEL_PRIORITY.off)
    );
  }

  function clearFlushTimer() {
    if (flushTimerId === null || !windowRef || typeof windowRef.clearTimeout !== "function") {
      flushTimerId = null;
      return;
    }
    windowRef.clearTimeout(flushTimerId);
    flushTimerId = null;
  }

  function buildOverflowEvent() {
    if (droppedEventCount <= 0) {
      return null;
    }
    const overflowEvent = createQueuedEvent(
      "warning",
      "Frontend log queue overflowed",
      {
        session: resolvedRuntimeConfig.sessionId,
        dropped_count: String(droppedEventCount),
      },
      now()
    );
    droppedEventCount = 0;
    return overflowEvent;
  }

  async function sendQueuedEvents(events, preferBeacon = false) {
    if (
      !resolvedRuntimeConfig.persist
      || !resolvedRuntimeConfig.transportEndpoint
      || !events.length
    ) {
      return false;
    }
    const body = JSON.stringify({ events });
    if (
      preferBeacon
      && !resolvedRuntimeConfig.apiToken
      && navigatorRef
      && typeof navigatorRef.sendBeacon === "function"
      && navigatorRef.sendBeacon(resolvedRuntimeConfig.transportEndpoint, body)
    ) {
      return true;
    }
    if (typeof fetchRef !== "function") {
      return false;
    }
    const headers = { "Content-Type": "application/json" };
    if (resolvedRuntimeConfig.apiToken) {
      headers["X-TNE-Session-Token"] = resolvedRuntimeConfig.apiToken;
    }
    try {
      await fetchRef(resolvedRuntimeConfig.transportEndpoint, {
        method: "POST",
        headers,
        body,
        keepalive: true,
      });
      return true;
    } catch {
      return false;
    }
  }

  async function flushQueue({ preferBeacon = false } = {}) {
    clearFlushTimer();
    const overflowEvent = buildOverflowEvent();
    if (!queuedEvents.length && overflowEvent === null) {
      return false;
    }
    const eventsToFlush = [...queuedEvents];
    queuedEvents = [];
    if (overflowEvent !== null) {
      eventsToFlush.push(overflowEvent);
    }
    return sendQueuedEvents(eventsToFlush, preferBeacon);
  }

  function scheduleFlush() {
    if (
      !resolvedRuntimeConfig.persist
      || !resolvedRuntimeConfig.transportEndpoint
      || queuedEvents.length === 0
    ) {
      return;
    }
    if (queuedEvents.length >= MAX_FLUSH_BATCH_SIZE) {
      void flushQueue();
      return;
    }
    if (
      flushTimerId !== null
      || !windowRef
      || typeof windowRef.setTimeout !== "function"
    ) {
      return;
    }
    flushTimerId = windowRef.setTimeout(() => {
      flushTimerId = null;
      void flushQueue();
    }, FLUSH_DELAY_MS);
  }

  function enqueueEvent(level, message, context) {
    if (
      !resolvedRuntimeConfig.persist
      || !resolvedRuntimeConfig.transportEndpoint
    ) {
      return;
    }
    const event = createQueuedEvent(level, message, context, now());
    if (queuedEvents.length < MAX_QUEUED_EVENTS) {
      queuedEvents.push(event);
    } else {
      droppedEventCount += 1;
    }
    scheduleFlush();
  }

  function syncPagehideListener() {
    if (!windowRef || typeof windowRef.addEventListener !== "function") {
      return;
    }
    if (pagehideHandler !== null) {
      if (typeof windowRef.removeEventListener === "function") {
        windowRef.removeEventListener("pagehide", pagehideHandler);
      }
      pagehideHandler = null;
    }
    if (
      !resolvedRuntimeConfig.persist
      || !resolvedRuntimeConfig.transportEndpoint
    ) {
      return;
    }
    pagehideHandler = () => {
      void flushQueue({ preferBeacon: true });
    };
    windowRef.addEventListener("pagehide", pagehideHandler);
  }

  function emit(level, message, context = {}) {
    if (!shouldEmit(level)) {
      return;
    }
    const mergedContext = {
      session: resolvedRuntimeConfig.sessionId,
      ...normalizeContext(context),
    };
    const dispatcher =
      dispatchers[level]
      || dispatchers.debug
      || dispatchers.info
      || dispatchers.warning
      || dispatchers.error;
    if (dispatcher) {
      dispatcher(formatLogMessage(message, mergedContext));
    }
    enqueueEvent(level, message, mergedContext);
  }

  function buildRequestTracker(requestContext = {}) {
    if (!resolvedRuntimeConfig.enabled) {
      return {
        requestId: requestContext.request_id || null,
        finish() {},
        fail() {},
      };
    }
    emit("debug", "API request started", requestContext);
    const startedAt = now();
    let settled = false;
    return {
      requestId: requestContext.request_id || null,
      finish(finishContext = {}) {
        if (settled) {
          return;
        }
        settled = true;
        emit("debug", "API request finished", {
          ...requestContext,
          ...finishContext,
          outcome: finishContext.outcome || "success",
          elapsed_ms:
            finishContext.elapsed_ms || elapsedMsText(startedAt, now()),
        });
      },
      fail(error, failureContext = {}) {
        if (settled) {
          return;
        }
        settled = true;
        const errorMessage =
          error && typeof error.message === "string" ? error.message : String(error);
        emit("warning", `API request failed: ${errorMessage}`, {
          ...requestContext,
          ...failureContext,
          outcome: failureContext.outcome || "error",
          elapsed_ms:
            failureContext.elapsed_ms || elapsedMsText(startedAt, now()),
        });
      },
    };
  }

  syncPagehideListener();

  return {
    isEnabled() {
      return resolvedRuntimeConfig.enabled;
    },
    getConfig() {
      return { ...resolvedRuntimeConfig };
    },
    refreshRuntimeConfig(nextRuntimeConfig = {}) {
      resolvedRuntimeConfig = normalizeRuntimeConfig({
        apiToken:
          typeof nextRuntimeConfig.apiToken === "string"
            ? nextRuntimeConfig.apiToken
            : typeof nextRuntimeConfig.api_token === "string"
              ? nextRuntimeConfig.api_token
              : resolvedRuntimeConfig.apiToken,
        sessionId:
          typeof nextRuntimeConfig.sessionId === "string"
            ? nextRuntimeConfig.sessionId
            : typeof nextRuntimeConfig.session_id === "string"
              ? nextRuntimeConfig.session_id
              : resolvedRuntimeConfig.sessionId,
        enabled:
          typeof nextRuntimeConfig.enabled === "boolean"
            ? nextRuntimeConfig.enabled
            : resolvedRuntimeConfig.enabled,
        level:
          typeof nextRuntimeConfig.level === "string"
            ? nextRuntimeConfig.level
            : resolvedRuntimeConfig.level,
        persist:
          typeof nextRuntimeConfig.persist === "boolean"
            ? nextRuntimeConfig.persist
            : resolvedRuntimeConfig.persist,
        transportEndpoint:
          typeof nextRuntimeConfig.transportEndpoint === "string"
            ? nextRuntimeConfig.transportEndpoint
            : typeof nextRuntimeConfig.transport_endpoint === "string"
              ? nextRuntimeConfig.transport_endpoint
              : resolvedRuntimeConfig.transportEndpoint,
      });
      syncPagehideListener();
      return { ...resolvedRuntimeConfig };
    },
    debug(message, context = {}) {
      emit("debug", message, context);
    },
    info(message, context = {}) {
      emit("info", message, context);
    },
    warn(message, context = {}) {
      emit("warning", message, context);
    },
    error(message, context = {}) {
      emit("error", message, context);
    },
    flush(options = {}) {
      return flushQueue(options);
    },
    startOperation(name, context = {}) {
      if (!resolvedRuntimeConfig.enabled) {
        return NOOP_OPERATION;
      }
      emit("debug", `${name} started`, context);
      const startedAt = now();
      return {
        branch(message, branchContext = {}) {
          emit("debug", message, {
            ...context,
            ...branchContext,
          });
        },
        finish(finishContext = {}) {
          emit("debug", `${name} finished`, {
            ...context,
            ...finishContext,
            outcome: finishContext.outcome || "success",
            elapsed_ms: elapsedMsText(startedAt, now()),
          });
        },
        fail(error, failureContext = {}) {
          const errorMessage =
            error && typeof error.message === "string"
              ? error.message
              : String(error);
          emit("warning", `${name} failed: ${errorMessage}`, {
            ...context,
            ...failureContext,
            outcome: failureContext.outcome || "error",
            elapsed_ms: elapsedMsText(startedAt, now()),
          });
        },
      };
    },
    startRequest(method, route, options = {}) {
      return buildRequestTracker({
        operation: options.operation || null,
        method,
        route,
        request_id: `req-${nextRequestNumber++}`,
        ...(options.context && typeof options.context === "object"
          ? options.context
          : {}),
      });
    },
  };
}
