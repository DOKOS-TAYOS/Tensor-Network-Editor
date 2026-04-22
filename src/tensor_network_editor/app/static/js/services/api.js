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

function requireJsonBody({ json }) {
  if (json === null) {
    throw new Error("Expected a JSON response.");
  }
  return json;
}

export async function apiGet(path) {
  const response = await fetch(path);
  const body = await readResponseBody(response);
  if (!response.ok) {
    throw new Error(buildErrorMessage(body));
  }
  return requireJsonBody(body);
}

export async function apiPost(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await readResponseBody(response);
  if (!response.ok) {
    throw new Error(buildErrorMessage(body));
  }
  return requireJsonBody(body);
}
