/**
 * Tiny fetch helper. Mirrors the JSON conventions used by the legacy
 * dashboard-app.js so components ported here behave the same way:
 *
 *   - GET-by-default. Pass `init` to override.
 *   - Parses JSON.
 *   - Throws an Error whose message is `data.message` or `data.error` when
 *     the response is non-OK; otherwise the HTTP status text.
 *
 * Endpoints expected to be served by web/app.py at the same origin as the
 * dashboard, so relative URLs ("/api/bot/status") are correct.
 */
export async function readJson(url, init) {
  const res = await fetch(url, init);
  let data = null;
  try {
    data = await res.json();
  } catch (_err) {
    // Non-JSON body — leave data null and fall through to status handling.
  }

  if (!res.ok) {
    const message =
      (data && (data.message || data.error)) ||
      res.statusText ||
      "Request failed";
    throw new Error(message);
  }

  return data;
}

/** Convenience for POSTs with an optional JSON body. */
export async function postJson(url, body) {
  return readJson(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
}
