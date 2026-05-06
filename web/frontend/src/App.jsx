import React, { useCallback, useEffect, useState } from "react";
import { readJson } from "./api.js";

/**
 * Proof-of-concept Status / Overview card.
 *
 * Pulls /api/bot/status (defined in web/app.py and backed by
 * web/services/bot_runner.py::BotRunner.status). Field shape:
 *
 *   {
 *     "running":     bool,
 *     "pid":         int | null,
 *     "start_time":  ISO-8601 string | null,
 *     "returncode":  int | null,
 *   }
 *
 * Visual structure intentionally reuses class names from the legacy
 * dashboard-app.js (dash-sticky-bar / dash-sticky-segment / status-dot /
 * status-on / status-off / muted / mono) so the FastAPI-served
 * dashboard.css applies as-is when this bundle runs inside the real page.
 */
export default function App() {
  const [status, setStatus] = useState({
    running: false,
    pid: null,
    start_time: null,
    returncode: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await readJson("/api/bot/status");
      setStatus({
        running: !!data.running,
        pid: data.pid ?? null,
        start_time: data.start_time ?? null,
        returncode: data.returncode ?? null,
      });
    } catch (err) {
      setError(err.message || "Failed to load /api/bot/status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    // Light polling so the proof-of-concept feels alive. Matches the legacy
    // refresh cadence loosely; the legacy app refreshes on user action and
    // via several other triggers.
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <article style={{ maxWidth: 960, margin: "1rem auto" }}>
      <h4 style={{ marginTop: 0 }}>Bot Status (Vite scaffold)</h4>
      <div
        className="dash-sticky-bar"
        role="status"
        aria-label="Bot status"
        style={{ marginBottom: "0.75rem" }}
      >
        <div className="dash-sticky-segment">
          <span
            className={
              "status-dot " + (status.running ? "status-on" : "status-off")
            }
          />
          <strong>{status.running ? "Running" : "Stopped"}</strong>
          {status.pid != null ? (
            <span
              className="muted mono"
              style={{ fontSize: "0.78rem" }}
            >
              {"PID " + status.pid}
            </span>
          ) : null}
        </div>

        {status.start_time ? (
          <div className="dash-sticky-segment">
            <span className="muted">Started</span>
            <span className="mono" style={{ fontSize: "0.85rem" }}>
              {status.start_time}
            </span>
          </div>
        ) : null}

        {status.returncode != null ? (
          <div className="dash-sticky-segment">
            <span className="muted">Exit</span>
            <span className="mono" style={{ fontSize: "0.85rem" }}>
              {String(status.returncode)}
            </span>
          </div>
        ) : null}

        <div className="dash-sticky-spacer" />

        <div className="dash-sticky-actions">
          <button
            type="button"
            className="secondary"
            onClick={refresh}
            disabled={loading}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error ? (
        <p className="muted" style={{ color: "var(--status-off)" }}>
          {error}
        </p>
      ) : null}

      <p className="muted" style={{ fontSize: "0.8rem" }}>
        This component is the Vite + JSX proof-of-concept. The canonical
        dashboard is still served by{" "}
        <code>web/static/dashboard/dashboard-app.js</code>.
      </p>
    </article>
  );
}
