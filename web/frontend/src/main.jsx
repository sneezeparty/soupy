import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

// Mount-node id is intentionally `react-root-v2` — distinct from the legacy
// `dashboard-root` used by web/static/dashboard/dashboard-app.js. Keeping the
// ids separate lets both scripts coexist on the same page during the
// piece-by-piece port. When this app supersedes the legacy UI, the FastAPI
// template can swap to `react-root-v2` (or the legacy id can be reused once
// dashboard-app.js is retired).
const rootEl = document.getElementById("react-root-v2");

if (!rootEl) {
  // Fail loudly in dev; in production the FastAPI template guarantees the node.
  // eslint-disable-next-line no-console
  console.error(
    "[soupy-dashboard] Mount node #react-root-v2 not found. " +
      "Add <div id=\"react-root-v2\"></div> to the host page."
  );
} else {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
