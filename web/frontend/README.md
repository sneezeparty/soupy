# Soupy Dashboard — Vite + React scaffold

This directory contains a fresh Vite + JSX build pipeline for the dashboard.
**It is not the canonical UI yet.** The canonical dashboard is still the five
script files under [`web/static/dashboard/`](../static/dashboard/)
(`dashboard-core.js`, `dashboard-media.js`, `dashboard-database.js`,
`dashboard-model.js`, `dashboard-app.js`): about 5,700 lines of `React.createElement(...)` calls served
directly by FastAPI, with no build step. They are the supported UI and change on
their own schedule; this scaffold does not track them.

The plan is to port one tab at a time into this app. This milestone (M6a)
landed only the build pipeline plus a single proof-of-concept Status card so a
future session can keep going.

## Setup

```bash
cd web/frontend
npm install        # one-time
```

## Develop

```bash
npm run dev
```

Vite serves on its own port (default `5173`) with hot-module reload. The dev
shell uses `index.html` in this folder, which mounts the React app into
`<div id="react-root-v2"></div>`. CSS variables come from the FastAPI template
in production; in dev you get a placeholder palette from
`public/dev-shell.css`, so colors will look approximate.

## Build

```bash
npm run build
```

Vite emits a hashed bundle into
[`web/static/dashboard/dist/`](../static/dashboard/dist/). FastAPI already
serves `web/static/` at `/static/`, so the bundle becomes available at
`/static/dashboard/dist/assets/dashboard-<hash>.js`.

The bundle is **not** wired into `web/templates/dashboard.html` yet — that
swap will happen in a later milestone, tab by tab.

## Mount-node id

This app mounts to `#react-root-v2`. The legacy `dashboard-app.js` mounts to
`#dashboard-root`. The two ids are intentionally distinct so the legacy and
new apps can coexist on the same page during the migration.

## API conventions

`src/api.js` mirrors the JSON conventions used throughout `dashboard-app.js`:
relative URLs, throw on non-OK responses, prefer `data.message`/`data.error`
for error text. Components ported from the legacy file should drop in cleanly.

## What's here

```
web/frontend/
├── package.json        # vite, @vitejs/plugin-react, react@18, react-dom@18
├── vite.config.js      # base = /static/dashboard/dist/, single bundle, no maps
├── index.html          # dev-mode host shell only
├── public/
│   └── dev-shell.css   # placeholder colors for dev mode
└── src/
    ├── main.jsx        # createRoot → App, mounts to #react-root-v2
    ├── App.jsx         # proof-of-concept Status card
    └── api.js          # readJson / postJson helpers
```

## Constraints

- The served dashboard scripts and `dashboard.css` are read-only for this
  scaffold. Do not edit them from here.
- `web/app.py` and `web/templates/` are also read-only — the FastAPI template
  still loads the legacy bundle.
- Use existing CSS class names from `dashboard.css` (e.g. `dash-sticky-bar`,
  `status-dot`, `status-on`, `muted`, `mono`) so ported components look like
  their legacy equivalents once rendered inside the real page.
