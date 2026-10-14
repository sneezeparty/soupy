# Vendored front-end libraries

The panel serves these itself so it loads on a machine with no internet access.
They are unmodified release builds, MIT licensed, with the license header kept
at the top of each file.

| File | Version | Source |
|------|---------|--------|
| `react.production.min.js` | React 18.3.1 | `https://unpkg.com/react@18.3.1/umd/react.production.min.js` |
| `react-dom.production.min.js` | ReactDOM 18.3.1 | `https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js` |
| `pico.min.css` | Pico CSS 2.1.1 | `https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css` |

To update one, download the new build over the old file and change the version
here. `web/templates/layout.html` loads Pico; `web/templates/dashboard.html`
loads React.
