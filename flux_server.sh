#!/usr/bin/env bash
# flux_server.sh — launch the local Flux (mflux) server with a chosen model.
#
# Usage:
#   ./flux_server.sh            # use FLUX_MODEL from .env-stable (the default)
#   ./flux_server.sh schnell    # FLUX.1 schnell  (gated on HF, guidance ~0.0)
#   ./flux_server.sh klein      # FLUX.2 klein-4B (ungated, guidance ~1.0)
#   ./flux_server.sh dev        # FLUX.1 dev      (gated)
#   ./flux_server.sh klein-9b   # FLUX.2 klein-9B (gated, ~20GB — likely too big)
#
# The model arg just sets FLUX_MODEL for this run, overriding .env-stable. Weights
# are cached under ~/.cache/huggingface after the first download, so switching back
# is fast. Only one model fits in RAM at a time — Ctrl-C this, then relaunch with a
# different arg to switch. (FLUX_GUIDANCE/FLUX_STEPS are bot-side; change those in
# .env-stable + restart the bot if you want them tuned per model.)

set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-}"
if [[ -n "$MODEL" ]]; then
  export FLUX_MODEL="$MODEL"
fi

# Prefer the project venv's python (has mflux); fall back to python3.
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

# Port from .env-stable (default 4942) — warn early if it's already bound.
PORT="$(grep -E '^FLUX_SERVER_PORT=' .env-stable 2>/dev/null | cut -d= -f2 || true)"
PORT="${PORT:-4942}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "⚠️  Port $PORT is already in use — another flux server is probably running."
  echo "    Stop it first (Ctrl-C in its terminal). It's currently serving:"
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "    (couldn't read /health)"
  echo
  exit 1
fi

echo "⚡ Launching Flux server  (model: ${FLUX_MODEL:-<from .env-stable>}, port: $PORT)"
exec "$PY" flux_server.py
