#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; always execute from repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate venv if present (adjust if your venv path differs)
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

# Load environment variables (Django + Schwab + VOL_DATA_DIR)
# Note: the management command also loads REPO_ROOT/.env, but this helps cron too.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Defaults (override by exporting env vars or passing args)
RUN_LABEL="${RUN_LABEL:-daily_close}"
TARGET_DTES="${TARGET_DTES:-30 45}"
STRIKE_COUNT="${STRIKE_COUNT:-18}"
STRICT_FLAG="${STRICT_FLAG:-}"   # set STRICT_FLAG="--strict" if you want strict mode
SYMBOL="${SYMBOL:-}"             # set SYMBOL="SPY" for debug

CMD=(python manage.py update_vol_snapshots
  --run-label "$RUN_LABEL"
  --target-dtes $TARGET_DTES
  --strike-count "$STRIKE_COUNT"
)

if [[ -n "$STRICT_FLAG" ]]; then
  CMD+=("$STRICT_FLAG")
fi

if [[ -n "$SYMBOL" ]]; then
  CMD+=(--symbol "$SYMBOL")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
