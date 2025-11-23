#!/bin/bash
# run_generate_motivation.sh
# safe wrapper for cron that activates venv, loads .env, runs the management command,
# and appends timestamped logs.

set -o pipefail

#PROJECT_DIR="/home/diegogalindo/my_stuff/01_Projects/antibes_webapp"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACT="$PROJECT_DIR/.venv/bin/activate"
ENVFILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/generate_motivation.log"

mkdir -p "$LOG_DIR"

# Load environment variables in .env (if present) and export them
if [ -f "$ENVFILE" ]; then
  # export all variables defined in .env
  set -a
  # shellcheck disable=SC1090
  source "$ENVFILE"
  set +a
fi

# Activate virtualenv
# shellcheck disable=SC1091
if [ -f "$VENV_ACT" ]; then
  # Use `.` instead of `source` for POSIX compatibility
  . "$VENV_ACT"
else
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR: virtualenv activate not found at $VENV_ACT" >> "$LOG_FILE"
  exit 1
fi

# Run the management command and log outcome
{
  echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') START generate_motivation ==="
  if python "$PROJECT_DIR/manage.py" generate_motivation ; then
    echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') SUCCESS generate_motivation ==="
  else
    echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') FAILURE generate_motivation (exit code $? ) ==="
  fi
  echo ""
} >> "$LOG_FILE" 2>&1
