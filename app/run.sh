#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_ACTIVATE="$SCRIPT_DIR/../.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "venv not found at: $VENV_ACTIVATE"
  echo "Create it with: python3.12 -m venv .venv"
  exit 1
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 8001
